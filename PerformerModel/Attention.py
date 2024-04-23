import math
from functools import partial

import torch
from einops import rearrange
from torch import nn
from local_attention import LocalAttention

from PerformerModel.kernel import generalized_kernel, softmax_kernel
from PerformerModel.performer_utils import default, exists, gaussian_orthogonal_random_matrix, empty, apply_rotary_pos_emb


def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


class FastAttention(nn.Module):
    def __init__(
            self,
            dim_heads,
            nb_features=None,
            ortho_scaling=0,
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            no_projection=False
    ):
        super().__init__()
        # number of features
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling

        self.create_projection = partial(
            gaussian_orthogonal_random_matrix,
            nb_rows=self.nb_features,
            nb_columns=dim_heads,
            scaling=ortho_scaling
        )
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        device = q.device

        if self.no_projection:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        out = linear_attention(q, k, v)
        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        heads=8,
        dim_head=64,
        local_heads=0,
        local_window_size=256,
        nb_features=None,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        dropout=0.,
        no_projection=False,
        qkv_bias=False,
        attn_out_bias=True,
        use_standard_transformer=False
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.use_standard_transformer = use_standard_transformer
        if not use_standard_transformer:
            self.fast_attention = FastAttention(
                dim_head,
                nb_features,
                generalized_attention=generalized_attention,
                kernel_fn=kernel_fn,
                no_projection=no_projection
            )

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, **kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if self.use_standard_transformer:
                scale = self.dim_head ** -0.5
                q = q * scale
                attn_scores = torch.einsum('b h i d, b h j d -> b h i j', q, k)
                if exists(mask):
                    mask = mask[:, None, None, :]
                    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_probs = torch.softmax(attn_scores, dim=-1)
                out = torch.einsum('b h i j, b h j d -> b h i d', attn_probs, v)
                attn_outs.append(out)
            else:
                if exists(context_mask):
                    global_mask = context_mask[:, None, :, None]
                    v.masked_fill_(~global_mask, 0.)

                if exists(pos_emb) and not cross_attend:
                    q, k = apply_rotary_pos_emb(q, k, pos_emb)

                out = self.fast_attention(q, k, v)
                attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)


class SelfAttention(Attention):
    def forward(self, *args, context = None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)
