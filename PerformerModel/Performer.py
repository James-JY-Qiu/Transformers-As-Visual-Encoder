from functools import partial

import torch
from torch import nn

from PerformerModel.Attention import SelfAttention, FastAttention
from PerformerModel.performer_utils import cast_tuple, default, shift, exists, find_modules, get_module_device, \
    route_args
from PerformerModel.Norm import ReZero, PreScaleNorm, PreLayerNorm


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        """A neural network module that splits the input tensor into multiple chunks
        and applies the function or module to each chunk.

        :param chunks: Number of chunks to split the input tensor into.
        :param fn:  The function or module to apply to each chunk
        :param along_dim: The dimension along which to split the tensor
        """
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        # split the input tensor into chunks
        chunks = x.chunk(self.chunks, dim=self.dim)
        # apply the function or module to each chunk
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        """A neural network module that applies a feedforward neural network to the input tensor

        :param dim: The input and output dimension of the input tensor
        :param mult: The multiplier for the intermediate dimension of the feedforward neural network
        :param dropout: The dropout probability
        :param activation: The activation function
        :param glu: Whether to use Gated Linear Units (GLU) activation function
        """
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        """A neural network module that shifts the tokens of the input tensor

        :param shifts: A tuple of integers representing the shifts to apply to each segment of the input tensor
        :param fn: The function or module to apply to the shifted input tensor"""
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        """Shift the tokens of the input tensor and apply the function or module to the shifted tensor"""
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        """A neural network module that updates the projection matrices of the attention layers

        :param instance: The instance of the model
        :param feature_redraw_interval: The interval at which to redraw the projection matrices of the attention layers
        """
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projections_(self):
        self.feature_redraw_interval = None

    def redraw_projections(self):
        model = self.instance

        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)
            fast_attentions = find_modules(model, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)
            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x):
        raise NotImplemented


class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route={}):
        """A neural network module that applies a sequence of layers to the input tensor

        :param layers: A list of layers to apply to the input tensor
        """
        super().__init__()
        assert all(len(route) == len(layers) for route in
                   args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


class Performer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads=0,
        local_window_size=256,
        causal=False,
        ff_mult=4,
        nb_features=None,
        feature_redraw_interval=1000,
        ff_chunks=1,
        generalized_attention=False,
        kernel_fn=nn.ReLU(),
        use_scalenorm=False,
        use_rezero=False,
        ff_glu=False,
        ff_dropout=0.,
        attn_dropout=0.,
        no_projection=False,
        auto_check_redraw=True,
        qkv_bias=True,
        attn_out_bias=True,
        shift_tokens=False,
        use_standard_transformer=False
    ):
        """Performer Model

        :param dim: Dimension of the input tensor
        :param depth: The depth of the model
        :param heads: The number of attention heads
        :param dim_head: The dimension of each attention head
        :param local_attn_heads: The number of local attention heads
        :param local_window_size: The size of the local window
        :param causal: Whether to use causal attention
        :param ff_mult: The multiplier for the intermediate dimension of the feedforward neural network
        :param nb_features: The number of random features for FastAttention
        :param feature_redraw_interval: The interval at which to redraw the projection matrices of the attention layers
        :param ff_chunks: The number of chunks to split the input tensor into
        :param generalized_attention: Whether to use generalized attention
        :param kernel_fn: The kernel function
        :param use_scalenorm: Whether to use ScaleNorm
        :param use_rezero: Whether to use ReZero
        :param ff_glu: Whether to use Gated Linear Units (GLU) activation function
        :param ff_dropout: The dropout probability for the feedforward neural network
        :param attn_dropout: The dropout probability for the attention layers
        :param no_projection: Whether to use projection matrices
        :param auto_check_redraw: Whether to automatically check and redraw the projection matrices
        :param qkv_bias: Whether to include bias to the query, key, and value matrices
        :param attn_out_bias: Whether to include bias to the output of the attention layer
        :param shift_tokens: Whether to shift the tokens of the input tensor
        :param use_standard_transformer: Whether to use the standard transformer
        """
        super().__init__()
        layers = nn.ModuleList([])
        # local attention heads per depth
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(
            local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads,
                       local_attn_heads)), 'local attention head value must be less than the total number of heads'

        # wrapper function for normalization
        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            # attention block
            attn = SelfAttention(
                dim,
                causal=causal,
                heads=heads,
                dim_head=dim_head,
                local_heads=local_heads,
                local_window_size=local_window_size,
                nb_features=nb_features,
                generalized_attention=generalized_attention,
                kernel_fn=kernel_fn,
                dropout=attn_dropout,
                no_projection=no_projection,
                qkv_bias=qkv_bias,
                attn_out_bias=attn_out_bias,
                use_standard_transformer=use_standard_transformer
            )
            # feedforward block
            ff = Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            attn, ff = map(wrapper_fn, (attn, ff))
            layers.append(nn.ModuleList([attn, ff]))

        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        self.net = SequentialSequence(layers, args_route={**attn_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)
