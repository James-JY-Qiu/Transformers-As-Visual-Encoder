import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def cast_tuple(val):
    """cast int to tuple if not already a tuple"""
    return (val,) if not isinstance(val, tuple) else val


def exists(val):
    """check if val is not None"""
    return val is not None


def default(val, d):
    """return d if val is None else return val"""
    return val if exists(val) else d


def empty(tensor):
    """check if tensor is empty"""
    return tensor.numel() == 0


def orthogonal_matrix_chunk(cols, device = None):
    """generate a random orthogonal matrix chunk"""
    # generate a random matrix
    unstructured_block = torch.randn((cols, cols), device=device)
    # qr decomposition
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    """generate a gaussian orthogonal random matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        # Generate a full orthogonal block
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    # Concatenate all blocks into the final matrix
    final_matrix = torch.cat(block_list)

    # Apply scaling to the orthogonal matrix based on the specified method
    if scaling == 0:
        # No scaling, normalize each row to unit length
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        # Scaling by the square root of the number of columns
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    # Scale the final matrix
    return torch.diag(multiplier) @ final_matrix


def rotate_every_two(x):
    """rotate every two elements in the last dimension of a tensor"""
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(q, k, sinu_pos):
    """apply rotary positional embedding to q and k tensors"""
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


def shift(t, amount, mask = None):
    """shift the elements of a tensor by a specified amount in the last dimension"""
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)


def find_modules(nn_module, type):
    """find all modules of a certain type in a nn.Module"""
    return [module for module in nn_module.modules() if isinstance(module, type)]


def get_module_device(module):
    """get the device of a module"""
    return next(module.parameters()).device


def route_args(router, args, depth):
    """Helper function to route arguments to the correct sublayer in a sequential model."""
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            # if route is None, then the argument is meant for the current layer
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            # update the argument dictionaries
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args