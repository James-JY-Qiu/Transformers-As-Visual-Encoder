import torch
from torch import nn
from einops import repeat


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4):
    """Apply the softmax kernel to the data

    :param data: The query or key tensor to be transformed
    :param projection_matrix: A random matrix used for projecting the input tensor to a lower-dimensional space
    :param is_query: Flag indicating whether the input data is a query
    :param normalize_data: If True, normalizes the data by the feature dimension
    :param eps: A small number to add to the denominator for numerical stability
    :return: The transformed data after applying the kernel function
    """
    b, h, *_ = data.shape

    # Normalization factor to scale the data if normalization is enabled
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    # Scaling factor for the softmax
    ratio = (projection_matrix.shape[0] ** -0.5)

    # Repeat and cast the projection matrix to match the batch and head dimensions of the data
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)

    # Project the normalized data into a lower-dimensional space using a random matrix
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    # Compute the squared sum of the data for normalization in the softmax-like operation
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    # Apply softmax-like function differently for queries and keys
    if is_query:
        # For queries, normalize across the last dimension
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        # For keys, normalize across the last two dimensions
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    """Apply the generalized attention kernel to the data

    :param data: The query or key tensor to be transformed
    :param projection_matrix: A random matrix used for projecting the input tensor to a lower-dimensional space
    :param kernel_fn: The kernel function to be applied
    :param kernel_epsilon: A small number added to the kernel function
    :param normalize_data: If True, normalizes the data by the feature dimension
    :param device: The device to run the computation on
    :return: The transformed data after applying the kernel function
    """
    b, h, *_ = data.shape

    # Normalization factor to scale the data if normalization is enabled
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    # Apply the kernel function to the data
    if projection_matrix is None:
        # If no projection matrix is provided, apply the kernel function directly to the data
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    # Repeat and cast the projection matrix to match the batch and head dimensions of the data
    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    # Project the normalized data into a lower-dimensional space using the projection matrix
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    # Apply the kernel function to the projected data
    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)