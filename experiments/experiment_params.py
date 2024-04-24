# ================= Performer parameters =================
from copy import deepcopy

from torch import nn

MNIST_PERFORMER = {
    'dataset_name': 'mnist',
    'patch_size': (4, 4),
    'linear_embedding_dim': 32,
    'performer_params': {
        'dim': 32,
        'depth': 2,
        'heads': 4,
        'dim_head': 16,
        'ff_dropout': 0.3,
    },
    'mlp_params': {
        'hidden_dim': 3072,
        'dropout_rate': 0.5
    }
}

CIFAR_PERFORMER = {
    'dataset_name': 'cifar',
    'patch_size': (4, 4),
    'linear_embedding_dim': 96,
    'performer_params': {
        'dim': 96,
        'depth': 4,
        'heads': 8,
        'dim_head': 32,
        'ff_dropout': 0.1,
    },
    'mlp_params': {
        'hidden_dim': 12288,
        'dropout_rate': 0.1
    }
}

MNIST_PERFORMER_RELU = deepcopy(MNIST_PERFORMER)
MNIST_PERFORMER_RELU['performer_params']['generalized_attention'] = True
MNIST_PERFORMER_RELU['performer_params']['kernel_fn'] = nn.ReLU()

CIFAR_PERFORMER_RELU = deepcopy(CIFAR_PERFORMER)
CIFAR_PERFORMER_RELU['performer_params']['generalized_attention'] = True
CIFAR_PERFORMER_RELU['performer_params']['kernel_fn'] = nn.ReLU()

MNIST_PERFORMER_APPROXIMATION_16 = deepcopy(MNIST_PERFORMER)
MNIST_PERFORMER_APPROXIMATION_16['performer_params']['generalized_attention'] = False
MNIST_PERFORMER_APPROXIMATION_16['performer_params']['nb_features'] = 16

CIFAR_PERFORMER_APPROXIMATION_16 = deepcopy(CIFAR_PERFORMER)
CIFAR_PERFORMER_APPROXIMATION_16['performer_params']['generalized_attention'] = False
CIFAR_PERFORMER_APPROXIMATION_16['performer_params']['nb_features'] = 16

MNIST_PERFORMER_APPROXIMATION_32 = deepcopy(MNIST_PERFORMER)
MNIST_PERFORMER_APPROXIMATION_32['performer_params']['generalized_attention'] = False
MNIST_PERFORMER_APPROXIMATION_32['performer_params']['nb_features'] = 32

CIFAR_PERFORMER_APPROXIMATION_32 = deepcopy(CIFAR_PERFORMER)
CIFAR_PERFORMER_APPROXIMATION_32['performer_params']['generalized_attention'] = False
CIFAR_PERFORMER_APPROXIMATION_32['performer_params']['nb_features'] = 32

MNIST_PERFORMER_APPROXIMATION_64 = deepcopy(MNIST_PERFORMER)
MNIST_PERFORMER_APPROXIMATION_64['performer_params']['generalized_attention'] = False
MNIST_PERFORMER_APPROXIMATION_64['performer_params']['nb_features'] = 64

CIFAR_PERFORMER_APPROXIMATION_64 = deepcopy(CIFAR_PERFORMER)
CIFAR_PERFORMER_APPROXIMATION_64['performer_params']['generalized_attention'] = False
CIFAR_PERFORMER_APPROXIMATION_64['performer_params']['nb_features'] = 64

MNIST_PERFORMER_APPROXIMATION_128 = deepcopy(MNIST_PERFORMER)
MNIST_PERFORMER_APPROXIMATION_128['performer_params']['generalized_attention'] = False
MNIST_PERFORMER_APPROXIMATION_128['performer_params']['nb_features'] = 128

CIFAR_PERFORMER_APPROXIMATION_128 = deepcopy(CIFAR_PERFORMER)
CIFAR_PERFORMER_APPROXIMATION_128['performer_params']['generalized_attention'] = False
CIFAR_PERFORMER_APPROXIMATION_128['performer_params']['nb_features'] = 128

MNIST_TRANSFORMER_LOCAL_ATTN_4 = deepcopy(MNIST_PERFORMER)
MNIST_TRANSFORMER_LOCAL_ATTN_4['performer_params']['use_standard_transformer'] = True
MNIST_TRANSFORMER_LOCAL_ATTN_4['performer_params']['local_attn_heads'] = 2
MNIST_TRANSFORMER_LOCAL_ATTN_4['performer_params']['local_window_size'] = 4

MNIST_TRANSFORMER_LOCAL_ATTN_16 = deepcopy(MNIST_PERFORMER)
MNIST_TRANSFORMER_LOCAL_ATTN_16['performer_params']['use_standard_transformer'] = True
MNIST_TRANSFORMER_LOCAL_ATTN_16['performer_params']['local_attn_heads'] = 2
MNIST_TRANSFORMER_LOCAL_ATTN_16['performer_params']['local_window_size'] = 16

CIFAR_PERFORMER_LOCAL_ATTN_4 = deepcopy(CIFAR_PERFORMER)
CIFAR_PERFORMER_LOCAL_ATTN_4['performer_params']['use_standard_transformer'] = True
CIFAR_PERFORMER_LOCAL_ATTN_4['performer_params']['local_attn_heads'] = 4
CIFAR_PERFORMER_LOCAL_ATTN_4['performer_params']['local_window_size'] = 4

CIFAR_PERFORMER_LOCAL_ATTN_16 = deepcopy(CIFAR_PERFORMER)
CIFAR_PERFORMER_LOCAL_ATTN_16['performer_params']['use_standard_transformer'] = True
CIFAR_PERFORMER_LOCAL_ATTN_16['performer_params']['local_attn_heads'] = 4
CIFAR_PERFORMER_LOCAL_ATTN_16['performer_params']['local_window_size'] = 16
