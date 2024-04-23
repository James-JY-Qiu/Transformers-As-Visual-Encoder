# ================= Performer parameters =================
from torch import nn

MNIST_PERFORMER_RELU = {
    'dataset_name': 'mnist',
    'patch_size': (4, 4),
    'linear_embedding_dim': 32,
    'performer_params': {
        'dim': 32,
        'depth': 2,
        'heads': 4,
        'dim_head': 16,
        'ff_dropout': 0.1,
        'generalized_attention': True,
        'kernel_fn': nn.ReLU(),
    },
    'mlp_params': {
        'hidden_dim': 3072,
        'dropout_rate': 0.1
    }
}

CIFAR_PERFORMER_RELU = {
    'dataset_name': 'cifar',
    'patch_size': (4, 4),
    'linear_embedding_dim': 96,
    'performer_params': {
        'dim': 96,
        'depth': 4,
        'heads': 8,
        'dim_head': 32,
        'ff_dropout': 0.1,
        'generalized_attention': True,
        'kernel_fn': nn.ReLU(),
    },
    'mlp_params': {
        'hidden_dim': 12288,
        'dropout_rate': 0.1
    }
}