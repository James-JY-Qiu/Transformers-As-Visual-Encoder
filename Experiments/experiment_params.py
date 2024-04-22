# ================= Performer parameters =================
MNIST_PERFORMER = {
    'dataset_name': 'mnist',
    'patch_size': (4, 4),
    'linear_embedding_dim': 64,
    'performer_params': {
        'dim': 64,
        'depth': 1,
        'heads': 4,
        'dim_head': 16,
        'ff_dropout': 0.5
    }
}