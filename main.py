from experiments.run_experiments import run_performer_experiment
from experiments.experiment_params import MNIST_PERFORMER_RELU, MNIST_PERFORMER_APPROXIMATION_16, MNIST_PERFORMER_APPROXIMATION_32, MNIST_PERFORMER_APPROXIMATION_64, MNIST_PERFORMER_APPROXIMATION_128, MNIST_TRANSFORMER_LOCAL_ATTN_4, MNIST_TRANSFORMER_LOCAL_ATTN_16
from experiments.experiment_params import CIFAR_PERFORMER_RELU, CIFAR_PERFORMER_APPROXIMATION_16, CIFAR_PERFORMER_APPROXIMATION_32, CIFAR_PERFORMER_APPROXIMATION_64, CIFAR_PERFORMER_APPROXIMATION_128, CIFAR_PERFORMER_LOCAL_ATTN_4, CIFAR_PERFORMER_LOCAL_ATTN_16, CIFAR_PERFORMER_LOCAL_ATTN_32

import os


if __name__ == '__main__':
    for params, name in [
        (MNIST_PERFORMER_RELU, 'MNIST_PERFORMER_RELU'),
        (MNIST_PERFORMER_APPROXIMATION_16, 'MNIST_PERFORMER_APPROXIMATION_16'),
        (MNIST_PERFORMER_APPROXIMATION_32, 'MNIST_PERFORMER_APPROXIMATION_32'),
        (MNIST_PERFORMER_APPROXIMATION_64, 'MNIST_PERFORMER_APPROXIMATION_64'),
        (MNIST_PERFORMER_APPROXIMATION_128, 'MNIST_PERFORMER_APPROXIMATION_128'),
        (MNIST_TRANSFORMER_LOCAL_ATTN_4, 'MNIST_TRANSFORMER_LOCAL_ATTN_4'),
        (MNIST_TRANSFORMER_LOCAL_ATTN_16, 'MNIST_TRANSFORMER_LOCAL_ATTN_16')
    ]:
        for repeat in range(10):
            subname = name + "_" + str(repeat)
            print(f"Running {subname} experiment")
            # bypass existing results
            if os.path.exists(f'results/{subname}.pkl'):
                print(f"Skipping {subname} experiment")
                continue
            run_performer_experiment(
                task='mnist',
                num_epochs=20,
                batch_size=32,
                learning_rate=1e-4,
                performer_encoder_params=params,
                experiment_name=subname,
                save_path='results'
            )

    for params, name in [
        (CIFAR_PERFORMER_RELU, 'CIFAR_PERFORMER_RELU'),
        (CIFAR_PERFORMER_APPROXIMATION_16, 'CIFAR_PERFORMER_APPROXIMATION_16'),
        (CIFAR_PERFORMER_APPROXIMATION_32, 'CIFAR_PERFORMER_APPROXIMATION_32'),
        (CIFAR_PERFORMER_APPROXIMATION_64, 'CIFAR_PERFORMER_APPROXIMATION_64'),
        (CIFAR_PERFORMER_APPROXIMATION_128, 'CIFAR_PERFORMER_APPROXIMATION_128'),
        (CIFAR_PERFORMER_LOCAL_ATTN_4, 'CIFAR_PERFORMER_LOCAL_ATTN_4'),
        (CIFAR_PERFORMER_LOCAL_ATTN_16, 'CIFAR_PERFORMER_LOCAL_ATTN_16'),
        (CIFAR_PERFORMER_LOCAL_ATTN_32, 'CIFAR_PERFORMER_LOCAL_ATTN_32')
    ]:
        for repeat in range(10):
            subname = name + "_" + str(repeat)
            print(f"Running {subname} experiment")
            # bypass existing results
            if os.path.exists(f'results/{subname}.pkl'):
                print(f"Skipping {subname} experiment")
                continue
            run_performer_experiment(
                task='mnist',
                num_epochs=200,
                batch_size=32,
                learning_rate=1e-4,
                performer_encoder_params=params,
                experiment_name=subname,
                save_path='results'
            )