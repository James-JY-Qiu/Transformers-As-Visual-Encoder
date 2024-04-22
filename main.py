from experiments.performer_experiments import run_performer_experiment
from experiments.experiment_params import MNIST_PERFORMER_RELU, CIFAR_PERFORMER_RELU


if __name__ == '__main__':
    repeat = 10
    for r in range(repeat):
        print(f'Running Performer ReLU experiment {r}')
        run_performer_experiment(
            task='mnist',
            performer_encoder_params=MNIST_PERFORMER_RELU,
            learning_rate=1e-3,
            experiment_name=f'MNIST_PERFORMER_RELU_{r}',
            save_path='results'
        )