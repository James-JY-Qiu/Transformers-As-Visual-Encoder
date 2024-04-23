from experiments.run_experiments import run_performer_experiment
from experiments.experiment_params import MNIST_PERFORMER_RELU, CIFAR_PERFORMER_RELU


if __name__ == '__main__':
    print(f'Running Performer ReLU experiment')
    run_performer_experiment(
        task='mnist',
        num_epochs=10,
        performer_encoder_params=MNIST_PERFORMER_RELU,
        learning_rate=1e-3,
        experiment_name=f'MNIST_PERFORMER_RELU',
        save_path='results'
    )