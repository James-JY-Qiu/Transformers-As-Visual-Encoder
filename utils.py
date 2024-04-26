import pickle
from collections import defaultdict

import torchvision
import torchvision.transforms as transforms


def load_data(data_names):
    datasets = {}

    if 'mnist' in data_names:
        # MNIST transformations
        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
        mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)
        datasets['mnist'] = (mnist_train, mnist_test)

    if 'cifar' in data_names:
        # CIFAR-10 transformations
        cifar_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
        cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)
        datasets['cifar'] = (cifar_trainset, cifar_testset)

    assert len(datasets) > 0, 'No datasets loaded'

    return datasets


def load_result(result_name):
    with open(f'results/{result_name}.pkl', 'rb') as f:
        return pickle.load(f)


def load_all_results():
    task = ['MNIST', 'CIFAR']
    models = ['PERFORMER_RELU', 'PERFORMER_APPROXIMATION_16', 'PERFORMER_APPROXIMATION_32', 'PERFORMER_APPROXIMATION_64',
              'PERFORMER_APPROXIMATION_128', 'TRANSFORMER_LOCAL_ATTN_4', 'TRANSFORMER_LOCAL_ATTN_16', 'TRANSFORMER_LOCAL_ATTN_32']
    results = {}

    for t in task:
        for m in models:
            name = f'{t}_{m}'
            results[name] = {}
            for i in range(10):
                result_name = f'{t}_{m}_{i}'
                try:
                    if t == 'MNIST' and m == 'TRANSFORMER_LOCAL_ATTN_32':
                        continue
                    result = load_result(result_name)
                    results[name][f'instance_{i}'] = result
                except FileNotFoundError:
                    continue

    return results
