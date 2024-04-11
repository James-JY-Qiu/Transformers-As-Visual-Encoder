import numpy as np
import torchvision
import torchvision.transforms as transforms


def load_data():
    # MNIST transformations
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # CIFAR-10 transformations
    cifar_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transform)
    cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transform)

    return {
        'mnist': (mnist_train, mnist_test),
        'cifar': (cifar_trainset, cifar_testset)
    }


if __name__ == '__main__':
    result = load_data()
    print(result)
