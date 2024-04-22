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

if __name__ == '__main__':
    result = load_data()
    print(result)
