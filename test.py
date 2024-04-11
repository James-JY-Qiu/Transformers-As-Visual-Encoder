import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import load_data
from PerformerEncoder import PerformerEncoder
from Classifier import FullyConnectedLayer


batch_size = 32
epochs = 10


if __name__ == '__main__':
    mnist_train = load_data()['mnist'][0]
    mnist_test = load_data()['mnist'][1]
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    performer_params = {
        'depth': 2,
        'heads': 8,
        'dim_head': 64,
    }
    input_dim = 28 * 28
    num_classes = 10

    model = FullyConnectedLayer(input_dim=input_dim, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            performer_encode_features = PerformerEncoder(
                'mnist', images, performer_params=performer_params
            ).preprocessing()
            outputs = model(performer_encode_features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}], Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            # 用 PerformerEncoder 处理 images
            performer_encode_features = PerformerEncoder(
                'mnist', images, performer_params=performer_params
            ).preprocessing()
            outputs = model(performer_encode_features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct / total}%')
