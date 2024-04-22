import torch


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FullyConnectedLayer, self).__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)