import torch.nn as nn


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.5):
        super(FullyConnectedLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.mean(dim=-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x