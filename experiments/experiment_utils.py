import os
import pickle
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def save_results(results, file_name, file_path):
    dataset_name, model_name, attribute = file_name.split('_')
    os.makedirs(os.path.join(file_path, f'{dataset_name}/{model_name}'), exist_ok=True)
    with open(os.path.join(file_path, f'{dataset_name}/{model_name}/experiment_{attribute}.pkl'), 'wb') as f:
        pickle.dump(results, f)