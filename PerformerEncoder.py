import torch
from PerformerModel.Performer import Performer
from utils import load_data
from torch.utils.data import DataLoader


class PerformerEncoder:
    def __init__(
            self,
            dataset_name,
            patch_size=(4, 4),
            linear_embedding_dim=256,
            performer_params=None,
            dtype=torch.float32
    ):
        # ================ Data ================
        assert dataset_name in ['mnist', 'cifar'], 'Dataset must be either "mnist" or "cifar"'
        self.dataset_name = dataset_name
        # ================ Parameters ================
        self.patch_height, self.patch_width = patch_size
        self.performer_params = performer_params if performer_params is not None else {
            'dim': linear_embedding_dim,
            'depth': 12,
            'heads': 8,
            'dim_head': 64
        }
        self.dtype = dtype
        # ================ Model ================
        # Initialize the performer model
        self.performer = Performer(**self.performer_params)
        # Initialize the linear layer
        linear_in_features = patch_size[0] * patch_size[1] * (3 if dataset_name == 'cifar' else 1)
        self.linear_embedding = torch.nn.Linear(in_features=linear_in_features, out_features=linear_embedding_dim)
        # Positional encoding
        num_patches = self._check_patch_size()
        self.positional_embedding = torch.nn.Parameter(torch.zeros(num_patches, linear_embedding_dim))
        torch.nn.init.uniform_(self.positional_embedding, -0.01, 0.01)

    def _check_patch_size(self):
        if self.dataset_name == 'mnist':
            height = width = 28
        elif self.dataset_name == 'cifar':
            height = width = 32
        else:
            raise ValueError(f'Unknown dataset: {self.dataset_name}')

        assert height % self.patch_height == 0 and width % self.patch_width == 0, "Image size must be divisible by patch size"

        return (height // self.patch_height) * (width // self.patch_width)

    def _get_patches(self, image, channels, num_patches):
        """Unfold the image into patches

        :param image: one image
        :param channels: number of channels
        :param num_patches: number of patches
        :return: patches
        """
        patches = image.unfold(1, self.patch_height, self.patch_height).unfold(2, self.patch_width, self.patch_width)
        patches = patches.contiguous().view(channels, num_patches, self.patch_height * self.patch_width)
        patches = patches.permute(1, 0, 2).reshape(num_patches, -1)
        return patches

    def _linear_embedding(self, batch):
        """Linearly embed the patches

        :param batch: batch of images
        :return: embeddings
        """
        batch_size, channels, height, width = batch.size()
        num_patches_height = height // self.patch_height
        num_patches_width = width // self.patch_width
        num_patches = num_patches_height * num_patches_width

        embeddings = []
        for img in batch:
            # Unfold the image into patches
            patches = self._get_patches(img, channels, num_patches)
            # Linearly embed the patches
            embedded_patches = self.linear_embedding(patches)
            embeddings.append(embedded_patches)

        embeddings = torch.stack(embeddings)

        return embeddings

    def encode(self, batch):
        """Encode the batch of images

        :param batch: batch of images
        :return: performer encoding
        """
        # Linearly embed the patches
        embeddings = self._linear_embedding(batch)
        # Add positional encoding
        embeddings += self.positional_embedding
        # Performer encoding
        encoding = self.performer(embeddings)
        return encoding


if __name__ == '__main__':
    data = load_data()
    data_loader = DataLoader(data['cifar'][0], batch_size=32, shuffle=True)
    encoder = PerformerEncoder(dataset_name='cifar')
    for batch, _ in data_loader:
        embeddings = encoder.encode(batch)
        print(embeddings.size())
        break






