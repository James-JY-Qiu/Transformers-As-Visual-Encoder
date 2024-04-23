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
        self.linear_embedding_dim = linear_embedding_dim
        self.performer_params = performer_params if performer_params is not None else {
            'dim': 16,
            'depth': 2,
            'heads': 8,
            'dim_head': 32
        }
        self.dtype = dtype
        # ================ Model ================
        # Initialize the performer model
        self.performer = Performer(**self.performer_params)
        # Initialize the linear layer
        linear_in_features = patch_size[0] * patch_size[1] * (3 if dataset_name == 'cifar' else 1)
        self.linear_embedding = torch.nn.Linear(in_features=linear_in_features, out_features=linear_embedding_dim)
        # Positional encoding
        self.num_patches = self._check_patch_size()
        self.positional_embedding = torch.nn.Parameter(torch.zeros(self.num_patches, linear_embedding_dim))
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

    def _get_patches(self, image, channels):
        """Unfold the image into patches

        :param image: one image
        :param channels: number of channels
        :return: patches
        """
        patches = image.unfold(1, self.patch_height, self.patch_height).unfold(2, self.patch_width, self.patch_width)
        patches = patches.contiguous().view(channels, self.num_patches, self.patch_height * self.patch_width)
        patches = patches.permute(1, 0, 2).reshape(self.num_patches, -1)
        return patches

    def _linear_embedding(self, batch_images):
        """Linearly embed the patches

        :param batch_images: batch of images
        :return: embeddings
        """
        channels = batch_images.size()[1]

        embeddings = []
        for img in batch_images:
            # Unfold the image into patches
            patches = self._get_patches(img, channels)
            # Linearly embed the patches
            embedded_patches = self.linear_embedding(patches)
            embeddings.append(embedded_patches)

        embeddings = torch.stack(embeddings)

        return embeddings

    def encode(self, batch_images):
        """Encode the batch of images

        :param batch_images: batch of images
        :return: performer encoding
        """
        # Linearly embed the patches
        embeddings = self._linear_embedding(batch_images)
        # Add positional encoding
        embeddings += self.positional_embedding
        # Performer encoding
        encoding = self.performer(embeddings)
        return encoding


if __name__ == '__main__':
    data = load_data('cifar')
    data_loader = DataLoader(data['cifar'][0], batch_size=32, shuffle=True)
    encoder = PerformerEncoder(dataset_name='cifar')
    for batch, _ in data_loader:
        result = encoder.encode(batch)
        print(result.size())
        break
