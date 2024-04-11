import numpy as np
import torch
from PerformerModel.Performer import Performer


class PerformerEncoder:
    def __init__(
            self,
            dataset_name,
            images,
            position_encoding_dim=128,
            performer_params=None,
            dtype=torch.float32
    ):
        assert dataset_name in ['mnist', 'cifar'], 'Dataset must be either "mnist" or "cifar"'
        self.dataset_name = dataset_name
        self.images = images
        self.position_encoding_dim = position_encoding_dim
        self.performer_params = performer_params.copy()
        self.dtype = dtype

        if dataset_name == 'mnist':
            self.input_shape = (1, 28, 28)
            self.performer_params['dim'] = 28 * 28
        elif dataset_name == 'cifar':
            self.input_shape = (3, 32, 32)
            self.performer_params['dim'] = 32 * 32 * 3
        else:
            raise ValueError('Invalid dataset name')

        position_enc = self.generate_positional_encoding()
        self.position_enc_tensor = torch.tensor(position_enc, dtype=self.dtype)
        self.performer = Performer(**self.performer_params)

    def flatten_images(self):
        """
        Flatten images from the data loader.

        Returns:
        - Flattened images of shape (N, C*H*W).
        """
        return self.images.view(self.images.size(0), -1)

    def generate_positional_encoding(self):
        """
        Generate positional encoding for an image.

        Returns:
        - Positional encoding for the image of shape (C*H*W, position_encoding_dim).
        """
        # Generate a grid of coordinates
        C, H, W = self.input_shape
        y, x, c = np.meshgrid(np.arange(H), np.arange(W), np.arange(C), indexing='ij')
        y = y.flatten()
        x = x.flatten()
        c = c.flatten()

        # Compute the positional encodings
        position_enc = np.zeros((H * W * C, self.position_encoding_dim))
        for i in range(self.position_encoding_dim // 3):
            div_term = np.power(10000, (3 * i) / self.position_encoding_dim)
            position_enc[:, 3 * i] = np.sin(y / div_term)
            position_enc[:, 3 * i + 1] = np.cos(x / div_term)
            position_enc[:, 3 * i + 2] = np.sin(c / div_term)

        return position_enc

    def preprocessing(self):
        """
        Preprocess the data.

        Returns:
        - Preprocessed data.
        """
        flattened_images = self.flatten_images()
        combine_feature = flattened_images.unsqueeze(-1) * self.position_enc_tensor.unsqueeze(0)

        return combine_feature

    def encoding(self):
        """
        Encode the data.

        Returns:
        - Encoded data.
        """
        combine_feature = self.preprocessing()
        return self.performer(combine_feature)



