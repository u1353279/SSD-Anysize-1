import torch
import torch.nn as nn
import numpy as np 
from torchsummary import summary

class BaseClass(nn.Module):
    def __init__(self, 
                input_dims: tuple, 
                first_out_layer: str=None, 
                second_out_layer: str=None,
                forward=None):

        super(BaseClass, self).__init__()
        self.input_dims = input_dims
        self.first_out_layer = first_out_layer 
        self.second_out_layer = second_out_layer
        self.forward = forward

    def _get_mock_image(self):
        mock_image = np.ones(self.input_dims)
        mock_image = np.dstack([mock_image] * 3)
        mock_image = mock_image.transpose()
        mock_image = mock_image[np.newaxis, ...]
        mock_image = torch.from_numpy(mock_image).float()

        return mock_image

    def _get_construction_info(self):
        """
        All backbones destined for an SSD framework need get_construction_info as a method 
        which will tell the SSD how to shape itself
        """
        mock_image = self._get_mock_image()
        features1, features2 = self.forward(mock_image)
        return features1.shape[1:], features2.shape[1:]
        