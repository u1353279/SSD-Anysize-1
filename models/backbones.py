import time
from itertools import product as product

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import mobilenet_v2
import torch.nn.functional as F
import torchvision
import numpy as np

# from models.backbone_baseclass import BaseClass


class MobileNetV1(nn.Module):
    def __init__(self, 
                input_dims, 
                out_layers=["7","13"],
                print_forward=False):

        super(MobileNetV1, self).__init__()
        
        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                                 nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 1),  # change stride to 1 from 2 so layer sizes match ssd layers
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1)
        )

        self.print_forward = print_forward
        self.input_dims = input_dims
        self.out_layers = out_layers
        self.out_shapes = self._get_construction_info()

    def forward(self, x):
        """
        This is the forward function for the SSD
        """
        out = []
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in self.out_layers:
                out.append(x)
            
            if self.print_forward:
                print(name, x.shape)

        return out

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
        out = self.forward(mock_image)
        return [o.shape[1:] for o in out]


class MobileNetV2(nn.Module):
    def __init__(self, 
                input_dims, 
                out_layers=["6", "13", "18"], 
                print_forward=False):

        super(MobileNetV2, self).__init__()

        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.print_forward = print_forward
        self.input_dims = input_dims
        self.out_layers = out_layers
        self.out_shapes = self._get_construction_info()

    def forward(self, x):
        out = []
        for name, layer in self.model.features.named_children():

            x = layer(x)
            if name in self.out_layers:
                out.append(x)
            
            if self.print_forward:
                print(name, x.shape)

        return out

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
        out = self.forward(mock_image)
        return [o.shape[1:] for o in out]
