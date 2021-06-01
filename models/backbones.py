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

from models.backbone_baseclass import BaseClass


class MobileNetV1(BaseClass):
    def __init__(self, 
                input_dims, 
                first_out_layer="7",
                second_out_layer="13",
                print_forward=False):

        super(MobileNetV1, self).__init__(
            input_dims, 
            first_out_layer, 
            second_out_layer, 
            self.forward
            )
        
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
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()
        self.first_out_layer = first_out_layer
        self.second_out_layer = second_out_layer
        

    def forward(self, x):
        """
        This is the forward function for the SSD
        """
        for name, layer in self.model.named_children():
            x = layer(x)
            if name == self.first_out_layer:
                out1 = x
            elif name == self.second_out_layer:
                out2 = x
                
            if self.print_forward:
                print(name, x.shape)

        return out1, out2


class MobileNetV2(BaseClass):
    def __init__(self, 
                input_dims, 
                first_out_layer="7", 
                second_out_layer="18",
                print_forward=False):

        super(MobileNetV2, self).__init__(
            input_dims, 
            first_out_layer, 
            second_out_layer, 
            self.forward
            )

        # Tentative, may need to replace
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.print_forward = print_forward
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()
        self.first_out_layer = first_out_layer
        self.second_out_layer = second_out_layer

        # The network can possibly be too small for the SSD. Make adjustments if this is the case
        if self.out_shape_2[-1] < 17:
            special_layer_1 = self._conv_upsample(
                in_channels=self.out_shape_1[0],
                out_channels=512,
                padding=1,
                stride=2)
            special_layer_2 = self._conv(in_channels=512,
                                         out_channels=self.out_shape_1[0],
                                         padding=1,
                                         stride=2)
            special_layer_3 = self._conv_upsample(in_channels=1280,
                                                  out_channels=1024,
                                                  padding=1,
                                                  stride=2)
            new_features = nn.Sequential(
                *self.model.features[:int(self.first_out_layer) + 1],
                special_layer_1, special_layer_2,
                *self.model.features[int(self.first_out_layer):],
                special_layer_3)

            self.model.features = new_features
            self.first_out_layer = str(int(self.first_out_layer) + 1)
            self.second_out_layer = str(int(self.second_out_layer) + 4)
            self.out_shape_1, self.out_shape_2 = self._get_construction_info()

    def forward(self, x):
        """
        This is the forward function for the SSD
        """
        for name, layer in self.model.features.named_children():

            x = layer(x)
            if name == self.first_out_layer:
                out1 = x
            if name == self.second_out_layer:
                out2 = x

            if self.print_forward:
                print(name, x.shape)

        return out1, out2

    def _conv_upsample(self, in_channels, out_channels, padding, stride):
        """
        If the input size is too small we need to add one last convolutional layer at the end to
        increase the output convoluted image size to anything but 10x10, so
        that's why this is here
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               padding,
                               stride,
                               bias=True), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def _conv(self, in_channels, out_channels, padding, stride):
        """
        If the input size is too small we need to add one last convolutional layer at the end to
        increase the output convoluted image size to anything but 10x10, so
        that's why this is here
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding, stride, bias=True),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


# class ResNet50(BaseClass):
#     def __init__(self, input_dims):
#         super(ResNet50, self).__init__()

#         model_with_classifier = torchvision.models.resnet50(pretrained=True)
#         self.model = nn.Sequential(*list(model_with_classifier.children())[:7])
#         self.out_shape_1, self.out_shape_2 = self._get_construction_info()

#     def forward(self, x):
#         """
#         This is the forward function for the SSD
#         """
#         for name, layer in self.model.named_children():
#             if name == "4":
#                 out1 = x
#             elif name == "6":
#                 out2 = x

#         return out1, out2
