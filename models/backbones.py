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
            conv_dw(1024, 1024, 1))

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

    def forward(self, x):
        """
        This is the forward function for the SSD
        """
        for name, layer in self.model.features.named_children():

            x = layer(x)
            if name == self.first_out_layer:
                m = nn.ReLU()
                out1 = m(x)
            if name == self.second_out_layer:
                out2 = x

            if self.print_forward:
                print(name, x.shape)

        return out1, out2
