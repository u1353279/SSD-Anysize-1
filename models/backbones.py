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

from utils import *
# from SSD.utils import decimate


class BaseClass(nn.Module):
    def __init__(self, 
                input_dims: tuple, 
                first_out_layer: str, 
                second_out_layer: str):

        super(BaseClass, self).__init__()
        self.input_dims = self.input_dims
        self.first_out_layer = None 
        self.second_out_layer = None


    def forward(self, x):
        if self.first_out_layer is None or self.second_out_layer is None:
            raise Exception("Must define first and second output layers for the SSD")
        return

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
        mock_image = self._get_mock_image(self.input_dims)
        features1, features2 = self.forward(mock_image)
        return features1.shape[1:], features2.shape[1:]


class MobileNetV1(nn.Module):
    """
    If attaching to a FasterRCNN cut off the FC layer: 
        backbone = MobileNetV1()
        truncated = nn.Sequential(*list(backbone.children())[:-1])

    NOTE: Not currently supported since can't find pretrained weights
    """


    def __init__(self, input_dims):
        super(MobileNetV1, self).__init__()

        self.first_out_layer = "7"
        self.second_out_layer = "13"

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
            conv_dw(1024, 1024, 1),
            # nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)  # Trim this off if attaching to a network

        self.in_dims = input_dims
        

    def forward(self, x):
        """
        This is the forward function for the SSD
        """
        for name, layer in self.model.named_children():

            x = layer(x)
            if name == self.first_out_layer:
                conv_7_feats = x
            elif name == self.second_out_layer:
                conv_13_feats = x

        return conv_7_feats, conv_13_feats

    def _get_construction_info(self):
        """
        All backbones destined for an SSD framework need get_construction_info as a method 
        which will tell the SSD how to shape itself
        """
        mock_image = get_mock_image(self.in_dims)
        features1, features2 = self.forward(mock_image)
        return features1.shape[1:], features2.shape[1:]


class MobileNetV2(nn.Module):
    def __init__(self, input_dims):
        super(MobileNetV2, self).__init__()

        # Tentative, may need to replace
        self.first_out_layer = "7"
        self.second_out_layer = "18"
        self.in_dims = input_dims
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()

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

        return out1, out2

    def _get_construction_info(self):
        """
        All backbones destined for an SSD framework need get_construction_info as a method 
        which will tell the SSD how to shape itself
        """
        mock_image = get_mock_image(self.in_dims)
        features1, features2 = self.forward(mock_image)
        return features1.shape[1:], features2.shape[1:]

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


class ResNet50(nn.Module):
    def __init__(self, input_dims):
        super(ResNet50, self).__init__()

        model_with_classifier = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(model_with_classifier.children())[:7])
        self.in_dims = input_dims
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()

    def forward(self, x):
        """
        This is the forward function for the SSD
        """
        for name, layer in self.model.named_children():
            if name == "4":
                out1 = x
            elif name == "6":
                out2 = x

        return out1, out2

    def _get_construction_info(self):
        """
        All backbones destined for an SSD framework need get_construction_info as a method 
        which will tell the SSD how to shape itself
        """
        mock_image = get_mock_image(self.in_dims)
        features1, features2 = self.forward(mock_image)
        return features1.shape[1:], features2.shape[1:]


if __name__ == "__main__":
    """
    This section is for bringing in new backbones, need to explore their structure to find 
    where to make SSD connections
    """

    dims = (560, 560)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    backbone = MobileNetV2(dims)
    backbone.to(device)

    print(summary(backbone, (3, *dims)))
    # print(summary(new_model, (3,300,300)))

    # print(special_layer_1)

    # new_layer = backbone._conv_upsample(in_channels=1280, out_channels=1280//4, padding=1, stride=1)
    # new_model = nn.Sequential(
    #     *backbone.model.features, new_layer
    # )
    # new_model = new_model.to(device)
    # print(summary(backbone, (3,300,300)))

    mock_image = get_mock_image(dims)
    x = mock_image.to(device)

    # # Need additional layers to translate output to what is expected
    for name, layer in backbone.model.features.named_children():
        x = layer(x)
        print(name)
        print(x.shape)
