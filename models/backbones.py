import time

import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import mobilenet_v2
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
import numpy as np

from utils import *


# from SSD.utils import decimate
def get_mock_image(dims):
    mock_image = np.ones(dims)
    mock_image = np.dstack([mock_image] * 3)
    mock_image = mock_image.transpose()
    mock_image = mock_image[np.newaxis, ...]
    mock_image = torch.from_numpy(mock_image).float()

    return mock_image


class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    def __init__(self, input_dims):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3,
                                 padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(
            kernel_size=2, stride=2,
            ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(
            kernel_size=3, stride=1,
            padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6,
                               dilation=6)  # atrous convolution
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.in_dims = input_dims
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()

    def forward(self, x):
        """
        This is the forward function for the SSD
        """

        for name, layer in self.named_children():

            if "conv" in name:  # if it's a conv layer apply relu
                x = F.relu(layer(x))

                if name == "conv4_3":
                    conv4_3_feats = x
                elif name == "conv7":
                    conv7_feats = x

            elif "pool" in name:
                x = layer(x)

        return conv4_3_feats, conv7_feats

    def _get_construction_info(self):
        """
        All backbones destined for an SSD framework need get_construction_info as a method 
        which will tell the SSD how to shape itself
        """
        mock_image = get_mock_image(self.in_dims)
        features1, features2 = self.forward(mock_image)
        return features1.shape[1:], features2.shape[1:]

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(
            pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(
                param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[
                pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(
            4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight,
                                              m=[4, None, 3,
                                                 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(
            4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight,
                                              m=[4, 4, None,
                                                 None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)


class MobileNetV1(nn.Module):
    """
    If attaching to a FasterRCNN cut off the FC layer: 
        backbone = MobileNetV1()
        truncated = nn.Sequential(*list(backbone.children())[:-1])

    NOTE: Not currently supported since can't find pretrained weights
    """
    def __init__(self, input_dims):
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
            conv_dw(
                256, 512, 1
            ),  # change stride to 1 from 2 so layer sizes match ssd layers
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            # nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024,
                            1000)  # Trim this off if attaching to a network

        self.in_dims = input_dims
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()

    def forward(self, x):
        """
        This is the forward function for the SSD
        """
        for name, layer in self.model.named_children():

            x = layer(x)
            if name == "7":
                conv_7_feats = x
            elif name == "13":
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
        self.last_out_layer = "18"
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
            self.last_out_layer = str(int(self.last_out_layer) + 4)
            self.out_shape_1, self.out_shape_2 = self._get_construction_info()

    def forward(self, x):
        """
        This is the forward function for the SSD
        """
        for name, layer in self.model.features.named_children():

            x = layer(x)
            if name == self.first_out_layer:
                out1 = x
            if name == self.last_out_layer:
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
