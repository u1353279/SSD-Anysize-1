import time
from itertools import product as product

import torch
import torch.nn as nn
import torchvision

from models.backbone_baseclass import BaseClass


class ResNet(BaseClass):
    def __init__(self, 
                input_dims, 
                architecture,
                first_out_layer="6", 
                second_out_layer="7"):

        super(ResNet, self).__init__(
            input_dims, 
            first_out_layer, 
            second_out_layer, 
            self.forward
            )

        if architecture == '18':
            model_with_classifier = torchvision.models.resnet18(pretrained=True)
        elif architecture == '34':
            model_with_classifier = torchvision.models.resnet34(pretrained=True)
        elif architecture == '50':
            model_with_classifier = torchvision.models.resnet50(pretrained=True)

        self.model = torch.nn.Sequential(*(list(model_with_classifier.children())[:-2]))        
        self.print_forward = False
        self.in_dims = input_dims
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()

    def forward(self, x):
        for name, layer in self.model.named_children():
            x = layer(x)
            if name == self.first_out_layer:
                out1 = x
            elif name == self.second_out_layer:
                out2 = x

            if self.print_forward:
                print(name, x.shape)

        return out1, out2


class MobileNetV1(BaseClass):
    def __init__(self, 
                input_dims, 
                first_out_layer="7",
                second_out_layer="13"):

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

        self.print_forward = False
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()
        self.first_out_layer = first_out_layer
        self.second_out_layer = second_out_layer
        

    def forward(self, x):
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
                use_alternative_structure=False):

        super(MobileNetV2, self).__init__(
            input_dims, 
            first_out_layer, 
            second_out_layer, 
            self.forward
            )

        # Experimental stuff
        if use_alternative_structure:
            inverted_residual_setting = [
                    # t, c, n, s
                    [1, 16, 1, 2],
                    [6, 24, 2, 1],
                    [6, 32, 3, 2],
                    [6, 64, 4, 1],
                    [6, 96, 3, 2],
                    [6, 160, 3, 1],
                    [6, 320, 1, 1],
                ]
        else:
            inverted_residual_setting = None

        width_mult = 1.0  # Adjusting this renders the pretrained weights useless

        self.model = torchvision.models.mobilenet_v2(
                                    pretrained=True,
                                    inverted_residual_setting=inverted_residual_setting,
                                    width_mult=width_mult)
        self.print_forward = False
        self.out_shape_1, self.out_shape_2 = self._get_construction_info()
        self.first_out_layer = first_out_layer
        self.second_out_layer = second_out_layer

    def forward(self, x):
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


###############################################################################
# WRAPPERS
###############################################################################


def mobilenetv1(dims):
    raise NotImplementedError

    # TODO: Not sure if this works, need to spend some time working with the checkpoints

    # backbone = MobileNetV1(CONFIG["input_dims"])
    # backbone.load_state_dict(
    #     torch.load("weights/mobilenet-v1-ssd-mp-0_675.pth"), strict=False)


def mobilenetv2(dims):
    return MobileNetV2(dims)

def mobilenetv2_alt(dims):
    """
    This iteration aims to expand the connective layers of mobilenetv2
    by modifying the strides of the network, at a cost of an ~80mb larger 
    network
    """
    return MobileNetV2(dims,
                        first_out_layer="9",
                        second_out_layer="18",
                        use_alternative_structure=True)

def resnet18(dims):
    return ResNet(dims, architecture="18")


def resnet34(dims):
    return ResNet(dims, architecture="34")


def resnet50(dims):
    return ResNet(dims, architecture="50")


AVAILABLE_MODELS = {
    "mobilenetv2": mobilenetv2,
    "mobilenetv2_alt": mobilenetv2_alt,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50
}