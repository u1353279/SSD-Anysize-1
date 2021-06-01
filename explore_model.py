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

from models.backbones import MobileNetV1, MobileNetV2
from config import CONFIG as CONFIG
            

dims = CONFIG["input_dims"]
device = torch.device("cuda:0")

if CONFIG["backbone"] == "mobilenetv2":
    backbone = MobileNetV2(dims, print_forward=True)
elif CONFIG["backbone"] == "mobilenetv1":
    backbone = MobileNetV1(dims, print_forward=True)

backbone.to(device)
print(backbone.out_shapes)
# print(summary(backbone, (3, *dims)))