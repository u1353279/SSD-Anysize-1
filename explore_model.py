import time
from itertools import product as product

import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torchvision

import numpy as np
from models.backbones import AVAILABLE_MODELS
from config import CONFIG as CONFIG
            

dims = CONFIG["input_dims"]
device = torch.device("cuda:0")

backbone = AVAILABLE_MODELS[CONFIG["backbone"]](dims)

backbone.to(device)
backbone.print_forward = True
backbone.to(device)
print(summary(backbone, (3, *dims)))

