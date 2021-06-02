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
from models.SSD import SSD
from config import CONFIG as CONFIG
            

def get_mock_image(dims):
        mock_image = np.ones(dims)
        mock_image = np.dstack([mock_image] * 3)
        mock_image = mock_image.transpose()
        mock_image = mock_image[np.newaxis, ...]
        mock_image = torch.from_numpy(mock_image).float()

        return mock_image


dims = CONFIG["input_dims"]
im = get_mock_image(dims).to(CONFIG["device"])

if CONFIG["backbone"] == "mobilenetv2":
    backbone = MobileNetV2(dims, print_forward=True)
elif CONFIG["backbone"] == "mobilenetv1":
    backbone = MobileNetV1(dims, print_forward=True)

ssd = SSD(backbone, CONFIG["device"], 3, CONFIG["batch_size"]).to(CONFIG["device"])

t = time.time()
ssd(im)
print(time.time() - t)
# adds = AdditionalLayers(backbone, desired_prediction_layers=8)

# backbone.to(CONFIG["device"])
# adds.to(CONFIG["device"])

# t = time.time()
# # out_backbone = backbone.forward(im)
# print(backbone)
# out_adds = adds.forward(out_backbone[-1])
# print(time.time() - t)
# print(len(out_adds))

# all_out = out_backbone + out_adds 
# for a in all_out:
#     print(a.shape)

# from models.SSD import create_extra_convolutions
# print(create_extra_convolutions(backbone))

# ssd = nn.Sequential(
#     backbone, create_extra_convolutions(backbone)
# )

# print(ssd)
