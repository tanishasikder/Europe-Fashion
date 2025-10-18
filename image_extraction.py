import torch
import torch.nn as n
import torchvisin.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet18(pretrained=True)

data