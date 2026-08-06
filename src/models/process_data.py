import mlflow.pytorch
import torch
import os
import copy
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import mlflow
import dagshub
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from dotenv import load_dotenv

load_dotenv()

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

color_dir = os.environ.get('COLOR_DIR')

def fashion_transform(data):
    # Fashion images have bigger transformations
    fashion_transforms = {
        'train' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val' : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    return fashion_transforms[data]

def color_transform(data):
    color_transforms = {
        'train' : transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val' : transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    return color_transforms[data]

def process_colors():
    color_dict = {}

    for path, mid, file in os.walk(color_dir):
        marker = path.find('colors\\')
        path = path[marker+7:]
        finish = path.find('\\')
        if finish == -1:
            continue
        else:
            path = path[:finish]
            if path in color_dict.keys():
                color_dict[path].append(file)
            else:
                color_dict[path] = file

    return color_dict

dict = process_colors()    

print(dict['Yellow'])
print(dict['Blue'])