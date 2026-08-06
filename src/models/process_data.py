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
from torch.utils.data import Dataset
from torch.utils.data import random_split

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
    paths = []

    for path, mid, file in os.walk(color_dir):
        paths.append(path) # For opening the image later one
        marker = path.find('colors\\')
        path = path[marker+7:]

        color_dict[path] = file

    return color_dict

dict = process_colors()    
print(dict.keys())

class ColorData(Dataset):
    def __init__(self, data, paths, transform):
        self.data = data
        self.colors = []
        self.paths = paths
        self.transform = transform

    def __len__(data):
        return len(data)

    def encoding(self, name):
        codes = {
            'Black' : 0,
            'Blue' : 1,
            'Gray' : 2,
            'Orange' : 3,
            'Pink' : 4,
            'Purple' : 5,
            'Skyblue' : 6,
            'White' : 6,
            'Yellow' : 7
        }

        if name in codes.keys():
            return codes[name]
        
    def make_data(self, data, paths, colors):
        for index, (key, value) in enumerate(data.items()):
            marker = key.find('\\')  # All this for class names
            if marker == -1:
                continue
            else:
                color = key[:marker]

            with open(f'{paths[index]}/key/{value}', 'rb') as f:
                img = Image.open(f)
                image = color_transform(img)
                image = torch.tensor(image)

            name = self.encoding(color)
            colors.append((image, name))


    def split_data(colors):
        train, test = random_split(colors, [0.8, 0.2])

        return train, test

            