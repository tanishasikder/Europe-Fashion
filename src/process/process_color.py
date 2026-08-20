import torch
import os
import copy
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from torch.utils.data import Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt

load_dotenv()

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

color_dir = os.environ.get('COLOR_DIR')

def color_transform():
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
    return color_transforms

def process_colors():
    color_dict = {}
    paths = []

    for path, mid, file in os.walk(color_dir):
        marker = path.find('colors\\')
        paths.append(path[:marker+7]) # For opening the image later on. Need full path
        path = path[marker+7:]

        color_dict[path] = file

    # Remove first value its the file path to whole folder
    (k := next(iter(color_dict)), color_dict.pop(k))

    return color_dict, paths

def encoding(name):
    codes = get_colors()

    if name in codes.keys():
        return codes[name]
        
def make_data(data, paths, transform):
    labels, images = [], []
    for index, (key, value) in enumerate(data.items()):
        marker = key.find('\\')  # All this for class names
        
        if marker == -1:
            continue
        else:
            color = key[:marker]

        for file in value:
            with open(f'{paths[index]}\\{key}\\{file}', 'rb') as f:
                img = Image.open(f)
                image_trans = transform['train']
                image = image_trans(img)

                name = encoding(color)
                labels.append(name)
                images.append(image)

    return labels, images

def split_data(colors):
    train, test = random_split(colors, [0.8, 0.2])
    return train, test

def get_colors():
    codes = { # Pytorch datasets encode labels with data
        'Black' : 0,
        'Blue' : 1,
        'Gray' : 2,
        'Orange' : 3,
        'Pink' : 4,
        'Purple' : 5,
        'Skyblue' : 7,
        'White' : 7,
        'Yellow' : 8
    }
    return codes

class ColorData(Dataset):
    def __init__(self, labels, images):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        sample = {
            'labels' : torch.tensor(self.labels[index]),
            'images' : torch.tensor(self.images[index])
        }
        return sample

def get_color_data():
    dict, paths = process_colors()    
    transform = color_transform()
    labels, images = make_data(dict, paths, transform)
    colors = ColorData(labels, images)
    train, test = split_data(colors)

    return train, test
