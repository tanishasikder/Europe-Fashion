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
from dotenv import load_dotenv
from torch.utils.data import Dataset
from torch.utils.data import random_split

load_dotenv()

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

color_dir = os.environ.get('COLOR_DIR')

def fashion_transform():
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
    return fashion_transforms

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
        paths.append(path) # For opening the image later on. Need full path
        marker = path.find('colors\\')
        path = path[marker+7:]
        slash = path.find('\\')

        if slash != -1:
            color_dict[path] = file

    # Remove first value its the file path to whole folder
    (k := next(iter(color_dict)), color_dict.pop(k))

    return color_dict, paths

class ColorData(Dataset):
    def __init__(self, data, paths, transform):
        self.data = data
        #self.colors = []
        self.paths = paths
        self.transform = transform

    def __len__(data):
        return len(data)

    def encoding(self, name):
        codes = { # Pytorch datasets encode labels with data
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
        
    def make_data(self, data, paths, colors, transform):
        print(data.keys())
        '''
        for index, (key, value) in enumerate(data.items()):
            marker = key.find('\\')  # All this for class names
            
            if marker == -1:
                continue
            else:
                color = key[:marker]

            hi = data.keys()
            idk = list(hi)[index]

            for file in value:
                with open(f'{paths[index]}\\{key}\\{file}', 'rb') as f:
                    img = Image.open(f)
                    image = transform['train']
                    image = image(img)
                    image = torch.tensor(image)

                    name = self.encoding(color)
                    colors.append((image, name)) # Store as tuples (important)
        '''
        return colors
    
    def split_data(colors):
        train, test = random_split(colors, [0.8, 0.2])
        return train, test

dict, paths = process_colors()    
transform = color_transform()
color = ColorData(dict, paths, transform)
colors = color.make_data(dict, paths, [], transform)

#train, test = color.split_data(colors)