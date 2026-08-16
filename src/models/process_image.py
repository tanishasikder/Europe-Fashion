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
import matplotlib.pyplot as plt

load_dotenv()

labels = os.environ.get('TYPE_LABEL')

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

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

def get_type_labels():
    objects = []
    detailed = []

    for file, mid, dirs in os.walk(labels):
        for d in dirs:
            with open(f'{labels}\\{d}', 'r', encoding='utf-8') as f:
                if d == 'objects.txt':
                    objects.append(f.read().splitlines())
                elif d == 'fine_details.txt':
                    detailed.append(f.read().splitlines())

    return objects, detailed

# Next you need to process all the clothing_labels.json to train and val on them
# Process clothing images to train and val on them