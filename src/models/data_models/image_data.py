from PIL import Image
from pathlib import Path
import os
import json
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import Dataset
import torch
import torchvision.transforms.v2 as transforms

load_dotenv()

cropped = os.environ.get('CROPPED_IMAGES')

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def fashion_transform():
    # Fashion images have bigger transformations
    fashion_transforms = {
        'train' : transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val' : transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    return fashion_transforms

class ImageData(Dataset):
    def __init__(self, dir=cropped, transform=None, extensions='.jpg'):
        self.dir = Path(dir)
        self.transform = transform
        self.image_paths = sorted([
            path for path in self.dir.iterdir()
            if path.suffix.lower() in extensions
        ]) # Loop through all images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_path[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, str(path) # Need to track filename to decode
