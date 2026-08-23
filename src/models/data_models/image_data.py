from PIL import Image
from pathlib import Path
import os
import json
import pandas as pd
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
from torchvision.transforms import v2
import csv
from sklearn.preprocessing import OneHotEncoder
#import torch.nn.functional as F

load_dotenv()

cropped = os.environ.get('CROPPED_IMAGES')
names = os.environ.get('CROPPED_CSV')

code = OneHotEncoder(sparse_output=False)

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def fashion_transform():
    # Fashion images have bigger transformations
    fashion_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std)
    ])

    return fashion_transforms

def sort():
    df = pd.read_csv(names)
    df = df.sort_values(by=df.columns[0])
    return df

def get_label_classes(encoder):
    # The labels are encoded so this makes a mapping of the decoded -> encoded
    mappings = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    return mappings

class ImageData(Dataset):
    def __init__(self, dir=cropped, transform=fashion_transform(), df = sort()):
        self.dir = Path(dir)
        self.transform = transform
        self.image_paths = sorted([
            path for path in self.dir.iterdir()
        ]) # Loop through all images
        self.image_labels = [
            (code.fit_transform(torch.tensor(row.iloc[1]), 
                                code.fit_transform(torch.tensor(row.iloc[2])))
                                for _, row in df.iterrows())
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.image_labels[idx]
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
