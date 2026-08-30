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
from sentence_transformers import SentenceTransformer

load_dotenv()
'''

Problem is that rows are tryna be cleaned from None to '' and theres problems with cleaning it
'''

cropped = os.environ.get('CROPPED_IMAGES')
names = os.environ.get('CROPPED_CSV')

code = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
    df = pd.read_csv(names, header=None)
    df = df.sort_values(by=df.columns[0])
    data = clean(df)
    return data

def clean(df):
    '''
    Replace all attributes with '' if none else leave it alone
    '''
    df.iloc[:, 2] = df.iloc[:, 2].fillna('')
    return df

def get_label_classes(encoder):
    # The labels are encoded so this makes a mapping of the decoded -> encoded
    mappings = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    return mappings

def image_label():
    data = sort()
    cat = data.iloc[:, 1].tolist() # Get all the categories and attributes
    att = data.iloc[:, 2].tolist()
    # Then encode and return as a list
    en_cat = code.encode(cat, batch_size=256, convert_to_tensor=True)
    en_att = code.encode(att, batch_size=256, convert_to_tensor=True)

    return list(zip(en_cat, en_att))

class ImageData(Dataset):
    def __init__(self, dir=cropped, transform=fashion_transform(), labels=image_label()):
        self.dir = Path(dir)
        self.transform = transform
        self.image_paths = sorted([
            path for path in self.dir.iterdir()
        ]) # Loop through all images
        self.image_labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.image_labels[idx]
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

hi = ImageData()
