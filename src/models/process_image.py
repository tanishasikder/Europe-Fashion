from PIL import Image
from pathlib import Path
import os
import json
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import Dataset

load_dotenv()

categories = os.environ.get('TYPE_LABEL')
cloth_labels = os.environ.get('FASHION_LABELS')
cloth_images = os.environ.get('IMAGE_FASHION_DIR')
cropped = os.environ.get('CROPPED_IMAGES')

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

    for file, mid, dirs in os.walk(categories):
        for d in dirs:
            with open(f'{categories}\\{d}', 'r', encoding='utf-8') as f:
                if d == 'objects.txt':
                    objects.append(f.read().splitlines())
                elif d == 'fine_details.txt':
                    detailed.append(f.read().splitlines())

    return objects, detailed

# Next you need to process all the clothing_labels.json to train and val on them
# Process clothing images to train and val on them
class FashionData(Dataset):
    def __init__(self, categories, attr):
        self.categories = categories
        self.attr = attr

    def __len__(self):
        return len(self.categories), len(self.attr)

    def __getitem__(self):
        pass

def image_labels():
    with open(cloth_labels, 'r') as f:
        labels = json.load(f)

    return labels

def extract_labels(labels, file):
    return labels.get(file) # These functions process the gotten index

def get_categories(values, file):
    idk = []
    ref = []
    if values is not None:
        idk.append(values[-1][-1])

    ref.append(values)
    print(len(idk))
    '''
    for val in values:
        if val is not None:
            if len(val) > 3:
                print(val[-1])
                #print(val)
    '''
#def save_image(val, file):



def pass_images():
    labels = image_labels() # Mapping of file -> categories, attributes
    idk = []
    ref = []
    nu = []
    for file, mid, dirs in os.walk(cloth_images):
        for d in dirs: # Gets all file names to map to clothing_labels
            values = extract_labels(labels, d) 
            #print(values)# Extracts the category and attributes of image
            #get_categories(values, d)
            if values is not None:
                idk.append(values[-1][-1])

            ref.append(values)
            if values is None:
                nu.append(None)

    print(len(idk))
    print(len(ref))
    print(len(nu))
pass_images()

'''
Open with PIL.Image.open("image.jpg"), crop with img.crop((xmin, ymin, xmax, ymax)), 
then transform to a tensor using torchvision.transforms.v2.functional.to_image.
'''
