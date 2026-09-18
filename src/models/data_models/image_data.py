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
l_path = os.environ.get('CLOTHING_EMBED')

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

def clean(df):
    '''
    Replace all attributes with '' if none else leave it alone
    '''
    df.iloc[:, 2] = df.iloc[:, 2].fillna('')
    return df

df = pd.read_csv(names, header=None)
df = df.sort_values(by=df.columns[0])
data = clean(df)

def get_label_classes(encoder):
    # The labels are encoded so this makes a mapping of the decoded -> encoded
    mappings = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    return mappings

def image_label(out_path=l_path):
    # Processes the labels once then use everytime
    cat = data.iloc[:, 1].tolist() # Get all the categories and attributes
    att = data.iloc[:, 2].tolist()
    # Then encode and return as a list
    en_cat = code.encode(cat, batch_size=256, convert_to_tensor=True)
    en_att = code.encode(att, batch_size=256, convert_to_tensor=True)

    torch.save({'cat': en_cat, 'attr': en_att}, out_path)

class ImageData(Dataset):
    def __init__(self, dir=cropped, transform=fashion_transform(), em_path=l_path):
        self.dir = Path(dir)
        self.transform = transform
        self.image_paths = sorted([
            path for path in self.dir.iterdir()
        ]) # Loop through all images
        labels = torch.load(em_path) # Load in premade labels
        self.image_labels = dict(zip(data.iloc[:, 0], list(zip(labels['cat'], labels['attr']))))
        nu = [k.partition('_')[2] for k in self.image_labels.keys()]
        idk = [p.partition('_')[2] for p in self.image_paths]
        lmao = 'e64c38c709dd92daa8092252800bafc6.jpg'
        for i in self.image_paths:
            poo = i.partition('final_img\\')[2]
            if poo.partition('_')[2] == lmao:
                print(poo)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        
        label = self.image_labels[path.name] # Explicit lookup
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # Heavy so load not at module import time. Needed only for labels not dataset
    #code = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    #image_label()
    hi = ImageData()
    #image_name_set = set(p.name for p in Path(cropped).iterdir() if p.name in hi)


    '''
    hi = set(data.iloc[:, 0])
    image_paths = set([p for p in Path(cropped).iterdir() if p.name in hi])
    print('overall', len(hi), len(image_paths))
    print(len(hi & image_paths))
    print(len(hi - image_paths))
    print(len(image_paths - hi))
    '''