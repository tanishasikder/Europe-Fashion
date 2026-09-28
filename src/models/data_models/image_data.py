from PIL import Image
from pathlib import Path
import os
import json
import pandas as pd
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import Dataset 
import torch
from torchvision.transforms import v2
from sentence_transformers import SentenceTransformer
from torchvision.transforms import v2

load_dotenv()

cropped = os.environ.get('CROPPED_IMAGES')
names = os.environ.get('CROPPED_CSV')
l_path = os.environ.get('CLOTHING_EMBED')

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class RandomGamma(torch.nn.Module):
    """random gamma transform"""
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            gamma = torch.empty(1).uniform_(*self.gamma_range).item()
            img = v2.functional.adjust_gamma(img, gamma=gamma)
        return img

class RandomRGBShift(torch.nn.Module):
    """simulates per-channel color-temperature shift"""
    def __init__(self, shift_limit=15/255, p=0.5):
        super().__init__()
        self.shift_limit = shift_limit
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            shift = (torch.rand(3, 1, 1) * 2 - 1) * self.shift_limit
            img = (img + shift).clamp(0, 1)
        return img

def fashion_transform():
    '''
    Fashionpedia does not really train on color. Must transform
    images to test on different shades. Map with HEXCodes later
    '''
    fashion_transforms = v2.Compose([
        v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # convert to [0,1] float BEFORE color ops
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        RandomGamma(gamma_range=(0.7, 1.5), p=0.5),
        RandomRGBShift(shift_limit=15/255, p=0.5),
        v2.RandomAutocontrast(p=0.2),        # closest native stand-in for CLAHE
        v2.GaussianNoise(mean=0.0, sigma=0.03),  # stand-in for ISONoise
        v2.Normalize(mean=mean, std=std),
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
        labels = torch.load(em_path) # Load in premade labels
        self.image_labels = dict(zip(data.iloc[:, 0], list(zip(labels['cat'], labels['attr']))))
        valid = set(data.iloc[:, 0])
        self.image_paths = sorted([
            path.name for path in self.dir.iterdir()
            if path.name in valid
        ]) # Loop through all images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Look up without using UUID
        #original_name = path.partition('_')[2]
        #label = self.image_labels[original_name] # Explicit lookup
        #image = Image.open(self.dir / path).convert('RGB')
        label = self.image_labels[path]
        image = Image.open(self.dir / path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # Heavy so load not at module import time. Needed only for labels not dataset
    code = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    image_label()
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