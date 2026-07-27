from matplotlib import pyplot as plt
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
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from dotenv import load_dotenv
import torchvision

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[1]

sys.path.insert(0, str(root_dir))

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#Data transformations for the train and val sets
data_transforms = {
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

load_dotenv()
data_dir = os.environ.get('DATA_DIR')

sets = ['train', 'val']

def get_valid_image(path):
    path_lower = path.lower()

    if path_lower.endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
        return True

    return False

if __name__ == '__main__':
    # Getting the data based on the train/val sets then doing transformations
    image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x],
                                                is_valid_file = get_valid_image)
                                                for x in sets}


    print(image_datasets)

    hi = datasets.ImageFolder(os.path.join(data_dir, 'train'))

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    num = 0

    for j in range(8):
        img, label = image_datasets['train'][j]
        class_name = image_datasets['train'].classes[label]

        img = img.permute(1, 2, 0).numpy()
        axes[j].imshow(img)
        
    plt.tight_layout()
    plt.show() 
    
    # Loading the data in batches. Separate dataloaders for color and type tests
    data_loaders = {
        'train' : DataLoader(image_datasets['train'], batch_size=32, 
                                shuffle=True, num_workers=4, pin_memory=True),
        'val' : DataLoader(image_datasets['val'], batch_size=32, 
                                shuffle=False, num_workers=4, pin_memory=True)                       
    }

    for images, label in data_loaders['train']:
        grid= torchvision.utils.make_grid(images, nrow=2)
        grid_np = np.transpose(grid.numpy(), (1, 2, 0))
        plt.show(grid_np)
        plt.show()