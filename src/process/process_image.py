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
import torch
import torchvision.transforms.v2 as transforms

load_dotenv()

categories = os.environ.get('TYPE_LABEL')
cloth_labels = os.environ.get('FASHION_LABELS')
cloth_images = os.environ.get('IMAGE_FASHION_DIR')

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

def image_labels():
    with open(cloth_labels, 'r') as f:
        labels = json.load(f)

    return labels

def extract_labels(labels, file):
    return labels.get(file) # These functions process the gotten index

def get_data(values, dirs, i):
    with open('image_crop.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for val in values:
            if isinstance(val[-1], list):
                crop = crop_image(val[-1], dirs)

                if crop == 'continue':
                    continue # Skip if things are wrong.

                file_name = f'{i}{dirs}'
                path = f'{cropped}\\{file_name}' # Save with a different index everytime

                crop.save(path)

                cat = val[-3]
                attr = val[-2]

                writer.writerow([file_name, cat, attr])


def crop_image(values, file):
    with open(f'{cloth_images}/{file}', 'rb') as f:
        img = Image.open(f)

        if len(values) < 4:
            return 'continue'
        
        x, y, w, h = values # Fashionpedia does not follow PIL format

        if w <= 0 or h <= 0:
            return 'continue'

        left = x
        top = y
        right = x + w
        bottom = y + h

        crop = img.crop([left, top, right, bottom])
        if crop:
            return crop
        
def pass_images():
    labels = image_labels() # Mapping of file -> categories, attributes

    for file, mid, dirs in os.walk(cloth_images):
        for i in range(len(dirs)): # Gets all file names to map to clothing_labels
            values = extract_labels(labels, dirs[i]) 
            if values:  # Most are lists values[-1][-1] but some are floats. find out which ones
                get_data(values, dirs[i], i)



'''
Open with PIL.Image.open("image.jpg"), crop with img.crop((xmin, ymin, xmax, ymax)), 
then transform to a tensor using torchvision.transforms.v2.functional.to_image.
'''
