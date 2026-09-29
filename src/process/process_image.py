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
import datetime
import torchvision.transforms.v2 as transforms

load_dotenv()

categories = os.environ.get('TYPE_LABEL')
cloth_labels = os.environ.get('FASHION_LABELS')
cloth_images = os.environ.get('IMAGE_FASHION_DIR')
crop_images = os.environ.get('CROPPED_IMAGES')

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

def get_data(values, dirs, mid):
    with open('image_crop.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for val in values:
            if isinstance(val[-1], list):
                crop, dimen = crop_image(val[-1], dirs, mid)

                if crop == 'continue':
                    continue # Skip if things are wrong.

                # Need to make the filenames unique so use dimen and separate with _
                # Having the dimensions makes the filenames always the same if you
                # Run the code again
                id = "".join(dimen)
                file_name = f'{id}_{dirs}'
                path = Path(crop_images) / file_name # Save with a different file everytime

                cat = val[-3]
                attr = val[-2]

                writer.writerow([file_name, cat, attr])
                crop.save(path)

def crop_image(values, file, mid):
    with open(os.path.join(mid, file), 'rb') as f:
        img = Image.open(f)

        if len(values) < 4:
            return 'continue', 0
        
        x, y, w, h = values # Fashionpedia does not follow PIL format

        if w <= 0 or h <= 0:
            return 'continue', 0

        left = x
        top = y
        right = x + w
        bottom = y + h
        dimen = [left, top, right, bottom]

        crop = img.crop(dimen)
        id = [str(x) for x in dimen]
        if crop:
            # Return the dimension to label the file later on
            return crop, id
        
def pass_images():
    labels = image_labels() # Mapping of file -> categories, attributes
    #required = set(f.split('_', 1)[1] for f in os.listdir(crop_images))
    for file, mid, dirs in os.walk(cloth_images):
        for i in dirs:
            values = extract_labels(labels, i) 
            if values:  # Most are lists values[-1][-1] but some are floats. find out which ones
                get_data(values, i, file)

#pass_images()
'''
Open with PIL.Image.open("image.jpg"), crop with img.crop((xmin, ymin, xmax, ymax)), 
then transform to a tensor using torchvision.transforms.v2.functional.to_image.
'''
