'''
Need to get every category of clothes in the clothing images dataset
'''

'''PUT THE DIMENSIONS OF THE CLOTHING FROM THE DATASET NOT ALL CLOTHING IMAGES ARE CROPPED
TO MATCH THE CLOTHING IT IS DESCRIBING'''
from dotenv import load_dotenv
import os 
import numpy as np
import pandas as pd
import json

load_dotenv()

annotations=os.getenv('ANNOTATION_DIR')
path = annotations + '\\train_annotations.json'

images = {}

def corresponding(file_names, img_value):
    name = next(file['name'] for file in file_names if file['id'] == img_value)
    return name

def get_images():
    '''
    Goes through every image and gets IDs of categories and 
    attributes. Stores everything in a dict
    '''
    images_temp = [] # Temporary hold all dicts before processing
    file_names = []

    with open(path, 'r', encoding='utf-8') as t:
        file = json.load(t)

    # They are just IDs so need to decode them later on
    for i in range(45623): # How many images in train
        values = {'image_id' : file['annotations'][i]['image_id'], 
                  'attribute_id' : file['annotations'][i]['attribute_ids'],
                  'category_id' : file['annotations'][i]['category_id'],
                  'bbox' : file['annotations'][i]['bbox']}

        images_temp.append(values)

    for j in range(45623): # How many images in train
        # Get file name and ID to match the other list with
        file_names.append({'name' : file['images'][j]['file_name'], 
                           'id' : file['images'][j]['id']})

    return file_names, images_temp

def process_values(file_names, images_temp):
    for value in images_temp:
        id = value['image_id']
        name = corresponding(file_names, id)
        images[name] = value

def get_cat(id, cats):
    cat = next(c['name'] for c in cats if c['id'] == id)
    return cat

def get_attr(attr_id, attrs):
    names = []
    
    for id in attr_id:
        names.append(next(a['name'] for a in attrs if a['id'] == id))

    return names

def decode_images():
    processed = {}

    # Decodes all of the IDs for each image
    with open(path, 'r', encoding='utf-8') as t:
        file = json.load(t)

    # Each image has 0 or more attributes in a list
    # Each image has only one category
    categories = file['categories']
    attributes = file['attributes']

    for img in images:
        cat = get_cat(images[img]['category_id'], categories)
        describe = images[img]['attribute_id']
        bbox = images[img]['bbox'] # No need to process this. Used to find stuff later

        if describe: # Not every img has an attribute
            attr = get_attr(describe, attributes)
        else:
            attr = None

        processed[img] = [cat, attr, bbox]

    return processed

file_names, images_temp = get_images()
process_values(file_names, images_temp)
processed = decode_images()
