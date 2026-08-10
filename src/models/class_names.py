'''
Need to get every category of clothes in the clothing images dataset
'''
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
        lines = file['images']

    # They are just IDs so need to decode them later on
    for i in range(len(lines)):
        values = {'image_id' : file['annotations'][i]['image_id'], 
                  'attribute_id' : file['annotations'][i]['attribute_ids'],
                  'category_id' : file['annotations'][i]['category_id']}

        images_temp.append(values)

    for j in range(len(lines)):
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
    cat = next(c for c in cats if c == id)
    return cat

def get_attr(id, attrs):
    attr = next(a for a in attrs if a == id)
    return attr

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
        attr = get_attr(images[img]['attribute_id'], attributes)
            

file_names, images_temp = get_images()
process_values(file_names, images_temp)
decode_images()