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

def corresponding(values):

def get_images():
    '''
    Goes through every image and gets IDs of categories and 
    attributes. Stores everything in a dict
    '''
    images_temp = [] # Temporary hold all dicts before processing

    with open(path, 'r', encoding='utf-8') as t:
        file = json.load(t)
        lines = file['images']

    # They are just IDs so need to decode them later on
    # THIS IS WRONG YOURE ASSUMING ALL THE IMAGES ARE IN ORDER
    # YOU NEED TO GET THE IMAGES BU IMAGE_ID AND GET THE CORRESPONDING CORRECT FILE NAME
    for i in range(len(lines)):
        values = {'image_id' : file['annotations'][i]['image_id'], 
                  'attribute_id' : file['annotations'][i]['attribute_ids'],
                  'category_id' : file['annotations'][i]['category_id']}

    images_temp[file['images'][i]['file_name']] = values

def decode_images():
    '''Decodes all of the IDs for each image'''
    with open(path, 'r', encoding='utf-8') as t:
        file = json.load(t)

    # Each image has 0 or more attributes in a list
    # Each image has only one category
    categories = file['categories']
    attributes = file['attributes']

    for img in images_temp:
        for dict in img:
            for cat in categories:
                   
        


get_images()

print(images_temp)