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

images_temp = {}

def get_images():
    path = annotations + '\\train_annotations.json'
 
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(path, 'r', encoding='utf-8') as t:
        file = json.load(t)

    #for key in file['images']:
    #    print(key['file_name'])
        #print(key['id'])

    for i in range(len(lines)):
        values = {'image_id' : file['annotations'][i]['image_id'], 
                  'attribute_id' : file['annotations'][i]['attribute_ids'],
                  'category_id' : file['annotations'][i]['category_id']}
        images_temp[file['images'][i]['file_name']] = values

    '''
    for cat in file['categories']:
        print(cat)  # MAP THE CATEGORY_ID IN THE DICTIONARY TO THESE LATER ON
    
    for annot in file['annotations']:
        #print(annot['image_id'])
        print(annot['category_id'])
        #print(annot['attribute_ids'])
    '''
get_images()

print(images_temp)