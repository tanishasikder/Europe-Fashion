from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from fastapi import File, HTTPException, UploadFile
from fastapi import FastAPI, Form
from pydantic import ValidationError
from enum import Enum
from PIL import Image
import io
#from validator import get_user_params
from src.services.model_service.predict import image_output
import os
from dotenv import load_dotenv

load_dotenv()

colors = os.getenv('COLOR_DIR')
labels = os.getenv('TYPE_LABEL')

# Use the enums to make sure model predictions are type safe
# Make enums based on looping through names and utilizing all caps

def read_file(path):
    contents = []

    with open(path, 'r') as f:
        for entry in f:
            contents.append(entry)

    return contents

def get_file_names():
    '''
    Loops through files and get names to create enums with
    '''
    color_names = []
    for root, dirs, files in os.walk(colors):
        for file in dirs:
            color_names.append(file)
            
    for root, dirs, files in os.walk(labels):
        for file in files:
            if dirs == 'fine_details.txt':
                path = os.path.join(root, file)
                attributes = read_file(path)
            elif dirs == 'objects.txt':
                path = os.path.join(root, file)
                categories = read_file(path)

    return color_names, attributes, categories

color_names, attributes, categories = get_file_names()

ColorParams = Enum(
    'colorparams',
    {color.upper() : color for color in color_names},
    type=str
)

CategoryParams = Enum(
    'categoryparams',
    {cat.upper() : cat for cat in categories},
    type=str
)

AttributeParams = Enum(
    'attributeparams',
    {attr.upper() : attr for attr in attributes},
    type=str
)

