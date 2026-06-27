from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi import APIRouter, HTTPException
from PIL import Image
import os
import io
import uuid
import torch
from torchvision import transforms
import torchvision.models as models
import numpy as np
from datetime import datetime, timedelta
from api.dependencies import get_image_model
from api.dependencies import get_stats_model
from pydantic import BaseModel
from typing import Optional, List

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Temporary storage for image features
# Store image_id, image characteristics, when uploaded
# (so it can expire after an hour) for predictions
image_storage = {}

# Transformations for user input images
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Initial parameters to predict with later on
class ClothingParameters(BaseModel):
    size : str = None
    catalog_price : float = None
    channel : str = None
    original_price : float = None

def remove_expired_images():
    # Remove images after an hour
    for key, data in list(image_storage.items()):
        upload = image_storage[key]['uploaded']
        if datetime.now() >= upload + timedelta(hours=1):
            image_storage.pop(key)

# Gets the model predictions for color and clothing type
def image_model_output(file: UploadFile) -> Image.Image:
    remove_expired_images()
    # Extract the image from content
    image = Image.open(io.BytesIO(UploadFile)).convert("RGB")
    # Create image ID
    image_id = str(uuid.uuid4())
    image = data_transforms(image).unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        color, cloth_type = get_image_model(image)

    image_storage[image_id] = {
        'features': [color, cloth_type],
        'uploaded': datetime.now()
    }
    return color, cloth_type

    
def get_user_params(
    # Select is the task the user wants
    matrix: List[ClothingParameters],
    select : int,
    color : str, 
    category : str
):
    numerical_outputs = []
    # Predict with the other model
    for data in matrix:
        user_params = {
            'size' : data.size,
            'catalog price' : data.catalog_price,
            'channel' : data.channel,
            'original price' : data.original_price
        }
        result = get_stats_model.clothing_predict([color, category, user_params, select])
        numerical_outputs.append(result.tolist())
        return numerical_outputs

# Loops through the file names and stores all colors and categories
def get_color_category():
    path = 'Fashion_Images/train'
    files = os.listdir(path)
    color = [file[:file.index('_')] for file in files]
    category = [file[file.index('_')+1:] for file in files]

    return color, category

def image_preds(file: UploadFile) -> Image.Image:
     # We need this to find the true english labels
    color_labels, category_labels = get_color_category()

    color_py, category_py = image_model_output(file)
    color_pred = torch.argmax(color_py, dim=1)
    cat_pred = torch.argmax(category_py, dim=1)
    
    # Find the color and category based on the gotten index
    color, category = color_labels[color_pred], category_labels[cat_pred]

    return color, category