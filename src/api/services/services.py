from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi import APIRouter, HTTPException
from PIL import Image
import os
from src.database import store_image, remove_expired
import torch
from torchvision import transforms
import torchvision.models as models
import numpy as np
from supabase import create_client, Client
from api.dependencies.dependencies import get_image_model
from api.dependencies.dependencies import get_stats_model
from pydantic import BaseModel
from typing import Optional, List
from schemas.input import ClothingRequest
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Transformations for user input images
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def get_user_params(
    matrix: List[ClothingRequest]
):
    numerical_outputs = []
    # Predict with the other model
    result = get_stats_model.clothing_predict(matrix)
    return result.tolist()

# Loops through the file names and stores all colors and categories
def get_color_category():
    path = 'Fashion_Images/train'
    files = os.listdir(path)
    color = [file[:file.index('_')] for file in files]
    category = [file[file.index('_')+1:] for file in files]

    return color, category

def image_preds(color, category):
    '''
    Predict what the color and category is in predict.py
    Then find the color with PyTorch from the matrix
    '''
     # We need this to find the true english labels
    color_labels, category_labels = get_color_category()

    color_pred = torch.argmax(color, dim=1)
    cat_pred = torch.argmax(category, dim=1)
    
    # Find the color and category based on the gotten index
    true_color, true_category = color_labels[color_pred], category_labels[cat_pred]

    return true_color, true_category