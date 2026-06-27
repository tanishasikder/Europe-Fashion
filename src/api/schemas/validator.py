'''
 Pydantic classes that define the exact shape of data coming in 
 and going out. Think of them as the data contract — every field, 
 its type, and its validation rules live here. Nothing else does.


NOTE WE NEED TO HAVE VALIDATION AND INPUT IN THE SERVICES LOGIC
'''

from datetime import datetime, timedelta
import os
import uuid
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import sys
from joblib import load
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

router = APIRouter()

app = FastAPI()
app.include_router(router)

# Basic health check to ensure server is functioning
@app.get("/health")
def root():
    return {"status" : "OK"}

# Initial parameters to predict with later on
class ClothingParameters(BaseModel):
    size : str = None
    catalog_price : float = None
    channel : str = None
    original_price : float = None

# Defines dictionary for acceptable ranges and types
def acceptable_values():
    template = {
        'size' : {'values': { 'XS', 'S', 'M', 'L', 'XL'}, 
                  'types': str},
        'catalog price' : {'values': range(5, 1000),
                           'types': (int, float)},
        'channel' : {'values': {'App Mobile', 'E-Commerce'}, 
                     'types': str},
        'original price' : {'values': range(5, 1000), 
                            'types': (int, float)}
    }

    return template

# Checks if theres an empty field
@field_validator()
@classmethod
def verify_inputs(user_param):
    cleaned = user_param.strip()

    # If the string is empty
    if not cleaned:
        raise ValueError("Field cannot be empty")

    return cleaned

# Checks if the user typed in acceptable parameters
# Then returns if its fine
async def get_user_params(
    # Select is the task the user wants
    matrix: List[ClothingParameters]):
    try:
        template = acceptable_values()
        # Combine the input parameters with calculated values
        for data in matrix:
            verify_inputs(data)
            user_params = {
                'size' : data.size,
                'catalog price' : data.catalog_price,
                'channel' : data.channel,
                'original price' : data.original_price
            }

            if user_params[data] not in template[data]:
                return "Not OK"
            if not isinstance(user_params[data], template[data]['types']):
                return "Not OK"
            '''
            Need put this somewhere else
            # Predict with the other model
            result = stats_model.clothing_predict([color, category, user_params, select])
            numerical_outputs.append(result.tolist())
            '''
        return 'OK'
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")