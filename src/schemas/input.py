from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from fastapi import File, HTTPException, UploadFile
from fastapi import FastAPI, Form
from pydantic import ValidationError
import json
from PIL import Image
import io
#from validator import get_user_params
from src.services.model_service.predict import image_output

# Use the enums to make sure model predictions are type safe
# Make enums based on looping through names and utilizing all caps
class ColorParams(str, Enum):
    white = "white"
    red   = "red"
    green = "green"
    blue  = "blue"
    black = "black"

class CategoryParams(str, Enum):
    tshirt    = "tshirt"
    sleepwear = "sleepwear"
    pants     = "pants"
    dress     = "dress"
    shoes     = "shoes"

class SizeParams(str, Enum):
    xs = "xs"
    s  = "s"
    m  = "m"
    l  = "l"
    xl  = "xl"

# This is what the caller must send in the request body
class ClothingRequest(BaseModel):
    color: ColorParams = Field(..., description='Clothing Color')
    category: CategoryParams = Field(..., description='Clothing Category')
    attribute: AttrParams = Field(..., description='CLothing Attributes')
    # Field level validator. Runs automatically

    # Checks if theres an empty field
    @field_validator("color", "category", "attr") # no original_price it is a float
    @classmethod
    def verify_inputs(cls, params):
        if not params:
           raise ValueError('Field cannot be empty')
        
        return params
   
def clean_domain(cls, v):
    return v.lower().strip().removeprefix("https://").removeprefix("www")

def get_upload( # Might delete this. I think something already takes care of this. 
                # Maybe delete it and copy some of the logic.
        contents : bytes
    ):
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        color, category = image_output(image)

        inputs = ClothingRequest(
            color = color,
            category = category,
            size = size,
            original_price = price
        )

        return inputs

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


