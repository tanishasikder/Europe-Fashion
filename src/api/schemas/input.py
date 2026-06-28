from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from fastapi import File, HTTPException, UploadFile
from fastapi import FastAPI, Form
from pydantic import ValidationError
import json
from PIL import Image
import io
from validator import get_user_params

app = FastAPI()

# Basic health check to ensure server is functioning
@app.get("/health")
def root():
    return {"status" : "OK"}

# Enums are safer
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
    size : SizeParams = Field(..., description='CLothing Size')
    original_price: float = Field(..., ge=0.0)
    # Field level validator. Runs automatically

    # Checks if theres an empty field
    @field_validator("color, category, size") # no original_price it is a float
    @classmethod
    def verify_inputs(params):
        param = param.strip().lower()

        if not param:
            raise ValueError('Field cannot be empty')
        
        return params
   
def clean_domain(cls, v):
    return v.lower().strip().removeprefix("https://").removeprefix("www")
    
@app.post("/upload")
async def upload(
    color: str = Form(...),
    category: str = Form(...),
    size : str = Form(...),
    original_price: str = Form(...),
    file: UploadFile = File(...)
    ):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = ClothingRequest(
            color = color,
            category = category,
            size = size,
            original_price = original_price
        )
        return {
            'numerical' : inputs.model_dump(),
            'image': image
        }
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


