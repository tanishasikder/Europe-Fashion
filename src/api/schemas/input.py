from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum

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
    @field_validator("domain")
    def clean_domain(cls, v):
        return v.lower().strip().removeprefix("https://").removeprefix("www")

