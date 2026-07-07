from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from api.services.rag import get_rag_response
from api.services.services import image_model_output
from api.services.services import get_user_params
from schemas.input import input
from schemas.input import ClothingRequest
from torchvision import transforms
import numpy as np

router = APIRouter()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Transformations for user input images
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

@router.get("/query/")
async def query_rag_system(query: str):
    try:
        response = await get_rag_response(query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#@app.post("/predict")
async def initialize_preds(numerical_outputs):
    query = (f"Using the following user parameters {numerical_outputs}"
            "Generate summary reports for how the clothing will do in"
            "the market. Predict profit margin, quantity, and item total"
            "All kinds of predictions use clothing type, color, size"
            "Only quantity and item total use original price alongside"
            "the other predictors")
        
    response = await query_rag_system(query)
    
    return response

# Gets the model predictions for color and clothing type
async def image_output(contents: Image.Image):
    try:
        transformed = data_transforms(contents)
        # Perform inference
        with torch.no_grad():
            color, cloth_type = image_model_output(transformed)
        
        return color, cloth_type
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")
    
async def get_user_params(
    # Select is the task the user wants
    # matrix has color, category, size, and price
    matrix: List[ClothingRequest],
    select : int,
):
    try:
        numerical_outputs = get_user_params(matrix)
        return numerical_outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")

