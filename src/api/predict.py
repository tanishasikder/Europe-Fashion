from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from rag import get_rag_response
from services import image_model_output
from services import get_user_params


# Initial parameters to predict with later on
class ClothingParameters(BaseModel):
    size : str = None
    catalog_price : float = None
    channel : str = None
    original_price : float = None

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
async def image_model_output(file: UploadFile) -> Image.Image:
    # Make sure the file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image. Try again")
    # Wait for the file to be read
    contents = await file.read()
    try:
        # Perform inference
        with torch.no_grad():
            color, cloth_type = image_model_output(contents)
        
        return color, cloth_type
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")
    
async def get_user_params(
    # Select is the task the user wants
    matrix: List[ClothingParameters],
    select : int,
    color : str, 
    category : str
):
    try:
        numerical_outputs = get_user_params(matrix, select, color, category)
        return numerical_outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")

