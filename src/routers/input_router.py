from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import FastAPI, Form, Request
from pydantic import ValidationError
import json
from PIL import Image
import io
#from validator import get_user_params
from src.services.model_service.predict import image_output
from src.main import limiter 
from src.schemas.tasks import process_image
router = APIRouter(prefix='preds')

# Basic health check to ensure server is functioning
@router.get("/health")
def root():
    return {"status" : "OK"}

@router.post("/upload") # Use with schemas/input function get_upload
@limiter.limit('3/minute') # How much we limit
async def upload(
        request : Request, # Need this or limiter will not work
        file: UploadFile = File(...)
    ):
    try:
        contents = await file.read()
        return contents

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

# Use with services/model_service/predict function query_rag_system
@router.get("/query")
@limiter.limit('3/minute')
async def get_query_rag(request : Request, query: str):
    try:
        return query
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gets the model predictions for color and clothing type
@router.post('/image_predict') # Somehow combine with the celery function below figure it out
async def get_image_model(request: Request):
    try:
        image_model = request.app.state.image_model
        return image_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/process/{image_id}') # CELERY FUNCTION NOT DONE
def image_modeling(image_id: str):
    task = process_image(image_id)
    return {"task_id": task.id}