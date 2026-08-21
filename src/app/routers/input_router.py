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
from src.app.services.model_service.predict import image_output
from main import limiter 

router = APIRouter(prefix='preds')

# Basic health check to ensure server is functioning
@router.get("/health")
def root():
    return {"status" : "OK"}

@router.post("/upload") # Use with schemas/input function get_upload
@limiter.limit('3/minute') # How much we limit
async def upload(
        request : Request, # Need this or limiter will not work
        file: UploadFile = File(...),
        size: str = Form(...),
        price: str = Form(...)
    ):
    try:
        contents = await file.read()
        return contents, size, price

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

# Use with services/model_service/predict function query_rag_system
@router.get("/query/")
@limiter.limit('3/minute')
async def get_query_rag(request : Request, query: str):
    try:
        return query
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))