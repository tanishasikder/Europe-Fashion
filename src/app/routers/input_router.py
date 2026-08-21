from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import FastAPI, Form
from pydantic import ValidationError
import json
from PIL import Image
import io
#from validator import get_user_params
from src.app.services.model_service.predict import image_output

router = APIRouter()

# Basic health check to ensure server is functioning
@router.get("/health")
def root():
    return {"status" : "OK"}

@router.post("/upload") # Use with schemas/input function get_upload
async def upload(
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
async def get_query_rag(query: str):
    try:
        return query
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))