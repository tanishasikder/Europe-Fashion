from src.core.celery_app import celery_app
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import FastAPI, Form, Request, status
from pydantic import ValidationError
import torch

@celery_app.task(name='predict-img') # Use the model and img from routers
def process_img(request: Request, tensor: torch.Tensor):
    try:
        image_model = request.app.state.image_model
        preds = image_model(tensor)
        return preds
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))