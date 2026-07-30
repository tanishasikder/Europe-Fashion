'''Ties everything together in fastapi'''

from fastapi.responses import RedirectResponse
from supabase import create_client, Client
from dotenv import load_dotenv
import os
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from PIL import Image
from src.api.schemas.input import ClothingRequest
from typing import Optional, List
from src.core import load_models
from src.api.schemas.input import upload
from core.lifespan import lifespan
from fastapi import FastAPI
import mlflow
import os

model_name = os.getenv('IMAGE_MODEL_NAME')

supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

app = FastAPI(lifespan=lifespan)

SUPABASE_BUCKET = supabase.storage.from_(os.getenv('BUCKET_NAME'))

client = mlflow.MlflowClient()
version = client.get_latest_versions(name=model_name)[0].version
model_uri = f'models:/{model_name}/{version}'

image_model = mlflow.keras.load_model(model_uri)


# GO ON CLAUDE AND FIX ERRORS


