from celery import Celery
from kombu import Queue
from fastapi import HTTPException, Request
import torch
from .config import Settings
from src.services.model_service.predict import image_output

celery_app = Celery('fashionproject')

# No need to load from env when you do this
celery_app.config_from_object(Settings)

celery_app.task_routes = {
    'tasks.predict_img' : {'queue' : 'queue_image'}
}

@celery_app.task(name='predict-img', queue='queue_image') # Use the model and img from routers
def process_img(image_bytes: bytes):
    try:
        preds = image_output(image_bytes) # Opens, preprocesses, and predicts
        # NEED TO DECODE THE PREDS YOU HAVE A FUNCTION TO DO THIS BUT USE
        # THE ENUMS MAYBE IDK RESEARCH OR FIX THE PREWRITTEN FUNCTIONS
        return preds
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))