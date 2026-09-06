from celery import Celery
from kombu import Queue
from fastapi import HTTPException, Request
import torch
from .config import Settings

celery_app = Celery('fashionproject')

# No need to load from env when you do this
celery_app.config_from_object(Settings)

celery_app.task_routes = {
    'tasks.predict_img' : {'queue' : 'queue_image'}
}

@celery_app.task(name='predict-img', queue='queue_image') # Use the model and img from routers
def process_img(request: Request, tensor: torch.Tensor):
    try:
        image_model = request.app.state.image_model
        preds = image_model(tensor)
        return preds
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))