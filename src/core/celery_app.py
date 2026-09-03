from celery import Celery
import os
from dotenv import load_dotenv
from kombu import Queue
from fastapi import HTTPException, Request
import torch

load_dotenv()

redis = os.getenv('REDIS_URL')

celery_app = Celery(
    'fashionproject',
    broker = f'{redis}/0',
    backend=f'{redis}/1'
)

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