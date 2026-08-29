from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

redis = os.getenv('REDIS_URL')

celery_app = Celery(
    'fashionproject',
    broker = f'{redis}/0',
    backend=f'{redis}/1'
)