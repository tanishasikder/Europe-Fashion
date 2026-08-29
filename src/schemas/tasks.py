from src.core.celery_app import celery_app

@celery_app.task
def process_image():
    pass