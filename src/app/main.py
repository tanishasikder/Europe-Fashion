from contextlib import asynccontextmanager
import joblib
import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.app.routers import clothing_router
from supabase import create_client, Client
import mlflow
from fastapi import APIRouter

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Loading in the custom model
from src.models import initialize_image_model
from src.models import initialize_stats_model

def database():
    supabase: Client = create_client(
        os.getenv('SUPABASE_URL'), # Initialize database
        os.getenv('SUPABASE_KEY')
    )

    bucket = supabase.storage.from_(os.getenv('BUCKET_NAME'))
    return bucket

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.image_model = initialize_image_model()
    app.state.stats_model = initialize_stats_model()
    yield

def cloud_image_model():
    model_name = os.getenv('IMAGE_MODEL_NAME')
    client = mlflow.MlflowClient()
    version = client.get_latest_versions(name=model_name)[0].version
    model_uri = f'models:/{model_name}/{version}'

    image_model = mlflow.keras.load_model(model_uri)

    return image_model

def start_app():
    app = FastAPI(lifespan=lifespan)

    #app.mount("/static", StaticFiles(directory="./"))
    #templates = Jinja2Templates(directory="./templates")

    app.include_router()