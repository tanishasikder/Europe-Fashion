from contextlib import asynccontextmanager
import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client
import mlflow
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Loading in the custom model
from src.models import initialize_image_model
from src.models import initialize_stats_model
from routers.input_router import router as input_router
from routers.db_router import router as db_router
from src.core.limiter import limiter

def database():
    supabase: Client = create_client(
        os.getenv('SUPABASE_URL'), # Initialize database
        os.getenv('SUPABASE_KEY')
    )

    bucket = supabase.storage.from_(os.getenv('BUCKET_NAME'))
    return bucket

@asynccontextmanager
async def lifespan(app: FastAPI):  # Problematic. Look more into this
    app.state.image_model = initialize_image_model()
    app.state.stats_model = initialize_stats_model()
    yield

def cloud_image_model(): # Gets model from DagsHub
    model_name = os.getenv('IMAGE_MODEL_NAME')
    client = mlflow.MlflowClient()
    version = client.get_latest_versions(name=model_name)[0].version
    model_uri = f'models:/{model_name}/{version}'

    image_model = mlflow.keras.load_model(model_uri)

    return image_model

app = FastAPI(lifespan=lifespan)
    
app.mount("/static", StaticFiles(directory="./"))

app.include_router(input_router)
app.include_router(db_router)

app.state.limiter = limiter # Initializes the rate limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="./templates") # Use this for db_router stuff

    