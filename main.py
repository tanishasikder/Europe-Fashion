from contextlib import asynccontextmanager
import joblib
import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routes import clothing_router

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Loading in the custom model
from models import initialize_image_model
from models import initialize_stats_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.image_model = initialize_image_model()
    app.state.stats_model = initialize_stats_model()
    yield

app = FastAPI(lifespan=lifespan)

#app.mount("/static", StaticFiles(directory="./"))
templates = Jinja2Templates(directory="./templates")

app.include_router(clothing_router.router)