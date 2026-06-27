from contextlib import asynccontextmanager
import joblib
import os
import sys
from fastapi import FastAPI

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

