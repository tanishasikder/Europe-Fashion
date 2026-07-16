from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.schemas.initialize import ImageService, StatsService
from src.api.services.state import initialize_stats_model, initialize_image_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.image_model = initialize_image_model(ImageService)
    app.state.stats_model = initialize_stats_model(StatsService)
    yield