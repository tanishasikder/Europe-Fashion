import torch
import pickle
from dotenv import load_dotenv
import os
from src.api.schemas import ImageService, StatsService
from src.api.services import initialize_stats_model, initialize_image_model

load_dotenv()

clothing_pred = None
sales_pred = None

def load_models():
    # Load models at startup
    global ImageService, StatsService

    clothing_pred = initialize_image_model(ImageService)
    sales_pred = initialize_stats_model(StatsService)

    return clothing_pred, sales_pred
    