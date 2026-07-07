import torch
import pickle
from dotenv import load_dotenv
import os
from api.services.initialize import initialize_image_model, initialize_stats_model

load_dotenv()

clothing_pred = None
sales_pred = None

def load_models():
    # Load models at startup
    global clothing_pred, sales_pred

    clothing_pred = initialize_image_model()
    sales_pred = initialize_stats_model()
    