from contextlib import asynccontextmanager
import os
import sys
from fastapi import FastAPI
import torch
from torchvision import transforms
import torchvision.models as models
# Loading in the custom model
from src.models.image_extraction import CNN
from pathlib import Path
import sys
from joblib import load
from PIL import Image
import numpy as np
from src.api.services.initialize import StatsService

# Makes python looks at the parent root directories to find the model
parent = Path(__file__).parent
path = parent / "stats_model.joblib"

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Transformations for user input images
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loops through the file names and stores all colors and categories
def get_color_category():
    path = 'Fashion_Images/train'
    files = os.listdir(path)
    color = [file[:file.index('_')] for file in files]
    category = [file[file.index('_')+1:] for file in files]

    return color, category

@asynccontextmanager
async def initialize_stats_model(app: FastAPI):
    path = parent / "stats_model.joblib"
    # deffo wrong try again
    stats_model = load(path)
    yield
    return StatsService(stats_model)

async def initialize_image_model(app: FastAPI):
    color, category = get_color_category()
    # Loading in the clothing predict model with error handling 
    image_model = CNN(color, category)
    # Loading in custom weights
    torch_path = parent / "image_extraction_model.pth"
    image_model.load_state_dict(torch.load(torch_path, map_location=device))
    yield
    image_model.eval()

    return image_model