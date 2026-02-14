from datetime import datetime, timedelta
import os
import uuid
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import sys
from joblib import load

# Loading in the custom model
from image_extraction import CNN

# Makes python looks at the parent root directories to find the model
DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(DIR, "business_logic", "stats_model.joblib")

try:
    stats_model = load(path)
except Exception as e:
    raise

app = FastAPI()

# Temporary storage for image features
# Store image_id, image characteristics, when uploaded
# (so it can expire after an hour) for predictions
image_storage = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initial parameters to predict with later on
class ClothingParameters(BaseModel):
    size : Optional[str] = None
    catalog_price : Optional[float] = None
    channel : Optional[str] = None
    original_price : Optional[float] = None
    unit_price : Optional[float] = None

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Transformations for user input images
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

async def image_model_output(file: UploadFile) -> Image.Image:
    # Make sure the file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image. Try again")
    # Wait for the file to be read
    contents = await file.read()
    try:
        path = 'Fashion_Images/train'
        files = os.listdir(path)

        color = []
        category = []
        color = [file[:file.index('_')] for file in files]
        category = [file[file.index('_')+1:] for file in files]

        # Loading in the clothing predict model with error handling 
        image_model = CNN(color, category)
        # Loading in custom weights
        image_model.load_state_dict(torch.load("image_extraction_model.pth", map_location='cpu'))
        image_model.eval()
        
        # Extract the image from content
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Create image ID
        image_id = str(uuid.uuid4())
        image = data_transforms(image).unsqueeze(0)
        # Perform inference
        with torch.no_grad():
            predictions = image_model(image)

        image_storage[image_id] = {
            'features': predictions,
            'uploaded': datetime.now()
        }
    except Exception as e:
        raise HTTPException(stats_code=500, detail="Sorry. Prediction Failed")
    
def remove_expired_images():
    # Remove images after an hour
    for key, data in image_storage:
        upload = image_storage[key]['uploaded']
        if upload >= upload + timedelta(hours=1):
            image_storage.pop(upload)

@app.post("/predict")
async def get_user_params(
    matrix: List[List[ClothingParameters]], 
    file : UploadFile = File(...), 
    select : int = Body(...)
):
    try:
        category, color = await image_model_output(file)
        numerical_outputs = []
        # Combine the input parameters with calculated values
        for data in matrix:
            user_params = [
                category,
                color,
                data.size or '',
                data.catalog_price or 0,
                data.channel or '',
                data.original_price or 0,
                data.unit_price or 0
            ]
            # Predict with the other model
            result = stats_model.clothing_predict([user_params, select])
            numerical_outputs.append(result.tolist())
        
        return category, color, numerical_outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")


