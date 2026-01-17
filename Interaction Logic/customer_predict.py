'''
This program is about using FastAPI to handle user requests
'''

from fastapi import Body, FastAPI, File, UploadFile
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Loading in the clothing predict model
image_model = models.vgg16(pretrained=False)
image_model.load_state_dict(torch.load("image_extraction_model"), map_location='cpu') 
image_model.eval()

# Loading in the stats predict model
with open('sales_predict.pkl', 'rb') as f:
    stats_model = pickle.load(f)

app = FastAPI()

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
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

async def image_model_output(file: UploadFile = File(...)):
    # Wait for the file to be read
    contents = await file.read()
    # Extract the image from content
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = data_transforms(image).unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        predictions = image_model(image)
    
    return predictions.cpu().numpy()

@app.post("/submit")
async def get_user_params(
    data : ClothingParameters, 
    file : UploadFile = File(...), 
    select : int = Body(...)
):
    outputs = await image_model_output(file)

    # Combine the input parameters with calculated values
    user_params = [
        data.size or '',
        data.catalog_price or 0,
        data.unit_price or 0,
        data.original_price or 0,
        data.channel or '',
        outputs
    ]
    # Predict with the other model
    result = stats_model.clothing_predict([user_params, select])
    return result.tolist()



