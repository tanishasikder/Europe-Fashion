import os
import sys
import torch
from torchvision import transforms
import torchvision.models as models
# Loading in the custom model
from src.models.image_extraction import CNN
from pathlib import Path
import sys
from joblib import load

# Makes python looks at the parent root directories to find the model
parent = Path(__file__).parent
path = parent / "stats_model.joblib"

# deffo wrong try again
stats_model = load(path)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Loops through the file names and stores all colors and categories
def get_color_category():
    path = 'Fashion_Images/train'
    files = os.listdir(path)
    color = [file[:file.index('_')] for file in files]
    category = [file[file.index('_')+1:] for file in files]

    return color, category

def initialize_stats_model():
    path = parent / "stats_model.joblib"
    # deffo wrong try again
    stats_model = load(path)
    return stats_model

def initialize_image_model():
    color, category = get_color_category()
    # Loading in the clothing predict model with error handling 
    image_model = CNN(color, category)
    # Loading in custom weights
    torch_path = parent / "image_extraction_model.pth"
    image_model.load_state_dict(torch.load(torch_path, map_location='cpu'))
    image_model.eval()

    return image_model