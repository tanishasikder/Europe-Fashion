from contextlib import asynccontextmanager
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
from PIL import Image

# Makes python looks at the parent root directories to find the model
parent = Path(__file__).parent
path = parent / "stats_model.joblib"

# deffo wrong try again
stats_model = load(path)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

class ImageService:
    def __init__(self, model_path, transform):
        self.model = torch.load(model_path)
        self.transform = transform
    
    def predict(self, image: Image.Image):
        tensor = self.transform(image)
        with torch.no_grad():
            color, category = self.model(tensor)
            # Predict class from logits
            co_pred = color.argmax(dim=1).item()
            ca_pred = category.argmax(dim=1).item()

        return co_pred, ca_pred
    
class StatsService:
    def __init__(self, model, color_la, type_la):
        self.model = model
        self.color_labels = color_la
        self.type_label = type_la

    def predict(self, labels):
        pred = self.model(labels)

        return pred