import mlflow.pytorch
import torch
import os
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import mlflow
import dagshub
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[1]

sys.path.insert(0, str(root_dir)) # Fix the path before importing below classes

from src.process.process_color import color_transform, get_colors
from src.core.tracking_config import Dagshub_Track
from src.models.image_extraction import CNN
from src.process.process_color import ColorData, get_color_data
from src.process.process_image import get_type_labels, fashion_transform

load_dotenv()

image_dir = os.environ.get('IMAGE_FASHION_DIR')
annotation = os.environ.get('ANNOTATION_DIR') # Might not need this we'll see
image_path = os.environ.get('IMAGE_MODEL')

train_labels = os.environ.get('TRAIN_LABELS')
sets = ['train', 'val']

def get_valid_image(path):
    path_lower = path.lower()

    if path_lower.endswith(('.jpg')):
        return True

    return False

# Train on the last layer of the resnet to learn clothing features
def train_model(model, criterion, optimizer, scheduler, num_epochs=None):
    with mlflow.start_run():
        best_model = model.state_dict()
        best_accuracy = 0.0
        best_model = None

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            # Switch between training and validation
            for phase in sets:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                run_loss = 0.0
                loss = 0.0
                correct = 0

                # Loop over the labels and the images in the dataloader
                for input, label in fashion_loaders[phase]:

                    with torch.set_grad_enabled(phase=='train'):
                        # Gets the outputs from resnet model
                        color, cat, attr = model(input)

                        color_labels = get_colors() 
                        cat_labels, attr_labels = get_type_labels()

                        color_labels = color_labels.values()

                        # Gets the largest score then calculates loss
                        _, color_pred = torch.max(color, 1)
                        color_loss = criterion(color_pred, color_labels)

                        _, cat_pred = torch.max(cat, 1)
                        cat_loss = criterion(cat_pred, cat_labels)

                        _, attr_pred = torch.max(attr, 1)
                        attr_loss = criterion(attr_pred, attr_labels)

                        # Overall loss from both predictions
                        loss = cat_loss + color_loss + attr_loss

                        # Optimizes and backward propagates if it is training
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    
                    # Calculates the loss and correct labels
                    run_loss += loss.item() * input.size(0)
                    correct += torch.sum(color_pred == (color_labels))
                    correct += torch.sum(cat_pred == (cat_labels))
                    correct += torch.sum(attr_pred == (attr_labels))
            
            # Overall loss and accuracy of this model
            epoch_loss = run_loss / dataset_sizes[phase]
            #print(f'data_size phase{dataset_sizes}')
            epoch_accuracy = correct / (2* dataset_sizes[phase])

            print(f'{phase} Loss : {epoch_loss:.3f} Accuracy : {epoch_accuracy:.3f}')

            # If it is validation, find the best model by finding the best accuracy
            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                mlflow.log_metric('best accuracy', best_accuracy)
                model_info = mlflow.pytorch.log_model(model, name='europe-fashion-image-extract')
                mlflow.register_model(model_uri=f"models:/{model_info.model_id}", name='europe-fashion-image-extract')
                best_model = mlflow.pytorch.load_model(model_uri=f"models:/{model_info.model_id}")

            scheduler.step()
        print(f'Best model accuracy: {best_accuracy:.3f}')
        return best_model.state_dict()   # return the best model's weights and params

# Required for multi-processing in Windows
# Executes code that start everything
if __name__ == '__main__':
    # Use the DagsHub Mlflow server to log things
    track = Dagshub_Track()
    track.initialize()
    
    # Getting the data based on the train/val sets then doing transformations
    image_datasets = {x : datasets.ImageFolder(os.path.join(image_dir, x),
                                            fashion_transform(x),
                                            is_valid_file = get_valid_image)
                                            for x in sets}

    # Use the custom class and functions for the color data
    train, test = get_color_data()
     
    # Loading the data in batches. Separate dataloaders for color and type tests
    fashion_loaders = {
        'train' : DataLoader(image_datasets['train'], batch_size=32, 
                                shuffle=True, num_workers=4, pin_memory=True),
        'val' : DataLoader(image_datasets['val'], batch_size=32, 
                                shuffle=False, num_workers=4, pin_memory=True)                       
    }

    color_loaders = {
        'train' : DataLoader(train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True),
        'val' : DataLoader(test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    }

    color = ColorData()
    dataset_sizes = {x : len(image_datasets[x]) for x in sets}

    # Configuring with color and clothing classes. Removing dashes
    codes = get_colors() # Dict where the values have the colors as strings
    clothing, attr = get_type_labels()

    model = CNN(list(codes.values()), clothing, attr)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Every 7 epochs the learning rate is multiplied by gamma
    step_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
    # Initializing the final model with all the parameters
    model = train_model(model, criterion, optimizer, step_lr, num_epochs=10)
    # Save the model locally too
    torch.save(model, image_path)