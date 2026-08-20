import mlflow.pytorch
import torch
import os
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
import torch.optim as optim
from torch.optim import lr_scheduler
import mlflow
import dagshub
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split, TensorDataset

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[2]

sys.path.insert(0, str(root_dir)) # Fix the path before importing below classes

from src.process.process_color import get_color_data, get_colors
from src.core.tracking_config import Dagshub_Track
from src.models.architecture.image_extraction import CNN
from src.process.process_color import ColorData, get_color_data
from src.process.process_image import get_type_labels
from src.models.data_models.image_data import ImageData

load_dotenv()

image_dir = os.environ.get('CROPPED_IMAGES')
annotation = os.environ.get('ANNOTATION_DIR') # Might not need this we'll see
image_path = os.environ.get('IMAGE_MODEL')

train_labels = os.environ.get('TRAIN_LABELS')
sets = ['train', 'test']

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

                        # Gets the largest score then calculates loss
                        #_, color_pred = torch.max(color, 1)
                        #color_loss = criterion(color_pred, label[:, 0])

                        _, cat_pred = torch.max(cat, 1)
                        cat_loss = criterion(cat_pred, label[:, 1])

                        _, attr_pred = torch.max(attr, 1)
                        attr_loss = criterion(attr_pred, label[:, 2])

                        # Overall loss from both predictions
                        loss = cat_loss #+ color_loss + attr_loss

                        # Optimizes and backward propagates if it is training
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    
                    # Calculates the loss and correct labels
                    run_loss += loss.item() * input.size(0)
                    #correct += torch.sum(color_pred == (label[:, 0]))
                    correct += torch.sum(cat_pred == (label[:, 1]))
                    correct += torch.sum(attr_pred == (label[:, 2]))
            
            # Overall loss and accuracy of this model
            epoch_loss = run_loss / dataset_sizes[phase]
            #print(f'data_size phase{dataset_sizes}')
            epoch_accuracy = correct / (2* dataset_sizes[phase])

            print(f'{phase} Loss : {epoch_loss:.3f} Accuracy : {epoch_accuracy:.3f}')

            # If it is validation, find the best model by finding the best accuracy
            if phase == 'test' and epoch_accuracy > best_accuracy:
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
    track = Dagshub_Track() # Comment this out for now. Uncomment when everything works
    track.initialize()

    # Finding all the images in the folder
    dataset = ImageData()

    # Splitting the dataset into train test
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    train, test = random_split(dataset, [train_size, test_size])
    # Use the custom class and functions for the color data
    co_train, co_test = get_color_data()
     
    # Loading the data in batches. Separate dataloaders for color and type tests
    fashion_loaders = {
        'train' : DataLoader(train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True),
        'test' : DataLoader(test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)                       
    }

    color_loaders = { # Gets custom class of loaded in color images
        'train' : DataLoader(co_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True),
        'test' : DataLoader(co_test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {x : (train_size, test_size) for x in sets}

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