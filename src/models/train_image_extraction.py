import mlflow.pytorch
import torch
import os
import copy
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

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[1]

sys.path.insert(0, str(root_dir))

from src.core.tracking_config import Dagshub_Track
from src.models.image_extraction import CNN
# Push to GPU if it is available, CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Fashion images have bigger transformations
fashion_transforms = {
    'train' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

color_transforms = {
    'train' : transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

image_dir = os.environ.get('IMAGE_FASHION_DIR')
color_dir = os.environ.get('COLOR_DIR')
annotation = os.environ.get('ANNOTATION_DIR')
image_path = os.environ.get('IMAGE_MODEL')

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
                for input, label in data_loaders[phase]:
                    inputs = input.to(device)
                    label = label.to(device)

                    with torch.set_grad_enabled(phase=='train'):
                        # Gets the outputs from resnet model
                        color, category, apparel, fg = model(inputs)

                        # Labels is a tensor of indices from the original file name
                        # Must separate labels to match color and clothing type
                        color_labels = torch.tensor([color_names.index(image_datasets[phase].classes[l].split('_')[0])
                                                    for l in label])
                        type_labels = torch.tensor([type_names.index(image_datasets[phase].classes[l].split('_')[1])
                                                    for l in label])

                        color_labels = color_labels.to(device)
                        type_labels = type_labels.to(device)

                        # Gets the largest score then calculates loss
                        _, color_pred = torch.max(color, 1)
                        color_loss = criterion(color, color_labels)

                        _, type_pred = torch.max(category, 1)
                        type_loss = criterion(category, type_labels)

                        # Overall loss from both predictions
                        loss = type_loss + color_loss

                        # Optimizes and backward propagates if it is training
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    
                    # Calculates the loss and correct labels
                    run_loss += loss.item() * inputs.size(0)
                    correct += torch.sum(color_pred == (color_labels))
                    correct += torch.sum(type_pred == (type_labels))
            
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
                                            fashion_transforms[x],
                                            is_valid_file = get_valid_image)
                                            for x in sets}

    # Need to separate stuff into train/val datasets
    color_datasets = {x : datasets.ImageFolder(os.path.join(color_dir, x),
                                               color_transforms[x],
                                               is_valid_file=get_valid_image)
                                               for x in sets}   
     
    # Loading the data in batches. Separate dataloaders for color and type tests
    fashion_loaders = {
        'train' : DataLoader(image_datasets['train'], batch_size=32, 
                                shuffle=True, num_workers=4, pin_memory=True),
        'val' : DataLoader(image_datasets['val'], batch_size=32, 
                                shuffle=False, num_workers=4, pin_memory=True)                       
    }

    dataset_sizes = {x : len(image_datasets[x]) for x in sets}
    color_names = image_datasets['train'].classes.copy()
    type_names = image_datasets['train'].classes.copy()

    # Configuring with color and clothing classes. Removing dashes
    for color in color_names:
        dash = color.index('_')
        color_index = color_names.index(color)
        replace_color = color[0:dash]
        color_names[color_index] = replace_color

    for cat in type_names:
        dash = cat.index('_')
        type_index = type_names.index(cat)
        replace_type = cat[dash+1:]
        type_names[type_index] = replace_type

    model = CNN(color_names, type_names)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Every 7 epochs the learning rate is multiplied by gamma
    step_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
    # Initializing the final model with all the parameters
    model = train_model(model, criterion, optimizer, step_lr, num_epochs=10)
    # Save the model locally too
    torch.save(model, image_path)