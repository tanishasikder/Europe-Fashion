import torch
import os
import copy
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image

# Push to GPU if it is available, CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#Data transformations for the train and val sets
data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
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

data_dir = 'Europe-Fashion/Fashion_images'
sets = ['color_train', 'type_train', 'color_val', 'type_val']

# Getting the data based on the train/val sets then doing transformations
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),
                                           data_transforms[x])
                                           for x in sets}

# Loading the data in batches. Separate dataloaders for color and type tests
data_loaders = {
    'color_train' : DataLoader(image_datasets['color_train'], batch_size=32, 
                               shuffle=True, num_workers=4, pin_memory=True),
    'type_train' : DataLoader(image_datasets['type_train'], batch_size=32, 
                              shuffle=True, num_workers=4, pin_memory=True),
    'color_val' : DataLoader(image_datasets['color_val'], batch_size=32, 
                             shuffle=False, num_workers=4, pin_memory=True),                          
    'type_val' : DataLoader(image_datasets['type_val'], batch_size=32, 
                            shuffle=False, num_workers=4, pin_memory=True)
}

dataset_sizes = {x : len(image_datasets[x]) for x in sets}
color_names = image_datasets['color_train'].classes
type_names = image_datasets['type_train'].classes

# CNN class to classify image features
class CNN(nn.Module):
    def __init__(self, color_names, type_names, device):
        super().__init()
        # Load in the pretrained resnet model
        model = models.resnet18(pretrained=True)
        self.resnet = model

        # Freeze parameters
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Assign a fully connected layer containing the class names
        num_features = self.resnet.fc.in_features
        # Head to classify the color
        self.fc_color = nn.Linear(num_features, len(color_names))
        # Head to classify the clothing type
        self.fc_type = nn.Linear(num_features, len(type_names))
        self.resnet.to(device)
    
    def forward(self, x):
        # Gather features and assign it to the color and type heads
        features = self.resnet.features(x)
        color = self.fc_color(features)
        type = self.fc_type(features)
        # Return the classification
        return color, type

model = CNN(color_names, type_names, device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Every 7 epochs the learning rate is multiplied by gamma
step_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train on the last layer of the resnet to learn clothing features
def train_model(model, criterion, optimizer, scheduler, num_epochs=None):
    best_model = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    best_model = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        # Switch between training and validation
        for phase in sets:
            if phase == 'color_train' or 'type_train':
                model.train()
            else:
                model.eval()

        loss = 0.0
        correct = 0

        # Loop over the labels and the images in the dataloader
        # Changes between train/val/color/type
        for input, label in data_loaders[phase]:
            inputs = input.to(device)
            labels = label.to(device)

            with torch.set_grad_enabled(phase=='train'):
                # Gets the outputs from resnet model
                output = None
                color, type = model(inputs)
                track = phase.replace('_train', '')
                if track == 'color': 
                    output = color 
                else: 
                    output = type
                # Gets the largest score then calculates loss
                _, preds = torch.max(output, 1)
                loss = criterion(output, labels)

                # Optimizes and backward propagates if it is training
                if phase in ['color_train','type_train']:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Calculates the loss and correct labels
            run_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)

        if phase in ['color_train', 'type_train']:
            scheduler.step()
        
        # Overall loss and accuracy of this model
        epoch_loss = run_loss / dataset_sizes[phase]
        epoch_accuracy = correct.double() / dataset_sizes[phase]

        print(f'{phase} Loss : {epoch_loss:.3f} Accuracy : {epoch_accuracy:.3f}')

        # If it is validation, find the best model by finding the best accuracy
        if phase in ['color_val', 'type_val'] and epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model = copy.deepcopy(model.state_dict())

    print(f'Best model accuracy: {best_accuracy:.3f}')
    model.load_state_dict(best_model)
    return model

# Initializing the final model with all the parameters
model = train_model(model, criterion, optimizer, step_lr, num_epochs=20)

# Save the model
torch.save(model.state_dict(), "resnet18_clothing_model.pth")