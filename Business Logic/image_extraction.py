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

data_dir = r'C:\Users\Tanis\Downloads\Europe-Fashion\Fashion_Images'
sets = ['train', 'val']



# TESTING
for set_name in ['train', 'val']:
    set_path = os.path.join(data_dir, set_name)
    print(f"\n{'='*50}")
    print(f"Checking: {set_path}")
    print(f"Exists: {os.path.exists(set_path)}")
    
    if os.path.exists(set_path):
        class_folders = os.listdir(set_path)
        print(f"Found {len(class_folders)} class folders")
        
        for class_folder in class_folders[:3]:  # Check first 3 folders
            class_path = os.path.join(set_path, class_folder)
            if os.path.isdir(class_path):
                files = os.listdir(class_path)
                print(f"\n  {class_folder}/")
                print(f"    Total items: {len(files)}")
                
                # Show first 5 files
                for file in files[:5]:
                    file_path = os.path.join(class_path, file)
                    print(f"    - {file} (is_file: {os.path.isfile(file_path)})")





# Getting the data based on the train/val sets then doing transformations
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),
                                           data_transforms[x],
                                           is_valid_file = lambda path: path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
                                           for x in sets}

# Loading the data in batches. Separate dataloaders for color and type tests
data_loaders = {
    'train' : DataLoader(image_datasets['train'], batch_size=32, 
                               shuffle=True, num_workers=4, pin_memory=True),
    'val' : DataLoader(image_datasets['val'], batch_size=32, 
                             shuffle=False, num_workers=4, pin_memory=True)                       
}

for input, label in data_loaders['train']:
    print(f"input:{input}")
    print(f"label:{label}")

dataset_sizes = {x : len(image_datasets[x]) for x in sets}
color_names = image_datasets['train'].classes
type_names = image_datasets['train'].classes

for color in color_names:
    dash = color.index('_')
    color_index = color_names.index(color)
    replace_color = color[0:dash]
    color_names[color_index] = replace_color

for type in type_names:
    dash = type.index('_')
    type_index = type_names.index(type)
    replace_type = type[dash:]
    type_names[type_index] = replace_type




# TESTING
print(f"Looking for images in: {data_dir}")
for set_name in sets:
    path = os.path.join(data_dir, set_name)
    print(f"\n{set_name} path: {path}")
    print(f"Exists: {os.path.exists(path)}")
    if os.path.exists(path):
        print(f"Contents: {os.listdir(path)}")




# CNN class to classify image features
class CNN(nn.Module):
    def __init__(self, color_names, type_names, device):
        super().__init__()
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
        features = self.resnet(x)
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
    run_loss = 0.0
    best_model = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        # Switch between training and validation
        for phase in sets:
            if phase == 'train':
                model.train()
            else:
                model.eval()

        loss = 0.0
        correct = 0

        # Loop over the labels and the images in the dataloader
        for input, label in data_loaders[phase]:
            inputs = input.to(device)
            labels = label.to(device)

            with torch.set_grad_enabled(phase=='train'):
                # Gets the outputs from resnet model
                color, type = model(inputs)

                # Gets the largest score then calculates loss
                _, color_pred = torch.max(color, 1)
                color_loss = criterion(color, labels)
                
                _, type_pred = torch.max(type, 1)
                type_loss = criterion(type, labels)

                # Overall loss from both predictions
                loss = type_loss + color_loss
                preds = color_pred + type_pred

                # Optimizes and backward propagates if it is training
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Calculates the loss and correct labels
            run_loss += loss.item() * inputs.size(0)
            correct += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()
        
        # Overall loss and accuracy of this model
        epoch_loss = run_loss / dataset_sizes[phase]
        epoch_accuracy = correct.double() / dataset_sizes[phase]

        print(f'{phase} Loss : {epoch_loss:.3f} Accuracy : {epoch_accuracy:.3f}')

        # If it is validation, find the best model by finding the best accuracy
        if phase == 'val' and epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model = copy.deepcopy(model.state_dict())

    print(f'Best model accuracy: {best_accuracy:.3f}')
    model.load_state_dict(best_model)
    return model

# Initializing the final model with all the parameters
model = train_model(model, criterion, optimizer, step_lr, num_epochs=20)

# Save the model
torch.save(model.state_dict(), "resnet18_clothing_model.pth")