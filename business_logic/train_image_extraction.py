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
import torch.multiprocessing as mp
import numpy as np
from PIL import Image
from image_extraction import CNN

# Push to GPU if it is available, CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Used the normalize the inputs
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#Data transformations for the train and val sets
data_transforms = {
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

data_dir = r'C:\Users\Tanis\Downloads\Europe-Fashion\Fashion_Images'
sets = ['train', 'val']

def get_valid_image(path):
    path_lower = path.lower()

    if path_lower.endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
        return True

    return False

# Train on the last layer of the resnet to learn clothing features
def train_model(model, criterion, optimizer, scheduler, num_epochs=None):
    best_model = copy.deepcopy(model.state_dict())
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
                    color, clothing_type = model(inputs)

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

                    _, type_pred = torch.max(clothing_type, 1)
                    type_loss = criterion(clothing_type, type_labels)

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
            best_model = copy.deepcopy(model.state_dict())

        scheduler.step()
    print(f'Best model accuracy: {best_accuracy:.3f}')
    model.load_state_dict(best_model)
    return model

# Required for multi-processing in Windows
# Executes code that start everything
if __name__ == '__main__':
    # Getting the data based on the train/val sets then doing transformations
    image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x],
                                            is_valid_file = get_valid_image)
                                            for x in sets}
        
    # Loading the data in batches. Separate dataloaders for color and type tests
    data_loaders = {
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

    for type in type_names:
        dash = type.index('_')
        type_index = type_names.index(type)
        replace_type = type[dash+1:]
        type_names[type_index] = replace_type

    model = CNN(color_names, type_names)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Every 7 epochs the learning rate is multiplied by gamma
    step_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
    # Initializing the final model with all the parameters
    model = train_model(model, criterion, optimizer, step_lr, num_epochs=10)
    # Save the model
    torch.save(model.state_dict(), 'C:/Users/Tanis/Downloads/Europe-Fashion/business_logic/image_extraction_model.pth')
