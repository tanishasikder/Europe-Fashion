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

# Push to GPU if it is available, CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN class to classify image features
class CNN(nn.Module):
    def __init__(self, color_names, type_names):
        super().__init__()
        # Load in the pretrained resnet model
        model = models.vgg16(pretrained=True)

        # Freeze parameters
        for param in model.features.parameters():
            param.requires_grad = False

        self.vgg16_features = model.features
        self.avgpool = model.avgpool
        # Assign a fully connected layer containing the class names
        num_features = 512 * 7 * 7
        # Head to classify the color
        # len of color_names and type_names is 25 each
        self.fc_color = nn.Linear(num_features, len(color_names))
        #self.dropout1 = nn.Dropout(0.5)
        # Head to classify the clothing type
        self.fc_type = nn.Linear(num_features, len(type_names))
        #self.dropout2 = nn.Dropout(0.5)
        self.to(device)
    
    def forward(self, x):
        # Gather features and assign it to the color and type heads
        x = self.vgg16_features(x)
        x = self.avgpool(x)
        # Flatten the features so it can be used in linear layers
        # Goes from [batch, 512, 1, 1] to [batch, 512]
        x = torch.flatten(x, 1)
        #x = self.dropout1(x)
        color = self.fc_color(x)
        #x = self.dropout2(x)
        type = self.fc_type(x)
        # Return the classification
        return color, type
