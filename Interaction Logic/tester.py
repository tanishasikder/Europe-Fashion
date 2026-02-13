from datetime import datetime, timedelta
import os
import uuid
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import sys

path = 'Fashion_Images/train'
files = os.listdir(path)

color = []
category = []
'''
for file in files:
    underscore = file.index('_')
    color.append(file[: underscore])
    category.append(file[underscore + 1:])

print(color)
print(category)
'''
color = [file[:file.index('_')] for file in files]
category = [file[file.index('_')+1:] for file in files]

print(color)
print(category)