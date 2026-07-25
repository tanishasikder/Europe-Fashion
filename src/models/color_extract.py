import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path
from PIL import Image

def read_in(file_path):
    #Read in image and convert to RGB
    image_list = []

    for root, dirs, files in os.walk(file_path):
        for n in files:
            #n = str(n)
            fp = os.path.join(root, n)
            #fp = fp.replace('\\', '/')
            img = Image.open(fp).convert('RGB')
            image_list.append(img)

    return image_list

def preprocess(img):
    # Image transformations, resize image
    resized = img.resize((50, 200))

    arr = np.array(resized)
    float = arr.astype(np.float32) / 255.0

    h, w, c = float.shape # Saving the original shape
    pixels = float.reshape(-1, c)  # Making it 2d dimensions for KMeans

    return pixels, (h, w, c)

def fit_model(pixels, dimen):
    h, w, c, = dimen
    kmeans = KMeans(n_clusters=7, random_state=42, n_init="auto")
    kmeans.fit(pixels)

    labels = kmeans.labels_  # Cluster assignment for each point
    centroids = kmeans.cluster_centers_  # Coordinates of cluster centers

    segmented_pixels = centroids[labels]
    segmented_img = segmented_pixels.reshape(h, w, c)
    return segmented_img

def list_files(train_path, val_path):
    train_names = [item.name for item in train_path.iterdir()]
    val_names = [item.name for item in val_path.iterdir()]

    return train_names, val_names


def looper():
    # Get directory of current folder
    current_dir = Path(__file__).resolve().parent

    # Go up top-most folder
    root_dir = current_dir.parents[1]

    train_path = root_dir / 'data' / 'Fashion_Images' / 'train'
    val_path = root_dir / 'data' / 'Fashion_Images' / 'val'

    train_names, val_names = list_files(train_path, val_path)

    train_images = read_in(train_path)

    #train = train_path.replace("\\\\", "/")
    #train = train.replace("\\", "/")
    '''
    # Doing train for now because not all images are finished
    train_images = read_in(train)
    '''

    for img in train_images:
        img.show()

looper()

