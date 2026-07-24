import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path

def read_in(file):
    #Read in image and convert to RGB
    img = Image.open(file).convert('RGB')

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
    kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
    kmeans.fit(pixels)

    labels = kmeans.labels_  # Cluster assignment for each point
    centroids = kmeans.cluster_centers_  # Coordinates of cluster centers

    segmented_pixels = centroids[labels]
    segmented_img = segmented_pixels.reshape(h, w, c)

def list_files():
    # Get directory of current folder
    current_dir = Path(__file__).resolve().parent

    # Go up top-most folder
    root_dir = current_dir.parents[1]

    train_path = root_dir / 'data' / 'Fashion_Images' / 'train'
    val_path = root_dir / 'data' / 'Fashion_Images' / 'val'

    train_files = [item.name for item in train_path.iterdir()]
    val_files = [item.name for item in val_path.iterdir()]

    print(train_files)
    print(val_files)

list_files()
    