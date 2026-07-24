import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances

#Read in image and convert to RGB
img = Image.open('data/Fashion_Images/train/black_dress/download (1).jpg').convert('RGB')

# Image transformations, resize image
resized = img.resize((50, 200))

arr = np.array(resized)
float = arr.astype(np.float32) / 255.0

h, w, c, = float.shape # Saving the original shape
pixels = float.reshape(-1, c)  # Making it 2d dimensions for KMeans

kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
kmeans.fit(pixels)

labels = kmeans.labels_  # Cluster assignment for each point
centroids = kmeans.cluster_centers_  # Coordinates of cluster centers

segmented_pixels = centroids[labels]
segmented_img = segmented_pixels.reshape(h, w, c)


