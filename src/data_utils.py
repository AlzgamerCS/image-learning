from PIL import Image
import torch
import numpy as np


def load_image(path, grayscale=False):
    img = Image.open(path).convert('RGB')
    img_array = np.array(img)
    return img_array


def prepare_dataset(image_np):
    H, W, C = image_np.shape
    rows, cols = np.indices((H,W))
    X = np.stack([rows.ravel(), cols.ravel()], axis = 1)
    Y = image_np.reshape(-1, C)
    Y = Y.astype(float)/255.0
    return X, Y



