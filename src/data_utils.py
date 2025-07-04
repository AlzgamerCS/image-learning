from PIL import Image
import torch
import numpy as np


def load_image(path, grayscale=True):
    img = Image.open(path)
    img = img.convert("L" if grayscale else "RGB")
    img_array = np.array(img)
    if grayscale:
        img_array = img_array[..., np.newaxis]
    return img_array


def prepare_dataset(image_np):
    H, W, C = image_np.shape
    rows, cols = np.indices((H, W))
    max_dim = max(H - 1, W - 1)  # Use max(H, W) to keep proportions (aspect ratio)
    
    rows = rows / max_dim
    cols = cols / max_dim

    X = np.stack([rows.ravel(), cols.ravel()], axis=1)
    Y = image_np.reshape(-1, C).astype(float) / 255.0

    return X, Y
