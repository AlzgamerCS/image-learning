import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from src.data_utils import load_image, prepare_dataset
from src.model import ImageMLP
from src.train import train, plot_loss

# Hyperparameters
IMAGE_PATH = "data/darwin.jpg"
EPOCHS = 15
LR = 0.005
HIDDEN_WIDTH = 128
HIDDEN_DEPTH = 4
batch_size = 1024
grayscale = True
ch = 1 if grayscale else 3  # Number of channels (1 for grayscale, 3 for RGB)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Image
img = load_image(IMAGE_PATH, grayscale)  # (H, W, ch), values in [0, 255]
print("Step1")
# 2. Prepare Dataset
X, Y = prepare_dataset(img)  # X: (N, 2), Y: (N, ch), both X, Y normalized
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
print("Step2")

# 3. Define Model
model = ImageMLP(2,HIDDEN_WIDTH, HIDDEN_DEPTH, ch).to(device)
print("Step3")

# 4. Train Model
losses = train(model, X_tensor, Y_tensor, EPOCHS, LR, device, batch_size= batch_size)
print("Step4")

# 5. Plot Loss
plot_loss(losses)

# 6. Generate Output Image
# 6. Generate Output Image (Grayscale)
with torch.no_grad():
    output = model(X_tensor).cpu().numpy()  # shape (N, 1)

H, W, _ = img.shape

# Сжимаем последний размерности канала: (H*W, 1) → (H*W,)
gray_flat = output.squeeze(-1)
# Возвращаем 2D-форму (H, W)
gray_image = (gray_flat.reshape(H, W) * 255).astype(np.uint8)

# Сохраняем как grayscale
plt.imsave(
    os.path.join(OUTPUT_DIR, "reconstructed_gray.png"),
    gray_image,
    cmap="gray",
    vmin=0,
    vmax=255
)

# И выводим на экран в gray
plt.figure(figsize=(6,6))
plt.imshow(gray_image, cmap="gray", vmin=0, vmax=255)
plt.title("Reconstructed Grayscale Image")
plt.axis("off")
plt.show()
