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
EPOCHS = 5000
LR = 1e-3
HIDDEN_WIDTH = 256
HIDDEN_DEPTH = 3
grayscale = True
ch = 1 if grayscale else 3  # Number of channels (1 for grayscale, 3 for RGB)
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Image
img = load_image(IMAGE_PATH, grayscale)  # (H, W, ch), values in [0, 255]

# 2. Prepare Dataset
X, Y = prepare_dataset(img)  # X: (N, 2), Y: (N, ch), both X, Y normalized
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

# 3. Define Model
model = ImageMLP(HIDDEN_WIDTH, HIDDEN_DEPTH, ch).to(device)

# 4. Train Model
losses = train(model, X_tensor, Y_tensor, EPOCHS, LR, device)

# 5. Plot Loss
plot_loss(losses)

# 6. Generate Output Image
with torch.no_grad():
    output = model(X_tensor).cpu().numpy()

H, W, _ = img.shape
output_image = (output.reshape(H, W, -1) * 255).astype(np.uint8)

plt.imsave(os.path.join(OUTPUT_DIR, "reconstructed.png"), output_image)
# Display the reconstructed image
plt.imshow(output_image)
plt.title("Reconstructed Image")
plt.axis("off")
plt.show()

print("✅ Training complete. Output saved in 'outputs/' folder.")
