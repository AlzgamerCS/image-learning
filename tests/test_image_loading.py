import os
import sys
from src.data_utils import load_image
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_load_and_show_image():
    image_path = "data/darwin.png"  # <-- Replace with your actual image file

    assert os.path.exists(image_path), f"Image not found: {image_path}"

    # Test grayscale load
    gray_img = load_image(image_path, grayscale=True)
    assert gray_img.ndim == 2, "Expected 2D grayscale image"
    assert gray_img.max() <= 1.0 and gray_img.min() >= 0.0

    plt.imshow(gray_img, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")
    plt.show()

    # Test RGB load
    rgb_img = load_image(image_path, grayscale=False)
    assert rgb_img.ndim == 3 and rgb_img.shape[2] == 3, "Expected 3D RGB image"
    assert rgb_img.max() <= 1.0 and rgb_img.min() >= 0.0

    plt.imshow(rgb_img)
    plt.title("RGB Image")
    plt.axis("off")
    plt.show()

    print("✅ Image loading and display test passed.")
