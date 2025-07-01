import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Ensure the root directory is in the path (now the test will work)

import numpy as np
from src.data_utils import load_image


def test_load_and_show_image():
    image_path = "data/darwin.jpg"  # <-- Replace with your actual image file

    assert os.path.exists(image_path), f"Image not found: {image_path}"

    # Test grayscale load
    gray_img = load_image(image_path, grayscale=True)
    assert gray_img.ndim == 2, "Expected 2D grayscale image"
    assert 0 <= gray_img.min() <= gray_img.max() <= 255
    # Test RGB load
    rgb_img = load_image(image_path, grayscale=False)
    assert rgb_img.ndim == 3 and rgb_img.shape[2] == 3, "Expected 3D RGB image"
    assert 0 <= rgb_img.min() <= rgb_img.max() <= 255

    print("✅ Image loading and display test passed.")
