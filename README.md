# 🧠 Image Copy Net (image-learning)

This project explores a simple deep learning method to **memorize and recreate an image** using a coordinate-based neural network. The network receives pixel coordinates `(x, y)` as input and learns to predict either:

- Brightness (for grayscale images)
- RGB values (for color images)

This is a beginner-friendly project for learning **PyTorch**, **image processing**, and **GitHub collaboration**.

---

## 📌 Project Goals

- Train a neural network to "memorize" an image pixel-by-pixel.
- Understand how neural networks can approximate visual functions.
- Practice PyTorch, image preprocessing, unit testing, and Git/GitHub collaboration.

---

## 🗂️ Project Structure

image_learning/  
├── data/ # Input images (PNG, JPG)  
├── src/ # Source code  
│ ├── model.py # Neural network model  
│ ├── data_utils.py # Image loading and preprocessing  
│ ├── train.py # Training loop  
│ ├── test.py # Image reconstruction  
│ └── config.py # Configs and constants  
├── tests/ # Unit tests  
├── main.py # Entry point (WIP)  
├── requirements.txt # Python dependencies  
├── README.md # Project documentation  
└── .gitignore # Git ignore rules

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AlzgamerCS/image-learning.git
cd image-learning
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python src/train.py
```

### 5. Test Image Reconstruction

```bash
python src/test.py
```

### 🧪 Testing

Run all unit tests using:

```bash
pytest # inside the root directory
```

---

## 🧑‍💻 Development

### TODOs

data_utils.py: Add support for RGB and grayscale images ✅ Done  
model.py: Implement a simple MLP architecture ✅ Done  
train.py : Implement training loop with loss calculation  
main.py: Create an entry point to load an image, train the model, and visualize results ✅ Done

Run

```bash
pytest tests/test_image_loading.py
```

to test the image loading functionality (already implemented, working). However, no tests for the model for now.

👥 Contributors

Kuanysh Murat  
Merey Bissenbin
