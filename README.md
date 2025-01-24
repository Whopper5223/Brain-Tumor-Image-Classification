# Brain-Tumor-Image-Classification
TensorFlow-based CNN for binary classification to detect the presence or absence of tumors in images. Includes preprocessing, data augmentation, and optimization techniques like early stopping and learning rate scheduling to improve performance and prevent overfitting.
# Binary Image Classification for Tumor Detection

This repository implements a Convolutional Neural Network (CNN) using TensorFlow to classify images into two categories: **Absent** (no tumor) or **Present** (tumor detected). The project includes data preprocessing, augmentation, and training optimizations to ensure robust and accurate predictions.

---

## Features
- **Binary Classification**: Detects the presence or absence of tumors in image data.
- **Data Preprocessing**: Removes corrupt images and normalizes pixel values.
- **Data Augmentation**: Applies random transformations (flipping, rotation, zoom, contrast) to increase dataset variability.
- **Training Optimizations**:
  - Early Stopping
  - Learning Rate Scheduling
- **Metrics**: Includes precision, recall, and binary accuracy.

---

## Setup

### Prerequisites
- Python 3.8 or later
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/binary-tumor-detection.git
   cd binary-tumor-detection
