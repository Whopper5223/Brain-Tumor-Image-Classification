import tensorflow as tf
import cv2
import os
import numpy as np
import imghdr
import matplotlib
from matplotlib import pyplot as plt
# Import augmentation layers
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from keras import regularizers


Desk_path = os.path.expanduser('~/Desktop')
project_path = 'NN Mushroom'
data_path = 'dataset'
data_add = os.path.join(Desk_path, project_path, data_path)
viable_extensions = ['jpeg', 'jpg', 'png']


test_path = os.path.join(Desk_path, project_path, 'TEST')


model_path = os.path.join(Desk_path, project_path, 'Model', 'CatVsDogModel2.h5')
CatVsDogModel = load_model(model_path)

# Ensure all images in the TEST folder are valid
for subset in os.listdir(test_path):
    if subset != ".DS_Store":  # Skip system files
        subset_path = os.path.join(test_path, subset)
        for img in os.listdir(subset_path):
            img_path = os.path.join(subset_path, img)
            try:
                image = cv2.imread(img_path)
                img_ext = imghdr.what(img_path)
                if img_ext not in viable_extensions:
                    print(f"Not a viable extension: {img}")
                    os.remove(img_path)
            except:
                print(f"Cannot read image: {img}")
                os.remove(img_path)

# Initialize counters
correct_predictions = 0
total_predictions = 0

# Predict the class for each image in the TEST folder
for subset in os.listdir(test_path):
    if subset != ".DS_Store":
        print(f"\nPredicting images in folder: {subset}")
        subset_path = os.path.join(test_path, subset)
        for img in os.listdir(subset_path):
            img_path = os.path.join(subset_path, img)
            try:
                # Read and preprocess the image
                image = cv2.imread(img_path)
                resized_img = tf.image.resize(image, (256, 256))  # Resize to model input size
                scaled_img = resized_img / 255.0  # Normalize pixel values

                # Make prediction
                yhat = CatVsDogModel.predict(np.expand_dims(scaled_img, 0))
                predicted_class = "Dog" if yhat > 0.5 else "Cat"

                # Check if prediction is correct
                if predicted_class.lower() == subset.lower():  # Compare to ground truth (folder name)
                    correct_predictions += 1
                total_predictions += 1

                print(f"Image: {img} - Predicted class: {predicted_class}")
            except Exception as e:
                print(f"Error processing image {img}: {e}")

# Calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")
