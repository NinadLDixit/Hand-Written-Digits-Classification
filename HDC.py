import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import seaborn as sn

# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

# Normalize the data
X_train = X_train / 255
X_test = X_test / 255

# Flatten the data
X_train_flat = X_train.reshape(len(X_train), 784)
X_test_flat = X_test.reshape(len(X_test), 784)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(X_train_flat, Y_train, epochs=5)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_flat, Y_test)
print(f"Test Accuracy: {accuracy}")

# Save the trained model as .h5 file
model.save('model.h5')

