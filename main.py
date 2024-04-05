import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# Define the input tensor
input_tensor = Input(shape=(784,))

# Define the model with a single dense layer
output_tensor = Dense(10, activation='softmax')(input_tensor)

# Create the model object
model = Model(inputs=input_tensor, outputs=output_tensor)

# Print model summary
model.summary()