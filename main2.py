# Import standard dependencies
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
import os
import uuid

# Ensure GPU memory growth is enabled
gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus: ', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print('cur gpu: ', gpu)


# Define the embedding model
def make_embedding():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(pool_size=(2, 2), padding='same')(c1)

    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(pool_size=(2, 2), padding='same')(c2)

    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(pool_size=(2, 2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=inp, outputs=d1, name='embedding')


# Create the embedding model
embedding_model = make_embedding()

# Show a summary of the embedding model
embedding_model.summary()


# Siamese L1 Distance class
class L1Dist(Layer):

    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, inputs):
        # Unpack the inputs list into two separate tensors
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)


# Instantiate the L1 distance layer
l1 = L1Dist()
print('l1: ', l1)


# Define the siamese model
def make_siamese_model(embedding_model):
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100, 100, 3))

    # Validation image in the network
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    # Get the embeddings for the input and validation images
    embedded_input_image = embedding_model(input_image)
    embedded_validation_image = embedding_model(validation_image)

    # Combine siamese distance components
    siamese_layer = L1Dist()
    # Pass the embeddings into the siamese layer
    distances = siamese_layer([embedded_input_image, embedded_validation_image])

    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    # Create and return the Siamese Network model
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


# Create the siamese model using the embedding model
siamese_model = make_siamese_model(embedding_model)

# If you want to print the model summary to check its architecture
siamese_model.summary()