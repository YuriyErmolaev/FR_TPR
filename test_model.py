import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model

# Define the custom L1 Distance layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)

# Load the model with the custom layer
model = load_model('siamesemodelv5.keras', custom_objects={'L1Dist': L1Dist})

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

def verify(image_path, model, detection_threshold=0.99, verification_threshold=0.8):
    results = []
    input_img = preprocess(image_path)
    for image_name in os.listdir(os.path.join('application_data', 'verification_images')):
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image_name))
        result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold
    return results, verified

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[120:120+250, 200:200+250, :]
        cv2.imshow('Verification', cropped_frame)

        key = cv2.waitKey(1)  # Read the key pressed
        if key != -1:  # Check if a key was pressed
            if key & 0xFF == ord('v'):
                SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
                cv2.imwrite(SAVE_PATH, cropped_frame)
                results, verified = verify(SAVE_PATH, model)
                print('Verification1:', 'Verified' if verified else 'Unverified')
            if key & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
