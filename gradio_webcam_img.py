import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)

# Load the model with the custom layer
model = load_model('siamesemodelv5.keras', custom_objects={'L1Dist': L1Dist})

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(image, (100, 100))
    img = img / 255.0
    return img

def verify(image, model, detection_threshold=0.99, verification_threshold=0.8):
    input_img = preprocess(image)
    results = []
    verification_images = os.listdir(os.path.join('application_data', 'verification_images'))
    for image_name in verification_images:
        validation_image_path = os.path.join('application_data', 'verification_images', image_name)
        validation_img = preprocess(cv2.imread(validation_image_path))
        result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
        results.append(result)
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    verified = verification > verification_threshold
    return {"verification_results": str(results), "verified": "Verified" if verified else "Unverified"}

def capture_and_save(image):
    logging.info("Button clicked.")
    if image is None:
        logging.warning("No image data received.")
        return None, "No image captured"

    logging.info("Image data received.")
    # Crop the image as per the specified area
    cropped_image = image[120:120+250, 200:200+250]

    # Define the save path
    SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
    # Ensure the directory exists
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Convert the cropped image from RGBA to BGR for OpenCV compatibility
    image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(SAVE_PATH, image_bgr)
    logging.info(f"Image saved as {SAVE_PATH}.")

    # Perform verification
    verification_result = verify(image_bgr, model)
    logging.info(f"Verification result: {verification_result['verified']}")

    return image_bgr, verification_result

# Setup the Gradio interface
with gr.Blocks() as demo:
    webcam_input = gr.Image(sources="webcam", streaming=True, type="numpy", label="Webcam Input")
    output_img = gr.Image(label="Output Image")
    verification_output = gr.Textbox(label="Verification Output")
    capture_button = gr.Button("Capture and Verify Image")

    capture_button.click(
        fn=capture_and_save,
        inputs=webcam_input,
        outputs=[output_img, verification_output]
    )

# Launch the Gradio app
demo.launch()
