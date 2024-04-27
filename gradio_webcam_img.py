import gradio as gr
import cv2
import numpy as np
from datetime import datetime
import logging
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define the custom L1 Distance layer
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
    image = np.array(image, dtype=np.uint8)
    filename = f'captured_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
    cv2.imwrite(filename, image)
    logging.info(f"Image saved as {filename}.")

    # Perform verification
    verification_result = verify(image, model)
    logging.info(f"Verification result: {verification_result['verified']}")

    return image, verification_result


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
