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

def add_centered_rectangle(image):
    h, w, _ = image.shape
    start_x = w // 2 - 125
    start_y = h // 2 - 125
    end_x = start_x + 250
    end_y = start_y + 250
    color = (255, 0, 0)  # Red color for the rectangle
    thickness = 2
    return cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, thickness)

def zoom_out(image, zoom_factor=0.5):
    new_width = int(image.shape[1] * zoom_factor)
    new_height = int(image.shape[0] * zoom_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    zoomed_out_image = np.zeros_like(image)
    start_x = (image.shape[1] - new_width) // 2
    start_y = (image.shape[0] - new_height) // 2
    zoomed_out_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image
    return zoomed_out_image

def update_stream(image):
    zoomed_out_image = zoom_out(image)
    return add_centered_rectangle(zoomed_out_image)

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, (100, 100))
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
    return results, verified

def capture_and_save(image):
    logging.info("Button clicked.")
    if image is None:
        logging.warning("No image data received.")
        return None, "No image captured"
    logging.info("Image data received.")
    reduced_image = zoom_out(image)
    image_with_rectangle = add_centered_rectangle(reduced_image)
    h, w, _ = reduced_image.shape
    start_x = w // 2 - 125
    start_y = h // 2 - 125
    end_x = start_x + 250
    end_y = start_y + 250
    cropped_image = reduced_image[start_y:end_y, start_x:end_x]
    SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(SAVE_PATH, image_bgr)
    logging.info(f"Image saved as {SAVE_PATH}.")
    results, verified = verify(image_bgr, model)
    verification_status = "Verified" if verified else "Unverified"
    logging.info(f"Verification result: {verification_status}")
    return image_with_rectangle, verification_status

# Setup the Gradio interface
with gr.Blocks() as demo:
    webcam_input = gr.Image(sources="webcam", streaming=True, type="numpy", label="Webcam Input")
    output_img = gr.Image(label="Output Image")
    verification_output = gr.Textbox(label="Verification Output")
    capture_button = gr.Button("Capture and Verify Image")
    webcam_input.change(fn=update_stream, inputs=webcam_input, outputs=output_img)
    capture_button.click(fn=capture_and_save, inputs=webcam_input, outputs=[output_img, verification_output])

# Launch the Gradio app
demo.launch()
