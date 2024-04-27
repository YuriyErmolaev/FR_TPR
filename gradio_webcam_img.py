import gradio as gr
import cv2
import numpy as np
from datetime import datetime
import logging

# Configure logging to display information on the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def capture_and_save(image):
    logging.info("Button clicked.")
    # Check if any image data is received
    if image is None:
        logging.warning("No image data received.")
        return None, "No image captured"
    else:
        logging.info("Image data received.")

    try:
        # Ensure the image is a numpy array with the correct type
        image = np.array(image, dtype=np.uint8)
        logging.info("Image array converted successfully.")

        # Convert the image from RGBA to BGR for OpenCV compatibility
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        logging.info("Image converted from RGBA to BGR.")

        # Generate a timestamped filename and save the image
        filename = f'captured_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename, image_bgr)
        logging.info(f"Image saved as {filename}.")

        return image_bgr, f"Image saved as {filename}"
    except Exception as e:
        logging.error(f"Error processing the image: {e}")
        return None, f"Error: {str(e)}"


# Setup the Gradio interface
with gr.Blocks() as demo:
    webcam_input = gr.Image(sources="webcam", streaming=True, type="numpy", label="Webcam Input")
    output_img = gr.Image(label="Output Image")
    save_text = gr.Textbox(label="Save Status")
    capture_button = gr.Button("Capture Image")

    capture_button.click(
        fn=capture_and_save,
        inputs=webcam_input,
        outputs=[output_img, save_text]
    )

# Launch the Gradio app
demo.launch()
