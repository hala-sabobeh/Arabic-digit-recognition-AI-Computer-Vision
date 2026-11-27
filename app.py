import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr

from src.preprocess import preprocess_image


def create_demo(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)

    def predict(image):
        if image is None:
            raise ValueError("Draw or upload a digit before predicting.")
        batch = preprocess_image(image)[np.newaxis, ...]
        probabilities = model.predict(batch, verbose=0)[0]
        top_digits = np.argsort(probabilities)[-3:][::-1]
        return {str(digit): float(probabilities[digit]) for digit in top_digits}

    with gr.Blocks(title="Arabic Digit Recognition") as demo:
        gr.Markdown("# Arabic Handwritten Digit Recognition")
        with gr.Row():
            with gr.Column():
                sketch = gr.Sketchpad(label="Draw a digit", image_mode="RGB")
                sketch_button = gr.Button("Predict drawing")
                sketch_result = gr.Label(label="Top predictions", num_top_classes=3)
            with gr.Column():
                upload = gr.Image(label="Upload an image", type="numpy", image_mode="RGB")
                upload_button = gr.Button("Predict image")
                upload_result = gr.Label(label="Top predictions", num_top_classes=3)
        sketch_button.click(predict, inputs=sketch, outputs=sketch_result)
        upload_button.click(predict, inputs=upload, outputs=upload_result)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch the Gradio digit demo.")
    parser.add_argument(
        "--model-path",
        default="models_optimized_v2/best_model.keras",
        help="Path to a trained Keras model.",
    )
    args = parser.parse_args()
    create_demo(args.model_path).launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
