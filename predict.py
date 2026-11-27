import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocess import preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict an Arabic handwritten digit.")
    parser.add_argument("image", help="Path to an image containing a digit.")
    parser.add_argument(
        "--model-path",
        default="models_optimized_v2/best_model.keras",
        help="Path to a trained Keras model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    image = preprocess_image(args.image)[np.newaxis, ...]
    probabilities = model.predict(image, verbose=0)[0]
    top_digits = np.argsort(probabilities)[-3:][::-1]

    print(f"Prediction: {top_digits[0]}")
    print("Top 3:")
    for digit in top_digits:
        print(f"  {digit}: {probabilities[digit] * 100:.2f}%")


if __name__ == "__main__":
    main()
