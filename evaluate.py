import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_dataset, split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Arabic digit CNN.")
    parser.add_argument("--data-dir", required=True, help="Directory containing digit folders 0-9.")
    parser.add_argument(
        "--model-path",
        default="models_optimized_v2/best_model.keras",
        help="Path to a trained Keras model.",
    )
    parser.add_argument(
        "--results-dir",
        default="results_optimized_v2",
        help="Directory for the evaluation report and confusion matrix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(model_path)
    images, labels = load_dataset(args.data_dir)
    (_, _), (_, _), (x_test, y_test) = split_dataset(images, labels)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == y_test)
    report = classification_report(
        y_test,
        predictions,
        labels=list(range(10)),
        target_names=[str(digit) for digit in range(10)],
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=list(range(10)))

    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(report)
    with (results_dir / "classification_report.txt").open(
        "w", encoding="utf-8"
    ) as report_file:
        report_file.write(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        report_file.write(report)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=f"Confusion matrix (accuracy: {accuracy * 100:.2f}%)",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=range(10),
        yticks=range(10),
    )
    for row in range(10):
        for column in range(10):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    main()
