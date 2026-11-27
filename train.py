import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.cnn_model import create_cnn_model
from src.data_loader import load_dataset, split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Arabic digit CNN.")
    parser.add_argument("--data-dir", required=True, help="Directory containing digit folders 0-9.")
    parser.add_argument(
        "--model-path",
        default="models_optimized_v2/best_model.keras",
        help="Path for the best validation checkpoint.",
    )
    parser.add_argument(
        "--results-dir",
        default="results_optimized_v2",
        help="Directory for training plots and metrics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)
    tf.random.set_seed(42)

    model_path = Path(args.model_path)
    results_dir = Path(args.results_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    images, labels = load_dataset(args.data_dir)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(
        images, labels
    )
    y_train_categorical = tf.keras.utils.to_categorical(y_train, 10)
    y_val_categorical = tf.keras.utils.to_categorical(y_val, 10)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        shear_range=0.05,
        fill_mode="nearest",
    )
    datagen.fit(x_train)

    model = create_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=1e-4,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode="min",
        ),
    ]

    history = model.fit(
        datagen.flow(x_train, y_train_categorical, batch_size=32),
        validation_data=(x_val, y_val_categorical),
        epochs=45,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=len(x_train) // 32,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    model.save(model_path.with_name("final_model.keras"))
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Model accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.9, 1.0])

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set(title="Model loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(results_dir / "training_history.png", dpi=300)
    plt.close(figure)

    with (results_dir / "summary.txt").open("w", encoding="utf-8") as summary:
        summary.write(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)\n")
        summary.write(f"Test loss: {test_loss:.4f}\n")


if __name__ == "__main__":
    main()
