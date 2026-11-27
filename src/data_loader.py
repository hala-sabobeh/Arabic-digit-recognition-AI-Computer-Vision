from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess_image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def load_dataset(data_dir):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    images = []
    labels = []
    for digit in range(10):
        digit_dir = data_dir / str(digit)
        if not digit_dir.is_dir():
            raise FileNotFoundError(f"Missing digit folder: {digit_dir}")

        image_paths = sorted(
            path
            for path in digit_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            raise ValueError(f"No supported images found in {digit_dir}")

        for image_path in image_paths:
            images.append(preprocess_image(image_path))
            labels.append(digit)

    return np.asarray(images), np.asarray(labels, dtype=np.int64)


def split_dataset(images, labels):
    holdout_size = int(np.ceil(len(images) * 0.15))
    remaining_images, test_images, remaining_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=holdout_size,
        random_state=42,
        stratify=labels,
    )
    train_images, validation_images, train_labels, validation_labels = train_test_split(
        remaining_images,
        remaining_labels,
        test_size=holdout_size,
        random_state=42,
        stratify=remaining_labels,
    )
    return (
        (train_images, train_labels),
        (validation_images, validation_labels),
        (test_images, test_labels),
    )
