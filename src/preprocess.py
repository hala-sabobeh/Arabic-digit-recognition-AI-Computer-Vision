from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image, image_size=(64, 64)):
    if isinstance(image, dict):
        image = image.get("composite")
        if image is None:
            raise ValueError("The sketch does not contain a composite image.")

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            image = ImageOps.exif_transpose(opened_image).copy()

    if isinstance(image, Image.Image):
        image = ImageOps.exif_transpose(image)
        if "A" in image.getbands():
            foreground = image.convert("RGBA")
            background = Image.new("RGBA", foreground.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, foreground)
        image = np.asarray(image.convert("RGB"))

    image = np.asarray(image)
    if image.size == 0 or not np.issubdtype(image.dtype, np.number):
        raise ValueError("The input image must contain numeric pixel values.")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all():
            raise ValueError("The input image contains non-finite pixel values.")
        if image.max() <= 1:
            image = image * 255

    if image.ndim == 3:
        if image.shape[2] == 4:
            color = image[:, :, :3].astype(np.float32)
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            image = color * alpha + 255.0 * (1.0 - alpha)
        if image.shape[2] not in (3, 4):
            raise ValueError("Expected a grayscale, RGB, or RGBA image.")
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    elif image.ndim != 2:
        raise ValueError("Expected a grayscale, RGB, or RGBA image.")

    image = np.clip(image, 0, 255).astype(np.uint8)

    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    if np.median(border) < 127:
        image = cv2.bitwise_not(image)

    image = cv2.resize(image, image_size)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return (image.astype(np.float32) / 255.0)[:, :, np.newaxis]
