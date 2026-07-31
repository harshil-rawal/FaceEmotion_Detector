"""
Functions for loading and preprocessing images.
"""

import cv2
import numpy as np


def preprocess_image(image_path, image_size=(48, 48)):
    """
    Load an image from disk and preprocess it for emotion prediction.

    Steps:
    1. Read image
    2. Convert to grayscale
    3. Resize to 48x48
    4. Normalize pixel values
    5. Expand dimensions for CNN

    Returns:
        numpy.ndarray
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, image_size)

    gray = gray.astype("float32") / 255.0

    gray = np.expand_dims(gray, axis=-1)

    gray = np.expand_dims(gray, axis=0)

    return gray