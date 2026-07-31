import cv2
import numpy as np


def preprocess_image(image, image_size=(48, 48)):
    """
    Preprocess an image or webcam frame for emotion prediction.
    """

    if isinstance(image, str):
        image = cv2.imread(image)

    if image is None:
        raise ValueError("Unable to load image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, image_size)

    gray = gray.astype("float32") / 255.0

    gray = np.expand_dims(gray, axis=-1)

    gray = np.expand_dims(gray, axis=0)

    return gray