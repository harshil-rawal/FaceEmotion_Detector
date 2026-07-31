"""
Utility functions used throughout the Face Emotion Detector project.
"""

import cv2


def draw_test(window_name, predicted_label, image, true_label):
    """
    Display the prediction result on an image.

    Args:
        window_name (str): OpenCV window title.
        predicted_label (str): Emotion predicted by the model.
        image (numpy.ndarray): Input image.
        true_label (str): Actual emotion label.
    """

    border_color = [0, 0, 0]

    canvas = cv2.copyMakeBorder(
        image,
        top=160,
        bottom=0,
        left=0,
        right=300,
        borderType=cv2.BORDER_CONSTANT,
        value=border_color,
    )

    cv2.putText(
        canvas,
        f"Predicted : {predicted_label}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        canvas,
        f"Actual    : {true_label}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow(window_name, canvas)