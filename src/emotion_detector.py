"""
Emotion detection CNN model.
"""

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Activation,
)
from tensorflow.keras.regularizers import l2


class EmotionDetector:

    EMOTION_LABELS = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    ]

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):

        model = Sequential(name="EmotionDetectorCNN")

        model.add(Input(shape=(48, 48, 1)))

        model.add(
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(1e-4),
            )
        )

        model.add(
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(1e-4),
            )
        )

        model.add(MaxPooling2D((2, 2)))

        model.add(
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(1e-4),
            )
        )

        model.add(MaxPooling2D((2, 2)))

        model.add(
            Conv2D(
                128,
                (3, 3),
                activation="relu",
                kernel_regularizer=l2(1e-4),
            )
        )

        model.add(MaxPooling2D((2, 2)))

        model.add(
            Conv2D(
                7,
                (1, 1),
                activation="relu",
                kernel_regularizer=l2(1e-4),
            )
        )

        model.add(
            Conv2D(
                7,
                (4, 4),
                activation="relu",
                kernel_regularizer=l2(1e-4),
            )
        )

        model.add(Flatten())
        model.add(Activation("softmax"))

        return model

    def predict(self, image):

        probabilities = self.model.predict(image, verbose=0)

        class_id = int(np.argmax(probabilities))

        confidence = float(probabilities[0][class_id])

        return {
            "class_id": class_id,
            "emotion": self.EMOTION_LABELS[class_id],
            "confidence": confidence,
        }