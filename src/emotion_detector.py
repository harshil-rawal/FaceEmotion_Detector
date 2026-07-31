"""
Emotion detection CNN model.
"""

import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Activation,
    BatchNormalization,
    Dropout,
    Dense,
)
from tensorflow.keras.regularizers import l2

from config import MODELS_DIR


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

        model_path = MODELS_DIR / "emotion_model.keras"

        if model_path.exists():
            print(f"Loading trained model from {model_path}")
            self.model = load_model(model_path)
        else:
            print("Trained model not found. Building a new model.")
            self.model = self._build_model()

    def _build_model(self):

        model = Sequential(name="EmotionDetectorCNN_V2")

        model.add(Input(shape=(48, 48, 1)))

        # -------- Block 1 --------
        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                kernel_regularizer=l2(1e-4),
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(
            Conv2D(
                32,
                (3, 3),
                padding="same",
                kernel_regularizer=l2(1e-4),
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        # -------- Block 2 --------
        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                kernel_regularizer=l2(1e-4),
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(
            Conv2D(
                64,
                (3, 3),
                padding="same",
                kernel_regularizer=l2(1e-4),
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        # -------- Block 3 --------
        model.add(
            Conv2D(
                128,
                (3, 3),
                padding="same",
                kernel_regularizer=l2(1e-4),
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(
            Conv2D(
                128,
                (3, 3),
                padding="same",
                kernel_regularizer=l2(1e-4),
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))

        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))

        # -------- Classifier --------
        model.add(Flatten())

        model.add(
            Dense(
                256,
                activation="relu",
                kernel_regularizer=l2(1e-4),
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(7, activation="softmax"))

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