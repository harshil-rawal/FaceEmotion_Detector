"""
Emotion detection CNN model.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.regularizers import l2


class EmotionDetector:
    """
    CNN architecture for facial emotion recognition.
    """

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential(name="EmotionDetectorCNN")

        model.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                kernel_regularizer=l2(0.0001),
                input_shape=(48, 48, 1),
            )
        )

        model.add(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                kernel_regularizer=l2(0.0001),
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                128,
                kernel_size=(3, 3),
                activation="relu",
                kernel_regularizer=l2(0.0001),
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                128,
                kernel_size=(3, 3),
                activation="relu",
                kernel_regularizer=l2(0.0001),
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(
                7,
                kernel_size=(1, 1),
                activation="relu",
                kernel_regularizer=l2(0.0001),
            )
        )

        model.add(
            Conv2D(
                7,
                kernel_size=(4, 4),
                activation="relu",
                kernel_regularizer=l2(0.0001),
            )
        )

        model.add(Flatten())
        model.add(Activation("softmax"))

        return model

    def summary(self):
        """Print the model summary."""
        self.model.summary()