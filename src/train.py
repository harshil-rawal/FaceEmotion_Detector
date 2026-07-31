"""
Train the Facial Emotion Recognition model.
"""

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from emotion_detector import EmotionDetector
from dataset import get_data_generators
import os


def main():

    # Load dataset
    train_generator, validation_generator = get_data_generators()

    # Build model
    detector = EmotionDetector()
    model = detector.model

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath="models/emotion_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )
    if os.path.exists("models/emotion_model.keras"):
        os.remove("models/emotion_model.keras")

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        verbose=1,
    )

    # Train
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=100,      # Increase later
        callbacks=[
            checkpoint,
            early_stop,
            reduce_lr,
        ],
    )


if __name__ == "__main__":
    main()