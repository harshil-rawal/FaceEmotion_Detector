from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATASET_DIR

def get_data_generators():

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR / "train",
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        shuffle=True,
    )

    validation_generator = val_datagen.flow_from_directory(
        DATASET_DIR / "validation",
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=64,
        class_mode="categorical",
        shuffle=False,
    )
    print(train_generator.class_indices)
    return train_generator, validation_generator