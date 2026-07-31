from config import (
    PROJECT_ROOT,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    TEST_IMAGES_DIR,
    OUTPUTS_DIR,
    DATASET_DIR,
    EMOTION_MODEL,
)


def main():
    print("=" * 60)
    print("Face Emotion Detector")
    print("=" * 60)

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Models       : {MODELS_DIR}")
    print(f"Notebooks    : {NOTEBOOKS_DIR}")
    print(f"Test Images  : {TEST_IMAGES_DIR}")
    print(f"Outputs      : {OUTPUTS_DIR}")
    print(f"Dataset      : {DATASET_DIR}")
    print(f"Model File   : {EMOTION_MODEL}")


if __name__ == "__main__":
    main()