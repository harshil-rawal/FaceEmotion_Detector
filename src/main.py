from config import TEST_IMAGES_DIR
from data_loader import preprocess_image
from emotion_detector import EmotionDetector


def main():

    detector = EmotionDetector()

    image = preprocess_image(
        TEST_IMAGES_DIR / "39.jpg"
    )

    result = detector.predict(image)

    print("=" * 50)
    print("Prediction Pipeline")
    print("=" * 50)

    print(result)


if __name__ == "__main__":
    main()