from emotion_detector import EmotionDetector


def main():
    detector = EmotionDetector()

    print("=" * 50)
    print("Face Emotion Detector")
    print("=" * 50)
    print("CNN architecture created successfully.\n")

    detector.summary()


if __name__ == "__main__":
    main()