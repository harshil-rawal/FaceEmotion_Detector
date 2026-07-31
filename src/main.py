import argparse

from data_loader import preprocess_image
from emotion_detector import EmotionDetector


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Face Emotion Detector"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image",
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    detector = EmotionDetector()

    image = preprocess_image(args.image)

    result = detector.predict(image)

    print("=" * 50)
    print("Prediction Result")
    print("=" * 50)

    print(f"Image      : {args.image}")
    print(f"Emotion    : {result['emotion']}")
    print(f"Confidence : {result['confidence']:.4f}")


if __name__ == "__main__":
    main()