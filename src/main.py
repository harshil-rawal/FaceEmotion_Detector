import argparse

from data_loader import preprocess_image
from emotion_detector import EmotionDetector
from webcam import WebcamEmotionDetector


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Face Emotion Detector"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image",
    )

    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run live webcam detection",
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    # Webcam mode
    if args.webcam:
        webcam = WebcamEmotionDetector()
        webcam.run()
        return

    # Image mode
    if args.image:

        detector = EmotionDetector()

        image = preprocess_image(args.image)

        result = detector.predict(image)

        print("=" * 50)
        print("Prediction Result")
        print("=" * 50)

        print(f"Image      : {args.image}")
        print(f"Emotion    : {result['emotion']}")
        print(f"Confidence : {result['confidence']:.4f}")

        return

    print("Please specify either:")
    print("  --image <path>")
    print("or")
    print("  --webcam")


if __name__ == "__main__":
    main()