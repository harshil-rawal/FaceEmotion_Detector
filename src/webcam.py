"""
Live webcam face detection.
"""

import cv2

from face_detector import FaceDetector
from data_loader import preprocess_image
from emotion_detector import EmotionDetector


class WebcamEmotionDetector:

    def __init__(self):

        self.face_detector = FaceDetector()
        self.detector = EmotionDetector()

    def run(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam.")

        print("Press Q to quit.")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            faces = self.face_detector.detect_faces(frame)

            for (x, y, w, h) in faces:

    # Crop detected face
                face = frame[y:y+h, x:x+w]

                try:
                    # Preprocess for CNN
                    processed = preprocess_image(face)

        # Predict emotion
                    result = self.detector.predict(processed)

                    label = f"{result['emotion']} ({result['confidence']:.2f})"

                except Exception:
                    label = "Unknown"

    # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

    # Draw label
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Face Emotion Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()