"""
Live webcam face detection.
"""

import cv2

from face_detector import FaceDetector


class WebcamEmotionDetector:

    def __init__(self):

        self.face_detector = FaceDetector()

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

    # Draw face bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

    # Display label above the face
                cv2.putText(
                    frame,
                    "Face",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "Face Emotion Detector",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()