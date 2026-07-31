"""
Face detection module.
"""

import cv2

from config import FACE_CASCADE


class FaceDetector:
    """
    Detect human faces using OpenCV Haar Cascade.
    """

    def __init__(self):

        self.detector = cv2.CascadeClassifier(str(FACE_CASCADE))

        if self.detector.empty():
            raise RuntimeError(
                f"Unable to load Haar Cascade:\n{FACE_CASCADE}"
            )

    def detect_faces(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        return faces