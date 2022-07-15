"""This module contains the class with hand detector."""
import mediapipe as mp
import numpy as np


class HandDetector:
    """This class creates a hand detector.
    Args:
        mode: Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream.
        max_num_hands: Maximum number of hands to detect. See details in
        min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
            detection to be considered successful.
        min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
            hand landmarks to be considered tracked successfully.
    """

    def __init__(
        self,
        mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.hands = mp.solutions.hands.Hands(
            self.mode,
            self.max_num_hands,
            1,
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        """This method performs hand detection.

        Args:
            img: the image on which the detection will take place.
            draw: whether to draw the detected hands or not.

        Returns:
            Image with the detected hands.

        """
        results = self.hands.process(img)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmark, mp.solutions.hands.HAND_CONNECTIONS
                    )
        return img
