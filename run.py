"""This module runs the app."""
import time
from cv2 import cv2
from detection import HandDetector


def main():
    """This will be executed on run."""
    # pylint: disable=[invalid-name]
    current_time = 0
    previous_time = 0

    # pylint: disable=[c-extension-no-member]
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        _, img = cap.read()

        img = detector.detect_hands(img)
        current_time = time.time()
        fps = int(1 / (current_time - previous_time))
        previous_time = current_time

        img = cv2.flip(img, 1)
        cv2.putText(
            img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2
        )
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
