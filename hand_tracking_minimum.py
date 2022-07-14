"""This module contains a simple code with hand tracking minimum."""
import time

from cv2 import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for index, landmark in enumerate(hand_landmark.landmark):
                height, width, channel = img.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                print(index, center_x, center_y)
                if index == 0:
                    cv2.circle(img, (center_x, center_y), 25, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)

    current_time = time.time()
    previous_time = 0
    fps = int(1 / (current_time - previous_time))
    previous_time = current_time

    cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
