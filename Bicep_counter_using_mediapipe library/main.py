import imp
import cv2
from matplotlib.pyplot import draw
import mediapipe as mp
import numpy as np
from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_INITIATE


def calculate_distance(fingur, thumb):
    dist = np.sqrt(np.square(fingur[0]- thumb[0]) + np.square(fingur[1]- thumb[1]))
    # dist = dist * 0.1
    max = 0.033
    percent = (dist/max)* 100
    # percent = (percent/170)* 100

    if percent > 520:
        return 499
    elif percent < 30:
        return 30
    else:
        return percent

def draw_rect(image, percent):
    cv2.rectangle(image,(30,620), (500,570),(255,255,5), 2)
    cv2.rectangle(image, (30,620),(int(percent),570),(255,255,5), -2)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
blurImg = 0
hands = mp_hands.Hands()
while True:
    data, image = cap.read()
    image = cv2.resize(image, None, None, fx=1.5, fy=1.5)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # print(mp_hands.HAND_CONNECTIONS.INDEX_FINGER_TIP)
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            indexFingur = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            thumbTip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            percent = calculate_distance(indexFingur, thumbTip)

            print(percent)
            draw_rect(image, percent)
            image = cv2.blur(image,(int(percent * 0.08),10))
    cv2.imshow("handtracker", image)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

