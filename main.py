import cv2
from Tracking import Tracking

tr = Tracking()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    print(tr.get_pose_landmarks(img, True))
    print(tr.get_hand_landmarks(img, True))

    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break