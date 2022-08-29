import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    heith, width = img.shape[0], img.shape[1]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)
    
    if results.multi_hand_landmarks:
        for one_hand in results.multi_hand_landmarks:
            for num, cors in enumerate(one_hand.landmark):
                if num == 0:
                    cor_x, cor_y = int(cors.x*width), int(cors.y*heith)
                    cv2.circle(img, (cor_x, cor_y), 20, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, one_hand, mp_hands.HAND_CONNECTIONS)
            

    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break