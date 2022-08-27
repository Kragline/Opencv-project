import cv2

faces_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyes_db = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faces_db.detectMultiScale(gray_img, 1.1, 19)
    cv2.putText(img, 'Press Q to close', (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'X: {w} Y: {h}', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        gray_img_face = gray_img[y:y+h, x:x+w]
        eyes = eyes_db.detectMultiScale(gray_img_face, 1.1, 19)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x + ex, y + ey), (x +ex + ew, y + ey + eh), (0, 0, 255), 1)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break