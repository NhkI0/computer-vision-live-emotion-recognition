import cv2
import numpy as np


def adjusted_face_detect(frame: np.ndarray) -> np.ndarray:
    face_image = frame.copy()
    face_rect = face_cascade.detectMultiScale(face_image, 1.3, 5)

    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return face_image


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", adjusted_face_detect(frame))
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("preview")
