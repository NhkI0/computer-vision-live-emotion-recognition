import cv2
import numpy as np
from models.cnn.predict import predict_emotion

CLAHE = False
# Smoothing factor: 0 = no update, 1 = no smoothing
SMOOTHING = 0.3
prev_faces = []


def smooth_faces(new_faces: list, prev: list, alpha: float) -> list:
    """Smooth bounding boxes between frames using exponential moving average."""
    if not prev:
        return [np.array(f, dtype=float) for f in new_faces]

    smoothed = []
    for nf in new_faces:
        # Find closest previous face by center distance
        nf = np.array(nf, dtype=float)
        nc = np.array([nf[0] + nf[2] / 2, nf[1] + nf[3] / 2])
        best = None
        best_dist = float('inf')

        for pf in prev:
            pc = np.array([pf[0] + pf[2] / 2, pf[1] + pf[3] / 2])
            d = np.linalg.norm(nc - pc)
            if d < best_dist:
                best_dist = d
                best = pf

        if best is not None and best_dist < max(nf[2], nf[3]) * 1.5:
            smoothed.append(best * (1 - alpha) + nf * alpha)
        else:
            smoothed.append(nf)
    return smoothed


def adjusted_face_detect(im_frame: np.ndarray) -> np.ndarray:
    """Return the square drawn image located on the face of the provided frame."""
    global prev_faces
    face_image = im_frame.copy()
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_rect = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=7, minSize=(80, 80)
    )

    if len(face_rect) > 0:
        prev_faces = smooth_faces(list(face_rect), prev_faces, SMOOTHING)
    else:
        prev_faces = []

    for (x, y, w, h) in prev_faces:
        x, y, w, h = int(x), int(y), int(w), int(h)
        face_crop = im_frame[y:y + h, x:x + w].copy()

        classification, confidence = predict_emotion(face_crop)

        if classification == "Positive":
            color = (0, 255, 0)
        elif classification == "Negative":
            color = (0, 0, 255)
        else:
            color = (255, 180, 0)

        label = f"{classification} ({confidence:.0%})"
        cv2.putText(face_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        cv2.rectangle(face_image, (x, y), (x + w, y + h), color, 2)

    return face_image


face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

cv2.namedWindow("Webcam preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    frame = None

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

while rval:
    if CLAHE:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imshow("Webcam preview", adjusted_face_detect(frame))
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("Webcam preview")
