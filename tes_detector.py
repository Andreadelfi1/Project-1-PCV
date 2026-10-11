import cv2
from detector import HandDetector

det = HandDetector()
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    state, pos, white, skin = det.detect(frame)
    if pos:
        cv2.circle(frame, pos, 12, (0, 255, 0), -1)
    cv2.putText(frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("kamera", frame)
    cv2.imshow("mask putih", white)
    cv2.imshow("mask kulit", skin)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()