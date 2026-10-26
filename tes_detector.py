import cv2
from detector import HandDetector

det = HandDetector()
cap = cv2.VideoCapture(0)
frame_now = None


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and frame_now is not None:
        hsv = cv2.cvtColor(frame_now, cv2.COLOR_BGR2HSV)
        print(f"HSV di ({x},{y}) = {hsv[y, x].tolist()}")


cv2.namedWindow("kamera")
cv2.setMouseCallback("kamera", on_mouse)
print("C = ambil referensi (TANPA sarung tangan), klik = cetak HSV, ESC = keluar")
print("Batas putih saat ini  lo:", det.white_lo.tolist(), "hi:", det.white_hi.tolist())

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    frame_now = frame.copy()
    state, pos, white, black = det.detect(frame)
    h, w = frame.shape[:2]

    cv2.line(frame, (0, int(h * det.roi_top)), (w, int(h * det.roi_top)), (0, 0, 255), 2)
    if pos:
        cv2.circle(frame, pos, 12, (0, 255, 0), -1)
    cv2.putText(frame, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    info = "REFERENSI: ADA" if det.ref is not None else "REFERENSI: BELUM (tekan C)"
    cv2.putText(frame, info, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("kamera", frame)
    if det.fg_debug is not None:
        cv2.imshow("beda dari referensi", det.fg_debug)
    cv2.imshow("mask putih", white)
    cv2.imshow("mask hitam", black)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('c'):
        det.set_reference(frame_now)
        print("Referensi tersimpan")

cap.release()
cv2.destroyAllWindows()