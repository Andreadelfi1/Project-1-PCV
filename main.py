import cv2
from detector import HandDetector
from game import Game, W, H

det = HandDetector()
game = Game()

cap = cv2.VideoCapture(0)
paused = False

print("C = ambil referensi (tanpa sarung tangan), P = pause, R = restart, ESC = keluar")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera tidak terbaca")
        break
    frame = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    state, pos, white, black = det.detect(frame)

    target_y = None
    if pos is not None:
        top = int(fh * det.roi_top)
        ratio = (pos[1] - top) / (fh - top)
        target_y = max(0.0, min(1.0, ratio)) * H

    if not paused:
        game.update(state, target_y)

    canvas = game.draw(paused)

    preview = cv2.resize(frame, (240, 180))
    line_y = int(180 * det.roi_top)
    cv2.line(preview, (0, line_y), (240, line_y), (0, 0, 255), 1)
    if pos is not None:
        px = int(pos[0] / fw * 240)
        py = int(pos[1] / fh * 180)
        cv2.circle(preview, (px, py), 6, (0, 255, 0), -1)
    cv2.putText(preview, state, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    canvas[H - 180:H, W - 240:W] = preview

    if det.ref is None:
        cv2.putText(canvas, "Tekan C tanpa sarung tangan untuk ambil referensi",
                    (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.imshow("Gundam Dodge", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('r'):
        game.reset()
    elif key == ord('c'):
        det.set_reference(frame)

cap.release()
cv2.destroyAllWindows()