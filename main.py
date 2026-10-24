import cv2
from detector import HandDetector
from game import Game, W, H

det = HandDetector()
game = Game()

cap = cv2.VideoCapture(0)
paused = False

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera tidak terbaca")
        break
    frame = cv2.flip(frame, 1)   
    fh, fw = frame.shape[:2]

    state, pos, white, skin = det.detect(frame)

    # movement atas bawah
    target_y = None
    if pos is not None:
        top = int(fh * det.roi_top)
        ratio = (pos[1] - top) / (fh - top)
        target_y = max(0.0, min(1.0, ratio)) * H

    if not paused:
        game.update(state, target_y)

    canvas = game.draw(paused)

    # popup kamera
    preview = cv2.resize(frame, (240, 180))
    if pos is not None:
        px = int(pos[0] / fw * 240)
        py = int(pos[1] / fh * 180)
        cv2.circle(preview, (px, py), 6, (0, 255, 0), -1)
    cv2.putText(preview, state, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    canvas[H - 180:H, W - 240:W] = preview

    cv2.imshow("Gundam Dodge", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:            # ESC = exit
        break
    elif key == ord('p'):    # P = pause
        paused = not paused
    elif key == ord('r'):    # R = restart
        game.reset()

cap.release()
cv2.destroyAllWindows()