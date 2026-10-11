import json
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
samples = {"white": [], "skin": []}
mode = "white"
frame_now = None
frame_raw = None


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and frame_raw is not None:
        # ambil patch 21x21 piksel di sekitar klik
        patch = frame_raw[max(0, y - 10):y + 11, max(0, x - 10):x + 11]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        samples[mode].append(hsv)
        print(f"{mode}: {len(samples[mode])} sampel")


cv2.namedWindow("kalibrasi")
cv2.setMouseCallback("kalibrasi", on_mouse)

print("W = mode sarung tangan putih, K = mode kulit")
print("Klik di beberapa titik sarung tangan/kulit (5-8 klik tiap mode)")
print("S = simpan, ESC = keluar")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    frame_raw = frame.copy()
    view = frame.copy()
    cv2.putText(view, f"MODE: {mode.upper()}  (klik pada objek)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(view, f"putih:{len(samples['white'])} kulit:{len(samples['skin'])}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("kalibrasi", view)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('w'):
        mode = "white"
    elif key == ord('k'):
        mode = "skin"
    elif key == ord('s'):
        if not samples["white"] or not samples["skin"]:
            print("Ambil sampel putih DAN kulit dulu!")
            continue
        out = {}
        for name in ("white", "skin"):
            data = np.vstack(samples[name])
            lo = np.percentile(data, 5, axis=0)
            hi = np.percentile(data, 95, axis=0)
            pad = np.array([8, 25, 25])        # toleransi H, S, V
            lo = np.clip(lo - pad, 0, 255).astype(int).tolist()
            hi = np.clip(hi + pad, 0, 255).astype(int).tolist()
            hi[0] = min(hi[0], 179)
            out[name] = {"lo": lo, "hi": hi}
        with open("warna.json", "w") as f:
            json.dump(out, f)
        print("Tersimpan ke warna.json:", out)

cap.release()
cv2.destroyAllWindows()