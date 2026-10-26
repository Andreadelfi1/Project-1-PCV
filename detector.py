import json
import os
from collections import deque
import cv2
import numpy as np


class HandDetector:
    def __init__(self):
        self.white_lo = np.array([0, 0, 170])
        self.white_hi = np.array([179, 60, 255])
        self.black_lo = np.array([0, 0, 0])
        self.black_hi = np.array([179, 255, 60])

        if os.path.exists("warna.json"):
            with open("warna.json") as f:
                d = json.load(f)
            if "white" in d and "black" in d:
                self.white_lo = np.array(d["white"]["lo"])
                self.white_hi = np.array(d["white"]["hi"])
                self.black_lo = np.array(d["black"]["lo"])
                self.black_hi = np.array(d["black"]["hi"])
                print("Memakai kalibrasi dari warna.json")
            else:
                print("warna.json lama. Hapus lalu jalankan kalibrasi.py")
        else:
            print("warna.json tidak ada, jalankan kalibrasi.py")

        self.skin_lo = np.array([0, 55, 60])
        self.skin_hi = np.array([25, 200, 255])

        self.roi_top = 0.25
        self.min_area = 6000
        self.max_area_ratio = 0.55
        self.min_solidity = 0.35
        self.diff_thresh = 30
        self.stable_need = 4
        self.smooth = 0.5

        self.kernel = np.ones((9, 9), np.uint8)
        self.ref = None
        self.history = deque(maxlen=7)
        self.last_state = "none"
        self.smooth_pos = None
        self.fg_debug = None

    def set_reference(self, frame):
        self.ref = cv2.GaussianBlur(frame, (11, 11), 0)
        self.history.clear()
        self.last_state = "none"
        self.smooth_pos = None

    def _foreground(self, blur):
        """Piksel yang berbeda dari foto referensi."""
        if self.ref is None:
            return np.full(blur.shape[:2], 255, np.uint8)
        diff = cv2.absdiff(blur, self.ref)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(gray, self.diff_thresh, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        return fg

    def _clean(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _best(self, mask, max_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_area, best_pos = 0, None
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > max_area:
                continue
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0 or area / hull_area < self.min_solidity:
                continue
            if area > best_area:
                m = cv2.moments(c)
                if m["m00"] == 0:
                    continue
                best_area = area
                best_pos = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
        return best_area, best_pos

    def detect(self, frame):
        """Return (state, (cx, cy), white_mask, black_mask)"""
        h, w = frame.shape[:2]
        blur = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        fg = self._foreground(blur)
        self.fg_debug = fg

        white = cv2.inRange(hsv, self.white_lo, self.white_hi)
        skin = cv2.inRange(hsv, self.skin_lo, self.skin_hi)
        white = cv2.bitwise_and(white, cv2.bitwise_not(skin))   # buang kulit
        white = cv2.bitwise_and(white, fg)

        black = cv2.bitwise_and(cv2.inRange(hsv, self.black_lo, self.black_hi), fg)

        white = self._clean(white)
        black = self._clean(black)

        cut = int(h * self.roi_top)
        white[:cut, :] = 0
        black[:cut, :] = 0

        max_area = self.max_area_ratio * (h - cut) * w
        w_area, w_pos = self._best(white, max_area)
        b_area, b_pos = self._best(black, max_area)

        if w_area == 0 and b_area == 0:
            raw, raw_pos = "none", None
        elif w_area >= b_area:
            raw, raw_pos = "white", w_pos
        else:
            raw, raw_pos = "black", b_pos

        self.history.append(raw)
        for s in ("white", "black", "none"):
            if self.history.count(s) >= self.stable_need:
                self.last_state = s
                break
        state = self.last_state

        if state == "none":
            self.smooth_pos = None
        elif raw_pos is not None:
            if self.smooth_pos is None:
                self.smooth_pos = raw_pos
            else:
                a = self.smooth
                self.smooth_pos = (int(a * raw_pos[0] + (1 - a) * self.smooth_pos[0]),
                                   int(a * raw_pos[1] + (1 - a) * self.smooth_pos[1]))

        pos = self.smooth_pos if state != "none" else None
        return state, pos, white, black