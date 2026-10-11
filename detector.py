import json
import os
import cv2
import numpy as np


class HandDetector:
    def __init__(self):
        # Nilai awal (akan ditimpa warna.json jika ada)
        self.white_lo = np.array([0, 0, 170])
        self.white_hi = np.array([179, 60, 255])
        self.skin_lo = np.array([0, 40, 80])
        self.skin_hi = np.array([25, 180, 255])

        if os.path.exists("warna.json"):
            with open("warna.json") as f:
                d = json.load(f)
            self.white_lo = np.array(d["white"]["lo"])
            self.white_hi = np.array(d["white"]["hi"])
            self.skin_lo = np.array(d["skin"]["lo"])
            self.skin_hi = np.array(d["skin"]["hi"])
            print("Memakai kalibrasi dari warna.json")
        else:
            print("warna.json tidak ada, memakai nilai bawaan (run kalibrasi.py)")

        self.min_area = 8000       # abaikan objek kecil
        self.roi_top = 0.35       # abaikan 30% bagian atas frame
        self.kernel = np.ones((7, 7), np.uint8)

    def _clean(self, mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _largest(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_area:
            return 0, None
        m = cv2.moments(c)
        if m["m00"] == 0:
            return 0, None
        return area, (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

    def detect(self, frame):
        """Return (state, (cx, cy), white_mask, skin_mask)"""
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        white = cv2.inRange(hsv, self.white_lo, self.white_hi)
        skin = cv2.inRange(hsv, self.skin_lo, self.skin_hi)

        # Area yang cocok keduanya dianggap ambigu -> dibuang dari kulit
        overlap = cv2.bitwise_and(white, skin)
        skin = cv2.bitwise_and(skin, cv2.bitwise_not(overlap))

        white = self._clean(white)
        skin = self._clean(skin)

        cut = int(frame.shape[0] * self.roi_top)
        white[:cut, :] = 0
        skin[:cut, :] = 0

        w_area, w_pos = self._largest(white)
        s_area, s_pos = self._largest(skin)

        if w_area == 0 and s_area == 0:
            return "none", None, white, skin
        if w_area >= s_area:
            return "white", w_pos, white, skin
        return "skin", s_pos, white, skin