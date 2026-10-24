import json
import os
import cv2
import numpy as np


class HandDetector:
    def __init__(self):
        # Nilai awal (ditimpa warna.json jika ada)
        self.white_lo = np.array([0, 0, 170])
        self.white_hi = np.array([179, 60, 255])
        self.black_lo = np.array([0, 0, 0])
        self.black_hi = np.array([179, 255, 70])

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
                print("warna.json lama (format kulit). Jalankan kalibrasi.py lagi.")
        else:
            print("warna.json tidak ada, memakai nilai bawaan (jalankan kalibrasi.py)")

        self.min_area = 8000
        self.roi_top = 0.30        # abaikan 30% bagian atas frame (wajah, rambut)
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
        """Return (state, (cx, cy), white_mask, black_mask)
        state: 'white', 'black', atau 'none'"""
        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        white = cv2.inRange(hsv, self.white_lo, self.white_hi)
        black = cv2.inRange(hsv, self.black_lo, self.black_hi)

        white = self._clean(white)
        black = self._clean(black)

        cut = int(frame.shape[0] * self.roi_top)
        white[:cut, :] = 0
        black[:cut, :] = 0

        w_area, w_pos = self._largest(white)
        b_area, b_pos = self._largest(black)

        if w_area == 0 and b_area == 0:
            return "none", None, white, black
        if w_area >= b_area:
            return "white", w_pos, white, black
        return "black", b_pos, white, black