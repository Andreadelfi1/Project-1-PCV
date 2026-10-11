import cv2

for nama in ["background.png", "mc.png", "zaku 1.png", "zaku 2.webp"]:
    img = cv2.imread("assets/" + nama, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(nama, "-> TIDAK TERBACA")
    else:
        print(nama, "-> OK, ukuran:", img.shape)