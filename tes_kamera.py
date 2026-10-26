import cv2

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera tidak terbaca")
        break
    cv2.imshow("tes", frame)
    if cv2.waitKey(1) == 27:  
        break
cap.release()
cv2.destroyAllWindows()