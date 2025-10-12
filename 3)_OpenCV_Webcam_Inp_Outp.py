import cv2
webcam = cv2.VideoCapture(0)
cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
while True:
    ret, frame = webcam.read()
    cv2.imshow("webcam", frame)
    if cv2.waitKey(8) & 0xFF == ord("q"):
        break
webcam.release()
cv2.destroyAllWindows()