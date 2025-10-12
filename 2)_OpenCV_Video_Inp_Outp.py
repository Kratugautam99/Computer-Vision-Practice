import cv2, os
vid_path = os.path.join(".", "inputs", "2) sample_video.mp4")
video = cv2.VideoCapture(vid_path)
cv2.namedWindow("shanghai city", cv2.WINDOW_NORMAL)
ret = True
while ret:
    ret, frame = video.read()
    if ret:
        cv2.imshow("shanghai city", frame)
        cv2.waitKey(8)
video.release()
cv2.destroyAllWindows()