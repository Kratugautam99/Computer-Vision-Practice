import cv2
from PIL import Image
from utils import get_limits
color = [0, 165, 255] # works well with yellow, blue and green but not with red, dark or light colours. Background should be clean and different from object.

webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerlim, upperlim = get_limits(color)
    mask = cv2.inRange(hsvImage, lowerlim, upperlim)
    maskimg = Image.fromarray(mask)
    bbox = maskimg.getbbox()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
    cv2.imshow("1) Color Detection of Object Project", frame)
    key = cv2.waitKey(1) & 0xFF
    if chr(key).lower() == 'q':
        break
webcam.release()
cv2.destroyAllWindows()