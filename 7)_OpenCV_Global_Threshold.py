import os, cv2

img = cv2.imread(os.path.join('.', 'inputs', '4) bear.jpg'))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
blur_thresh = cv2.blur(thresh, (10, 10))
ret, thresh = cv2.threshold(blur_thresh, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(".","outputs","6) bear_segmented.jpg"), thresh)