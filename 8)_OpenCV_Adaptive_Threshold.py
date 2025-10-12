import os, cv2
img = cv2.imread(os.path.join('.', 'inputs', '5) handwritten_text.png'))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

adaptive_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)
ret, simple_thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)


cv2.imshow('img', img)
cv2.imshow('adaptive_thresh', adaptive_thresh)
cv2.imshow('simple_thresh', simple_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(".", "outputs", "7) handwritten_text_extracted_global.png"), simple_thresh)
cv2.imwrite(os.path.join(".", "outputs", "8) handwritten_text_extracted_adaptive.png"), adaptive_thresh)