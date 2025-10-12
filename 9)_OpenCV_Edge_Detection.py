import os, cv2
import numpy as np

img = cv2.imread(os.path.join('.', 'inputs', '6) messi.jpg'))
img_edge = cv2.Canny(img, 100, 250)
img_edge_d = cv2.dilate(img_edge, np.ones((3, 3), dtype=np.int8))
img_edge_e = cv2.erode(img_edge, np.ones((2, 1), dtype=np.int8))

cv2.namedWindow("GOAT", cv2.WINDOW_NORMAL)
cv2.imshow('GOAT', img)
cv2.namedWindow("GOAT_edge", cv2.WINDOW_NORMAL)
cv2.imshow('GOAT_edge', img_edge)
cv2.namedWindow("GOAT_edge_dilated", cv2.WINDOW_NORMAL)
cv2.imshow('GOAT_edge_dilated', img_edge_d)
cv2.namedWindow("GOAT_edge_eroded", cv2.WINDOW_NORMAL)
cv2.imshow('GOAT_edge_eroded', img_edge_e)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(os.path.join(".", "outputs", "9) messi_edge.jpg"), img_edge)
cv2.imwrite(os.path.join(".", "outputs", "10) messi_edge_dilated.jpg"), img_edge_d)
cv2.imwrite(os.path.join(".", "outputs", "11) messi_edge_eroded.jpg"), img_edge_e)