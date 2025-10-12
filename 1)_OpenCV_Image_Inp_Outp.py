import cv2, os
img_path = os.path.join(".", "inputs", "1) dragon.jpg")
img = cv2.imread(img_path)
print(type(img))
cv2.imwrite(os.path.join(".", "outputs", "1) dragon_bgr.jpg"), img)
cv2.namedWindow("Dragonz", cv2.WINDOW_NORMAL)
cv2.imshow("Dragonz", img)
cv2.waitKey(0)
cv2.destroyAllWindows()