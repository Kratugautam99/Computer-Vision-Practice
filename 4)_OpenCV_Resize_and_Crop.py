import cv2, os
img = cv2.imread(os.path.join(".", "inputs", "1) dragon.jpg"))
cv2.namedWindow("Big Dragon", cv2.WINDOW_NORMAL)
cv2.imshow("Big Dragon", img)
cv2.waitKey(0)
print("original_img.shape :",img.shape) # (1080, 1920, 3) H,W,C
resized_img = cv2.resize(img, (640, 480)) # W,H
cv2.namedWindow("Small Dragon", cv2.WINDOW_NORMAL)
cv2.imshow("Small Dragon", resized_img)
cv2.waitKey(0)
print("resized_img.shape :",resized_img.shape) 
cropped_img = img[100:600,800:1500] # H,W
cv2.namedWindow("Dragon Head", cv2.WINDOW_NORMAL)
cv2.imshow("Dragon Head", cropped_img)
cv2.waitKey(0)
print("cropped_img.shape :",cropped_img.shape)
cv2.destroyAllWindows()