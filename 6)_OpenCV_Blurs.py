import cv2, os

img = cv2.imread(os.path.join(".", "inputs", "1) dragon.jpg"))
noised_img = cv2.imread(os.path.join(".", "inputs", "3) cow_salt_pepper.png"))

k_size = 7
img_blur = cv2.blur(img, (k_size, k_size))
img_gaussian_blur = cv2.GaussianBlur(img, (k_size, k_size), 5)
img_median_blur = cv2.medianBlur(img, k_size)

cv2.namedWindow('img_median_blur', cv2.WINDOW_NORMAL)
cv2.imshow('img_median_blur', img_median_blur)

cv2.namedWindow('img_gaussian_blur', cv2.WINDOW_NORMAL)
cv2.imshow('img_gaussian_blur', img_gaussian_blur)
 
cv2.namedWindow('img_blur', cv2.WINDOW_NORMAL)
cv2.imshow('img_blur', img_blur)
 
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)

cv2.waitKey(0)

k_size = 7
noised_img_blur = cv2.blur(noised_img, (k_size, k_size))
noised_img_gaussian_blur = cv2.GaussianBlur(noised_img, (k_size, k_size), 5)
noised_img_median_blur = cv2.medianBlur(noised_img, k_size)

cv2.namedWindow('noised_img_median_blur', cv2.WINDOW_NORMAL)
cv2.imshow('noised_img_median_blur', noised_img_median_blur)

cv2.namedWindow('noised_img_gaussian_blur', cv2.WINDOW_NORMAL)
cv2.imshow('noised_img_gaussian_blur', noised_img_gaussian_blur)

cv2.namedWindow('noised_img_blur', cv2.WINDOW_NORMAL)
cv2.imshow('noised_img_blur', noised_img_blur)

cv2.namedWindow('noised_img', cv2.WINDOW_NORMAL)
cv2.imshow('noised_img', noised_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(".","outputs","5) cleaned_cow_salt_pepper.png"), noised_img_median_blur)
