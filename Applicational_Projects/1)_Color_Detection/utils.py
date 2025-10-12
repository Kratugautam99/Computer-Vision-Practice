import cv2
import numpy as np

def get_limits(color):
    val = np.uint8([[color]])
    hsvVal = cv2.cvtColor(val, cv2.COLOR_BGR2HSV)
    lowerlim = hsvVal[0][0][0] - 10,100,100
    upperlim = hsvVal[0][0][0] + 10,255,255
    lowerlim = np.array(lowerlim, dtype=np.uint8)
    upperlim = np.array(upperlim, dtype=np.uint8)
    return lowerlim, upperlim