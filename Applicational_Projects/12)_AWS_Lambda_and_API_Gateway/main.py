import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO


def convertToGrayAPI(img):
    API_ENDPOINT = None  # paste your endpoint here, through Amazon API Gateway, Use Lambda function to process and S3 bucket for dependencies.
    is_success, im_buf_arr = cv2.imencode(".png", img)
    byte_im = im_buf_arr.tobytes()
    r = requests.post(url=API_ENDPOINT, data=byte_im)
    img_ = Image.open(BytesIO(r.content))
    return np.asarray(img_)


def convertToGray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


if __name__ == "__main__":
    print("12) AWS Lambda and API Gateway Project")
    img_path = r'Applicational_Projects\12)_AWS_Lambda_and_API_Gateway\TestIMG.png'
    img = cv2.imread(img_path)
    img_gray = convertToGrayAPI(img)
    cv2.imwrite(r'Applicational_Projects\12)_AWS_Lambda_and_API_Gateway\OutputIMG.png', img_gray)