import json, cv2, base64, numpy as np

def decode(encoded_img):
    img_bytes = base64.b64decode(encoded_img)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)                                                                                                                                                                                                                                                                                                                 # r"https://jp2cxh6zn7.execute-api.eu-north-1.amazonaws.com/DEV"

def encode(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode("utf-8")

def lambda_handler(event, context):
    img = decode(event['body'])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_encoded = encode(gray)
    return {
        'statusCode': 200,
        'body': gray_encoded,
        'isBase64Encoded': True,
        'headers': {
            'Content-Type': 'image/png'
        }
    }
