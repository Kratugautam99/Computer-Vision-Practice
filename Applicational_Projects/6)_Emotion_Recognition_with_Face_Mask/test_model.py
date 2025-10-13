import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
import pickle, cv2
from utils import get_face_landmarks
emotions = ['HAPPY', 'SAD', 'SURPRISED'] # Changeable according to the Emotions

with open(r'Applicational_Projects\6)_Emotion_Recognition_with_Face_Mask\model.p', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while ret:
    ret, frame = cap.read()
    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)
    output = model.predict([face_landmarks])
    print("Raw prediction:", emotions[int(output[0])])
    cv2.putText(frame, emotions[int(output[0])],
               (10, frame.shape[0] - 1),
               cv2.FONT_HERSHEY_SIMPLEX,
               3,
               (255, 0, 0),
               5)
    cv2.imshow('6) Emotion Recognition with Face Mask', frame)
    key = cv2.waitKey(25) & 0xFF
    if chr(key).lower() == 'q':
        break


cap.release()
cv2.destroyAllWindows()