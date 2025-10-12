import os, cv2
import numpy as np
from utils import get_face_landmarks

data_dir = r'Applicational_Projects\6)_Emotion_Recognition_with_Face_Mask\data'

output = []
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        image = cv2.imread(image_path)

        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

np.savetxt(r'Applicational_Projects\6)_Emotion_Recognition_with_Face_Mask\data\data.txt', np.asarray(output))