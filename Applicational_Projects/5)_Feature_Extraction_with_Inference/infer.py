import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image


with open(r'Applicational_Projects\5)_Feature_Extraction_with_Inference\model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()
image_path = r'Applicational_Projects\5)_Feature_Extraction_with_Inference\data\weather_dataset\val\rain\rain37.jpg'
img = Image.open(image_path)
features = img2vec.get_vec(img)
pred = model.predict([features])
print(pred)