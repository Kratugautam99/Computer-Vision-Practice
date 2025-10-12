from PIL import Image
import os, pickle, shutil, random
from img2vec_pytorch import Img2Vec
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

## relocation, if this link ["https://data.mendeley.com/datasets/4drtyfjtfy/1"] is used to get data.

# src = r'Applicational_Projects\5)_Feature_Extraction_with_Inference\data'
# base = r'Applicational_Projects\5)_Feature_Extraction_with_Inference\data\weather_dataset'
# os.makedirs(base, exist_ok=True)

# for f in os.listdir(src):
#     if f.endswith('.jpg'):
#         cls = f.rstrip('0123456789.jpg')
#         split = 'val' if random.random() < 0.2 else 'train'
#         target_dir = os.path.join(base, split, cls)
#         os.makedirs(target_dir, exist_ok=True)
#         shutil.move(os.path.join(src, f), os.path.join(target_dir, f))



# prepare data
img2vec = Img2Vec()
data_dir = r'Applicational_Projects\5)_Feature_Extraction_with_Inference\data\weather_dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}
for j, dir_ in enumerate([train_dir, val_dir]):
    labels = []
    features = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_).convert('RGB')
            img_features = img2vec.get_vec(img)
            features.append(img_features)
            labels.append(category)
    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

# train model        
model = RandomForestClassifier(random_state=0)
model.fit(data["training_data"], data["training_labels"])

# test performance
y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])

print(score)

# save the model
with open(r'.\Applicational_Projects\5)_Feature_Extraction_with_Inference\model.p', 'wb') as f:
    pickle.dump(model, f)
    f.close()