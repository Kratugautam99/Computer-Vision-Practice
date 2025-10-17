import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# This Model is of (K, R, A) Alphabets
data_dict = pickle.load(open(r'Applicational_Projects\7)_Sign_Language_Detection_for_N_Alphabets\Data\data.pickle', 'rb'))

data, labels = [], []
expected_len = len(data_dict['data'][0])

for sample, label in zip(data_dict['data'], data_dict['labels']):
    if len(sample) == expected_len:
        data.append(sample)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open(r'Applicational_Projects\7)_Sign_Language_Detection_for_N_Alphabets\model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()