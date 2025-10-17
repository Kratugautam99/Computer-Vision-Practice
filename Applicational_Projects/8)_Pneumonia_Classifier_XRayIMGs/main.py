import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background(r'Applicational_Projects\8)_Pneumonia_Classifier_XRayIMGs\bgs\bg5.png')

# set title
st.markdown("<h1 style='color:black;'>8) Pneumonia Classifier XRayIMGs Project</h1>", unsafe_allow_html=True)

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model(r"Applicational_Projects\8)_Pneumonia_Classifier_XRayIMGs\model\pneumonia_classifier.h5")

# load class names
with open(r"Applicational_Projects\8)_Pneumonia_Classifier_XRayIMGs\model\labels.txt", 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, width=500)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.markdown(f"<h2 style='color:black;'>{class_name}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:black;'>Confidence Score (Postivity): {int(conf_score * 1000) / 10}%</h3>", unsafe_allow_html=True)

