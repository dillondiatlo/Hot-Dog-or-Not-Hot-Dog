import pickle
import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_modal import Modal
import tensorflow as tf

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array

from PIL import Image



# Function to load and preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of your CNN model
    image = image.resize((256, 256))  # Assuming your model expects 224x224 input
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the pixel values to be in the range [0, 1]
    image = image / 255.0
    # Expand the dimensions to create a batch of size 1
    image = np.expand_dims(image, axis=0)
    return image



st.title("Maybe Dog :hotdog:")
st.subheader("Is your photo a _hot dog_ or a _not dog_?")



uploaded_image = st.file_uploader("Upload your photo here...", type=['png', 'jpeg', 'jpg'])

if uploaded_image is not None:
    st.image(uploaded_image)


    image = Image.open(uploaded_image)
    preprocessed_image = preprocess_image(image)


    model = keras.models.load_model('models/hotdog_or_notdog.h5')
    pred = model.predict(preprocessed_image)[0][0]

    
    if (pred < 0.5):
        pred_image = 'img/itshotdog.png'
    else: 
        pred_image = 'img/nothotdog.png'




    modal = Modal(key="Demo Key",title="Prediction")
    open_modal = st.button(label='Predict', key='prediction')
    if open_modal:
        with modal.container():
            # st.markdown(pred_text)
            st.image(pred_image)
            # st.write(pred)

