# -*- coding: utf-8 -*-
"""
@author: HZU
"""
import streamlit as st
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from utils.plot_functions import plot_image_input, plot_predict_histo, plot_same_label_img

st.set_page_config(layout="wide")

st.header('THIS IS THE TITLE')
st.subheader(' a lot of text doing all the explanation, because this app has to explain itself ')


st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
st.write('\n')
##Loading the image to be analize
st.write('here something to load the image.')
image_file = st.file_uploader("Upload File",type=['jpg'])


if image_file is not None:
    file_details = {"FileName":image_file .name,"FileType":image_file.type,"FileSize":image_file.size}
    st.write(file_details)

#Loading the model and also all the classes images to plot
model = tf.keras.models.load_model('./utils/cnn_model.h5')

#One image for each class is loaded here named from 0 to 10
img_all_classes = np.load('./utils/img_all_classes.npy')
class_names = [0,1,2,3,4,5,6,7,8,9,10]

with st.expander('CNN method'):
    col_mid, col1, col2, col3, col_mid = st.columns(5)
    st.write('here is the method')
    if image_file is not None:
        data = image_file.read()
        dataBytesIO = io.BytesIO(data)
        img = Image.open(dataBytesIO)
        img = np.array(img)
        prediction = model.predict(img[None,:,:])
        plot_1 = plot_image_input(img, prediction)
        plot_2 = plot_predict_histo(prediction)
        plot_3 = plot_same_label_img(prediction)
        col1.pyplot(plot_1)
        col2.pyplot(plot_2)
        col3.pyplot(plot_3)



