# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:12:52 2021

@author: HZU
"""
import matplotlib.pyplot as plt
import numpy as np
import os

class_names = [0,1,2,3,4,5,6,7,8,9,10]
img_all_classes = np.load('./utils/img_all_classes.npy')

def plot_image_input(img, prediction):
    fig, ax = plt.subplots()
    predicted_label = np.argmax(prediction)
    plt.gca().set_title('Image to predict')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.xlabel("Label predicted : {}".format(class_names[predicted_label])+"\n"+
               "Similarity value : {:2.0f}".format(np.max(prediction))+"\n")
    return fig
    
def plot_predict_histo(prediction):
    fig, ax = plt.subplots()
    predicted_label = np.argmax(prediction.tolist()[0])
    plt.gca().set_title('Image to predict') 
    plt.grid(False)
    plt.xticks(range(len(class_names)))
    thisplot = ax.bar(class_names, prediction.tolist()[0], color="#777777")
    thisplot[predicted_label].set_color('red')
    return fig

def plot_same_label_img(prediction):
    fig, ax = plt.subplots()
    predicted_label = np.argmax(prediction)
    plt.gca().set_title('Random image with same class') 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    ax.imshow(img_all_classes[predicted_label], cmap='gray', vmin=0, vmax=255)
    plt.xlabel("Class value: {}".format(predicted_label))
    return fig