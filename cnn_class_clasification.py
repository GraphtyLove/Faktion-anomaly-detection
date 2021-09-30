# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:50:21 2021

@author: HZU
"""
import tensorflow as tf
from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import time
import numpy as np
import matplotlib.image as img
from scipy import signal
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import time
import pandas as pd



train_folder = os.path.abspath('./normal_dice/train')
# Flow training images in batches of 32 using train_datagen generator
train_generator = tf.keras.utils.image_dataset_from_directory(
                    directory = train_folder,
                    image_size = (128,128),
                    batch_size = 32,
                    shuffle = True,
                    seed = 42
                    )
    
validation_folder = os.path.abspath('./normal_dice/val')
# Flow training images in batches of 32 using train_datagen generator
validation_generator = tf.keras.utils.image_dataset_from_directory(
                    directory = validation_folder,
                    image_size = (128,128),
                    batch_size = 32,
                    shuffle = True,
                    seed = 42
                    )


test_folder = os.path.abspath('./normal_dice/test')
# Flow training images in batches of 32 using train_datagen generator
test_generator = tf.keras.utils.image_dataset_from_directory(
                    directory = test_folder,
                    image_size = (128,128),
                    batch_size = 1,
                    shuffle = False,
                    seed = 42
                    )



AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_generator.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = validation_generator.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = 11

model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds,
                    epochs=3, batch_size=32)




plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")



import matplotlib.pyplot as plt
 
class_names = [1,2,3,4,5,6,7,8,9,10,11]
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(12):
    # for i in range(len(labels)):
        if i in labels[i]:   
            ax = plt.subplot(4, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")



