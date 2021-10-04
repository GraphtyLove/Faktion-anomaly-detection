# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:08:32 2021

@author: HZU
"""
from PIL import Image
from numpy import asarray
from scipy.fft import fft, ifft
import numpy as np
import os
from skimage import exposure

##For normal dices
for j in range(0,11):
    folder_path = os.path.abspath('/Users/Corty/Downloads/')
    locationFiles=str(folder_path)+"/normal_dice/"+str(j)
    all_files = os.listdir(locationFiles)
    new_path = os.path.join(folder_path, 'fft_arrays/fft_normal_dice_arrays/arr_'+str(j))
    os.mkdir(new_path)
    text_files=[]
    for i in range(len(all_files)):
        if all_files[i][-4:]=='.jpg':
            # load the image
            image = Image.open(os.path.join(locationFiles, all_files[i]))
            image = image.convert('L')
            # convert image to numpy array
            data = asarray(image)
            data = exposure.adjust_gamma(data, gamma=1.1, gain=1.001)
            data = exposure.adjust_log(data, gain = 1.001)
            # data = exposure.adjust_sigmoid(data, cutoff=0.2, gain=10, inv=False)
            data = data[15:113, 15:113]
            data_fft = abs(fft(data))
            all_files[i] = all_files[i].replace('.jpg', '')
            np.save(os.path.join(new_path, all_files[i]), data_fft)
        else:
            continue

# x = np.load('./1001.npy')


##For anomalous_dices

folder_path = os.path.abspath('/Users/Corty/Downloads/')
locationFiles=str(folder_path)+"/anomalous_dice/"
all_files = os.listdir(locationFiles)
new_path = os.path.join(folder_path, 'fft_arrays/fft_anomalous_dice_arrays')
os.mkdir(new_path)
text_files=[]
for i in range(len(all_files)):
    if all_files[i][-4:]=='.jpg':
        # load the image
        image = Image.open(os.path.join(locationFiles, all_files[i]))
        image = image.convert('L')
        data = asarray(image)
        data = exposure.adjust_gamma(data, gamma=1.1, gain=1.001)
        data = exposure.adjust_log(data, gain=1.001)
        # data = exposure.adjust_sigmoid(data, cutoff=0.2, gain=10, inv=False)
        data = data[15:113, 15:113]
        data_fft = abs(fft(data))
        all_files[i] = all_files[i].replace('.jpg', '')
        np.save(os.path.join(new_path, all_files[i]), data_fft)
    else:
        continue
