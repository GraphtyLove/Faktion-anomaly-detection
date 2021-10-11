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

def preprocess_input(image : np.ndarray) -> np.ndarray:
    data = np.copy(asarray(image))
    data = exposure.adjust_gamma(data, gamma=1.1, gain=1.001)
    data = exposure.adjust_log(data, gain=1.001)
    data = data[15:113, 15:113]
    return abs(fft(data))

def cut_circle(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask

def set_max_pixel_and_cut(matrix):
    """
    cut in circle and set max pixel value.
    """
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if matrix[i][j]>190:
                matrix[i][j] = 190

    mask = cut_circle(matrix.shape,(64,64),60,(0,360))
    matrix[~mask] = 190
    return matrix

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
            data = np.copy(asarray(image))
            data = exposure.adjust_gamma(data, gamma=1.1, gain=1.001)
            data = exposure.adjust_log(data, gain = 1.001)
            # data = exposure.adjust_sigmoid(data, cutoff=0.75, gain=12, inv=False)
            data = data[15:113, 15:113]
            # data = set_max_pixel_and_cut(data)
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
new_path = os.path.join(folder_path, '../fft_arrays/fft_anomalous_dice_arrays')
os.mkdir(new_path)
text_files=[]
for i in range(len(all_files)):
    if all_files[i][-4:]=='.jpg':
        # load the image
        image = Image.open(os.path.join(locationFiles, all_files[i]))
        image = image.convert('L')
        data = np.copy(asarray(image))
        data = exposure.adjust_gamma(data, gamma=1.1, gain=1.001)
        data = exposure.adjust_log(data, gain=1.001)
        # data = exposure.adjust_sigmoid(data, cutoff=0.75, gain=12, inv=False)
        data = data[15:113, 15:113]
        # data = set_max_pixel_and_cut(data)
        data_fft = abs(fft(data))
        all_files[i] = all_files[i].replace('.jpg', '')
        np.save(os.path.join(new_path, all_files[i]), data_fft)
    else:
        continue
