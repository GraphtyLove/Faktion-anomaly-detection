# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:01:18 2021

@author: HZU
"""
from PIL import Image
from numpy import asarray
from scipy.fft import fft, ifft
import numpy as np
import os

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
    folder_path = os.path.abspath('./')
    locationFiles=str(folder_path)+"/normal_dice_for_FFT/"+str(j)
    all_files = os.listdir(locationFiles)
    new_path = os.path.join(folder_path, 'cutted_arr_'+str(j))
    os.mkdir(new_path)
    text_files=[]
    for i in range(len(all_files)):
        if all_files[i][-4:]=='.jpg':
            # load the image
            image = Image.open(os.path.join(locationFiles, all_files[i]))
            image_bw = image.convert('L')
            # convert image to numpy array
            data = asarray(image_bw)
            data = set_max_pixel_and_cut(data)
            all_files[i] = all_files[i].replace('.jpg', '')
            np.save(os.path.join(new_path, all_files[i]), data)
        else:
            next

##For anomalous_dices

folder_path = os.path.abspath('./')
locationFiles=str(folder_path)+"/anomalous_dice/"
all_files = os.listdir(locationFiles)
new_path = os.path.join(folder_path, 'anomalous_dice_preprocessed')
os.mkdir(new_path)
text_files=[]
for i in range(len(all_files)):
    if all_files[i][-4:]=='.jpg':
        # load the image
        image = Image.open(os.path.join(locationFiles, all_files[i]))
        image_bw = image.convert('L')
        # convert image to numpy array
        data = asarray(image_bw)
        data = set_max_pixel_and_cut(data)
        all_files[i] = all_files[i].replace('.jpg', '')
        np.save(os.path.join(new_path, all_files[i]), data)
    else:
        next
