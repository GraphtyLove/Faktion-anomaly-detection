# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:26:51 2021

@author: HZU
"""
from PIL import Image
from numpy import asarray
from scipy.fft import fft, ifft
import pandas as pd
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go


def plot_3dFFT(file_name:str):
    # load the image
    image = Image.open(file_name)
    image_bw = image.convert('L')
    # convert image to numpy array
    data = asarray(image_bw)
    data = abs(fft(data))
    data_df = pd.DataFrame(abs(data))
    
    fig = go.Figure(data=[go.Surface(z=data_df)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    
    fig.show()
