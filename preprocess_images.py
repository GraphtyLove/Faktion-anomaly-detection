from PIL import Image
import numpy as np
from scipy.fft import fft, ifft
import os
from skimage import exposure
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go

def process_image(path="/Users/Corty/Downloads/anomalous_dice/img_17450_cropped.jpg"):

    image_data = np.asarray(Image.open(path))
    exposure_image_data = adjust_exposure(image_data)

    Image.fromarray(exposure_image_data[15:113, 15:113]).show()


def adjust_exposure(image: np.ndarray):
    image = exposure.adjust_gamma(image, gamma=1.1, gain=1.001)
    image = exposure.adjust_log(image, gain = 1.001)
    image = exposure.adjust_sigmoid(image, cutoff=0.55, gain=5, inv=False)
    return image

process_image("/Users/Corty/Downloads/normal_dice/0/99.jpg")
process_image("/Users/Corty/Downloads/normal_dice/2/122.jpg")
process_image("/Users/Corty/Downloads/anomalous_dice/img_18068_cropped.jpg")