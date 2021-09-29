from PIL import Image
from scipy.fft import fft, ifft
import pandas as pd
import plotly.io as pio
import numpy as np
import os
from glob import glob
pio.renderers.default='browser'
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity


def img_to_fft(path):
    # load the image
    image_defect = Image.open('img_17413_cropped.jpg')
    image_normal = Image.open('674.jpg')
    image_defect_bw = image_defect.convert('L')
    image_normal_bw = image_normal.convert('L')

    # convert image to numpy array
    data_defect = np.asarray(image_defect_bw)
    data_normal = np.asarray(image_normal_bw)

    data_defect_fft = abs(fft(data_defect))
    data_normal_fft = abs(fft(data_normal))

def fft_loader(file_name):
    folder_path = os.path.abspath(f'./fft_arrays/**/')
    arr = None

    for file in glob(folder_path, recursive=True):
        if f"{file_name}.npy" in file:
            arr = np.load(file)
    print(arr)
    return arr


def fft_plot(file_name):

    arr = fft_loader(file_name)
    clipped_df = pd.DataFrame(abs(arr))

    fig = go.Figure(data=[go.Surface(z=clipped_df)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

    fig.show()

def fft_similarity(file_name_1, file_name_2):

    fft_1, fft_2 = fft_loader(file_name_1), fft_loader(file_name_2)

    return abs(fft_1[:, 0:1] - fft_2[:, 0:1]) + abs(fft_1[:, -1:] - fft_2[:, -1:])

# scores_same_class = []
# scores_dif_class = []
#
# for i in range(0, 101):
#     scores_dif_class.append(np.mean(abs(fft_similarity(f"{i}", f"{100+i+3}"))))
#
# for i in range(0, 101):
#     scores_same_class.append(np.mean(abs(fft_similarity(f"{i}", f"{i+1}"))))
#     print(i, i+1)
#     print(np.mean(abs(fft_similarity(f"{i}", f"{i+1}"))))
#
# print(np.mean(scores_same_class), np.min(scores_same_class), np.max(scores_same_class))
# print("------------------")
# print(np.mean(scores_dif_class), np.min(scores_dif_class), np.max(scores_dif_class))