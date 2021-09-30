from PIL import Image
from scipy.fft import fft, ifft
import pandas as pd
import plotly.io as pio
import numpy as np
import os
import re
from natsort import natsorted
from glob import glob
pio.renderers.default='browser'
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


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

def normalize_fft(fft):
    return MinMaxScaler().fit_transform(fft.reshape((128*128, 1))).reshape((128,128))

def load_fft(file_name):

    folder_path = os.path.abspath(f'./fft_arrays/**/')
    arr = None

    for file in glob(folder_path, recursive=True):
        pattern = re.compile(f"/{file_name}.npy")
        if pattern.search(file):
            arr = np.load(file)

    return arr


def plot_fft(file_name):

    arr = load_fft(file_name)
    clipped_df = pd.DataFrame(abs(arr))

    fig = go.Figure(data=[go.Surface(z=clipped_df)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

    fig.show()

def fft_similarity(fft_1, fft_2):

    return np.linalg.norm(fft_1 - fft_2)

def load_dataset(folder_path = os.path.abspath(f'./fft_arrays/**/')):
    data = {}
    anomalies = []
    for i in range(0,11):
        data[i] = []
        for path in natsorted(glob(folder_path, recursive=True)):
            pattern_1 = re.compile(f"/arr_{i}/.+")
            if pattern_1.search(path):
                arr = np.load(path)
                data[i].append(arr)

    for path in natsorted(glob(folder_path, recursive=True)):
        pattern_2 = re.compile(f"fft_anomalous_dice_arrays/img_")
        if pattern_2.search(path):
            arr = np.load(path)
            anomalies.append(arr)

    return data, anomalies

def create_class_avg(data):
    avg_dict = {}
    for _class, arr in data.items():
        avg_dict[_class] = sum(arr) / len(arr)
    return avg_dict

def predict_class(fft_test, models):
    scores = []
    for _class, fft in models.items():
        scores.append((fft_similarity(fft_test, fft)))
    if np.min(scores) < 35000:
        return np.min(scores), np.argmin(scores), scores
    else:
        return -1

data, anomalies = load_dataset()
models = create_class_avg(data)

for fft in anomalies:
    print(predict_class(fft, models))

print("------")

for fft in data[6]:
    print(predict_class(fft, models))