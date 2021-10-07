from PIL import Image
from scipy.fft import fft, ifft
import pandas as pd
import plotly.io as pio
import numpy as np
import os
import re
import time
from natsort import natsorted
from glob import glob
pio.renderers.default='browser'
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


def normalize_fft(fft : np.ndarray):
    return MinMaxScaler().fit_transform(fft.reshape((128*128, 1))).reshape((128,128))

def load_fft(file_name : str):
    '''Docstring'''
    folder_path = os.path.abspath(f'/Users/Corty/Downloads/fft_arrays/**/')
    arr = None

    for file in glob(folder_path, recursive=True):
        pattern = re.compile(f"/{file_name}.npy")
        if pattern.search(file):
            arr = np.load(file, allow_pickle=True)

    return arr


def plot_fft(file_name : str):
    '''Docstring'''
    arr = load_fft(file_name)
    clipped_df = pd.DataFrame(abs(arr))

    fig = go.Figure(data=[go.Surface(z=clipped_df)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

    fig.show()

def fft_similarity(fft_1 : np.ndarray, fft_2 : np.ndarray):
    '''Docstring'''
    score = np.linalg.norm(fft_1 - fft_2)
    cosine = 1
    # for i in range(5):
    #     cosine += cosine_similarity(fft_1[:, i : i+1].reshape(1, -1), fft_2[:, i : i+1].reshape(1, -1))
    return score * cosine

def load_dataset(folder_path : str = os.path.abspath(f'/Users/Corty/Downloads/fft_arrays/**/')):
    '''Docstring'''
    data = {}
    anomalies = []
    for i in range(0,11):
        data[i] = []
        for path in natsorted(glob(folder_path, recursive=True)):
            pattern_1 = re.compile(f"arr_{i}/.+")
            if pattern_1.search(path):
                arr = np.load(path, allow_pickle=True)
                data[i].append(arr)

    for path in natsorted(glob(folder_path, recursive=True)):
        pattern_2 = re.compile(f"fft_anomalous_dice_arrays/.+")
        if pattern_2.search(path):
            arr = np.load(path, allow_pickle=True)
            anomalies.append(arr)

    return data, anomalies

def create_class_avg(data : dict):
    '''Docstring'''
    avg_dict = {}
    for _class, arr in data.items():
        avg_dict[_class] = sum(arr) / len(arr)
    return avg_dict

def predict_class(fft_test : np.ndarray, models : dict):
    '''Docstring'''
    scores = []
    for _class, fft in models.items():
        scores.append((fft_similarity(fft_test, fft)))
    return np.min(scores), np.argmin(scores), scores

def create_metrics(data : dict, models : dict):
    '''Docstring'''
    max_dict = {}
    for _class, arr in data.items():
        _max = 0
        score = 0
        for fft in arr:
            if predict_class(fft, models):
                score = predict_class(fft, models)[0]
            if score > _max:
                _max = score
        max_dict[_class] = _max
    return max_dict

def predict_anomaly(fft_test : np.ndarray, models : dict, threshold_dict : dict):
    '''Docstring'''
    input_score, predicted_class, class_scores = predict_class(fft_test, models)
    if input_score > threshold_dict[predicted_class]:
        return True
    else:
        return False


start = time.time()


data, anomalies = load_dataset()
print(time.time() - start)
models = create_class_avg(data)
print(time.time() - start)
thresholds_dict = create_metrics(data, models)

print(time.time() - start)

detections = 0
for i, fft in enumerate(anomalies):
    print(i, predict_anomaly(fft, models, thresholds_dict), predict_class(fft, models)[1])
    print()
    detections += predict_anomaly(fft, models, thresholds_dict)

print("------")

print(thresholds_dict)
print(detections, (detections/91)*100)
for fft in data[0]:
    print(predict_class(fft, models))
    print(predict_anomaly(fft, models, thresholds_dict))
print(time.time() - start)