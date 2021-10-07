import pandas as pd
import plotly.io as pio
import numpy as np
import os
import re
import time
from natsort import natsorted
from glob import glob


def fft_similarity(fft_1 : np.ndarray, fft_2 : np.ndarray):
    '''Docstring'''
    score = np.linalg.norm(fft_1 - fft_2)
    return score


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