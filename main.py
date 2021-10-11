from utils.fft import fft_detector
from utils.save_fft import preprocess_input

import numpy as np

matrix_image = np.array([]) # To get from the upload Streamlit interface

predictive_strength = 0.9 # To play with between 0 (Everything is an anomaly) and 1 (No False Positives on Training)

processed_image = preprocess_input(matrix_image) # Apply the preprocessing on the matrix image

detected_anomaly, detected_class, false_positives_on_training_set = fft_detector(processed_image, predictive_strength)