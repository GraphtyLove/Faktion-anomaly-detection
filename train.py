import streamlit as st
from gan.model import DCGAN
import plotly.express as px

import numpy as np
import time

st.set_page_config(
    page_title="DCGAN Dashboard"
)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

st.title("DCGAN Dashboard")
sb = st.sidebar
gan = DCGAN(ngpu=0)
if st.button(label="Train DCGAN"):
    gan.fit(
        epochs=10,
        dataset_path="data/dice/"
    )
