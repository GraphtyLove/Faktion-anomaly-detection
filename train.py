import streamlit as st
from gan.model import DCGAN

st.set_page_config(
    page_title="DCGAN Dashboard"
)

st.title("DCGAN Dashboard")
sb = st.sidebar
gan = DCGAN(ngpu=0)
if st.button(label="Train DCGAN"):
    gan.fit(
        epochs=10,
        dataset_path="data/dice/"
    )
