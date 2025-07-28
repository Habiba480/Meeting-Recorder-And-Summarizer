import streamlit as st
from faster_whisper import WhisperModel

@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")
