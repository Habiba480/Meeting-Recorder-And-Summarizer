import streamlit as st
import tempfile
import os
import requests
import  moviepy
from moviepy import VideoFileClip

from src.ui_module import set_page_config_and_style, setup_sidebar
from src.whisper_module import load_whisper_model
from src.llm_module import LLM_API_URL, MODEL_NAME

# UI setup
set_page_config_and_style()
setup_sidebar()

# Load Whisper model
whisper_model = load_whisper_model()

# Sidebar for saved chat titles (already handled in ui_module too)
for title in st.session_state.chat_titles:
    if st.sidebar.button(title):
        st.session_state.current_chat = title

# Main title
st.title(" AI Meeting Summarizer & Chat Assistant")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a meeting video or audio file",
    type=["mp4", "mov", "mkv", "mp3", "wav", "m4a"]
)

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    # If video, extract audio
    if uploaded_file.type.startswith("video"):
        st.info("Extracting audio from video file...")
        video = VideoFileClip(temp_file_path)
        audio_path = temp_file_path + ".mp3"
        video.audio.write_audiofile(audio_path, logger=None)
    else:
        audio_path = temp_file_path

    # Transcribe audio using Whisper
    st.info("Transcribing audio, please wait...")
    segments, _ = whisper_model.transcribe(audio_path)
    full_transcript = "\n".join(segment.text for segment in segments)

    # Display transcript
    st.subheader("Full Transcript")
    st.text_area("Transcript", full_transcript, height=300)

    # Generate summary with LLM
    if st.button("Generate Summary"):
        st.info("Summarizing meeting...")
        try:
            response = requests.post(LLM_API_URL, json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": f"Please summarize the following meeting:\n\n{full_transcript}"}
                ]
            })
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"]
            st.subheader("Meeting Summary")
            st.text_area("Summary", summary, height=250)

            # Save chat to history
            new_title = f"Chat {len(st.session_state.chat_titles)+1}"
            st.session_state.chat_titles.append(new_title)
            st.session_state[new_title] = {
                "transcript": full_transcript,
                "summary": summary
            }

        except Exception as e:
            st.error(f"Error generating summary: {e}")
