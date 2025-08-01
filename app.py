import streamlit as st
import tempfile
import os
import requests
from moviepy import VideoFileClip

from src.ui_module import set_page_config_and_style, setup_sidebar
from src.whisper_module import load_whisper_model
from src.llm_module import LLM_API_URL, MODEL_NAME

from src.diarize import diarize_audio

# UI setup
set_page_config_and_style()
setup_sidebar()

# Load Whisper model
whisper_model = load_whisper_model()

# Sidebar for saved chat titles
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

    #  Speaker Diarization
    st.info("Identifying who spoke when...")
    diarized_segments = diarize_audio(audio_path, num_speakers=2)  # You can later detect this dynamically

    # Whisper transcription
    st.info("Transcribing audio, please wait...")
    segments, _ = whisper_model.transcribe(audio_path)
    whisper_segments = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text.strip()
        }
        for s in segments
    ]

    # Step 3: Align transcription to speakers
    st.info("Aligning transcription with speakers...")
    final_transcript = []

    for dseg in diarized_segments:
        speaker_text = ""
        for wseg in whisper_segments:
            # If there's overlap between diarized segment and Whisper segment
            if not (wseg["end"] < dseg["start"] or wseg["start"] > dseg["end"]):
                speaker_text += " " + wseg["text"]
        dseg["text"] = speaker_text.strip()
        if dseg["text"]:  # only add non-empty
            final_transcript.append(f"{dseg['speaker']}: {dseg['text']}")

    speaker_transcript = "\n".join(final_transcript)

    # Display speaker-attributed transcript
    st.subheader("️ Speaker-Attributed Transcript")
    st.text_area("Transcript", speaker_transcript, height=300)

    # Step 4: Summarization
    if st.button("Generate Summary"):
        st.info("Summarizing meeting...")
        try:
            response = requests.post(LLM_API_URL, json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": f"Please summarize the following meeting transcript with speaker information:\n\n{speaker_transcript}"}
                ]
            })
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"]
            st.subheader("Meeting Summary")
            st.text_area("Summary", summary, height=250)

            # Save chat
            new_title = f"Chat {len(st.session_state.chat_titles)+1}"
            st.session_state.chat_titles.append(new_title)
            st.session_state[new_title] = {
                "transcript": speaker_transcript,
                "summary": summary
            }

        except Exception as e:
            st.error(f"Error generating summary: {e}")
