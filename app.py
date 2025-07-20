import os
import tempfile
import streamlit as st
import moviepy
from faster_whisper import WhisperModel
import requests

# LM Studio API endpoint and model name
LLM_API_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load Whisper model once and reuse it for transcription
@st.cache_resource
def load_whisper_model():
    # Load the 'base' Whisper model locally, adjust model size as needed
    return WhisperModel("base", device="cpu", compute_type="int8")

whisper_model = load_whisper_model()

# Basic page config and styles for clean look
st.set_page_config(page_title="AI Meeting Summarizer", layout="wide")
st.markdown("""
    <style>
    /* Add some padding and style */
    .block-container { padding-top: 1rem; }
    .sidebar .sidebar-content { background-color: #a2d2ff; }
    </style>
""", unsafe_allow_html=True)

# Sidebar for saved chat titles
st.sidebar.title(" Meeting Chats")
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = []

for title in st.session_state.chat_titles:
    if st.sidebar.button(title):
        st.session_state.current_chat = title

# Main title of the app
st.title(" AI Meeting Summarizer & Chat Assistant")

# File uploader accepts both video and audio formats
uploaded_file = st.file_uploader(
    "Upload a meeting video or audio file",
    type=["mp4", "mov", "mkv", "mp3", "wav", "m4a"]
)

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    # Check file type: if video, extract audio
    if uploaded_file.type.startswith("video"):
        st.info("Extracting audio from video file...")
        video = VideoFileClip(temp_file_path)
        audio_path = temp_file_path + ".mp3"
        # Export audio as mp3 for whisper
        video.audio.write_audiofile(audio_path, logger=None)
        audio_file_for_transcription = audio_path
    else:
        # If audio file, use it directly
        audio_file_for_transcription = temp_file_path

    # Transcribe audio using Whisper model
    st.info("Transcribing audio with Whisper...")
    segments, _ = whisper_model.transcribe(audio_file_for_transcription, beam_size=5)
    transcript = " ".join([segment.text for segment in segments])

    st.subheader(" Full Transcript")
    st.text_area("Transcript", transcript, height=250)

    # Helper function to split transcript into manageable chunks for LLaMA
    def chunk_text(text, max_words=800):
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    transcript_chunks = chunk_text(transcript)

    # Summarize each chunk by querying LM Studio
    summaries = []
    st.info("Summarizing transcript in chunks using LLaMA 3.1...")
    for i, chunk in enumerate(transcript_chunks):
        prompt = f"Summarize the following meeting transcript chunk clearly and professionally:\n\n{chunk}"
        response = requests.post(
            LLM_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 512
            }
        )
        # Parse summary from response
        try:
            summary_text = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            summary_text = f"⚠️ Error generating summary for chunk {i+1}: {e}"
        summaries.append(summary_text)

    # Combine chunk summaries into one full summary
    full_summary = "\n\n".join(summaries)
    st.subheader(" Meeting Summary")
    st.text_area("Summary", full_summary, height=250)

    # Allow user to name and save this chat summary in the sidebar
    chat_title = st.text_input("Name this meeting chat (for sidebar history)", value=f"Meeting {len(st.session_state.chat_titles)+1}")
    if st.button("Save Chat Summary"):
        if chat_title and chat_title not in st.session_state.chat_titles:
            st.session_state.chat_titles.append(chat_title)
            st.success(f"Chat titled '{chat_title}' saved!")

    # Chat interface for user to ask questions about the meeting
    st.subheader(" Ask Questions About the Meeting")

    # Initialize chat history if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "system",
                "content": f"You are a helpful assistant answering questions based on the following meeting summary:\n\n{full_summary}"
            }
        ]

    # Display chat messages
    for message in st.session_state.chat_history[1:]:  # Skip system prompt
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input for new user question
    user_question = st.chat_input("Ask a question about the meeting...")

    if user_question:
        # Add user's question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Prepare messages payload for LM Studio: system prompt + last few chat messages
        messages_to_send = st.session_state.chat_history[-6:]  # Limit context length

        # Call LM Studio chat completions API
        response = requests.post(
            LLM_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "messages": messages_to_send,
                "temperature": 0.5,
                "max_tokens": 512
            }
        )

        # Parse assistant reply
        try:
            assistant_reply = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            assistant_reply = f"⚠️ Error getting response: {e}"

        # Display assistant reply
        st.chat_message("user").markdown(user_question)
        st.chat_message("assistant").markdown(assistant_reply)

        # Add assistant reply to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
