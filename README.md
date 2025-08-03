
# Meeting Recorder and Summarizer

## Overview

**Meeting Recorder and Summarizer** is an AI-driven application that automates the transcription and summarization of meetings. It supports audio and video uploads, identifies different speakers, and generates concise summaries to enhance meeting productivity and accessibility.

---

## Features

- Upload audio or video meeting files.
- Automatically extract audio from video files.
- Transcribe audio using OpenAI Whisper.
- Perform speaker diarization using Resemblyzer.
- Align speaker segments with transcriptions.
- Generate accurate summaries via a large language model (LLM).
- Simple, interactive user interface powered by Streamlit.
- Maintain chat history for previous sessions.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Habiba480/Meeting-Recorder-And-Summarizer.git
   cd Meeting-Recorder-And-Summarizer


2. **(Optional) Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To run the application:

1. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open the link shown in the terminal (usually `http://localhost:8501`) in your browser.

3. On the UI:

   * Upload an audio (`.mp3`, `.wav`) or video (`.mp4`) file.
   * The app will automatically extract audio (if video) and begin transcription.
   * Speaker diarization is applied to identify and label different speakers.
   * A speaker-attributed transcript is generated and displayed.
   * You can then click "Summarize" to get a clean, concise summary of the meeting.

---

## Project Structure

```
Meeting-Recorder-And-Summarizer/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── diarize.py              # Speaker diarization logic (Resemblyzer)
├── src/
│   ├── whisper_module.py   # Whisper transcription integration
│   ├── llm_module.py       # Summary generation with LLM API
│   └── ui_module.py        # UI components and layout helpers
```

---

## Dependencies

The project uses the following key Python libraries:

* `openai-whisper` — For transcription of audio.
* `resemblyzer` — For speaker diarization and voice embeddings.
* `streamlit` — For the interactive web interface.
* `moviepy` — For extracting audio from video files.
* `scikit-learn` — For clustering speaker embeddings.
* `requests` — For interacting with LLM APIs.
* `torch` — Required by Whisper and Resemblyzer.

All dependencies are listed in `requirements.txt`.

---

## Configuration

* Make sure your LLM endpoint and API key are correctly set in `src/llm_module.py`.
* Long meetings may require additional system resources due to Whisper and clustering steps.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---


