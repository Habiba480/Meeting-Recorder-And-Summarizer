from resemblyzer import preprocess_wav, VoiceEncoder
from resemblyzer.hparams import sampling_rate
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def diarize_audio(audio_path, num_speakers=2):

    try:
        wav = preprocess_wav(Path(audio_path))
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")

    encoder = VoiceEncoder()
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

    # If fewer segments than speakers, lower num_speakers to avoid crash
    if len(cont_embeds) < num_speakers:
        num_speakers = max(1, len(cont_embeds))

    clustering = AgglomerativeClustering(n_clusters=num_speakers)
    labels = clustering.fit_predict(cont_embeds)

    segments = []
    for (start, end), label in zip(wav_splits, labels):
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": f"Speaker {label + 1}",
            "text": ""
        })

    return segments


if __name__ == "__main__":
    audio_file = "meeting.wav"
    diarized_segments = diarize_audio(audio_file, num_speakers=2)
    for seg in diarized_segments:
        print(f"[{seg['start']} - {seg['end']}] {seg['speaker']}")
