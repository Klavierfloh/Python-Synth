import os
import librosa
import soundfile as sf
import numpy as np

# Directory with WAV files
directory = "./finalTestOutput"
target_sr = 48000  # Target sample rate

# Collect audio data and original sample rate from first file
audio_data = []
min_len = None

# Step 1: Load all .wav files, normalize, resample
for filename in sorted(os.listdir(directory)):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(directory, filename)
        audio, sr = librosa.load(filepath, sr=None)  # Use original SR
        audio = librosa.util.normalize(audio)  # Normalize to [-1, 1]
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if min_len is None or len(audio) < min_len:
            min_len = len(audio)
        audio_data.append((filename, audio))

# Step 2: Trim and write back to disk
for filename, audio in audio_data:
    trimmed_audio = audio[:min_len]
    out_path = os.path.join(directory, filename)
    sf.write(out_path, trimmed_audio, target_sr)
    print(f"Processed: {filename}")

print("✅ All files normalized, resampled to 48kHz, and trimmed to the same length.")
