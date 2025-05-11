import librosa

# Load audio file
PATH = "KlavierA(backup).wav"
data, sr = librosa.load(PATH, sr=44100)

# Normalize and clip the audio
data = data.astype('float32')
data = data.clip(-1.0, 1.0)

# Write each byte value as string to compare2.txt
with open("compare2.txt", "w") as f:
    for b in data.tobytes():
        f.write(str(b))