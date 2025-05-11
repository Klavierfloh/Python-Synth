import numpy as np
import librosa
import soundfile as sf  # Import soundfile to save wav files
import matplotlib.pyplot as plt

name1 = 'Piano.mf.A3-[AudioTrimmer.com].wav'
name2 = 'output_generated.wav'

ref, _ = librosa.load(name1, sr=48000)
test, _ = librosa.load(name2, sr=48000)

# Trim the beginning and end to the same length
min_len = min(len(ref), len(test))
ref = ref[:min_len]
test = test[:min_len]

# Save the trimmed audio using soundfile
sf.write(name1, ref, 48000)
sf.write(name2, test, 48000)

# Load the reference and test signals again, now using librosa
reference, sr_ref = librosa.load(name1, sr=None)
test, sr_test = librosa.load(name2, sr=None)

# Resample if needed (to the same sample rate)
if sr_ref != sr_test:
    test = librosa.resample(test, sr_test, sr_ref)

# Calculate RMS error
rms_error = np.sqrt(np.mean((reference - test)**2))

# Plot both signals to visually compare
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Reference")
plt.plot(reference[:1000])
plt.subplot(1, 2, 2)
plt.title("Test")
plt.plot(test[:1000])
plt.show()

print(f"RMS Error between signals: {rms_error}")
