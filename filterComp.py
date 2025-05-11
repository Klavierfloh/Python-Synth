import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter
import scipy.signal as signal

# === Load and prepare both audio files ===
file1 = 'KlavierA.wav'
file2 = 'KlavierA(backup).wav'

y1, sr1 = librosa.load(file1, sr=None)
y2, sr2 = librosa.load(file2, sr=None)

# === Resample if needed ===
if sr1 != sr2:
    y2 = librosa.resample(y2, orig_sr=sr2, target_sr=sr1)
    sr2 = sr1

# === Trim to same length ===
min_len = min(len(y1), len(y2))
y1 = y1[:min_len]
y2 = y2[:min_len]

t = np.linspace(0, len(y1) / sr1, len(y1), endpoint=False)

# === FFTs ===
fft1 = np.fft.fft(y1)
fft2 = np.fft.fft(y2)

frequencies = np.fft.fftfreq(len(fft1), d=1/sr1)
halb_freqs = frequencies[:len(frequencies)//2]
halb_fft1 = np.abs(fft1[:len(fft1)//2])
halb_fft2 = np.abs(fft2[:len(fft2)//2])

# === STFTs ===
D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)

# === Visualization ===
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Waveform comparison
axs[0, 0].plot(t, y1, label="Datei 1", alpha=0.7)
axs[0, 0].plot(t, y2, label="Datei 2", alpha=0.7)
axs[0, 0].set_title("Übereinandergelegte Wellen")
axs[0, 0].set_xlabel("Zeit (s)")
axs[0, 0].set_ylabel("Lautstärke")
axs[0, 0].legend()
axs[0, 0].grid()

# FFT comparison
axs[0, 1].plot(halb_freqs, halb_fft1, label="Datei 1", alpha=0.7)
axs[0, 1].plot(halb_freqs, halb_fft2, label="Datei 2", alpha=0.7)
axs[0, 1].set_xscale("log")
axs[0, 1].set_yscale("log")
axs[0, 1].set_title("FFT Spektrum Vergleich")
axs[0, 1].set_xlabel("Frequenz (Hz)")
axs[0, 1].set_ylabel("Lautstärke")
axs[0, 1].legend()
axs[0, 1].grid()

# Spectrogram 1
img1 = librosa.display.specshow(D1, sr=sr1, x_axis="time", y_axis="log", ax=axs[1, 0])
axs[1, 0].set_title("Spektrogramm - Datei 1")
fig.colorbar(img1, ax=axs[1, 0], format="%+2.0f dB")

# Spectrogram 2
img2 = librosa.display.specshow(D2, sr=sr1, x_axis="time", y_axis="log", ax=axs[1, 1])
axs[1, 1].set_title("Spektrogramm - Datei 2")
fig.colorbar(img2, ax=axs[1, 1], format="%+2.0f dB")


plt.tight_layout()
plt.show()
