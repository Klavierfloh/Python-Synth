import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

# Load audio file
audio_file = "KlavierA.wav"
signal_data, sample_rate = sf.read(audio_file)

# Convert to mono if stereo
if len(signal_data.shape) > 1:
    signal_data = signal_data.mean(axis=1)

# Different window sizes for different frequency resolutions
n_fft_small = 1024  # Small window → better time resolution
n_fft_big = 8192    # Large window → better frequency resolution
hop_length = 128    # Keeping hop-length same for fair comparison

# Compute STFT with different window sizes
fft_spectrum_small_win = librosa.stft(signal_data, n_fft=n_fft_small, hop_length=hop_length)
fft_spectrum_big_win = librosa.stft(signal_data, n_fft=n_fft_big, hop_length=hop_length)

# Compute frequency axes based on window sizes
frequencies_small = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft_small)
frequencies_big = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft_big)

# Compute magnitude spectra (mean over time frames)
magnitude_small = np.abs(fft_spectrum_small_win).mean(axis=1)
magnitude_big = np.abs(fft_spectrum_big_win).mean(axis=1)

# Create figure with two subplots sharing x- and y-axes
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)

# Plot small window size result
axs[0].plot(frequencies_small, magnitude_small, label="kleines Fenster "+ str(n_fft_small), color="b", alpha=0.7)
axs[0].set_title("Frequenz Spektrum mit kleiner Fenstergröße (Niedrige Frequenzauflösung)")
axs[1].set_xlabel("Frequenz (Hz)")
axs[0].set_ylabel("Größe")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)
axs[0].legend()

# Plot big window size result
axs[1].plot(frequencies_big, magnitude_big, label="großes Fenster " + str(n_fft_big) , color="r", alpha=0.7)
axs[1].set_title("Frequenz Spektrum mit großer Fenstergröße (Hohe Frequenzauflösung)")
axs[1].set_xlabel("Frequenz (Hz)")
axs[1].set_ylabel("Größe")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)
axs[1].legend()

# Adjust layout and show
plt.tight_layout()
plt.show()
