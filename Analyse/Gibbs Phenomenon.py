import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as signal
import pyaudio

# Load an audio file
filename = 'KlavierA(backup).wav'
y, sr = librosa.load(filename, sr=None)  # Load with original sampling rate
t = np.linspace(0, len(y) / sr, len(y), endpoint=False)  # Time axis

# Compute FFT of original signal
fft_spectrum = np.fft.fft(y)
frequencies = np.fft.fftfreq(len(fft_spectrum), d=1/sr)

# Keep only the positive half of the spectrum
half_spectrum_orig = np.abs(fft_spectrum[:len(fft_spectrum)//2])
half_frequencies_orig = frequencies[:len(frequencies)//2]

# Compute STFT (Short-Time Fourier Transform)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

nyquist = sr / 2

def normalizedFrequencyBaseTone(freq):
    return freq / nyquist

nFreqLow = normalizedFrequencyBaseTone(500)
nFreqHigh = normalizedFrequencyBaseTone(4048)
num_taps = 100  # Fewer taps make the Gibbs effect more pronounced
cutoff = [nFreqLow, nFreqHigh]  # Band-pass range

# Compute filter coefficients with a rectangular window (Gibbs phenomenon visible)
fir_coeffs_gibbs = signal.firwin(num_taps, cutoff, pass_zero=False, window='boxcar')
y_filtered_gibbs = signal.lfilter(fir_coeffs_gibbs, 1.0, y)

# Compute filter coefficients with a Hamming window (smoother response)
fir_coeffs_hamming = signal.firwin(num_taps, cutoff, pass_zero=False, window='hamming')
y_filtered_hamming = signal.lfilter(fir_coeffs_hamming, 1.0, y)

# Compute FFTs of filtered signals
fft_filtered_gibbs = np.fft.fft(y_filtered_gibbs)
half_spectrum_filtered_gibbs = np.abs(fft_filtered_gibbs[:len(fft_filtered_gibbs)//2])

fft_filtered_hamming = np.fft.fft(y_filtered_hamming)
half_spectrum_filtered_hamming = np.abs(fft_filtered_hamming[:len(fft_filtered_hamming)//2])

# Compute STFTs of filtered signals
D_filtered_gibbs = librosa.amplitude_to_db(np.abs(librosa.stft(y_filtered_gibbs)), ref=np.max)
D_filtered_hamming = librosa.amplitude_to_db(np.abs(librosa.stft(y_filtered_hamming)), ref=np.max)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 8))

# FIR filter frequency response (Gibbs effect visible)
w, h_gibbs = signal.freqz(fir_coeffs_gibbs, worN=8000, fs=sr)
axs[0].plot(w, np.abs(h_gibbs), color='red')
axs[0].set_title("FIR Filter Antwort (Gibssches Phänomen sichtbar)")
axs[0].set_xlabel("Frequenz (Hz)")
axs[0].set_ylabel("Verstärkung")
axs[0].grid()

# FIR filter frequency response (Hamming window, smoother)
w, h_hamming = signal.freqz(fir_coeffs_hamming, worN=8000, fs=sr)
axs[1].plot(w, np.abs(h_hamming), color='blue')
axs[1].set_title("FIRFilter Antwort (Hamming Window)")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Verstärkung")
axs[1].grid()

plt.tight_layout()
plt.show()
