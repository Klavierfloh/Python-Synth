import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, lfilter
import scipy.signal as signal
import pyaudio

# Lade eine Audiodatei
dateiname = 'KlavierA(backup).wav'
y, sr = librosa.load(dateiname, sr=None)  # Laden mit ursprünglicher Abtastrate
t = np.linspace(0, len(y) / sr, len(y), endpoint=False)  # Zeitachse

# Berechnung der FFT des ursprünglichen Signals
fft_spektrum = np.fft.fft(y)
frequenzen = np.fft.fftfreq(len(fft_spektrum), d=1/sr)

# Behalte nur die positive Hälfte des Spektrums
halb_spektrum_orig = np.abs(fft_spektrum[:len(fft_spektrum)//2])
halb_frequenzen_orig = frequenzen[:len(frequenzen)//2]

# Berechnung der STFT (Short-Time Fourier Transform)
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

nyquist = 22050

# Funktion zur Normalisierung der Frequenz
def normalisierteFrequenzGrundton(freq):
    return freq / nyquist

nFreqLow = normalisierteFrequenzGrundton(5080)
nFreqHigh = normalisierteFrequenzGrundton(5120)
num_taps = 100  # Anzahl der Koeffizienten (höher = schärferer Übergang)
cutoff = [nFreqLow, nFreqHigh]  # Bandpass-Bereich (normalisiert auf Nyquist-Frequenz)
fenster = 'hamming'  # Hamming-Fenster zur Glättung der Filterantwort

# Berechnung der Filterkoeffizienten
fir_koeffs = signal.firwin(num_taps, cutoff, pass_zero=False, window=fenster)



# Anwendung des Filters auf das Signal
daten_gefiltert = signal.lfilter(fir_koeffs, 1.0, y)
y_gefiltert = daten_gefiltert

# Berechnung der FFT des gefilterten Signals
fft_gefiltert = np.fft.fft(y_gefiltert)
halb_spektrum_gefiltert = np.abs(fft_gefiltert[:len(fft_gefiltert)//2])

# Berechnung der STFT des gefilterten Signals
D_gefiltert = librosa.amplitude_to_db(np.abs(librosa.stft(y_gefiltert)), ref=np.max)

# Erstellen von Subplots in einem 3x3-Raster
fig, axs = plt.subplots(3, 3, figsize=(15, 12))

# Plot 1: Ursprüngliche Tonsignal-Wellenform
axs[0, 0].plot(t, y, label="Ursprüngliches Signal", color='gray')
axs[0, 0].set_xlabel("Zeit (s)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].set_title("Wellenform des ursprünglichen Tonsignals")
axs[0, 0].legend()
axs[0, 0].grid()

# Plot 2: FFT des ursprünglichen Signals
axs[0, 1].plot(halb_frequenzen_orig, halb_spektrum_orig, color='blue')
axs[0, 1].set_xlabel("Frequenz (Hz)")
axs[0, 1].set_ylabel("Magnitude")
axs[0, 1].set_title("FFT-Spektrum des ursprünglichen Tons")
axs[0, 1].set_xscale("log")  # Setze x-Achse auf logarithmische Skala
axs[0, 1].set_yscale("log")
axs[0, 1].grid()

# Plot 3: Spektrogramm des ursprünglichen Signals
axs[0, 2].set_title("Spektrogramm eines Tons")
img_orig = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=axs[0, 2])
fig.colorbar(img_orig, ax=axs[0, 2], label="dB")

# Plot 4: Frequenzgang des FIR-Filters
w, h = freqz(fir_koeffs, worN=8000, fs=sr)
axs[2, 0].plot(w, np.abs(h), color='red')
axs[2, 0].set_xlabel("Frequenz (Hz)")
axs[2, 0].set_ylabel("Verstärkung")
axs[2, 0].set_title("Frequenzgang des FIR-Filters")
axs[2, 0].grid()

# Plot 5: Gefiltertes vs. ursprüngliches Tonsignal
axs[1, 0].plot(t, y, label="Ursprüngliches Signal", alpha=0.5, color='gray')
axs[1, 0].plot(t, y_gefiltert, label="Gefiltertes Signal", color='red', linewidth=2)
axs[1, 0].set_xlabel("Zeit (s)")
axs[1, 0].set_ylabel("Amplitude")
axs[1, 0].set_title("Vergleich: Gefiltertes vs. ursprüngliches Tonsignal")
axs[1, 0].legend()
axs[1, 0].grid()

# Plot 6: FFT des gefilterten Signals
axs[1, 1].plot(halb_frequenzen_orig, halb_spektrum_gefiltert, color='green')
axs[1, 1].set_xscale("log")  # Setze x-Achse auf logarithmische Skala
axs[1, 1].set_xlabel("Frequenz (Hz)")
axs[1, 1].set_ylabel("Magnitude")
axs[1, 1].set_title("FFT-Spektrum des gefilterten Tons")
axs[1, 1].grid()
axs[1, 1].set_yscale("log")

# Plot 7: Vergleich der FFT des ursprünglichen und gefilterten Signals
axs[2, 1].plot(halb_frequenzen_orig, halb_spektrum_orig, label="Ursprüngliches Signal", color='green')
axs[2, 1].plot(halb_frequenzen_orig, halb_spektrum_gefiltert, label="Gefiltertes Signal", color='red')
axs[2, 1].set_xscale("log")  # Setze x-Achse auf logarithmische Skala
axs[2, 1].set_xlabel("Frequenz (Hz)")
axs[2, 1].set_ylabel("Magnitude")
axs[2, 1].set_title("FFT-Vergleich: Ursprünglich vs. Gefiltert")
axs[2, 1].set_yscale("log")
axs[2, 1].legend()
axs[2, 1].grid()

# Plot 8: Spektrogramm des gefilterten Tons
axs[1, 2].set_title("Spektrogramm des gefilterten Tons")
img_filtered = librosa.display.specshow(D_gefiltert, sr=sr, x_axis="time", y_axis="log", ax=axs[1, 2])
fig.colorbar(img_filtered, ax=axs[1, 2], label="dB")

# Layout anpassen und das Diagramm anzeigen
plt.tight_layout()
plt.show()
