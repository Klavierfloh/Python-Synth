import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf

audio_file = "KlavierA(backup).wav"
signal_data, sample_rate = sf.read(audio_file)

# nur Mono Signal
if len(signal_data.shape) > 1:
    signal_data = signal_data.mean(axis=1)

# fft
n = len(signal_data)
frequencies = np.fft.fftfreq(n, 1/sample_rate)  
fft_values = np.fft.fft(signal_data)  
magnitude = np.abs(fft_values)  

# Wirklich nur Mono
print(f"Shape of magnitude: {magnitude.shape}")
if len(magnitude.shape) > 1:
    raise ValueError("Magnitude is not a 1D array. Something went wrong with the FFT.")

# fft hat positive und negatve Frequenze, brauche aber nur positive
magnitude = magnitude[:n//2]
frequencies = frequencies[:n//2]

# dominante Frequenzen
peak_indices, _ = signal.find_peaks(magnitude, height=0.1)  # Find peaks in the positive frequencies
peak_frequencies = frequencies[peak_indices]
print("Dominant Frequencies:", peak_frequencies)

# 4. Harmonic Structure: Calculate Tonality based on harmonic relations
def check_harmonics(peak_frequencies, tolerance=0.1):
    tonal_value = 0
    fundamental = peak_frequencies[0]  # Assume the first peak is the fundamental frequency

    # Loop through each peak and check if it is a harmonic of the fundamental
    for freq in peak_frequencies[1:]:
        # Check if the frequency is an integer multiple of the fundamental frequency
        harmonic_ratio = freq / fundamental
        if np.isclose(harmonic_ratio, round(harmonic_ratio), atol=tolerance):  # Allow for small floating point differences
            tonal_value += 1  # Count harmonic pairs
    
    return tonal_value

tonal_value = check_harmonics(peak_frequencies)
print(f"Tonal Value (Number of Harmonic Pairs): {tonal_value}")



# Plot fft
plt.figure(figsize=(10, 6))
plt.plot(frequencies, magnitude)  # Only plot positive frequencies
plt.title("Frequency Spectrum of the Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xscale("log")
plt.grid(True)
plt.show()
