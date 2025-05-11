import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt

def calculate_tonality_metric(signal_data, sr):
    # Step 1: Fourier Transform to get frequency components
    n = len(signal_data)
    freqs = np.fft.fftfreq(n, 1/sr)
    fft_vals = np.fft.fft(signal_data)
    magnitude = np.abs(fft_vals)
    
    # Only consider positive frequencies
    positive_freqs = freqs[:n//2]
    positive_magnitude = magnitude[:n//2]

    # Step 2: Identify peaks (dominant frequencies)
    peak_indices, _ = signal.find_peaks(positive_magnitude, height=0.1)
    peak_frequencies = positive_freqs[peak_indices]
    
    # Step 3: Fundamental frequency detection (the lowest peak frequency)
    fundamental_freq = peak_frequencies[0] if len(peak_frequencies) > 0 else 0
    
    # Step 4: Harmonic Series Calculation
    harmonics = []
    for i in range(1, 10):  # Check up to 10 harmonics
        harmonic = fundamental_freq * i
        if harmonic <= positive_freqs[-1]:
            harmonics.append(harmonic)
    
    # Step 5: Tonality Metric Calculation
    tonal_value = 0
    harmonic_magnitudes = []
    
    # Loop through the harmonics and compute their magnitude
    for harmonic in harmonics:
        closest_freq_index = np.argmin(np.abs(positive_freqs - harmonic))
        harmonic_magnitude = positive_magnitude[closest_freq_index]
        harmonic_magnitudes.append(harmonic_magnitude)
        tonal_value += harmonic_magnitude  # Add magnitude to tonal value
    
    # Normalize the tonal value by the sum of all magnitudes
    total_magnitude = np.sum(positive_magnitude)
    normalized_tonal_value = tonal_value / total_magnitude if total_magnitude > 0 else 0
    
    return normalized_tonal_value, peak_frequencies, harmonics, harmonic_magnitudes


# Load the audio file (replace "audio_file_path" with your actual file path)
audio_file = "KlavierA(backup).wav"
signal_data, sample_rate = sf.read(audio_file)

# Normalize stereo to mono if needed (just to simplify the analysis)
if len(signal_data.shape) > 1:
    signal_data = signal_data.mean(axis=1)

# Calculate the Aures Tonality Metric
tonal_value, peak_frequencies, harmonics, harmonic_magnitudes = calculate_tonality_metric(signal_data, sample_rate)

# Output the results
print(f"Aures Tonality Metric *1000: {tonal_value *1000}")
print(f"Dominant Frequencies (Peaks): {peak_frequencies}")
print(f"Harmonics Detected: {harmonics}")
print(f"Magnitudes of Harmonics: {harmonic_magnitudes}")

# Plotting the frequency spectrum and harmonic markers

plt.figure(figsize=(10, 6))
plt.plot(signal_data, label="Ton im Zeitbereich", color='blue')
plt.plot(np.fft.fftfreq(len(signal_data), 1/sample_rate)[:len(signal_data)//2], np.abs(np.fft.fft(signal_data))[:len(signal_data)//2])
plt.scatter(peak_frequencies, np.abs(np.fft.fft(signal_data))[:len(signal_data)//2][np.isin(np.fft.fftfreq(len(signal_data), 1/sample_rate)[:len(signal_data)//2], peak_frequencies)], color='red')
plt.title("Zeitdarstellung eines Tons")
plt.xlabel("Zeit (Sample)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
