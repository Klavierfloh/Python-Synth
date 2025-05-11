import numpy as np
import scipy.signal as signal
import soundfile as sf

def calculate_snr_in_signal(signal_data, sample_rate):
    """
    Calculate the Signal-to-Noise Ratio (SNR) within a single audio signal.
    The noise is estimated as the difference between the original signal and a low-pass filtered version of it.
    
    :param signal_data: The input audio signal (1D or 2D numpy array)
    :param sample_rate: The sample rate of the audio signal
    :return: SNR in decibels (dB)
    """
    # Ensure that the signal is mono by averaging stereo channels if necessary
    if signal_data.ndim > 1:
        signal_data = np.mean(signal_data, axis=1)  # Convert to mono by averaging channels

    # 1. Smooth the signal (using a low-pass filter to get the "signal" without noise)
    nyquist = 0.5 * sample_rate  # Nyquist frequency
    cutoff_freq = 5000  # A low-pass filter cutoff frequency in Hz
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(1, normal_cutoff, btype='low', analog=False)
    smoothed_signal = signal.filtfilt(b, a, signal_data)

    # 2. Estimate the noise as the difference between the original and smoothed signal
    noise = signal_data - smoothed_signal

    # 3. Calculate the power of the signal and the noise
    signal_power = np.sum(smoothed_signal ** 2) / len(smoothed_signal)
    noise_power = np.sum(noise ** 2) / len(noise)

    # 4. Calculate the SNR in dB
    if noise_power == 0:
        # Prevent division by zero if noise power is zero (ideal case, but rare)
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

# Example usage
audio_file = 'KlavierA.wav'
signal_data, sample_rate = sf.read(audio_file)

# Calculate the SNR for the signal
snr = calculate_snr_in_signal(signal_data, sample_rate)

# Output the result
print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB")
