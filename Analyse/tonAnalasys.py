from librosa import load
import numpy as np

PATH_1 = "KlavierA.wav"
PATH_2 = "KlavierA(backup).wav"

DATA_1, SR_1 = load(PATH_1, sr=44100)
DATA_2, SR_2 = load(PATH_2, sr=44100)


min_length = min(len(DATA_1), len(DATA_2))

# Trim both signals to the same length
DATA_1_trimmed = DATA_1[:min_length]
DATA_2_trimmed = DATA_2[:min_length]

numpy_correlation = np.corrcoef(DATA_1_trimmed, DATA_2_trimmed)[0, 1]
print('NumPy Correlation:', numpy_correlation)