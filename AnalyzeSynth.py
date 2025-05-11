import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import re

# --- Parameters ---
original_dir = "./compOrigs"
compare_dir = "./finalOutput"

note_pattern = re.compile(r'([A-G]#?\d)')  # Matches note names like C#4

# --- Expected Frequencies Map ---
expected_freqs = {
    "A2": 110.00, "A#2": 116.54, "B2": 123.47,
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56,
    "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00, "G#3": 207.65,
    "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13,
    "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30,
    "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25,
    "E5": 659.25, "F5": 698.46, "F#5": 739.99, "G5": 783.99,
    "G#5": 830.61, "A5": 880.00
}

# --- Roughness Function ---
def compute_roughness(freqs, mags):
    roughness = 0
    idx = np.argsort(mags)[-10:]
    freqs, mags = freqs[idx], mags[idx]
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            f1, f2 = freqs[i], freqs[j]
            a1, a2 = mags[i], mags[j]
            s = 0.24 / (0.021 * min(f1, f2) + 19)
            df = abs(f1 - f2)
            dissonance = (a1 * a2) * np.exp(-3.5 * s * df) * np.exp(-5.75 * s * df)
            roughness += dissonance
    return roughness

# --- Improved Aures Approximation ---
def compute_improved_aures(y, sr, tonal_threshold=-30, spectral_resolution=2048, weighting_exponent=1.0):
    D = librosa.stft(y, n_fft=spectral_resolution)
    magnitude, _ = librosa.magphase(D)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    tonal_components = centroid > tonal_threshold
    tonality_score = np.sum(tonal_components * (centroid ** weighting_exponent))
    return tonality_score

# --- Feature Extraction ---
def extract_features(file_path):
    print(f"Extracting features from: {file_path}")
    y, sr = librosa.load(file_path, sr=None)
    
    # Print to debug
    
    f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
    avg_f0 = np.mean(f0)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y)[0].mean()
    harmonic, percussive = librosa.effects.hpss(y)
    hnr = 10 * np.log10(np.var(harmonic) / (np.var(percussive) + 1e-6))
    S = np.abs(librosa.stft(y))
    freqs, mags = librosa.piptrack(S=S, sr=sr)
    freqs, mags = freqs.flatten(), mags.flatten()
    valid = mags > np.median(mags)
    freqs, mags = freqs[valid], mags[valid]
    roughness = compute_roughness(freqs, mags)
    tonality_score = compute_improved_aures(y, sr)

    # Debug: Print the extracted features
    print(f"Avg F0: {avg_f0}, Spectral Centroid: {centroid}, Spectral Flatness: {flatness}, Zero Crossing Rate: {zcr}, HNR: {hnr}, Roughness: {roughness}, Tonality: {tonality_score}")
    
    return avg_f0, centroid, flatness, hnr, zcr, roughness, tonality_score

# --- Process and Compare ---
results = []

for file in sorted(os.listdir(original_dir)):
    match = note_pattern.search(file)
    if not match:
        continue
    note = match.group(1)
    if note not in expected_freqs:
        continue

    orig_path = os.path.join(original_dir, file)
    comp_path = os.path.join(compare_dir, file)
    if not os.path.exists(comp_path):
        continue

    orig_feats = extract_features(orig_path)
    comp_feats = extract_features(comp_path)
    expected = expected_freqs[note]

    results.append((note, expected, *orig_feats, *comp_feats))
    print(f"Compared {file} ({note})")

# Convert to structured NumPy array
results = np.array(results, dtype=object)
results = sorted(results, key=lambda x: float(x[1]))


# --- Extract and Plot ---
notes = [r[0] for r in results]
expected = [float(r[1]) for r in results]

# Original
orig_pitch     = [float(r[2]) for r in results]
orig_centroid  = [float(r[3]) for r in results]
orig_flatness  = [float(r[4]) for r in results]
orig_hnr       = [float(r[5]) for r in results]
orig_zcr       = [float(r[6]) for r in results]
orig_roughness = [float(r[7]) for r in results]
orig_tonality  = [float(r[8]) for r in results]

# Compared
comp_pitch     = [float(r[9]) for r in results]
comp_centroid  = [float(r[10]) for r in results]
comp_flatness  = [float(r[11]) for r in results]
comp_hnr       = [float(r[12]) for r in results]
comp_zcr       = [float(r[13]) for r in results]
comp_roughness = [float(r[14]) for r in results]
comp_tonality  = [float(r[15]) for r in results]

# --- Plotting ---
fig, axs = plt.subplots(7, 1, figsize=(14, 28), sharex=True)

metrics = [
    ("Pitch (Hz)", orig_pitch, comp_pitch),
    ("Spectral Centroid", orig_centroid, comp_centroid),
    ("Spectral Flatness", orig_flatness, comp_flatness),
    ("Harmonic-to-Noise Ratio", orig_hnr, comp_hnr),
    ("Zero Crossing Rate", orig_zcr, comp_zcr),
    ("Roughness", orig_roughness, comp_roughness),
    ("Improved Aures Tonality", orig_tonality, comp_tonality)
]

for i, (title, orig_data, comp_data) in enumerate(metrics):
    axs[i].plot(notes, orig_data, 'o-', label='Original')
    axs[i].plot(notes, comp_data, 'x--', label='Compared')
    axs[i].set_title(title)
    axs[i].grid()
    axs[i].legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
