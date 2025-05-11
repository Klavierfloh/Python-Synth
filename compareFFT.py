import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

def generate_note_range(start_note="A2", end_note="A4"):
    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    start_note_name, start_oct = start_note[:-1], int(start_note[-1])
    end_note_name, end_oct = end_note[:-1], int(end_note[-1])

    if start_note_name not in NOTES:
        start_note_name = start_note[:-2]
        start_oct = int(start_note[-1])
    if end_note_name not in NOTES:
        end_note_name = end_note[:-2]
        end_oct = int(end_note[-1])

    note_range = []
    for oct in range(start_oct, end_oct + 1):
        for note in NOTES:
            full_note = f"{note}{oct}"
            if oct == start_oct and NOTES.index(note) < NOTES.index(start_note_name):
                continue
            if oct == end_oct and NOTES.index(note) > NOTES.index(end_note_name):
                break
            note_range.append(full_note)
    return note_range

NOTE_RANGE = generate_note_range("A2", "A4")
print(NOTE_RANGE)

def load_and_preprocess(filepath, n_samples=44100):
    rate, data = wav.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    data = data[:n_samples]
    data = data / np.max(np.abs(data))
    return data

def rms_error(x, y):
    min_len = min(len(x), len(y))
    x, y = x[:min_len], y[:min_len]
    return np.sqrt(np.mean((x - y)**2))

def correlation_coefficient(x, y):
    min_len = min(len(x), len(y))
    return np.corrcoef(x[:min_len], y[:min_len])[0, 1]

from scipy.fft import fft

def spectral_centroid(signal, sr=44100):
    spectrum = np.abs(fft(signal))[:len(signal)//2]
    freqs = np.linspace(0, sr/2, len(spectrum))
    return np.sum(freqs * spectrum) / np.sum(spectrum)

def spectral_centroid_diff(x, y):
    return abs(spectral_centroid(x) - spectral_centroid(y))



orig_path = "compOrigs"
out_path = "finalOutput"

results = []

for note in NOTE_RANGE:
    orig_file = os.path.join(orig_path, f"Piano.mf.{note}.wav")
    gen_file = os.path.join(out_path, f"{note}.wav")

    if os.path.exists(orig_file) and os.path.exists(gen_file):
        try:
            data1 = load_and_preprocess(orig_file)
            data2 = load_and_preprocess(gen_file)

            corr = correlation_coefficient(data1, data2)
            centroid_diff = spectral_centroid_diff(data1, data2)

            results.append((note, corr, centroid_diff))
        except Exception as e:
            print(f"Fehler bei {note}: {e}")
    else:
        print(f"Überspringe {note}: Datei fehlt")


if results:
    results.sort(key=lambda x: NOTE_RANGE.index(x[0]))
    notes, corrs, centroid_diffs = zip(*results)


    plt.bar(notes, corrs, color="darkblue")
    plt.ylabel("Korrelationskoeffizient")
    plt.title("Lineare Ähnlichkeit (Original vs. Generiert)")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Keine gültigen Vergleichspaare gefunden.")
