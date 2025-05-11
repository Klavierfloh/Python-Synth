import InputHandler
from rtmidi.midiutil import open_midiinput
from time import perf_counter, process_time
from librosa import load

PATH = "./KlavierA.wav"  # Weg zum Sample

DATA, SR = load(PATH, sr=None)

handler = InputHandler.InputHandler()
t1 = perf_counter()
handler.play_wave(60, 70, DATA)
t2 = perf_counter()
print(f"Time for play_wave: {t2-t1:.6f}s")