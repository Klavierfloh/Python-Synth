from __future__ import print_function

import keyboard

from KeyboardInputHandler import KeyboardInputHandler
from MidiInputHandler import MidiInputHandler
import matplotlib.pyplot as plt
from librosa import load
import logging
import sys
import time
from rtmidi.midiutil import open_midiinput, list_input_ports
import scipy
# Setzt wichtige Konstanten

PATH = "Piano.mf.A3-[AudioTrimmer.com].wav"   # Weg zum Sample
FREQU = 440                     # Grundfrequenz des Samples
CHUNK = 512
MIDI = True
KEYS = {".": 47, ",": 46, "m": 45, "n": 44,
        "a": 48, "s": 49, "d": 50, "f": 51,
        "g": 52, "h": 53, "j": 54, "k": 55,
        "l": 56, "ö": 57, "ä": 58, "#": 59, "y": 60,
        "A": 61, "S": 62, "D": 63, "F": 64,
        "G": 65, "H": 66, "J": 67, "K": 68,
        "L": 69, "Ö": 70, "Ä": 71, "'": 72, "Y": 73}
DATA, SR = load(PATH, sr=None)

# Initialisiert die Logger
log = logging.getLogger('midiin_callback')
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
numba_logger = logging.getLogger('numba')  # Deaktiviert störende Ausgaben
numba_logger.setLevel(logging.WARNING)

try:

    list_input_ports()
    # fragt den Port ab, über den Midi-Inputs gelesen werden
    port = input("Wich Midi-port do you choose? For Keyboard controle type \"k\": ")
    if port.isnumeric():
        # versucht, den Midi-Port zu öffnen
        midiin, port_name = open_midiinput(6)#open_midiinput(port)
    else:
        MIDI = False
except (EOFError, KeyboardInterrupt):
    sys.exit()

# legt fest, die __call__-Funktion welcher Klasse ausgeführt werden soll, sobald ein Input vorliegt
if MIDI:
    midiin.set_callback(MidiInputHandler(DATA, SR, CHUNK,FREQU))
else:
    keyboard.hook(KeyboardInputHandler(DATA, SR, CHUNK,KEYS,FREQU))

try:
    #  Hier passiert nichts mehr, weil alles über den Callback des midiin-Objektes gesteuert wird
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    log.debug("Interruption!")

finally:
    #  Sobald das Programm beendet wird, wird der Midi-Port sauber geschlossen und gelöscht
    log.debug("Exit")
    midiin.close_port()
    del midiin
