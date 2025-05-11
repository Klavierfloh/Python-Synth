from __future__ import print_function
import time
import numpy as np
import pyaudio
from scipy import signal
import librosa


class InputHandler:
    def __init__(self, sr=44100, chunk_size=512, baseFrequency=440):
        self.DECAY = 30
        self.BASE_FREQUENCY = baseFrequency
        self.OVERTONES = 7

        self.sr = sr # Abtastrate des ursprünlichen Samples
        self.nyquist = sr/2 #ich gehe davon aus, dass es richtig eingespielt wurde
        self.chunk_size = chunk_size # Wieviel wird abgespielt pro Zeiteinheit
        self.p = pyaudio.PyAudio()
        self.active_notes = {}  # Dictionary, um spielende Noten zu verwalten
        self.notes_to_remove_from_active = []
        self.notes_to_remove_from_decaying = []

        self.decaying_notes = {} #Töne, die losgelassen wurden
 
        self.decayFactors = 0.9**np.arange(self.DECAY)
        self.overtoneFactors = 1+0.7**np.arange(self.OVERTONES)

        self.cachedResamples = {}        

        fs = 44100  # Sampling rate
        T60 = 0.5  # 500ms reverb time
        N = int(fs * T60)  # FIR filter length

        # Generiert Reverb response
        alpha = 3 * np.log(10) / (T60 * fs)
        ReverbFactors = np.exp(-alpha * np.arange(N))

        # Normalize
        self.reverbeFactors_normalized = ReverbFactors / np.max(np.abs(ReverbFactors))


        # Es gibt nur einen Output Stream, in den alles geschrieben 
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sr,
            output=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()


    def play_wave(self, note, strength, data):
        """Fügt neuen Ton zu den Aktiven noten hinzu. Berechnet dafür die Welleninformationen neu.
        führt fft, phaseVocoder und ifft zusammen"""
        def normalizedFrequencyBaseTone(freq):
            return freq/self.nyquist
        # berechnet neue Abtastrate für Tonhöhe

        rate_factor = (2**(-(57 - note)/12))

        frequency = self.BASE_FREQUENCY*2**((note - 57)/12)
        
        # berechnet neue Audiodaten für Tonhöhe
        # res_type ist der Algorithmus. Da noch schauen, welcher der beste ist  
        new_sr = int(self.sr * rate_factor)

        filteredSignals =  []

        if note not in self.cachedResamples:
            # Stretch factor (how much to stretch the audio)
            time_of_note = len(data)/self.sr * (frequency/self.BASE_FREQUENCY)**-0.5
            stretch_factor = self.sr/new_sr
            data_stretched = librosa.effects.time_stretch(y=data, rate=stretch_factor)

            data_stretched = np.interp(np.arange(0, len(data_stretched), new_sr / self.sr), np.arange(0, len(data_stretched)), data_stretched)


            # Apply time stretching (pitch is preserved)

            print("Length: " + str(data_stretched.shape))                 

            self.cachedResamples[note] = {
                            "streched" : data_stretched
                        }
            for i in range(1,self.OVERTONES):
                nFreqLow = i*normalizedFrequencyBaseTone(frequency-20)
                nFreqHigh = i*normalizedFrequencyBaseTone(frequency+20)

                num_taps = 100       # Number of coefficients (higher = sharper transition)
                cutoff = [nFreqLow, nFreqHigh]  # Band-pass range (normalized to Nyquist frequency)
                window = 'hamming'   # Hamming window to smooth response

                # Compute filter coefficients
                coeffs = signal.firwin(num_taps, cutoff, pass_zero=False, window=window)

                data_stretched_filtered = signal.lfilter(coeffs,1.0,data_stretched)
                filteredSignals.append(data_stretched_filtered)

                self.cachedResamples[note]["filtered_"+str(i)] = data_stretched_filtered * self.overtoneFactors[i]

        else:
            data_stretched = self.cachedResamples[note]["streched"]
            for i in range(1,self.OVERTONES):
                filteredSignals.append(self.cachedResamples[note]["filtered_"+str(i)])
        
        

        # Nimmt Anschlagstärke mit in Berechnung
        volume = strength / 127.0

        data_stretched = volume*(sum(filteredSignals)+data_stretched)
        print(sum(filteredSignals))
        
        #data_stretched = self.fast_convolve(data_stretched, self.reverbeFactors_normalized)
        # Speichert die neue Note zum Spielen in active_notes
        self.active_notes[note] = {
            "waveform": data_stretched,
            "frequency" : frequency,  # berechnet mit f=440Hz*2**(-x)/12
            "position": 0, # Wo im playback?,
            "timeSinceLetGo" : 0 #Decay
        }

    def stop_wave(self, note):
        """Entfernt Ton aus aktiven Noten. Werden zu decaying Noten."""
        if note in self.active_notes:
            note_data = self.active_notes[note]
            self.notes_to_remove_from_active.append((note, note_data))

    def apply_decay(self,waveform, time_since_let_go):
        """Zerfall wird mit voerberechneten Werten angewendet. Dadurch schneller, als alles live zu berechnen."""  
        if time_since_let_go >= self.DECAY:
            return waveform * self.decayFactors[-1]  # benutzt letzte Stelle für Fehlerfrei
        return waveform * self.decayFactors[time_since_let_go]
    
    # Struktur vorgegeben durhc PyAudio
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Berechnet für allen aktiven Töne, wie sie ausgegeben werden. löscht fertige Töne"""
        output_buffer = np.zeros(frame_count, dtype=np.float32) # neuer leerer Buffer der größe frame_count


        # Mixt alle aktiven Töne
        for note, note_data in list(self.active_notes.items()): #ließt aus, welche es gibt
            waveform = note_data["waveform"]
            waveform_len = len(waveform)
            pos = note_data["position"]

            # Ton ist am Ende nicht länger als möglich
            end_pos = min(pos + frame_count, waveform_len)
            if end_pos == waveform_len:
                self.notes_to_remove_from_active.append((note, note_data))

            waveform_slice = waveform[pos:end_pos]

            output_len = end_pos - pos
            output_buffer[:output_len] += waveform_slice[:output_len]

            # Updated playback position
            self.active_notes[note]["position"] = end_pos

        #handeled Decay
        for note, note_data in list(self.decaying_notes.items()):
            waveform = note_data["waveform"]
            waveform_len = len(waveform)
            pos = note_data["position"]
            timeSinceLetGo = note_data["timeSinceLetGo"]
            frequency = note_data["frequency"]


            # Ton ist am Ende nicht länger als möglich
            end_pos = min(pos + frame_count, len(waveform))

            if end_pos == waveform_len or timeSinceLetGo >= self.DECAY:
                self.notes_to_remove_from_decaying.append(note)


            waveform_slice = waveform[pos:end_pos]
            waveform_slice_dampened = self.apply_decay(waveform_slice, timeSinceLetGo)

            output_len = end_pos - pos
            output_buffer[:output_len] += waveform_slice_dampened[:output_len]
            # Updated playback position
            self.decaying_notes[note]["position"] = end_pos  # Updated die Abspielposition
            self.decaying_notes[note]["timeSinceLetGo"] += 1

        # löscht fertige Noten
        for note in self.notes_to_remove_from_active:
            note, note_data = note
            if note in self.active_notes:
                self.decaying_notes[note] = note_data
                del self.active_notes[note]
        
        for note in self.notes_to_remove_from_decaying:
            if note in self.decaying_notes:
                del self.decaying_notes[note]

        self.notes_to_remove_from_active.clear()
        self.notes_to_remove_from_decaying.clear()

        # normalisiert den Buffer (wie starker Compressor)
        output_buffer = np.clip(output_buffer, -1.0, 1.0)

        # geht zurüc und sagt, dass abgespielt werden soll, was gerade berechnet wurde
        return (output_buffer.tobytes(), pyaudio.paContinue)

    def close(self):
        """Schließt alles ordentlich"""
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

from InputHandler import InputHandler


class KeyboardInputHandler(InputHandler):

    def __init__(self, DATA, SR, CHUNK, KEYS, BaseFrequency):
        super().__init__(sr=SR, chunk_size=CHUNK, baseFrequency=BaseFrequency)
        self.DATA, self.KEYS = DATA, KEYS
        self.pressedKeys = {}

        self.strenght = 100

        print("To Change Volume press \"+\" or \'-\'")

    def __call__(self, key):
        """Verwaltet Tastaturinputs"""
        key_str = str(key)
        # Das Aufhören zu spielen, wenn losgelassen funktioniert noch nicht so gut
        # Tastatur zu viele Inputs pro Sekunde nötig
        if key_str.endswith("up)"):  # Key losgelassen
            key_value = key_str.split(" ")[0].split("(")[1]
            if key_value in self.KEYS:
               self.stop_wave(self.KEYS[key_value])  # Stoppt note
            self.pressedKeys[key_value] = False

            pass
        elif key_str.endswith("down)"):  # Wenn Taste gedrückt
            key_value = key_str.split(" ")[0].split("(")[1]
            if key_value in self.KEYS:
                # um zu mgehen, dass ein Tin mehrmals gespieltwird, weil der Finger liegen bliebt
                if  key_value not in self.pressedKeys or self.pressedKeys[key_value] != True:
                    if self.KEYS[key_value] in self.active_notes:
                        return
                    #print(f"Playing note: {self.KEYS[key_value]}")
                    self.pressedKeys[key_value] = True
                    self.play_wave(self.KEYS[key_value], self.strenght, self.DATA)  # Startet note
            
            elif key_value ==  "+":
                self.strenght += 5
                print("louder!")
            
            elif key_value == "-":
                self.strenght -= 5
                print("Quiet!")
            else:
                print("Invalid Key")

class MidiInputHandler(InputHandler):
    def __init__(self, DATA, SR, CHUNK, FREQU):
        super().__init__(sr=SR, chunk_size=CHUNK, baseFrequency=FREQU)
        self.DATA, self.SR, self.CHUNK = DATA, SR, CHUNK

    def __call__(self, event, data=None):  # Funktion, die vom midiin-Objekt aufgerufen wird, bei Midi-Input
        """Verwaltet Midiinputs"""
        message, deltatime = event  # entpackt Midi-Event
        #print(event)
        print(message)
        #print("%r" % message)  # (self.port, self._wallclock)
        if message[2] != 0:  # Wenn eine Taste gedrückt wird

            print("Trying to play MIDI")
            if 91 >= message[1] >= 9:  # Wenn eine Taste gedrückt wird, die auf der Klaviertastatur liegt
                # 96, 9, aber wegen invalid samplerate ist Obergrenze 91
                frequency = self.BASE_FREQUENCY * 2 ** ((57 - (57 - (message[1] - 57))) / 12)  # berechnet mit f=440Hz*2**(-x)/12
                # x wird anhand von message[1] berechnet
                #print(f"Frequenz: {frequency}")
                # spielt den Ton ab
                self.play_wave(message[1],  message[2], self.DATA)  # Start note
        else:
           # AUch hier, stop on nicht mehr Drücken funktiert noch nicht ganz
           self.stop_wave(note=message[1]) 
           pass

import keyboard
import matplotlib.pyplot as plt
from librosa import load
import logging
import sys
import time
from rtmidi.midiutil import open_midiinput, list_input_ports
import scipy
# Setzt wichtige Konstanten

PATH = "KlavierA(backup).wav"   # Weg zum Sample
FREQU = 440                     # Grundfrequenz des Samples
CHUNK = 512
MIDI = True
KEYS = {"a": 48, "s": 49, "d": 50, "f": 51,
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
    


