import time
import numpy as np
import pyaudio
from scipy import signal
import librosa
import wave



class InputHandler:
    def __init__(self, sr=44100, chunk_size=512, baseFrequency=440):
        
        self.COMPARE = False
        self.RECORDE = True
        self.recording = self.RECORDE
        self.compareFile = None
        
        self.start = 0
        self.end = 0
        self.firsttest=True
        
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

        if self.COMPARE:
            self.compareFile = open("compare.txt","a")
        
        if self.RECORDE:
            self.wav_output = wave.open("output_generated.wav", "wb")
            self.wav_output.setnchannels(1)
            self.wav_output.setsampwidth(2)  # Use 2 for int16
            self.wav_output.setframerate(self.sr)



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
        
    def deleteCache(self):
        self.cachedResamples = {}
        print("Deleted Cache")

    def stop_recording(self):
        if self.RECORDE: 
            self.wav_output.close()
            self.recording = False

    def new_recording(self):
            self.wav_output = wave.open("output_generated.wav", "wb")
            self.wav_output.setnchannels(1)
            self.wav_output.setsampwidth(2)  # Use 2 for int16
            self.wav_output.setframerate(self.sr)
            self.recording = True

    def play_wave(self, note, strength, data):
        """Fügt neuen Ton zu den Aktiven noten hinzu. Berechnet dafür die Welleninformationen neu.
        führt fft, phaseVocoder und ifft zusammen"""
        self.start = time.time()
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
            '''time_of_note = len(data)/self.sr * (frequency/self.BASE_FREQUENCY)**-0.5
            time_orig = len(data)/self.sr
            strech_factor = time_of_note/time_orig'''

            data_stretched = np.interp(np.arange(0, len(data), new_sr / self.sr), np.arange(0, len(data)), data)


            d = librosa.stft(data_stretched, n_fft=4096, hop_length=1024)


            # streckt den stft mithilfe phase vocoder in richtige Länge


            dFast = librosa.phase_vocoder(d, rate=1/rate_factor, hop_length=1024)


            # führt einen istft durch, um aus der frequency-domaine zu kommen


            data_stretched = librosa.istft(dFast, hop_length=1024)


            print("Length: " + str(data_stretched.shape) + " Frequency = " + str(frequency))                 

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
        if self.COMPARE:
            if any(output_buffer):  # Only write if there's non-zero data
                for i in output_buffer.tobytes():
                    self.compareFile.write(str(i))

        if self.RECORDE and self.recording:
            int16_buffer = np.int16(output_buffer * 32767)  # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
            if any(int16_buffer):
                self.wav_output.writeframes(int16_buffer.tobytes())
                
        self.end = time.time()
        if self.firsttest and self.start > 0 and any(output_buffer) :
            print("Time till first note: " + str(self.end-self.start))
            self.start=0
        return (output_buffer.tobytes(), pyaudio.paContinue)

    def close(self):
        """Schließt alles ordentlich"""
        if self.COMPARE:
            self.compareFile.close()
        if self.RECORDE: 
            self.wav_output.close()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

