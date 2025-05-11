from InputHandler import InputHandler
import threading


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