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
            
            elif key_value == "p":
                print("Stop Recording")
                self.stop_recording()
            
            elif key_value == "q":
                self.deleteCache()
            
            elif key_value == "r":
                print("New Recording")
                self.new_recording()
            else:
                print("Invalid Key")