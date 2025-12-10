import numpy as np
import sounddevice as sd
import time


class SoundGenerator:
    
    def __init__(self, sample_rate=44100, tempo=120, instrument=str):
        self.sample_rate = sample_rate
        self.tempo = tempo
        self.sound_type = instrument
        
        # Note frequencies in Hz
        self.notes = {
            'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
            'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
            'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
        }
    
    def generate_sine_wave(self, freq, duration):
        """Generate a basic sine wave"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        wave = np.sin(2 * np.pi * freq * t)
        return wave
    
    def generate_piano(self, freq, duration):
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        signal = np.zeros(len(t))
        
        for n in range(1, 9):
            amplitude = 1.0 / n
            signal += amplitude * np.sin(2 * np.pi * freq * n * t)
        
        # Fade out
        fade_len = int(len(signal) * 0.3)
        signal[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        return signal * 0.2
    
    def generate_guitar(self, freq, duration):
        delay = int(self.sample_rate / freq)
        noise = np.random.uniform(-1, 1, delay)
        
        output = np.zeros(int(self.sample_rate * duration))
        buffer = noise.copy()
        
        for i in range(len(output)):
            output[i] = buffer[0]
            avg = 0.5 * (buffer[0] + buffer[1])
            buffer = np.append(buffer[1:], avg * 0.996)
        
        # Fade out
        fade_len = int(len(output) * 0.3)
        output[-fade_len:] *= np.linspace(1, 0, fade_len)
        
        return output * 0.3
    
    def play(self, note, duration):
        """Play a note"""
        if note not in self.notes:
            return
        
        freq = self.notes[note]
        duration_sec = (60.0 / self.tempo) * duration
        
        if self.sound_type == 'piano':
            audio = self.generate_piano(freq, duration_sec)
        elif self.sound_type == 'guitar':
            audio = self.generate_guitar(freq, duration_sec)
        else:
            audio = self.generate_sine_wave(freq, duration_sec)
        
        sd.play(audio.astype(np.float32), self.sample_rate)
        sd.wait()
    
    def rest(self, duration):
        """Silence"""
        duration_sec = (60.0 / self.tempo) * duration
        time.sleep(duration_sec)


def main(instrument=SoundGenerator, notes_list=list[str]):
    "Takes in a list of notes and plays them sequentially as half notes"
    for note in notes_list:
        if note == 'rest':
            instrument.rest(0.5)
        instrument.play(note, 0.5)

# Example
if __name__ == "__main__":
    
    gen = SoundGenerator(tempo=100,instrument='guitar')
    
    #play scale
    notes_list = ['C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3', 'rest', 'C4', 'E4', 'G4', 'rest', 'C5']
#    main(instrument=gen, notes_list=notes_list)

    #play happy birthday
    notes_list = ['C4', 'C4', 'D4', 'C4', 'F4', 'E4', 'rest',
                  'C4', 'C4', 'D4', 'C4', 'G4', 'F4', 'rest',
                  'C4', 'C4', 'C5', 'A4', 'F4', 'E4', 'D4', 'rest',
                  'B4', 'B4', 'A4', 'F4', 'G4', 'F4']
    main(instrument=gen, notes_list=notes_list)