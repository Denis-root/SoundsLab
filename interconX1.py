import numpy as np
import sounddevice as sd
from icecream import ic

def generate_tone(freq, duration=0.5, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def play_sequence():
    sample_rate = 44100
    tones = [
        (2100, 1.0),  # Carrier tone
        (1300, 0.4),  # Answer tone
        (1500, 0.4),  # Training
        (800, 0.3),  # FSK tone
        (1000, 0.3),  # FSK tone
        (1200, 0.3),
        (1400, 0.3),
        (1000, 0.1),  # rapid sequence
        (1200, 0.1),
        (1000, 0.1),
        (1200, 0.1),
    ]

    audio = np.concatenate([generate_tone(freq, dur, sample_rate) for freq, dur in tones])

    sd.play(audio, sample_rate)
    sd.wait()


play_sequence()
