import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
TONE_DURATION = 0.3
TONE_AMPLITUDE = 0.8

START_FREQ = 1200
MARK_FREQ = 1400
SPACE_FREQ = 1000
END_FREQ = 1600

def generate_tone(freq, duration=TONE_DURATION, sample_rate=SAMPLE_RATE, amplitude=TONE_AMPLITUDE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)

def send_afsk_sequence(bitstring):
    audio = generate_tone(START_FREQ)
    for bit in bitstring:
        freq = MARK_FREQ if bit == '1' else SPACE_FREQ
        audio = np.concatenate((audio, generate_tone(freq)))
    audio = np.concatenate((audio, generate_tone(END_FREQ, duration=0.6)))  # tono final más largo
    print(f"🔊 Enviando secuencia: start + {bitstring} + fin")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    print("✅ Secuencia enviada.\n")

def text_to_bits(text):
    return ''.join(f"{ord(c):08b}" for c in text)

if __name__ == "__main__":
    # text = input("Ingresa texto para enviar: ")
    # bits = text_to_bits(text)
    bits = text_to_bits("hola")
    send_afsk_sequence(bits)
