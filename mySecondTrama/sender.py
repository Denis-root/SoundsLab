import numpy as np
import sounddevice as sd

# --- Configuración general ---
SAMPLE_RATE = 44100
BAUD_RATE = 10  # 10 bits por segundo
BIT_DURATION = 1 / BAUD_RATE
AMPLITUDE = 0.8

# --- Frecuencias ---
START_FREQ = 1200
MARK_FREQ = 1400  # Bit 1
SPACE_FREQ = 1000  # Bit 0
END_FREQ = 1600


PRE_TONE_FREQ = 1100

def generate_tone(freq, duration=BIT_DURATION, sample_rate=SAMPLE_RATE, amplitude=AMPLITUDE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)

def text_to_bits(text):
    return ''.join(f"{ord(c):08b}" for c in text)

def send_afsk(text):
    bitstring = text_to_bits(text)
    print(f"🔊 Enviando secuencia: start + {bitstring} + fin")

    audio = []

    # ⚠️ Pre-roll de silencio para evitar fade-in
    # audio.append(np.zeros(int(SAMPLE_RATE * 1), dtype=np.float32))  # 100 ms de silencio
    audio.append(generate_tone(PRE_TONE_FREQ, 1))

    # Tono de inicio
    audio.append(generate_tone(START_FREQ))

    # Bits
    for bit in bitstring:
        freq = MARK_FREQ if bit == '1' else SPACE_FREQ
        audio.append(generate_tone(freq))

    # Tono de fin
    audio.append(generate_tone(END_FREQ))

    final_wave = np.concatenate(audio)
    sd.play(final_wave, SAMPLE_RATE)
    sd.wait()
    print("✅ Secuencia enviada.\n")


if __name__ == "__main__":
    send_afsk("hola")

# 01101000011011110110110001100
# 01101000011011110110110001100001