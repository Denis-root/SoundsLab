import numpy as np
import sounddevice as sd

def generate_tone(freq, duration, sample_rate=44100, amplitude=0.6):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def generate_chirp(f_start, f_end, duration, sample_rate=44100, amplitude=0.6):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    freqs = np.linspace(f_start, f_end, t.shape[0])
    return amplitude * np.sin(2 * np.pi * freqs * t)

def generate_noise(duration, sample_rate=44100, amplitude=0.5):
    return amplitude * (2 * np.random.rand(int(sample_rate * duration)) - 1)

def play_modem_handshake():
    sample_rate = 44100
    parts = []

    # Inicio - tono sostenido
    parts.append(generate_tone(2100, 1.5, sample_rate))

    # Respuesta - tono medio
    parts.append(generate_tone(1300, 1.0, sample_rate))

    # Tonos intercalados
    parts.append(generate_tone(1200, 0.3, sample_rate))
    parts.append(generate_tone(1400, 0.3, sample_rate))
    parts.append(generate_tone(1000, 0.2, sample_rate))

    # Chirp ascendente (barrido de frecuencia)
    parts.append(generate_chirp(800, 3000, 1.5, sample_rate))

    # Chirp descendente
    parts.append(generate_chirp(3000, 500, 1.2, sample_rate))

    # Ruido tipo "krshhhh"
    parts.append(generate_noise(1.5, sample_rate))

    # Tonos entrecortados finales
    for _ in range(5):
        parts.append(generate_tone(np.random.choice([1000, 1500, 2000]), 0.1, sample_rate))

    # Mezclar todo
    audio = np.concatenate(parts).astype(np.float32)

    print("🎧 Reproduciendo imitación de sonido de módem...")
    sd.play(audio, sample_rate)
    sd.wait()

play_modem_handshake()
