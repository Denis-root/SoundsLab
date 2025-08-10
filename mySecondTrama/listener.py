import numpy as np
import sounddevice as sd

# --- Configuración general ---
SAMPLE_RATE = 44100
BAUD_RATE = 10  # 10 bits por segundo => cada bit dura 0.1s
SAMPLES_PER_BIT = int(SAMPLE_RATE / BAUD_RATE)
TONE_DURATION = 1 / BAUD_RATE
TOLERANCE_HZ = 50  # tolerancia para reconocer el tono

# --- Frecuencias ---
START_FREQ = 1200
MARK_FREQ = 1400  # Bit 1
SPACE_FREQ = 1000  # Bit 0
END_FREQ = 1600

FREQ_TO_BIT = {
    MARK_FREQ: '1',
    SPACE_FREQ: '0'
}

ALL_TONES = [START_FREQ, MARK_FREQ, SPACE_FREQ, END_FREQ]

# --- Función para detectar frecuencia dominante ---
def detect_frequency(samples, sample_rate=SAMPLE_RATE):
    windowed = samples * np.hamming(len(samples))
    fft_data = np.fft.fft(windowed)
    magnitude = np.abs(fft_data[:len(fft_data)//2])
    freqs = np.fft.fftfreq(len(samples), d=1/sample_rate)[:len(fft_data)//2]
    peak_index = np.argmax(magnitude)
    peak_freq = freqs[peak_index]
    return abs(peak_freq)

# --- Buscar la frecuencia más cercana ---
def match_frequency(freq):
    closest = min(ALL_TONES, key=lambda f: abs(f - freq))
    if abs(closest - freq) <= TOLERANCE_HZ:
        return closest
    return None

# --- Decodificador principal ---
def listen_afsk():
    print("🎙️ Escuchando secuencia AFSK...")

    duration = 60  # duración máxima de escucha en segundos
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()

    samples = recording[:, 0]
    num_bits = len(samples) // SAMPLES_PER_BIT

    bit_buffer = []
    detected_start = False

    for i in range(num_bits):
        start_idx = i * SAMPLES_PER_BIT
        end_idx = start_idx + SAMPLES_PER_BIT
        chunk = samples[start_idx:end_idx]

        if len(chunk) < SAMPLES_PER_BIT:
            break

        freq = detect_frequency(chunk)
        matched = match_frequency(freq)

        if not detected_start:
            if matched == START_FREQ:
                print("✅ Tono START detectado.")
                detected_start = True
        else:
            if matched == END_FREQ:
                print("✅ Tono END detectado. Fin de secuencia.")
                break
            elif matched in FREQ_TO_BIT:
                bit = FREQ_TO_BIT[matched]
                bit_buffer.append(bit)
                print(f"🔢 Bit detectado: {bit}")
            else:
                print(f"⚠️ Frecuencia no válida: {freq:.1f} Hz")

    bitstring = ''.join(bit_buffer)
    print(f"\n📥 Secuencia completa recibida: {bitstring}")

    if len(bitstring) % 8 != 0:
        padding = 8 - (len(bitstring) % 8)
        print(f"🛠️ Rellenando {padding} bits para completar byte")
        bitstring += '0' * padding

    chars = [bitstring[i:i+8] for i in range(0, len(bitstring), 8)]
    text = ''
    for byte in chars:
        try:
            text += chr(int(byte, 2))
        except ValueError:
            text += '?'

    print(f"📝 Texto decodificado: '{text}'")
    return text

if __name__ == "__main__":
    listen_afsk()
