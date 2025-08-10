import numpy as np
import sounddevice as sd

# === CONFIGURACIÓN GLOBAL ===
SAMPLE_RATE = 44100
TONE_FREQ = 1600       # Frecuencia del tono de inicio de trama
TONE_DURATION = 0.5    # Duración del tono en segundos
TONE_AMPLITUDE = 0.8   # Volumen
TOLERANCE_HZ = 50      # Rango de detección permitido


# === FUNCIÓN: Generar una onda senoidal ===
def generate_tone(freq, duration, sample_rate=SAMPLE_RATE, amplitude=TONE_AMPLITUDE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)


# === MÉTODO: Enviar tono de inicio ===
def send_tone_start():
    print(f"🔊 Enviando tono de inicio de trama ({TONE_FREQ} Hz por {TONE_DURATION}s)...")
    tone = generate_tone(TONE_FREQ, TONE_DURATION)
    sd.play(tone, samplerate=SAMPLE_RATE)
    sd.wait()
    print("✅ Tono enviado.\n")


# === FUNCIÓN: Detectar frecuencia dominante con FFT ===
def detect_frequency(data, sample_rate=SAMPLE_RATE):
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1 / sample_rate)
    magnitude = np.abs(fft_data)
    peak_index = np.argmax(magnitude)
    return abs(freqs[peak_index])


# === MÉTODO: Escuchar tono de inicio desde el micrófono ===
def listen_for_tone_start():
    print(f"🎙️ Escuchando tono de inicio ({TONE_FREQ} Hz ± {TOLERANCE_HZ})...")

    detected = False  # 🔁 Bandera de detección

    def callback(indata, frames, time, status):
        nonlocal detected
        if status:
            print("⚠️", status)

        mono = indata[:, 0]
        freq = detect_frequency(mono)
        print(f"🔍 Frecuencia detectada: {freq:.1f} Hz")

        if abs(freq - TONE_FREQ) < TOLERANCE_HZ:
            print("✅ ¡Tono detectado! Terminando escucha.")
            detected = True
            raise sd.CallbackStop()

    try:
        with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=2048):
            print("🔊 Esperando detección...")
            while not detected:
                sd.sleep(100)
    except sd.CallbackStop:
        pass

    print("🎤 Escucha finalizada.\n")
