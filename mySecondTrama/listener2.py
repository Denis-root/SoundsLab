import numpy as np
import sounddevice as sd
from scipy.fftpack import fft

SAMPLE_RATE = 44100
BAUD_RATE = 10
BIT_DURATION = 1 / BAUD_RATE  # 0.1 segundos por bit
BUFFER_SIZE = int(SAMPLE_RATE * BIT_DURATION)  # muestras por bit

START_FREQ = 1200
MARK_FREQ = 1400
SPACE_FREQ = 1000
END_FREQ = 1600

FREQ_TOLERANCE = 50  # Hz
AMPLITUDE_THRESHOLD = 0.1  # Ajustar según mic

TONE_FREQS = {
    MARK_FREQ: '1',
    SPACE_FREQ: '0'
}

def detect_frequency(chunk, sample_rate=SAMPLE_RATE):
    window = np.hanning(len(chunk))
    spectrum = np.abs(fft(chunk * window))
    freq_bin = np.argmax(spectrum[:len(spectrum)//2])
    frequency = freq_bin * sample_rate / len(chunk)
    amplitude = np.max(spectrum)
    return frequency, amplitude / len(chunk)

def match_frequency(freq):
    for target in [START_FREQ, MARK_FREQ, SPACE_FREQ, END_FREQ]:
        if abs(freq - target) <= FREQ_TOLERANCE:
            return target
    return None

def listen_afsk():
    in_transmission = False
    bitstream = ""

    def callback(indata, frames, time, status):
        nonlocal in_transmission, bitstream

        if status:
            print("⚠️", status)

        mono = indata[:, 0]
        freq, amp = detect_frequency(mono)

        if amp < AMPLITUDE_THRESHOLD:
            #print("Silencio o ruido bajo")
            return

        matched = match_frequency(freq)

        print(f"Freq detectada: {freq:.1f} Hz, Amplitud: {amp:.3f}, Matched: {matched}")

        if matched is None:
            return

        if not in_transmission:
            if matched == START_FREQ:
                print("📶 Inicio de transmisión detectado.")
                in_transmission = True
                bitstream = ""
        else:
            if matched == END_FREQ:
                print("🏁 Fin de transmisión detectado.")
                print("🧩 Bits recibidos:", bitstream)
                # Decodificar
                chars = [chr(int(bitstream[i:i+8], 2)) for i in range(0, len(bitstream), 8)]
                print("📥 Texto decodificado:", ''.join(chars))
                in_transmission = False
            elif matched in TONE_FREQS:
                bitstream += TONE_FREQS[matched]
                print(f"🔢 Bit agregado: {TONE_FREQS[matched]}, Bitstream: {bitstream}")

    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE):
        input("⏳ Presiona Enter para detener la escucha...\n")

if __name__ == "__main__":
    listen_afsk()
