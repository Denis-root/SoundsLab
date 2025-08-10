import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 44100
TOLERANCE_HZ = 70

START_FREQ = 1200
MARK_FREQ = 1400
SPACE_FREQ = 1000
END_FREQ = 1600

FREQ_TO_BIT = {
    MARK_FREQ: '1',
    SPACE_FREQ: '0',
}

ALL_TONES = [START_FREQ, MARK_FREQ, SPACE_FREQ, END_FREQ]

def detect_frequency(data, sample_rate=SAMPLE_RATE):
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data), d=1/sample_rate)
    magnitude = np.abs(fft_data)
    peak_index = np.argmax(magnitude)
    peak_freq = abs(freqs[peak_index])
    closest_freq = min(ALL_TONES, key=lambda x: abs(x - peak_freq))
    if abs(closest_freq - peak_freq) > TOLERANCE_HZ:
        return None
    return closest_freq

def listen_for_afsk_sequence(max_timeout_after_start=30, blocks_per_bit=10):
    print("🎙️ Escuchando secuencia AFSK...")

    detected_start = False
    bit_buffer = []
    finished = False
    prev_freq = None
    start_detect_time = None

    freq_window = []

    def callback(indata, frames, time_info, status):
        nonlocal detected_start, finished, prev_freq, start_detect_time, freq_window
        if status:
            print("⚠️", status)

        mono = indata[:, 0]
        freq = detect_frequency(mono)
        if freq is None:
            return

        freq_window.append(freq)

        # Solo procesar cuando acumulamos suficiente bloques para 1 bit
        if len(freq_window) >= blocks_per_bit:
            # Calcular la frecuencia predominante en la ventana
            avg_freq = np.median(freq_window)
            freq_window = []

            # Redondear al tono más cercano
            closest_freq = min(ALL_TONES, key=lambda x: abs(x - avg_freq))

            print(f"🔍 Frecuencia ventana detectada: {avg_freq:.1f} Hz, aproximada a {closest_freq} Hz")

            if not detected_start:
                if closest_freq == START_FREQ:
                    print("✅ Tono START detectado.")
                    detected_start = True
                    start_detect_time = time.time()
            else:
                if closest_freq == END_FREQ:
                    print("✅ Tono END detectado. Fin de secuencia.")
                    finished = True
                    raise sd.CallbackStop()
                elif closest_freq in (MARK_FREQ, SPACE_FREQ):
                    bit = FREQ_TO_BIT[closest_freq]
                    print(f"🔢 Bit detectado: {bit}")
                    bit_buffer.append(bit)

            if detected_start and (time.time() - start_detect_time > max_timeout_after_start):
                print(f"⏰ Timeout después del START. Finalizando escucha.")
                finished = True
                raise sd.CallbackStop()

    try:
        with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=2048):
            while not finished:
                sd.sleep(100)
    except sd.CallbackStop:
        pass

    bitstring = ''.join(bit_buffer)
    print(f"📥 Secuencia completa recibida: {bitstring}\n")

    if len(bitstring) % 8 != 0:
        padding = 8 - (len(bitstring) % 8)
        print(f"🛠️ Rellenando {padding} bits para completar byte")
        bitstring += '0' * padding

    chars = [bitstring[i:i+8] for i in range(0, len(bitstring), 8)]
    text = ''
    for c in chars:
        try:
            text += chr(int(c, 2))
        except ValueError:
            text += '?'

    print(f"📝 Texto decodificado: '{text}'")
    return text


if __name__ == "__main__":
    listen_for_afsk_sequence()
