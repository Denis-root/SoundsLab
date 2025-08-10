# fft_mp3_linea_bandas.py
import os
os.environ['MPLBACKEND'] = 'TkAgg'
import sys, queue
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from pydub import AudioSegment

# ------------------ Parámetros ------------------
SR = 44100
N = 4096
HOP = 512
BLOCKSIZE = 2048
Y_SPAN_DB = 120
DEVICE_OUT = None
FILE = sys.argv[1] if len(sys.argv) > 1 else "PXRKX, Nitro Station - 90s Sunshine.mp3"

# Definición de bandas (ajusta a gusto)
# Nota: el límite superior real es SR/2 (Nyquist)
BANDS = [
    ("Bajos", 20, 250),
    ("Medios", 250, 4000),
    ("Altos", 4000, 20000),
]

# ------------------ Cargar MP3 ------------------
seg = AudioSegment.from_file(FILE).set_frame_rate(SR).set_channels(1)
sw = seg.sample_width
dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
arr = np.frombuffer(seg.raw_data, dtype=dtype_map[sw]).astype(np.float32)
if sw == 1:
    arr = arr - 128.0
max_int = float(2**(8*sw - 1))
audio = (arr / max_int).astype(np.float32)

# ------------------ FFT & buffers ------------------
window = np.hanning(N).astype(np.float32)
freqs = np.fft.rfftfreq(N, 1.0/SR)
EPS = 1e-12

# Precalcular índices de bins por banda (para no hacerlo en cada frame)
band_bins = []
nyq = SR / 2.0
for name, f_lo, f_hi in BANDS:
    lo = max(0.0, f_lo)
    hi = min(f_hi, nyq)
    idx = np.where((freqs >= lo) & (freqs < hi))[0]
    band_bins.append(idx)

read_idx = 0
ring = np.zeros(N, dtype=np.float32)
ring_fill = 0
fft_q = queue.Queue(maxsize=8)

# ------------------ Callback de audio ------------------
def callback(outdata, frames, time_info, status):
    global read_idx, ring_fill, ring
    end = read_idx + frames
    chunk = audio[read_idx:end]
    read_idx = end

    if len(chunk) < frames:
        out = np.zeros(frames, dtype=np.float32)
        out[:len(chunk)] = chunk
        outdata[:] = out.reshape(-1, 1)
        raise sd.CallbackStop()
    else:
        outdata[:] = chunk.reshape(-1, 1)

    ptr = 0
    while ptr < frames:
        take = min(frames - ptr, N - ring_fill)
        ring[ring_fill:ring_fill+take] = chunk[ptr:ptr+take]
        ring_fill += take
        ptr += take
        if ring_fill == N:
            frame = ring.copy()
            if HOP < N:
                ring[:N-HOP] = ring[HOP:]
                ring_fill = N - HOP
            else:
                ring_fill = 0
            if not fft_q.full():
                fft_q.put_nowait(frame)

# ------------------ Plot: espectro (línea) ------------------
plt.ion()
fig1, ax1 = plt.subplots()
(line,) = ax1.plot(freqs, np.zeros_like(freqs))
ax1.set_xlim(0, SR/2)
ax1.set_ylim(-Y_SPAN_DB, 0)
ax1.set_xlabel("Hz"); ax1.set_ylabel("dB")
ax1.set_title(f"Espectro (línea) — {FILE}")

# ------------------ Plot: bandas (barras) ------------------
fig2, ax2 = plt.subplots()
band_names = [b[0] for b in BANDS]
band_heights = np.zeros(len(BANDS))
bars = ax2.bar(band_names, band_heights)
ax2.set_ylim(-60, 0)  # rango dB para las bandas (ajustable)
ax2.set_ylabel("dB (energía por banda)")
ax2.set_title("Bajos / Medios / Altos")

# ------------------ Loop principal + stream ------------------
with sd.OutputStream(samplerate=SR,
                     channels=1,
                     dtype='float32',
                     blocksize=BLOCKSIZE,
                     device=DEVICE_OUT,
                     callback=callback):
    while plt.fignum_exists(fig1.number) and plt.fignum_exists(fig2.number):
        try:
            frame = fft_q.get(timeout=0.05)
        except queue.Empty:
            plt.pause(0.001)
            continue

        # FFT del frame
        X = np.fft.rfft(frame * window)
        mag = np.abs(X)

        # Espectro en dB (línea)
        db = 20*np.log10(mag + EPS)
        line.set_ydata(db)
        plt.pause(0.001)

        # Energía por banda:
        # Usamos potencia promedio en cada banda: 10*log10(mean(mag^2))
        band_db = []
        power_ref = EPS  # referencia para evitar -inf (relativo)
        for idx in band_bins:
            if len(idx) == 0:
                band_db.append(-60.0)
                continue
            band_power = np.mean((mag[idx]**2)) + power_ref
            band_db.append(10*np.log10(band_power))

        # Actualizar barras
        for bar, h in zip(bars, band_db):
            bar.set_height(h)

        # Pequeño respiro para el GUI
        plt.pause(0.001)

plt.ioff()
plt.show()
