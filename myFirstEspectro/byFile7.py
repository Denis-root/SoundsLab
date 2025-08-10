# fft_mp3_linea_iso_bandas_norm.py
import os
os.environ['MPLBACKEND'] = 'TkAgg'
import sys, queue
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from pydub import AudioSegment
import matplotlib; print("Backend:", matplotlib.get_backend())


# ------------------ Parámetros ------------------
SR = 44100
N = 8192                 # sube a 4096 si querés más resolución en frecuencia
HOP = 512
BLOCKSIZE = 4096
Y_SPAN_DB = 120
DEVICE_OUT = None
FILE = sys.argv[1] if len(sys.argv) > 1 else "PXRKX, Nitro Station - 90s Sunshine.mp3"

# Bandas ISO (logarítmicas)
BPO = 3                  # 1 = octavas, 3 = tercios
FMIN = 40
FMAX = 20000.0
SMOOTH_ALPHA = 0.25      # suavizado EMA (0..1)
MIN_BINS_PER_BAND = 2    # ignora bandas con muy pocos bins

# Normalización (parches):
NORM_MODE = "psd"        # "psd" = calibrada (recomendada), "simple" = magnitud clásica
RELATIVE_PER_FRAME = False  # True = normaliza por frame (máximo de bandas a 0 dB)

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
df = SR / N
EPS = 1e-12

read_idx = 0
ring = np.zeros(N, dtype=np.float32)
ring_fill = 0
fft_q = queue.Queue(maxsize=8)

# ------------------ Utilidades: bandas ISO ------------------
def make_iso_bands(freqs, bpo=3, fmin=20.0, fmax=None):
    if fmax is None:
        fmax = freqs[-1]
    centers = []
    k = -200
    while True:
        fc = 1000.0 * (2.0 ** (k / bpo))
        if fc > fmax * (2 ** (1/(2*bpo))):
            break
        if fc >= fmin / (2 ** (1/(2*bpo))):
            centers.append(fc)
        k += 1
    centers = np.array(centers, dtype=float)
    kfac = 2.0 ** (1.0 / (2.0 * bpo))
    edges_lo = centers / kfac
    edges_hi = centers * kfac
    valid = (edges_hi >= fmin) & (edges_lo <= fmax)
    centers = centers[valid]
    edges_lo = edges_lo[valid]
    edges_hi = edges_hi[valid]
    band_bins = []
    for lo, hi in zip(edges_lo, edges_hi):
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        band_bins.append(idx)
    return centers, edges_lo, edges_hi, band_bins

centers_hz, lo, hi, band_bins = make_iso_bands(
    freqs, bpo=BPO, fmin=FMIN, fmax=min(FMAX, freqs[-1])
)
band_labels = [f"{int(round(c))} Hz" if c < 1000 else f"{c/1000:.1f} kHz" for c in centers_hz]
band_db_prev = None

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

# ------------------ Plot: solo bandas ISO (barras) ------------------
plt.ion()
fig2, ax2 = plt.subplots()
xpos = np.arange(len(centers_hz))

BANDS_YMIN = -30 if RELATIVE_PER_FRAME else -100   # piso del eje Y
ax2.set_ylim(BANDS_YMIN, 0)

bars = ax2.bar(xpos, np.zeros_like(xpos, dtype=float), bottom=BANDS_YMIN, align="center")
ax2.set_xticks(xpos)
ax2.set_xticklabels(band_labels, rotation=45, ha="right")
ax2.set_ylabel("dB por banda (promedio de potencia)")
ax2.set_title(f"Bandas ISO ({'octava' if BPO==1 else '1/3 octava'})")

# ------------------ Loop principal + stream ------------------
with sd.OutputStream(samplerate=SR,
                     channels=1,
                     dtype='float32',
                     blocksize=BLOCKSIZE,
                     device=DEVICE_OUT,
                     callback=callback):
    while plt.fignum_exists(fig2.number):
        try:
            frame = fft_q.get(timeout=0.05)
        except queue.Empty:
            plt.pause(0.001)
            continue

        # ----- FFT (ventaneada) -----
        Xw = np.fft.rfft(frame * window)

        # ----- Bandas ISO -----
        if NORM_MODE == "psd":
            U = (window**2).mean()
            ps = (np.abs(Xw)**2) / (N**2 * U)
            ps[1:-1] *= 2.0
            band_vals_db = []
            for idx in band_bins:
                if len(idx) < MIN_BINS_PER_BAND:
                    band_vals_db.append(-120.0)
                else:
                    band_power = (np.sum(ps[idx]) * df)  # integra potencia por banda
                    band_vals_db.append(10*np.log10(band_power + EPS))
        else:
            mag = np.abs(Xw)
            band_vals_db = []
            for idx in band_bins:
                if len(idx) < MIN_BINS_PER_BAND:
                    band_vals_db.append(-120.0)
                else:
                    band_power = np.mean((mag[idx]**2)) + EPS
                    band_vals_db.append(10*np.log10(band_power))

        # Suavizado EMA
        if band_db_prev is not None:
            band_vals_db = SMOOTH_ALPHA*np.array(band_vals_db) + (1-SMOOTH_ALPHA)*band_db_prev
        band_db_prev = np.array(band_vals_db)

        # Normalización relativa por frame (opcional)
        if RELATIVE_PER_FRAME:
            band_vals_db = band_vals_db - np.max(band_vals_db)

        # Alturas positivas respecto a BANDS_YMIN
        heights = np.maximum(0.0, np.array(band_vals_db) - BANDS_YMIN)

        # Actualiza barras
        for b, h in zip(bars, heights):
            b.set_height(float(h))

        plt.pause(0.001)

plt.ioff()
plt.show()
