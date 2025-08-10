# fft_mp3_linea_iso_bandas.py
import os
os.environ['MPLBACKEND'] = 'TkAgg'
import sys, queue
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from pydub import AudioSegment

# ------------------ Parámetros ------------------
SR = 44100
N = 2048
HOP = 512
BLOCKSIZE = 2048
Y_SPAN_DB = 120
DEVICE_OUT = None
FILE = sys.argv[1] if len(sys.argv) > 1 else "PXRKX, Nitro Station - 90s Sunshine.mp3"

# Bandas ISO (logarítmicas)
BPO = 3          # bands per octave: 1 (octava) o 3 (tercios)
FMIN = 20.0      # Hz
FMAX = 20000.0   # Hz
SMOOTH_ALPHA = 0.25  # 0..1 (suavizado EMA para las bandas)

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

read_idx = 0
ring = np.zeros(N, dtype=np.float32)
ring_fill = 0
fft_q = queue.Queue(maxsize=8)

# ------------------ Utilidades: bandas ISO ------------------
def make_iso_bands(freqs, bpo=3, fmin=20.0, fmax=None):
    """Genera centros y bordes de bandas tipo ISO (octava/tercios) y mapea bins FFT a cada banda."""
    if fmax is None:
        fmax = freqs[-1]
    centers = []
    k = -200  # arranque bajo; subimos hasta pasar fmax
    while True:
        fc = 1000.0 * (2.0 ** (k / bpo))  # familia ISO basada en 1 kHz
        if fc > fmax * (2 ** (1/(2*bpo))):
            break
        if fc >= fmin / (2 ** (1/(2*bpo))):
            centers.append(fc)
        k += 1
    centers = np.array(centers, dtype=float)

    kfac = 2.0 ** (1.0 / (2.0 * bpo))  # factor medio-banda
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

def band_levels_db_from_mag(mag, band_bins, eps=1e-12, smooth_prev=None, alpha=0.25):
    """Convierte |FFT| -> dB por banda (promedio de potencia). Opcional: suavizado EMA."""
    out = []
    for idx in band_bins:
        if len(idx) == 0:
            out.append(-120.0)
        else:
            power = np.mean((mag[idx] ** 2)) + eps
            out.append(10.0 * np.log10(power))
    out = np.asarray(out, dtype=float)
    if smooth_prev is not None:
        out = alpha * out + (1.0 - alpha) * smooth_prev
    return out

# Precalcular bandas ISO según tu FFT y SR
centers_hz, lo, hi, band_bins = make_iso_bands(
    freqs, bpo=BPO, fmin=FMIN, fmax=min(FMAX, freqs[-1])
)
band_labels = [f"{int(round(c))} Hz" if c < 1000 else f"{c/1000:.1f} kHz" for c in centers_hz]
band_db_prev = None  # para suavizado

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

# ------------------ Plot: bandas ISO (barras) ------------------
fig2, ax2 = plt.subplots()
xpos = np.arange(len(centers_hz))
bars = ax2.bar(xpos, np.zeros_like(xpos, dtype=float))
ax2.set_ylim(-60, 0)  # ajusta a gusto
ax2.set_xticks(xpos)
ax2.set_xticklabels(band_labels, rotation=45, ha="right")
ax2.set_ylabel("dB (promedio de potencia)")
ax2.set_title(f"Bandas ISO ({'octava' if BPO==1 else '1/3 octava'})")

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

        X = np.fft.rfft(frame * window)
        mag = np.abs(X)

        # Espectro en dB (línea)
        db = 20*np.log10(mag + EPS)
        line.set_ydata(db)

        # Niveles por banda ISO
        band_db = band_levels_db_from_mag(
            mag, band_bins,
            eps=EPS,
            smooth_prev=band_db_prev,
            alpha=SMOOTH_ALPHA
        )
        band_db_prev = band_db

        # Actualiza barras
        for b, h in zip(bars, band_db):
            b.set_height(float(h))

        plt.pause(0.001)

plt.ioff()
plt.show()
