"""
No reproduce audio
"""

import sys, time
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

# ---------- Parámetros ----------
SR = 44100          # resample a este SR
N = 1024            # tamaño de ventana (FFT)
HOP = N             # salto entre ventanas (usa N//2 para 50% solape)
Y_MAX_DB = 0        # techo eje Y
Y_SPAN_DB = 120     # rango dinámico mostrado
FILE = sys.argv[1] if len(sys.argv) > 1 else "Afro Medusa - Pasilda (Knee Deep Mix).mp3"

# ---------- Carga y preproceso ----------
# Lee MP3 con pydub (requiere FFmpeg)
seg = AudioSegment.from_file(FILE)
seg = seg.set_frame_rate(SR).set_channels(1)  # forzamos mono y SR
sample_width = seg.sample_width               # bytes por muestra (2=16-bit, etc.)

# A numpy float32 [-1, 1]
samples = np.frombuffer(seg.raw_data, dtype={1: np.int8, 2: np.int16, 4: np.int32}[sample_width]).astype(np.float32)
if sample_width == 1:
    # 8-bit pydub es unsigned; centra a signed
    samples = samples - 128.0
max_int = float(2**(8*sample_width - 1))
x = samples / max_int

# ---------- Ventaneo / FFT ----------
window = np.hanning(N).astype(np.float32)
freqs = np.fft.rfftfreq(N, 1.0/SR)
EPS = 1e-10

# ---------- Plot ----------
plt.ion()
fig, ax = plt.subplots()
(line,) = ax.plot(freqs, np.zeros_like(freqs))
ax.set_xlim(0, SR/2)
ax.set_ylim(Y_MAX_DB - Y_SPAN_DB, Y_MAX_DB)
ax.set_xlabel("Frecuencia (Hz)")
ax.set_ylabel("Magnitud (dB)")
ax.set_title(f"Espectro FFT en tiempo real — {FILE}")

# ---------- Loop “tiempo real” ----------
frame_duration_s = HOP / SR
i = 0
t0 = time.perf_counter()

while i + N <= len(x) and plt.fignum_exists(fig.number):
    frame = x[i:i+N]
    i += HOP

    X = np.fft.rfft(frame * window)
    mag = np.abs(X)
    db = 20*np.log10(mag + EPS)

    line.set_ydata(db)
    # ax.set_ylim(db.max()-Y_SPAN_DB, db.max()+5)  # autoscale opcional

    plt.pause(0.001)

    # Sincroniza al “tiempo real”
    expected_t = (i // HOP) * frame_duration_s
    now = time.perf_counter() - t0
    delay = expected_t - now
    if delay > 0:
        time.sleep(delay)

plt.ioff()
plt.show()
