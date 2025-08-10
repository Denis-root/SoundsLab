import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from queue import Queue

print(sd.query_devices())

# ----- Parámetros -----
SR = 44100          # sample rate
N = 1024            # tamaño de ventana/FFT (potencia de 2)
HOP = N             # salto (usa N para “ventanas” sin solape; puedes bajar a N//2)
DEVICE = 1       # None = dispositivo por defecto; o pon el índice/nombre

# ----- Estado -----
q = Queue(maxsize=8)
window = np.hanning(N)
freqs = np.fft.rfftfreq(N, 1/SR)

# Evita log(0)
EPS = 1e-10

def audio_callback(indata, frames, time, status):
    if status:
        # print(status)   # si quieres ver underruns/overruns
        pass
    # indata llega como float32 [-1,1], shape: (frames, channels)
    mono = indata.mean(axis=1)  # mezcla a mono
    # Bufferiza en bloques exactos de N (por si frames != N)
    # Aquí partimos el chunk entrante en piezas de N y las enviamos a la cola.
    start = 0
    while start + N <= len(mono):
        chunk = mono[start:start+N]
        start += HOP
        if not q.full():
            q.put_nowait(chunk.copy())

# ---- Inicia stream ----
stream = sd.InputStream(
    samplerate=SR,
    blocksize=HOP,
    channels=1,          # si tu micro es mono, pon 1
    callback=audio_callback,
    device=DEVICE
)

# ---- Plot inicial ----
plt.ion()
fig, ax = plt.subplots()
(line,) = ax.plot(freqs, np.zeros_like(freqs))
ax.set_xlim(0, SR/2)
ax.set_ylim(-120, 0)            # dBFS aprox
ax.set_xlabel("Frecuencia (Hz)")
ax.set_ylabel("Magnitud (dB)")
ax.set_title("Espectro (FFT) en tiempo real")

with stream:
    while plt.fignum_exists(fig.number):
        try:
            x = q.get(timeout=0.1)  # siguiente bloque
        except:
            plt.pause(0.01)
            continue

        xw = x * window
        X = np.fft.rfft(xw)
        mag = np.abs(X)
        db = 20*np.log10(mag + EPS)

        line.set_ydata(db)
        # (Opcional) auto-escalar lento:
        # ax.set_ylim(max(db)-100, max(db)+5)

        plt.pause(0.001)  # refresco
