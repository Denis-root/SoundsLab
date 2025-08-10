# fft_mp3_play_callback.py
import sys            # Para leer argumentos de la línea de comandos
import queue          # Para crear una cola que comparta datos entre el callback y el hilo principal
import numpy as np    # Librería para cálculos numéricos, FFT, etc.
import matplotlib.pyplot as plt  # Para graficar el espectro
import sounddevice as sd         # Para reproducir audio y manejar callbacks en tiempo real
from pydub import AudioSegment   # Para cargar archivos MP3 y convertirlos a un formato utilizable

# -------- Parámetros --------
SR = 44100                     # Frecuencia de muestreo objetivo (Hz)
N = 2048                       # Tamaño de la ventana para FFT (cuántas muestras analizamos por bloque)
HOP = 512                       # Salto entre ventanas para FFT (solape: N - HOP muestras se repiten)
BLOCKSIZE = 1024               # Tamaño de bloque que se envía al dispositivo de audio en cada callback
Y_SPAN_DB = 120                 # Rango dinámico mostrado en dB en el eje Y
DEVICE_OUT = None              # None = salida de audio por defecto (puedes poner el índice de sd.query_devices())
FILE = sys.argv[1] if len(sys.argv) > 1 else "Afro Medusa - Pasilda (Knee Deep Mix).mp3"  # Ruta del archivo MP3, tomada de argumentos o valor fijo

# -------- Cargar MP3 a mono float32 --------
seg = AudioSegment.from_file(FILE)     # Carga el archivo MP3 usando pydub
seg = seg.set_frame_rate(SR)           # Fuerza la frecuencia de muestreo a SR
seg = seg.set_channels(1)              # Convierte a mono
sw = seg.sample_width                  # Obtiene el número de bytes por muestra (1, 2 o 4)
dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}  # Mapeo de bytes por muestra a tipo NumPy
arr = np.frombuffer(seg.raw_data, dtype=dtype_map[sw]).astype(np.float32)  # Convierte el audio a un array NumPy
if sw == 1:                             # Si el audio es de 8 bits (unsigned)
    arr = arr - 128.0                   # Lo centramos a valores con signo (-128 a 127)
max_int = float(2**(8*sw - 1))          # Valor máximo posible según la resolución (para normalizar)
audio = (arr / max_int).astype(np.float32)  # Normaliza el audio a rango [-1, 1] como float32

# -------- Buffers para callback/FFT --------
read_idx = 0                            # Índice de lectura actual en el array de audio
ring = np.zeros(N, dtype=np.float32)    # Buffer circular (ring buffer) para acumular muestras hasta llenar N
ring_fill = 0                           # Cuántas posiciones del ring buffer están ocupadas

fft_q = queue.Queue(maxsize=4)                 # Cola para pasar ventanas de audio completas al hilo principal
window = np.hanning(N).astype(np.float32)      # Ventana de Hann para suavizar los extremos antes de FFT
freqs = np.fft.rfftfreq(N, 1.0/SR)              # Vector de frecuencias correspondientes a cada bin de la FFT
EPS = 1e-10                                     # Pequeño valor para evitar log(0) al calcular dB

# -------- Callback de audio --------
def callback(outdata, frames, time_info, status):
    global read_idx, ring_fill, ring            # Variables globales que vamos a modificar
    end = read_idx + frames                     # Calcula el índice final de lectura para este bloque
    chunk = audio[read_idx:end]                 # Extrae el bloque de audio correspondiente
    read_idx = end                              # Avanza el índice de lectura
    if len(chunk) < frames:                     # Si llegamos al final del audio y el bloque es incompleto
        out = np.zeros(frames, dtype=np.float32)  # Crea un bloque lleno de ceros
        out[:len(chunk)] = chunk                 # Copia las muestras que quedan
        outdata[:] = out.reshape(-1, 1)          # Lo escribe al buffer de salida (1 canal)
        raise sd.CallbackStop()                  # Detiene el stream al acabar el audio
    else:
        outdata[:] = chunk.reshape(-1, 1)        # Copia el bloque completo al buffer de salida

    ptr = 0                                      # Puntero local para recorrer el bloque
    while ptr < frames:                          # Mientras queden muestras en este bloque
        take = min(frames - ptr, N - ring_fill)  # Cuántas muestras podemos meter en el ring sin pasarnos
        ring[ring_fill:ring_fill+take] = chunk[ptr:ptr+take]  # Copiamos al ring buffer
        ring_fill += take                        # Avanzamos el contador de llenado
        ptr += take                              # Avanzamos en el bloque actual
        if ring_fill == N:                       # Si el ring buffer está lleno (tenemos N muestras)
            frame = ring.copy()                  # Hacemos una copia para procesar FFT
            if HOP < N:                          # Si usamos solape
                ring[:N-HOP] = ring[HOP:]        # Desplazamos el buffer para dejar las últimas N-HOP muestras
                ring_fill = N - HOP              # Actualizamos el llenado del buffer
            else:
                ring_fill = 0                    # Sin solape: vaciamos el buffer
            if not fft_q.full():                 # Si la cola no está llena
                fft_q.put_nowait(frame)          # Enviamos la ventana al hilo principal para graficar

# -------- Configuración inicial del plot --------
plt.ion()                                        # Modo interactivo para refrescar en tiempo real
fig, ax = plt.subplots()                         # Crea la figura y el eje
(line,) = ax.plot(freqs, np.zeros_like(freqs))   # Línea inicial con ceros
ax.set_xlim(0, SR/2)                             # Eje X: 0 a SR/2 (frecuencia de Nyquist)
ax.set_ylim(-Y_SPAN_DB, 0)                       # Eje Y: de -Y_SPAN_DB a 0 dB
ax.set_xlabel("Hz")                              # Etiqueta eje X
ax.set_ylabel("dB")                              # Etiqueta eje Y
ax.set_title(f"FFT en tiempo real — {FILE}")     # Título de la gráfica

# -------- Stream de salida con callback --------
with sd.OutputStream(samplerate=SR,              # Frecuencia de muestreo
                     channels=1,                 # Un canal (mono)
                     dtype='float32',            # Tipo de dato
                     blocksize=BLOCKSIZE,        # Tamaño de bloque de audio
                     device=DEVICE_OUT,          # Dispositivo de salida
                     callback=callback):         # Callback que alimenta el audio
    while plt.fignum_exists(fig.number):         # Mientras la ventana de la gráfica exista
        try:
            frame = fft_q.get(timeout=0.05)      # Obtiene una ventana desde la cola (espera máx 0.05s)
        except queue.Empty:                      # Si no hay datos en ese tiempo
            plt.pause(0.001)                     # Espera un poco y vuelve a intentar
            continue
        X = np.fft.rfft(frame * window)          # Calcula la FFT de la ventana (con ventana de Hann)
        db = 20*np.log10(np.abs(X) + EPS)         # Convierte la magnitud a decibelios
        line.set_ydata(db)                       # Actualiza la línea del gráfico con los nuevos valores
        plt.pause(0.001)                         # Refresca el gráfico sin bloquear

plt.ioff()                                       # Desactiva el modo interactivo
plt.show()                                       # Muestra la gráfica final y espera cierre
