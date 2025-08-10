# fft_mp3_barras.py
import sys                          # Para leer argumentos de la línea de comandos
import queue                        # Para la cola entre callback (audio) y UI (gráfico)
import numpy as np                  # Cálculos numéricos (FFT, arrays, etc.)
import matplotlib.pyplot as plt     # Graficación
import sounddevice as sd            # Salida de audio + callback en tiempo real
from pydub import AudioSegment      # Carga y conversión de MP3 (requiere FFmpeg en PATH)

# -------- Parámetros generales --------
SR = 44100                          # Frecuencia de muestreo objetivo (Hz)
N = 2048                            # Tamaño de ventana para la FFT (muestras por análisis)
HOP = 512                           # Paso entre ventanas (solape = N - HOP)
BLOCKSIZE = 2048                    # Tamaño del bloque que consume el dispositivo por callback (sube si “patea”)
Y_SPAN_DB = 120                     # Rango dinámico del eje Y en dB (mostramos [-Y_SPAN_DB, 0])
DEVICE_OUT = None                   # Dispositivo de salida (None = predeterminado; usa sd.query_devices() si querés fijar uno)

# -------- Parámetros de visualización en barras --------
GROUP_SIZE = 4                      # Cuántos bins de FFT se agrupan en UNA barra (4=menos barras, más “gordas”)
BAR_GAP = 0.9                       # Factor de ancho de barra respecto al espacio entre bins agrupados (0..1)

# -------- Archivo a reproducir --------
FILE = sys.argv[1] if len(sys.argv) > 1 else "PXRKX, Nitro Station - 90s Sunshine.mp3"  # Obtiene la ruta del MP3 desde argumentos o usa uno por defecto

# -------- Cargar y normalizar el MP3 --------
seg = AudioSegment.from_file(FILE)         # Carga el archivo con pydub (FFmpeg)
seg = seg.set_frame_rate(SR)               # Asegura frecuencia SR
seg = seg.set_channels(1)                  # Convierte a mono
sw = seg.sample_width                      # Bytes por muestra (1, 2 o 4)
dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}  # Mapeo de tamaño de muestra a tipo NumPy
arr = np.frombuffer(seg.raw_data, dtype=dtype_map[sw]).astype(np.float32)  # Crea array NumPy de las muestras crudas
if sw == 1:                                 # Si el formato era 8-bit (unsigned)
    arr = arr - 128.0                       # Centra a signed
max_int = float(2**(8*sw - 1))              # Valor máximo según resolución (para normalizar a [-1,1])
audio = (arr / max_int).astype(np.float32)  # Normaliza a float32 en [-1, 1]

# -------- Preparación FFT --------
window = np.hanning(N).astype(np.float32)   # Ventana de Hann para reducir leakage
freqs = np.fft.rfftfreq(N, 1.0 / SR)        # Frecuencias (Hz) de cada bin de rFFT (0..SR/2)
EPS = 1e-10                                  # Evita log(0) en dB
read_idx = 0                                 # Índice de lectura global del buffer de audio

# -------- Ring buffer para construir ventanas de N muestras con solape --------
ring = np.zeros(N, dtype=np.float32)        # Buffer circular tamaño N
ring_fill = 0                                # Cuántas muestras del ring están actualmente llenas

# -------- Cola para pasar ventanas del callback al hilo de UI --------
fft_q = queue.Queue(maxsize=8)              # Cola no bloqueante para ventanas listas para FFT

# -------- Función para agrupar bins de FFT y reducir número de barras --------
def group_bins(values, g):
    """
    Agrupa el vector 'values' de a 'g' en promedio por grupo.
    Maneja el caso en el que la longitud no sea múltiplo de g.
    """
    L = len(values)                                              # Longitud total
    if g <= 1:                                                   # Si g==1, no agrupa
        return values
    k = L // g                                                   # Cuántos grupos completos caben
    rem = L % g                                                  # Resto (bins sobrantes)
    if k > 0:
        base = values[:k*g].reshape(k, g).mean(axis=1)           # Promedio por grupo en bloques completos
        if rem:
            tail = values[k*g:].mean(keepdims=False)             # Promedio de los bins sobrantes
            return np.concatenate([base, np.array([tail])])      # Concatena base + último grupo parcial
        return base                                              # Solo grupos completos
    else:
        return np.array([values.mean()])                         # Si hay menos de g elementos, promedia todo

# -------- Precomputo de frecuencias agrupadas para ubicar barras --------
freqs_grp = group_bins(freqs, GROUP_SIZE)   # Agrupa el eje de frecuencias para posicionar cada barra
if len(freqs_grp) > 1:                      # Si hay al menos 2 barras
    step = freqs_grp[1] - freqs_grp[0]      # Distancia entre barras en Hz
else:
    step = freqs[-1] if len(freqs) > 1 else SR/2  # Fallback si sólo hay una barra
bar_width = step * BAR_GAP                   # Ancho de barra proporcional al paso

# -------- Callback de audio (tiempo real) --------
def callback(outdata, frames, time_info, status):
    """
    En cada llamada:
    - Escribe 'frames' muestras al dispositivo (audio fluido).
    - Llena el ring buffer; cuando hay N, encola una ventana para la UI.
    """
    global read_idx, ring, ring_fill                              # Usamos y actualizamos índices/buffers globales
    end = read_idx + frames                                       # Calcula el final del bloque a reproducir
    chunk = audio[read_idx:end]                                   # Extrae el trozo de audio
    read_idx = end                                                # Avanza el índice global
    if len(chunk) < frames:                                       # Si nos quedamos sin audio antes de completar el bloque
        out = np.zeros(frames, dtype=np.float32)                  # Rellena el resto con ceros
        out[:len(chunk)] = chunk                                  # Copia lo disponible
        outdata[:] = out.reshape(-1, 1)                           # Escribe al buffer de salida (mono)
        raise sd.CallbackStop()                                   # Señala que terminó la reproducción
    else:
        outdata[:] = chunk.reshape(-1, 1)                         # Escribe el bloque completo (mono)

    ptr = 0                                                       # Puntero local dentro de 'chunk'
    while ptr < frames:                                           # Procesa todo el bloque recién enviado
        take = min(frames - ptr, N - ring_fill)                   # Cuántas muestras caben en el ring
        ring[ring_fill:ring_fill + take] = chunk[ptr:ptr + take]  # Copia esa porción al ring
        ring_fill += take                                         # Actualiza cuánto del ring está lleno
        ptr += take                                               # Avanza el puntero en el 'chunk'
        if ring_fill == N:                                        # Si el ring se llenó: tenemos una ventana completa
            frame = ring.copy()                                   # Copia segura de la ventana
            if HOP < N:                                           # Si hay solape
                ring[:N - HOP] = ring[HOP:]                       # Desplaza el ring para conservar N-HOP últimas muestras
                ring_fill = N - HOP                               # Actualiza el llenado (queda espacio para HOP nuevas)
            else:
                ring_fill = 0                                     # Sin solape: vacía el ring
            if not fft_q.full():                                  # Si la cola tiene espacio
                fft_q.put_nowait(frame)                           # Encola la ventana (no bloquear el audio)

# -------- Configuración del gráfico (barras) --------
plt.ion()                                                         # Modo interactivo (refresco sin bloquear)
fig, ax = plt.subplots()                                          # Figura y ejes
initial_heights = np.zeros_like(freqs_grp)                        # Alturas iniciales de barras (todo 0)
bars = ax.bar(freqs_grp,                                          # Posición X de cada barra (Hz agrupados)
              initial_heights + Y_SPAN_DB,                        # Altura inicial (0 real => altura Y_SPAN_DB por 'bottom')
              width=bar_width,                                    # Ancho visual de cada barra
              bottom=-Y_SPAN_DB)                                  # Base inferior de las barras en -Y_SPAN_DB
ax.set_xlim(0, SR / 2)                                            # X de 0 a Nyquist
ax.set_ylim(-Y_SPAN_DB, 0)                                        # Y en decibelios: -Y_SPAN_DB .. 0
ax.set_xlabel("Hz")                                               # Etiqueta eje X
ax.set_ylabel("dB")                                               # Etiqueta eje Y
ax.set_title(f"Espectro (barras) en tiempo real — {FILE}")        # Título

# -------- Bucle principal con stream de salida --------
with sd.OutputStream(samplerate=SR,                               # Frecuencia de muestreo
                     channels=1,                                  # Mono
                     dtype='float32',                             # Tipo de dato de las muestras
                     blocksize=BLOCKSIZE,                         # Tamaño de bloque por callback
                     device=DEVICE_OUT,                           # Dispositivo de salida (o None)
                     callback=callback):                          # Callback de reproducción/encolado
    while plt.fignum_exists(fig.number):                          # Mientras la ventana de la gráfica siga abierta
        try:
            frame = fft_q.get(timeout=0.05)                       # Toma una ventana lista desde la cola (espera breve)
        except queue.Empty:                                       # Si no llegó nada a tiempo
            plt.pause(0.001)                                      # Cede tiempo al GUI y reintenta
            continue                                              # Vuelve al inicio del loop

        X = np.fft.rfft(frame * window)                           # FFT de la ventana con Hann
        mag = np.abs(X)                                           # Magnitud de cada bin
        db = 20 * np.log10(mag + EPS)                             # Magnitud en dB

        db_grp = group_bins(db, GROUP_SIZE)                       # Agrupa dB por GROUP_SIZE para reducir barras
        heights = db_grp + Y_SPAN_DB                              # Convertimos dB reales a altura relativa desde 'bottom'

        for bar, h in zip(bars, heights):                         # Recorre barra a barra
            bar.set_height(max(0.0, float(h)))                    # Actualiza altura (clamp >= 0 para evitar negativos)

        plt.pause(0.001)                                          # Refresca dibujo sin bloquear demasiado

# -------- Cierre limpio de la UI --------
plt.ioff()                                                        # Desactiva modo interactivo
plt.show()                                                        # Mantiene la ventana hasta que el usuario cierre
