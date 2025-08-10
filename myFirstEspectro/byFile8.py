# fft_mp3_iso_pyqtgraph_fixedpath.py
import sys, queue
import numpy as np
import sounddevice as sd
from pydub import AudioSegment

# --- Qt / PyQtGraph ---
try:
    from PySide6 import QtWidgets, QtCore, QtGui
except ImportError:
    from PyQt5 import QtWidgets, QtCore, QtGui  # type: ignore
import pyqtgraph as pg

# ------------------ Parámetros ------------------
SR = 44100
N = 8192
HOP = 512
BLOCKSIZE = 4096
DEVICE_OUT = None
FILE = sys.argv[1] if len(sys.argv) > 1 else "PXRKX, Nitro Station - 90s Sunshine.mp3"

BPO = 3
FMIN = 40.0
FMAX = 20000.0
SMOOTH_ALPHA = 0.25
MIN_BINS_PER_BAND = 2

NORM_MODE = "psd"
RELATIVE_PER_FRAME = False

# ------------------ Cargar MP3 ------------------
seg = AudioSegment.from_file(FILE).set_frame_rate(SR).set_channels(1)
sw = seg.sample_width
dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
arr = np.frombuffer(seg.raw_data, dtype=dtype_map[sw]).astype(np.float32)
if sw == 1:
    arr = arr - 128.0
max_int = float(2 ** (8 * sw - 1))
audio = (arr / max_int).astype(np.float32)

# ------------------ FFT & buffers ------------------
window = np.hanning(N).astype(np.float32)
freqs = np.fft.rfftfreq(N, 1.0 / SR)
df = SR / N
EPS = 1e-12
read_idx = 0
ring = np.zeros(N, dtype=np.float32)
ring_fill = 0
fft_q = queue.Queue(maxsize=32)


# ------------------ Bandas ISO ------------------
def make_iso_bands(freqs, bpo=3, fmin=20.0, fmax=None):
    if fmax is None:
        fmax = freqs[-1]
    centers = []
    k = -200
    while True:
        fc = 1000.0 * (2.0 ** (k / bpo))
        if fc > fmax * (2 ** (1 / (2 * bpo))): break
        if fc >= fmin / (2 ** (1 / (2 * bpo))): centers.append(fc)
        k += 1
    centers = np.array(centers, dtype=float)
    kfac = 2.0 ** (1.0 / (2.0 * bpo))
    edges_lo, edges_hi = centers / kfac, centers * kfac
    valid = (edges_hi >= fmin) & (edges_lo <= fmax)
    centers, edges_lo, edges_hi = centers[valid], edges_lo[valid], edges_hi[valid]
    band_bins = [np.where((freqs >= lo) & (freqs < hi))[0] for lo, hi in zip(edges_lo, edges_hi)]
    return centers, edges_lo, edges_hi, band_bins


centers_hz, lo, hi, band_bins = make_iso_bands(
    freqs, bpo=BPO, fmin=FMIN, fmax=min(FMAX, freqs[-1])
)
band_labels = [f"{int(round(c))} Hz" if c < 1000 else f"{c / 1000:.1f} kHz" for c in centers_hz]
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
        ring[ring_fill:ring_fill + take] = chunk[ptr:ptr + take]
        ring_fill += take
        ptr += take
        if ring_fill == N:
            frame = ring.copy()
            if HOP < N:
                ring[:N - HOP] = ring[HOP:]
                ring_fill = N - HOP
            else:
                ring_fill = 0
            if not fft_q.full():
                fft_q.put_nowait(frame)


# ------------------ UI PyQtGraph (con Path de rectángulos) ------------------
app = QtWidgets.QApplication(sys.argv)
pg.setConfigOptions(antialias=True)

plot = pg.PlotWidget(title=f"Bandas ISO ({'octava' if BPO == 1 else '1/3 octava'}) — {FILE}")
plot.setLabel('left', 'dB por banda (promedio de potencia)')
axis = plot.getPlotItem().getAxis('bottom')
axis.setTicks([list(enumerate(band_labels))])

# Rango X fijo para que se vean desde el inicio
xpos = np.arange(len(centers_hz), dtype=float)
plot.setXRange(-0.5, len(centers_hz) - 0.5, padding=0)
plot.setMouseEnabled(x=True, y=False)
plot.showGrid(x=False, y=True)

# Y inicial razonable
INIT_YMIN = -120.0
plot.setYRange(INIT_YMIN, 0.0, padding=0.0)

# Item de barras como Path (siempre visible)
path_item = QtWidgets.QGraphicsPathItem()
path_item.setBrush(pg.mkBrush(200, 200, 255))
path_item.setPen(pg.mkPen(230, 230, 255, width=1))
path_item.setZValue(10)
plot.addItem(path_item)

# ------------------ Stream + Update Timer ------------------
stream = sd.OutputStream(samplerate=SR, channels=1, dtype='float32',
                         blocksize=BLOCKSIZE, device=DEVICE_OUT, callback=callback)
stream.start()


def on_close():
    try:
        stream.stop();
        stream.close()
    except Exception:
        pass


app.aboutToQuit.connect(on_close)

width = 0.8  # ancho de cada barra en coordenadas X


def build_path(heights, y0):
    """Construye un QPainterPath con rectángulos de barras desde y0 hasta y0+height."""
    p = QtGui.QPainterPath()
    # si heights llega vacío, devuelvo un rectángulo mínimo para que algo se vea
    if len(heights) == 0:
        p.addRect(QtCore.QRectF(-0.5, y0, 1.0, 1.0))
        return p
    for i, h in enumerate(heights):
        if h <= 0:
            continue
        x0 = xpos[i] - width / 2.0
        x1 = xpos[i] + width / 2.0
        y1 = y0 + float(h)
        # RectF(x, y, w, h)
        p.addRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
    # si todas eran <=0, al menos pone una mínima para evidenciar algo
    if p.isEmpty():
        p.addRect(QtCore.QRectF(-0.5, y0, 1.0, 1.0))
    return p


def update():
    global band_db_prev
    # Drenar cola, quedarnos con el último frame
    frame = None
    try:
        while True:
            frame = fft_q.get_nowait()
    except queue.Empty:
        pass
    if frame is None:
        return

    Xw = np.fft.rfft(frame * window)

    # dB por banda
    if NORM_MODE == "psd":
        U = (window ** 2).mean()
        ps = (np.abs(Xw) ** 2) / (N ** 2 * U)
        ps[1:-1] *= 2.0
        vals = []
        for idx in band_bins:
            if len(idx) < MIN_BINS_PER_BAND:
                vals.append(-150.0)
            else:
                band_power = np.sum(ps[idx]) * df
                vals.append(10 * np.log10(band_power + EPS))
    else:
        mag = np.abs(Xw)
        vals = []
        for idx in band_bins:
            if len(idx) < MIN_BINS_PER_BAND:
                vals.append(-150.0)
            else:
                band_power = np.mean((mag[idx] ** 2)) + EPS
                vals.append(10 * np.log10(band_power))

    b = np.array(vals, dtype=float)
    if band_db_prev is not None:
        b = SMOOTH_ALPHA * b + (1.0 - SMOOTH_ALPHA) * band_db_prev
    band_db_prev = b

    # Rango Y: fijo el techo en 0 dB, piso dinámico con límites seguros
    if RELATIVE_PER_FRAME:
        b = b - np.max(b)
        y_top = 0.0
        y_bottom = -30.0
    else:
        y_top = 0.0
        y_bottom = float(np.clip(np.min(b) - 5.0, -150.0, -10.0))

    plot.setYRange(y_bottom, y_top, padding=0.0)

    # Alturas positivas respecto al piso actual
    heights = np.maximum(0.0, b - y_bottom)

    # Actualizar path de barras
    path = build_path(heights, y_bottom)
    path_item.setPath(path)


timer = QtCore.QTimer()
timer.setTimerType(QtCore.Qt.PreciseTimer)
timer.timeout.connect(update)
timer.start(16)  # ~60 FPS

plot.resize(1200, 500)
plot.show()
try:
    sys.exit(app.exec())
except AttributeError:
    sys.exit(app.exec_())
