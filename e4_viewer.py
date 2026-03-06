# --- HR/RR display smoothing / cadence ---
HR_UPDATE_S = 2.0
RR_UPDATE_S = 5.0
MAX_HR_JUMP_BPM = 10.0
MAX_RR_JUMP_BRPM = 10.0

#!/usr/bin/env python3
"""
Empatica E4 real-time viewer (reverse-engineering helper)

Adds:
- Scan + Connect button (no hard-coded MAC)
- One window with subplots: BVP, ACC magnitude, Temperature, EDA
- Temperature (3ea6) low-level decode: 8x uint16 samples + uint32 counter (20B packet)
- EDA (3ea8) low-level decode: 8x uint16 samples + uint32 counter (20B packet)
- Contact = YES/NO derived from EDA with hysteresis + optional motion gate

Install:
  pip install bleak numpy scipy matplotlib

Notes on EDA/contact:
- Off-wrist EDA often does NOT go to 0. Depending on front-end design it can float/saturate.
- Therefore contact detection should generally use hysteresis and you may need to tune thresholds
  on your device/firmware. Start with the defaults and adjust while watching the live EDA trace.
"""

import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
from bleak import BleakClient, BleakScanner

from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# -------------------- BLE UUIDs (Empatica E4) --------------------
CTRL_CHAR = "00003e71-0000-1000-8000-00805f9b34fb"
START_CMD = b"\x02"

BVP_CHAR = "00003ea1-0000-1000-8000-00805f9b34fb"
ACC_CHAR = "00003ea3-0000-1000-8000-00805f9b34fb"
TEMP_CHAR = "00003ea6-0000-1000-8000-00805f9b34fb"
EDA_CHAR = "00003ea8-0000-1000-8000-00805f9b34fb"
BUTTON_CHAR = "00003eb2-0000-1000-8000-00805f9b34fb"  # Button / event marker

CONNECT_TIMEOUT = 25.0

# -------------------- Rates / packet formats (inferred) --------------------
FS_BVP = 64.0  # BVP expected
BVP_DATA_BYTES = 18
BVP_SAMPLES_PER_PACKET = 12  # 18 bytes -> 12 samples (12-bit packing)

# -------------------- BVP decode mode --------------------
#  - "12*12bit": use first 18 bytes as packed 12-bit samples (12 samples/packet)
#  - "10*16bit": use all 20 bytes as 10x int16 little-endian samples (10 samples/packet)
BVP_MODE_LABELS = ["12*12bit", "10*16bit", "10*16bit_cleanS9", "10*16bit_dropS9"]
bvp_mode = "12*12bit"
_bvp_mode_lock = threading.Lock()

# Dynamic BVP fs estimate (EMA from packet cadence)
BVP_FS_EMA_ALPHA = 0.15
_latest_bvp_pkt_t = None
_bvp_fs_est = FS_BVP  # starts at nominal

FS_ACC = 32.0
ACC_TRIPLES_PER_PACKET = 3
ACC_SAMPLES_PER_PACKET = 3

# These two are inferred from your CSV: 20B per notify with ~1.5–2.0s cadence and 8 samples.
TEMP_SAMPLES_PER_PACKET = 8
EDA_SAMPLES_PER_PACKET = 8

# -------------------- Temperature conversion (unknown) --------------------
# Raw-to-°C mapping is device/firmware specific; keep raw visible until calibrated.
# If you later determine a linear mapping, set:
TEMP_SCALE = 0.02864
TEMP_OFFSET = -400.9

# -------------------- Contact detection tuning --------------------
# Your observation: off-wrist EDA proxy can be HIGHER than on-wrist (open-circuit floats/rails high).
# Therefore we use an *inverted* hysteresis threshold + a short-window variability guard (Fix #2).
#
# State machine (hysteresis):
#   - If currently OFF: declare ON only when (proxy <= EDA_ON_MAX) AND (std_recent >= EDA_STD_MIN)
#   - If currently ON:  declare OFF when (proxy >= EDA_OFF_MIN)
#
# Tune these on your device:
EDA_ON_MAX = 9200.0  # contact becomes YES when proxy falls *below* this (and variability passes)
EDA_OFF_MIN = 9800.0  # contact becomes NO when proxy rises *above* this (hysteresis; should be > EDA_ON_MAX)

# Variability guard over a short window of recent packets (does not assume a known sample rate).
EDA_STD_POINTS = 8  # ~8 packets ≈ 12–20 seconds depending on cadence
EDA_STD_MIN = 15.0  # minimum std(proxy) over window to accept ON transition

# -------------------- Buffers --------------------
BVP_SECONDS = 180
bvp_buf = deque(maxlen=int(BVP_SECONDS * FS_BVP))
t_bvp = deque(maxlen=int(BVP_SECONDS * FS_BVP))

ACC_SECONDS = 60
acc_mag = deque(maxlen=int(ACC_SECONDS * FS_ACC))
t_acc = deque(maxlen=int(ACC_SECONDS * FS_ACC))

# For TEMP/EDA we store decoded "proxy" values at packet cadence with per-sample timestamps.
TEMP_SECONDS = 10 * 60
temp_raw_buf = deque(maxlen=int(TEMP_SECONDS * 4))  # loose; real fs inferred at runtime
t_temp = deque(maxlen=int(TEMP_SECONDS * 4))

EDA_SECONDS = 10 * 60
eda_proxy_buf = deque(maxlen=int(EDA_SECONDS * 4))
t_eda = deque(maxlen=int(EDA_SECONDS * 4))

latest = {
    "status": "idle",
    "device": None,
    "addr": None,

    "hr_bpm": None,
    "rr_brpm": None,

    "motion_ok": True,
    "contact": None,  # True/False/None
    "eda_proxy": None,
    "temp_raw": None,
    "button_count": 0,
    "button_last_epoch": 0.0,
}

stop_flag = threading.Event()


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# -------------------- Decoders --------------------
def unpack_bvp_12bit(payload18: bytes) -> np.ndarray:
    # 18 bytes -> 12 samples, values 0..4095
    out = np.empty(12, dtype=np.float64)
    j = 0
    for i in range(0, 18, 3):
        b0, b1, b2 = payload18[i], payload18[i + 1], payload18[i + 2]
        s0 = b0 | ((b1 & 0x0F) << 8)
        s1 = ((b1 >> 4) & 0x0F) | (b2 << 4)
        out[j] = s0
        out[j + 1] = s1
        j += 2
    return out


def decode_bvp_i16x10(payload20: bytes) -> np.ndarray:
    # 20 bytes -> 10 samples, int16 little-endian
    if len(payload20) != 20:
        return np.empty(0, dtype=np.float64)
    v = np.frombuffer(payload20, dtype="<i2", count=10).astype(np.float64)
    return v



def decode_bvp_u16x10(payload20: bytes) -> np.ndarray:
    """20 bytes -> 10 samples, uint16 little-endian (raw BVP)."""
    if len(payload20) != 20:
        return np.empty(0, dtype=np.float64)
    v = np.frombuffer(payload20, dtype="<u2", count=10).astype(np.float64)
    return v


def clean_s9_nibble(v10_u16: np.ndarray) -> np.ndarray:
    """Return a copy of v10 where S9's high-byte low nibble is masked (assumed 4-bit counter)."""
    if v10_u16.size != 10:
        return v10_u16.astype(np.float64, copy=True)
    v = v10_u16.astype(np.int32, copy=True)
    s9 = int(v[9])
    s9_l = s9 & 0x00FF
    s9_h = (s9 >> 8) & 0x00FF
    s9_h_clean = s9_h & 0xF0
    v[9] = s9_l + (s9_h_clean << 8)
    return v.astype(np.float64)

# --- Drop-S9 reconstruction state (one-packet latency) ---
_pending_bvp_9 = None          # np.ndarray shape (9,)
_pending_bvp_last_s8 = None    # float
_bvp_time_next = None          # float epoch seconds for next sample
def decode_acc_triples(payload18: bytes) -> np.ndarray:
    v = np.frombuffer(payload18, dtype="<i2").astype(np.float64)
    if v.size != 9:
        return np.empty((0, 3), dtype=np.float64)
    return v.reshape(3, 3)


def decode_u16x8_u32(payload20: bytes) -> tuple[np.ndarray, int] | None:
    """20 bytes: first 16 = 8x uint16 little-endian; last 4 = uint32 little-endian counter."""
    if len(payload20) != 20:
        return None
    s = np.frombuffer(payload20[:16], dtype="<u2").astype(np.float64)  # shape (8,)
    c = int.from_bytes(payload20[16:20], "little", signed=False)
    return s, c


def dominant_u16_value(u16_vals: np.ndarray) -> float:
    """Return the mode-ish value by picking the most frequent element in the 8-sample packet."""
    # Many of your EDA packets have a value repeated 3 times; this stabilizes the proxy.
    vals = u16_vals.astype(np.int64)
    uniq, counts = np.unique(vals, return_counts=True)
    return float(uniq[int(np.argmax(counts))])


# -------------------- 3EA8 alternate proxy (24-bit rotate + avg of 6 sessions) --------------------
def rot24_nibbles_12_from3(b3: bytes) -> int:
    """3 bytes -> 24-bit int, then rotate left by 12 bits: 0xABCDEF -> 0xDEFABC."""
    if len(b3) < 3:
        return 0
    v = int.from_bytes(b3[:3], "big", signed=False) & 0xFFFFFF
    return ((v << 12) & 0xFFFFFF) | (v >> 12)


def eda_proxy_avg6_rot12(payload: bytes) -> float | None:
    """Use first 18 bytes as 6 sessions of 3 bytes; rotate each (12-bit) then average."""
    if len(payload) < 18:
        return None
    vals = [rot24_nibbles_12_from3(payload[i:i+3]) for i in range(0, 18, 3)]
    return float(int(round(sum(vals) / 6.0)))

# -------------------- Filters / estimation --------------------
def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray | None:
    if len(x) < int(fs * 3):
        return None
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def estimate_hr_from_bvp(bvp: np.ndarray, fs: float) -> float | None:
    """Robust HR estimate from BVP using Welch PSD in the cardiac band.
    Returns bpm or None if confidence is low.
    """
    if len(bvp) < int(fs * 12):
        return None
    x = bvp.astype(np.float64)
    x = x - np.mean(x)

    # Cardiac band (typical adult)
    xf = bandpass(x, fs, 0.7, 3.0, order=3)
    if xf is None:
        return None

    from scipy.signal import welch
    f, Pxx = welch(xf, fs=fs, nperseg=min(2048, len(xf)))
    band = (f >= 0.8) & (f <= 2.5)  # 48–150 bpm
    if not np.any(band):
        return None

    p_band = Pxx[band]
    f_band = f[band]
    i = int(np.argmax(p_band))
    f_peak = float(f_band[i])
    bpm = f_peak * 60.0

    # Confidence gate: peak prominence vs median power in band
    prom = float(np.max(p_band) / (np.median(p_band) + 1e-12))
    if prom < 1.6:
        return None
    return float(bpm)
def estimate_rr_from_bvp(bvp: np.ndarray, fs: float) -> float | None:
    """Respiration rate estimate from BVP using combined surrogates:
    - RIIV: baseline (low-frequency) modulation of the raw BVP
    - RIAV: beat-to-beat amplitude modulation from detected pulses
    Returns breaths/min or None if confidence is low.
    """
    if len(bvp) < int(fs * 60):
        return None
    x = bvp.astype(np.float64)
    x = x - np.mean(x)

    from scipy.signal import butter, filtfilt, find_peaks, welch

    # --- RIIV (baseline modulation) ---
    b_lp, a_lp = butter(3, 0.5 / (fs / 2), btype="low")
    riiv = filtfilt(b_lp, a_lp, x)

    # --- Find beats for RIAV / RSA ---
    xf = bandpass(x, fs, 0.7, 3.0, order=3)
    if xf is None:
        return None

    prom = max(np.std(xf) * 0.6, 1e-6)
    min_dist = int(0.4 * fs)  # up to 150 bpm
    peaks, props = find_peaks(xf, distance=max(1, min_dist), prominence=prom)
    if len(peaks) < 8:
        # fallback: use only RIIV
        peaks = None

    # RIAV: pulse amplitude series (peak-to-trough within a short window)
    riav_t = None
    riav = None
    if peaks is not None:
        amps = []
        times = []
        w = int(0.25 * fs)
        for p in peaks:
            lo = max(0, p - w)
            hi = min(len(xf), p + w)
            trough = float(np.min(xf[lo:hi]))
            amp = float(xf[p] - trough)
            amps.append(amp)
            times.append(p / fs)
        if len(amps) >= 8:
            riav = np.array(amps, dtype=np.float64)
            riav_t = np.array(times, dtype=np.float64)

    # Helper: PSD peak in respiration band + confidence
    def rr_psd(sig, sig_fs):
        f, Pxx = welch(sig, fs=sig_fs, nperseg=min(1024, len(sig)))
        band = (f >= 0.10) & (f <= 0.50)  # 6–30 brpm
        if not np.any(band):
            return None, 0.0
        p = Pxx[band]
        fb = f[band]
        k = int(np.argmax(p))
        rr = float(fb[k] * 60.0)
        conf = float(np.max(p) / (np.median(p) + 1e-12))
        return rr, conf

    # RR from RIIV (uniform at fs)
    rr1, c1 = rr_psd(riiv, fs)

    # RR from RIAV (irregular, interpolate to 4 Hz)
    rr2, c2 = (None, 0.0)
    if riav is not None and riav_t is not None:
        fs2 = 4.0
        t_u = np.arange(riav_t[0], riav_t[-1], 1.0 / fs2)
        if t_u.size >= 32:
            riav_u = np.interp(t_u, riav_t, riav)
            riav_u = riav_u - np.mean(riav_u)
            rr2, c2 = rr_psd(riav_u, fs2)

    # Fuse: prefer agreement; otherwise pick higher confidence
    c1 = float(c1)
    c2 = float(c2)
    if rr1 is None and rr2 is None:
        return None
    if rr1 is not None and rr2 is not None:
        if abs(rr1 - rr2) <= 2.0:  # within 2 brpm
            rr = 0.5 * (rr1 + rr2)
            conf = max(c1, c2)
        else:
            rr = rr1 if c1 >= c2 else rr2
            conf = max(c1, c2)
    else:
        rr = rr1 if rr1 is not None else rr2
        conf = c1 if rr1 is not None else c2

    if conf < 1.4:
        return None
    return float(rr)
def motion_ok_from_acc() -> bool:
    win = int(FS_ACC * 2.0)
    if len(acc_mag) < win:
        return True
    a = np.asarray(list(acc_mag)[-win:], dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < win // 2:
        return True
    return np.std(a) < 2000.0


def eda_std_recent() -> float | None:
    """Std of recent EDA proxy values over the last EDA_STD_POINTS packets."""
    if len(eda_proxy_buf) < 2:
        return None
    n = min(EDA_STD_POINTS, len(eda_proxy_buf))
    x = np.asarray(list(eda_proxy_buf)[-n:], dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    return float(np.std(x))


def contact_from_eda_proxy_fix2(prev: bool | None, proxy: float) -> bool:
    """Fix #2: inverted hysteresis + variability guard on the OFF->ON transition."""
    # Default to OFF until we have evidence of contact.
    if prev is None:
        prev = False

    if prev:
        # ON -> OFF when proxy rises high (open-circuit float/rail)
        return not (proxy >= EDA_OFF_MIN)

    # OFF -> ON only when proxy is low enough AND recent variability is present
    stdv = eda_std_recent()
    if stdv is None:
        return False
    return (proxy <= EDA_ON_MAX) and (stdv >= EDA_STD_MIN)


# -------------------- BLE controller thread (async loop inside) --------------------
class BLEController(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = None
        self.client: BleakClient | None = None
        self.task: asyncio.Task | None = None
        self.last_error: Exception | None = None

    def run(self):
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        except Exception as e:
            self.last_error = e
        finally:
            try:
                if self.loop and self.loop.is_running():
                    self.loop.stop()
            except Exception:
                pass

    # ---------- public entrypoints (thread-safe) ----------
    def connect(self):
        if not self.loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._connect_flow(), self.loop)
        return fut

    def disconnect(self):
        if not self.loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._disconnect_flow(), self.loop)
        return fut

    # ---------- async internals ----------
    async def _disconnect_flow(self):
        latest["status"] = "disconnecting"
        try:
            if self.client and self.client.is_connected:
                try:
                    await self.client.stop_notify(BVP_CHAR)
                except Exception:
                    pass
                try:
                    await self.client.stop_notify(ACC_CHAR)
                except Exception:
                    pass
                try:
                    await self.client.stop_notify(TEMP_CHAR)
                except Exception:
                    pass
                try:
                    await self.client.stop_notify(EDA_CHAR)
                    await self.client.stop_notify(BUTTON_CHAR)
                except Exception:
                    pass

                await self.client.disconnect()
        finally:
            self.client = None
            latest["status"] = "idle"
            latest["device"] = None
            latest["addr"] = None

    async def _connect_flow(self):
        # If already connected, do nothing
        if self.client and self.client.is_connected:
            return

        latest["status"] = "scanning"

        devices = await BleakScanner.discover(timeout=6.0)

        # Prefer names containing E4 / Empatica; otherwise pick first.
        picked = None
        for d in devices:
            nm = (d.name or "").lower()
            if "empatica" in nm or "e4" in nm:
                picked = d
                break
        if picked is None and devices:
            picked = devices[0]

        if picked is None:
            latest["status"] = "no device found"
            return

        latest["device"] = picked.name
        latest["addr"] = picked.address
        latest["status"] = f"connecting ({picked.name or 'unknown'})"

        self.client = BleakClient(picked.address, timeout=CONNECT_TIMEOUT)
        try:
            await self.client.connect()
            if not self.client.is_connected:
                latest["status"] = "connect failed"
                return

            latest["status"] = "connected"

            # ---- Notification handlers ----
            def on_bvp(_sender, data: bytearray):
                # E4 BVP notify is 20 bytes
                if len(data) != 20:
                    return

                bb = bytes(data)

                # Choose decode mode
                with _bvp_mode_lock:
                    mode = bvp_mode

                try:
                    if mode == "10*16bit_cleanS9":
                        # Strategy: treat packet as 10x uint16 samples, but mask S9_H[3:0] (suspected 4-bit counter).
                        v10 = decode_bvp_u16x10(bb)
                        if v10.size != 10:
                            return
                        samples = clean_s9_nibble(v10)
                        spp = 10

                    elif mode in ("10*16bit_dropS9", "10*16bit_cleanS9"):

                        # Strategy: treat packet as 10x uint16 samples, but S9 high-byte low nibble is a packet counter.
                        # We reconstruct the missing/corrupted S9 using linear interpolation between:
                        #   prev packet S8 and current packet S0.
                        # This introduces a 1-packet latency but preserves a 64 Hz uniform time base.
                        v10 = decode_bvp_u16x10(bb)
                        if v10.size != 10:
                            return

                        global _pending_bvp_9, _pending_bvp_last_s8
                        s0_curr = float(v10[0])
                        s8_curr = float(v10[8])
                        cur9 = v10[:9].astype(np.float64)

                        if _pending_bvp_9 is None:
                            # Prime the pipeline; can't reconstruct previous S9 yet.
                            _pending_bvp_9 = cur9
                            _pending_bvp_last_s8 = s8_curr
                            return

                        # Reconstruct previous S9 (the 10th sample) and emit 10 samples for the previous packet.
                        s9_hat = 0.5 * (float(_pending_bvp_last_s8) + s0_curr)
                        samples = np.concatenate([_pending_bvp_9, np.array([s9_hat], dtype=np.float64)])
                        spp = 10

                        # Shift pipeline state to current packet (to be emitted on next notify)
                        _pending_bvp_9 = cur9
                        _pending_bvp_last_s8 = s8_curr

                    elif mode == "10*16bit":
                        # Legacy: interpret as 10x int16 little-endian (kept for experimentation)
                        samples = decode_bvp_i16x10(bb)              # 10 samples
                        spp = 10
                    else:
                        payload = bb[:BVP_DATA_BYTES]               # 18 bytes
                        samples = unpack_bvp_12bit(payload)         # 12 samples
                        spp = BVP_SAMPLES_PER_PACKET
                except Exception:
                    return

                # Update dynamic fs estimate from packet cadence
                global _latest_bvp_pkt_t, _bvp_fs_est
                t_now = time.time()
                if _latest_bvp_pkt_t is not None:
                    dt_pkt = t_now - _latest_bvp_pkt_t
                    if 0.02 < dt_pkt < 2.0:
                        fs_pkt = spp / dt_pkt
                        _bvp_fs_est = (1.0 - BVP_FS_EMA_ALPHA) * _bvp_fs_est + BVP_FS_EMA_ALPHA * fs_pkt
                _latest_bvp_pkt_t = t_now
                latest["bvp_fs"] = float(_bvp_fs_est)

                # Timestamp samples.
                # For dropS9 mode we enforce a uniform 64 Hz time base to avoid packet-jitter artifacts.
                if mode in ("10*16bit_dropS9", "10*16bit_cleanS9"):
                    global _bvp_time_next
                    dt = 1.0 / FS_BVP
                    if _bvp_time_next is None:
                        # Backdate so the last sample aligns close to arrival time.
                        _bvp_time_next = t_now - (samples.size - 1) * dt
                    for i in range(samples.size):
                        t_sample = _bvp_time_next
                        _bvp_time_next += dt
                        bvp_buf.append(float(samples[i]))
                        t_bvp.append(t_sample)
                else:
                    # Default: use estimated fs from packet cadence (fallback to nominal)
                    fs_use = float(_bvp_fs_est) if np.isfinite(_bvp_fs_est) and _bvp_fs_est > 1 else FS_BVP
                    dt = 1.0 / fs_use
                    for i in range(samples.size):
                        t_sample = t_now - (samples.size - 1 - i) * dt
                        bvp_buf.append(float(samples[i]))
                        t_bvp.append(t_sample)

            def on_acc(_sender, data: bytearray):
                if len(data) < 18:
                    return
                triples = decode_acc_triples(bytes(data)[:18])

                t_now = time.time()
                dt = 1.0 / FS_ACC
                for i in range(triples.shape[0]):
                    x, y, z = triples[i]
                    mag = float(np.sqrt(x * x + y * y + z * z))
                    if np.isfinite(mag):
                        t_sample = t_now - (triples.shape[0] - 1 - i) * dt
                        acc_mag.append(mag)
                        t_acc.append(t_sample)

            def on_temp(_sender, data: bytearray):
                bb = bytes(data)
                dec = decode_u16x8_u32(bb)
                if dec is None:
                    return
                u16s, _ctr = dec
                t_now = time.time()

                # infer per-sample timing assuming 8 samples evenly spaced over packet interval (~2s)
                # we don't know exact fs, so just spread over last 2 seconds for visualization
                dt = 2.0 / TEMP_SAMPLES_PER_PACKET
                for i in range(TEMP_SAMPLES_PER_PACKET):
                    t_sample = t_now - (TEMP_SAMPLES_PER_PACKET - 1 - i) * dt
                    raw = float(u16s[i])
                    temp_raw_buf.append(raw)
                    t_temp.append(t_sample)

                latest["temp_raw"] = float(np.median(u16s))
            def on_eda(_sender, data: bytearray):
                            bb = bytes(data)
                            t_now = time.time()

                            # Alternate 3EA8 proxy:
                            # - first 18 bytes = 6 sessions * 3 bytes
                            # - rotate each 24-bit value by 12 bits (abcdef -> defabc)
                            # - average the 6 rotated values
                            proxy = eda_proxy_avg6_rot12(bb)
                            if proxy is None:
                                return

                            # push one proxy point per packet (don’t pretend we know the real per-sample timing)
                            eda_proxy_buf.append(float(proxy))
                            t_eda.append(t_now)

                            latest["eda_proxy"] = float(proxy)
                            latest["eda_proxy_hex"] = f"0x{int(proxy) & 0xFFFFFF:06x}"

                            prev = latest["contact"]
                            now = contact_from_eda_proxy_fix2(prev, float(proxy))

                            latest["contact"] = now

            def on_button(_sender, data: bytearray):
                bb = bytes(data)
                if len(bb) != 20:
                    return
                event_code = bb[0]
                if event_code != 1:
                    return

                # Debounce: ignore repeats within 300 ms
                t_now = time.time()
                if t_now - float(latest.get("button_last_epoch", 0.0)) < 0.3:
                    return

                latest["button_last_epoch"] = t_now
                latest["button_count"] = int(latest.get("button_count", 0)) + 1
                print(f"Button Pressed: {latest['button_count']}")

            # ---- Subscribe ----
            await self.client.start_notify(BVP_CHAR, on_bvp)
            await self.client.start_notify(ACC_CHAR, on_acc)
            await self.client.start_notify(TEMP_CHAR, on_temp)
            await self.client.start_notify(EDA_CHAR, on_eda)
            await self.client.start_notify(BUTTON_CHAR, on_button)

            # ---- Start streaming ----
            await self.client.write_gatt_char(CTRL_CHAR, START_CMD, response=True)
            latest["status"] = "streaming"

        except Exception as e:
            self.last_error = e
            latest["status"] = f"error: {type(e).__name__}"
            try:
                await self.client.disconnect()
            except Exception:
                pass
            self.client = None
            latest["device"] = None
            latest["addr"] = None


# -------------------- UI / Plotting --------------------
# Option A: keep a single shared time window across all subplots so BVP/ACC don't
# visually shrink when TEMP/EDA history grows.
PLOT_WINDOW_S = 30.0  # seconds shown on ALL plots (shared x-axis)

def main():
    ctrl = BLEController()
    ctrl.start()

    # One window, multiple "subscreens"
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(4, 1, hspace=0.35)

    ax_bvp = fig.add_subplot(gs[0, 0])
    ax_acc = fig.add_subplot(gs[1, 0], sharex=ax_bvp)
    ax_temp = fig.add_subplot(gs[2, 0], sharex=ax_bvp)
    ax_eda = fig.add_subplot(gs[3, 0], sharex=ax_bvp)

    # Lines
    (line_bvp,) = ax_bvp.plot([], [], linestyle="-")
    (line_acc,) = ax_acc.plot([], [], linestyle="-")
    (line_temp,) = ax_temp.plot([], [], linestyle="-")
    (line_eda,) = ax_eda.plot([], [], linestyle="-")

    ax_bvp.set_title("Empatica E4 (BVP/ACC/TEMP/EDA) — single window")
    ax_bvp.set_ylabel("BVP (bandpassed)")
    ax_acc.set_ylabel("|acc|")
    ax_temp.set_ylabel("Temp (raw / T*)")
    ax_eda.set_ylabel("EDA proxy (avg6 rot12)")
    ax_eda.set_xlabel("Time (local)")

    for ax in (ax_bvp, ax_acc, ax_temp, ax_eda):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # Show EDA proxy as hex (24-bit)
    ax_eda.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"0x{int(x) & 0xFFFFFF:06x}"))

    fig.autofmt_xdate()

    # Status text
    txt = ax_bvp.text(0.01, 0.95, "", transform=ax_bvp.transAxes, va="top")
    txt_temp = ax_temp.text(0.01, 0.90, "", transform=ax_temp.transAxes, va="top")

    # Buttons
    ax_btn_conn = fig.add_axes([0.82, 0.93, 0.08, 0.05])
    ax_btn_disc = fig.add_axes([0.91, 0.93, 0.08, 0.05])
    btn_conn = Button(ax_btn_conn, "Connect")
    btn_disc = Button(ax_btn_disc, "Disconnect")

    def on_connect(_evt):
        ctrl.connect()

    def on_disconnect(_evt):
        ctrl.disconnect()

    btn_conn.on_clicked(on_connect)
    btn_disc.on_clicked(on_disconnect)

    # BVP decode mode selector (acts like a dropdown for quick switching)
    from matplotlib.widgets import RadioButtons
    ax_mode = fig.add_axes([0.68, 0.93, 0.12, 0.05])  # [left, bottom, width, height]
    ax_mode.set_title("BVP mode", fontsize=9)
    mode_radio = RadioButtons(ax_mode, BVP_MODE_LABELS, active=BVP_MODE_LABELS.index(bvp_mode))

    def on_mode(label):
        global _latest_bvp_pkt_t, _bvp_fs_est
        with _bvp_mode_lock:
            globals()["bvp_mode"] = str(label)

        # Clear BVP buffers so we don't mix formats
        bvp_buf.clear()
        t_bvp.clear()
        latest["hr_bpm"] = None
        latest["rr_brpm"] = None
        latest["_t_hr"] = 0.0
        latest["_t_rr"] = 0.0
        _latest_bvp_pkt_t = None
        _bvp_fs_est = FS_BVP
        latest["bvp_fs"] = float(_bvp_fs_est)

    mode_radio.on_clicked(on_mode)

    def update(_):
        if ctrl.last_error is not None:
            # hard crash in controller thread
            txt.set_text(f"BLE controller crashed: {ctrl.last_error}")
            return (line_bvp, line_acc, line_temp, line_eda, txt, txt_temp)

        # Update motion gate
        latest["motion_ok"] = motion_ok_from_acc()

        # --- BVP plot (shared window) ---
        t_now = time.time()
        x_min_epoch = t_now - PLOT_WINDOW_S

        if len(t_bvp) >= 2 and len(bvp_buf) >= 2:
            t_np = np.asarray(t_bvp, dtype=np.float64)
            y_np = np.asarray(bvp_buf, dtype=np.float64)

            mask = t_np >= x_min_epoch
            if np.count_nonzero(mask) >= int(FS_BVP * 3):
                t_dt = [datetime.fromtimestamp(tt).astimezone() for tt in t_np[mask]]
                y = y_np[mask].astype(np.float64)
                y = y - np.mean(y)
                yf = bandpass(y, FS_BVP, 0.7, 3.5, order=3)
                if yf is None:
                    yf = y
                line_bvp.set_data(t_dt, yf)

                # autoscale Y only; keep X fixed (shared across subplots)
                ax_bvp.relim()
                ax_bvp.autoscale_view(scalex=False, scaley=True)

        # --- ACC plot (shared window) ---
        if len(t_acc) >= 2 and len(acc_mag) >= 2:
            ta = np.asarray(t_acc, dtype=np.float64)
            aa = np.asarray(acc_mag, dtype=np.float64)

            mask = ta >= x_min_epoch
            if np.count_nonzero(mask) >= int(FS_ACC * 1):
                ta_dt = [datetime.fromtimestamp(tt).astimezone() for tt in ta[mask]]
                line_acc.set_data(ta_dt, aa[mask])

                ax_acc.relim()
                ax_acc.autoscale_view(scalex=False, scaley=True)

        # --- TEMP plot (shared window, raw) ---
        if len(t_temp) >= 2 and len(temp_raw_buf) >= 2:
            t_list = np.asarray(t_temp, dtype=np.float64)
            y_list = np.asarray(temp_raw_buf, dtype=np.float64)

            mask = t_list >= x_min_epoch
            if np.any(mask):
                t_dt = [datetime.fromtimestamp(tt).astimezone() for tt in t_list[mask]]
                line_temp.set_data(t_dt, y_list[mask])

                ax_temp.relim()
                ax_temp.autoscale_view(scalex=False, scaley=True)

        # --- EDA plot (shared window, proxy) ---
        if len(t_eda) >= 2 and len(eda_proxy_buf) >= 2:
            t_list = np.asarray(t_eda, dtype=np.float64)
            y_list = np.asarray(eda_proxy_buf, dtype=np.float64)

            mask = t_list >= x_min_epoch
            if np.any(mask):
                t_dt = [datetime.fromtimestamp(tt).astimezone() for tt in t_list[mask]]
                line_eda.set_data(t_dt, y_list[mask])

                ax_eda.relim()
                ax_eda.autoscale_view(scalex=False, scaley=True)

        # --- Apply shared X limits (fixed window) ---
        try:
            x_min_dt = datetime.fromtimestamp(x_min_epoch).astimezone()
            x_max_dt = datetime.fromtimestamp(t_now).astimezone()
            ax_bvp.set_xlim(x_min_dt, x_max_dt)  # shared -> applies to all subplots
        except Exception:
            pass

        # --- Estimation (always; no contact/motion gating) ---
        hr = latest["hr_bpm"]
        rr = latest["rr_brpm"]
        mot = latest["motion_ok"]
        contact = latest["contact"]
        status = latest["status"]
        dev = latest["device"]
        addr = latest["addr"]

        fs_bvp_live = float(latest.get('bvp_fs') or FS_BVP)
        if len(bvp_buf) > int(fs_bvp_live * 10):
            bvp_np = np.asarray(bvp_buf, dtype=np.float64)

            # Update HR at a fixed cadence to avoid jittery re-estimation
            if (t_now - float(latest.get("_t_hr", 0.0))) >= HR_UPDATE_S:
                hr_new = estimate_hr_from_bvp(bvp_np[-int(fs_bvp_live * 30):], fs_bvp_live)
                if hr_new is not None:
                    # Clamp sudden jumps (often harmonics / motion artifacts)
                    if hr is not None and abs(hr_new - hr) > MAX_HR_JUMP_BPM:
                        hr_new = hr + np.sign(hr_new - hr) * MAX_HR_JUMP_BPM
                    latest["hr_bpm"] = float(hr_new)
                    hr = float(hr_new)
                latest["_t_hr"] = float(t_now)

            # Update RR at a slower cadence (needs more data + is noisier)
            if (t_now - float(latest.get("_t_rr", 0.0))) >= RR_UPDATE_S:
                rr_new = estimate_rr_from_bvp(bvp_np[-int(fs_bvp_live * 60):], fs_bvp_live)
                if rr_new is not None:
                    if rr is not None and abs(rr_new - rr) > MAX_RR_JUMP_BRPM:
                        rr_new = rr + np.sign(rr_new - rr) * MAX_RR_JUMP_BRPM
                    latest["rr_brpm"] = float(rr_new)
                    rr = float(rr_new)
                latest["_t_rr"] = float(t_now)

        hr_s = f"HR: {hr:.1f} bpm" if hr is not None else "HR: (warming up)"
        rr_s = f"RR: {rr:.1f} brpm" if rr is not None else "RR: (need ~60s)"

        contact_s = "contact: YES" if contact else ("contact: NO" if contact is not None else "contact: (n/a)")
        stdv = eda_std_recent()
        eda_s = (f"EDA proxy: {latest.get('eda_proxy_hex', '(n/a)')} ({latest['eda_proxy']:.0f}) | std({min(EDA_STD_POINTS, len(eda_proxy_buf))}): {stdv:.1f}"
                 if (latest["eda_proxy"] is not None and stdv is not None)
                 else (f"EDA proxy: {latest.get('eda_proxy_hex', '(n/a)')} ({latest['eda_proxy']:.0f}) | std: (n/a)" if latest[
                                                                                   "eda_proxy"] is not None else "EDA proxy: (n/a)"))
        temp_s = ("T raw: (n/a)" if latest[
                                        "temp_raw"] is None else f"T raw: {latest['temp_raw']:.0f} | T*: {latest['temp_raw'] * TEMP_SCALE + TEMP_OFFSET:.2f}")
        button_s = f"Button Pressed: {int(latest.get('button_count', 0))}"

        # Show temperature in the temperature subplot
        txt_temp.set_text(temp_s)

        motion_s = "motion: OK" if mot else "motion: HIGH"
        dev_s = f"dev: {dev} @ {addr}" if addr else "dev: (not connected)"

        txt.set_text(
            f"{dev_s}\n"
            f"status: {status}\n"            
            f"BVP mode: {bvp_mode} | fs~{float(latest.get('bvp_fs') or FS_BVP):.1f} Hz\n"
            f"{contact_s} | {motion_s}\n"
            f"{button_s}\n"
            f"{eda_s} | {temp_s}\n"
            f"{hr_s} | {rr_s}\n"
            f"UTC: {iso_utc()}"
        )

        return (line_bvp, line_acc, line_temp, line_eda, txt, txt_temp)

    def on_close(_evt):
        stop_flag.set()
        try:
            ctrl.disconnect()
        except Exception:
            pass
        try:
            plt.close(fig)
        except Exception:
            pass

    fig.canvas.mpl_connect("close_event", on_close)

    _ani = FuncAnimation(fig, update, interval=250, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        try:
            ctrl.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
