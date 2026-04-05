#!/usr/bin/env python3
"""
Empatica E4 real-time viewer (reverse-engineering helper)

- Scan + Connect button (no hard-coded MAC)
- One window with subplots: BVP, ACC magnitude, Temperature, EDA
- Temperature (3ea6) low-level decode: 8x uint16 samples + uint32 counter (20B packet)
- EDA (3ea8) low-level decode: 8x uint16 samples + uint32 counter (20B packet)
- Contact = YES/NO derived from EDA with hysteresis + optional motion gate
- BVP decoded exclusively as 10*uint16, S9 high-nibble masked (4-bit counter)
- Save CSV button: snapshots all raw buffers to a timestamped CSV file

Install:
  pip install bleak numpy scipy matplotlib

Notes on EDA/contact:
- Off-wrist EDA often does NOT go to 0. Depending on front-end design it can float/saturate.
- Contact detection uses inverted hysteresis; tune EDA_ON_MAX / EDA_OFF_MIN on your device.
"""

import asyncio
import csv
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

# -------------------- HR/RR display smoothing --------------------
HR_UPDATE_S = 2.0
RR_UPDATE_S = 5.0
MAX_HR_JUMP_BPM = 10.0
MAX_RR_JUMP_BRPM = 10.0

# -------------------- BLE UUIDs (Empatica E4) --------------------
CTRL_CHAR   = "00003e71-0000-1000-8000-00805f9b34fb"
START_CMD   = b"\x02"
BVP_CHAR    = "00003ea1-0000-1000-8000-00805f9b34fb"
ACC_CHAR    = "00003ea3-0000-1000-8000-00805f9b34fb"
TEMP_CHAR   = "00003ea6-0000-1000-8000-00805f9b34fb"
EDA_CHAR    = "00003ea8-0000-1000-8000-00805f9b34fb"
BUTTON_CHAR = "00003eb2-0000-1000-8000-00805f9b34fb"

CONNECT_TIMEOUT = 25.0

# -------------------- BVP rates --------------------
# BVP is always decoded as 10 × uint16 (cleanS9 mode):
#   S9 high-byte low nibble is a 4-bit packet counter → masked before use.
FS_BVP = 64.0          # nominal sample rate (Hz)
BVP_SAMPLES_PER_PACKET = 10

# Dynamic BVP fs estimate (EMA from packet cadence)
BVP_FS_EMA_ALPHA = 0.15
_latest_bvp_pkt_t = None
_bvp_fs_est = FS_BVP
_bvp_time_next = None   # enforces uniform 64 Hz time base across packets

FS_ACC = 32.0
ACC_SAMPLES_PER_PACKET = 3

TEMP_SAMPLES_PER_PACKET = 8
EDA_SAMPLES_PER_PACKET  = 8

# -------------------- Temperature conversion --------------------
TEMP_SCALE  =  0.02      # MLX90614: raw in 0.02 K increments
TEMP_OFFSET = -273.15   # convert Kelvin to Celsius

# -------------------- Contact detection --------------------
EDA_ON_MAX   = 9200.0
EDA_OFF_MIN  = 9800.0
EDA_STD_POINTS = 8
EDA_STD_MIN    = 15.0

# -------------------- Buffers --------------------
BVP_SECONDS = 180
bvp_buf = deque(maxlen=int(BVP_SECONDS * FS_BVP))
t_bvp   = deque(maxlen=int(BVP_SECONDS * FS_BVP))

ACC_SECONDS = 60
acc_mag = deque(maxlen=int(ACC_SECONDS * FS_ACC))
t_acc   = deque(maxlen=int(ACC_SECONDS * FS_ACC))

TEMP_SECONDS = 10 * 60
temp_raw_buf = deque(maxlen=int(TEMP_SECONDS * 4))
t_temp       = deque(maxlen=int(TEMP_SECONDS * 4))

EDA_SECONDS = 10 * 60
eda_proxy_buf = deque(maxlen=int(EDA_SECONDS * 4))
t_eda         = deque(maxlen=int(EDA_SECONDS * 4))

latest = {
    "status": "idle",
    "device": None,
    "addr": None,
    "hr_bpm": None,
    "rr_brpm": None,
    "motion_ok": True,
    "contact": None,
    "eda_proxy": None,
    "temp_raw": None,
    "button_count": 0,
    "button_last_epoch": 0.0,
}

stop_flag = threading.Event()


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# -------------------- Decoders --------------------
def decode_bvp_u16x10(payload20: bytes) -> np.ndarray:
    """20 bytes → 10 samples as uint16 little-endian."""
    if len(payload20) != 20:
        return np.empty(0, dtype=np.float64)
    return np.frombuffer(payload20, dtype="<u2", count=10).astype(np.float64)


def clean_s9_nibble(v10_u16: np.ndarray) -> np.ndarray:
    """Mask S9 high-byte low nibble (suspected 4-bit packet counter)."""
    if v10_u16.size != 10:
        return v10_u16.astype(np.float64, copy=True)
    v = v10_u16.astype(np.int32, copy=True)
    s9   = int(v[9])
    s9_l = s9 & 0x00FF
    s9_h = (s9 >> 8) & 0xF0   # keep only high nibble; zero the counter nibble
    v[9] = s9_l | (s9_h << 8)
    return v.astype(np.float64)


def decode_acc_triples(payload18: bytes) -> np.ndarray:
    v = np.frombuffer(payload18, dtype="<i2").astype(np.float64)
    if v.size != 9:
        return np.empty((0, 3), dtype=np.float64)
    return v.reshape(3, 3)


def decode_u16x8_u32(payload20: bytes) -> tuple[np.ndarray, int] | None:
    """20 bytes: first 16 = 8×uint16; last 4 = uint32 counter."""
    if len(payload20) != 20:
        return None
    s = np.frombuffer(payload20[:16], dtype="<u2").astype(np.float64)
    c = int.from_bytes(payload20[16:20], "little", signed=False)
    return s, c


def dominant_u16_value(u16_vals: np.ndarray) -> float:
    vals = u16_vals.astype(np.int64)
    uniq, counts = np.unique(vals, return_counts=True)
    return float(uniq[int(np.argmax(counts))])


# -------------------- 3EA8 EDA proxy --------------------
def rot24_nibbles_12_from3(b3: bytes) -> int:
    if len(b3) < 3:
        return 0
    v = int.from_bytes(b3[:3], "big", signed=False) & 0xFFFFFF
    return ((v << 12) & 0xFFFFFF) | (v >> 12)


def eda_proxy_avg6_rot12(payload: bytes) -> float | None:
    if len(payload) < 18:
        return None
    vals = [rot24_nibbles_12_from3(payload[i:i + 3]) for i in range(0, 18, 3)]
    return float(int(round(sum(vals) / 6.0)))


# -------------------- Filters / estimation --------------------
def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> np.ndarray | None:
    if len(x) < int(fs * 3):
        return None
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def estimate_hr_from_bvp(bvp: np.ndarray, fs: float) -> float | None:
    if len(bvp) < int(fs * 12):
        return None
    x = bvp.astype(np.float64) - np.mean(bvp)
    xf = bandpass(x, fs, 0.7, 3.0, order=3)
    if xf is None:
        return None
    from scipy.signal import welch
    f, Pxx = welch(xf, fs=fs, nperseg=min(2048, len(xf)))
    band = (f >= 0.8) & (f <= 2.5)
    if not np.any(band):
        return None
    p_band = Pxx[band]
    i = int(np.argmax(p_band))
    bpm = float(f[band][i]) * 60.0
    prom = float(np.max(p_band) / (np.median(p_band) + 1e-12))
    if prom < 1.6:
        return None
    return float(bpm)


def estimate_rr_from_bvp(bvp: np.ndarray, fs: float) -> float | None:
    if len(bvp) < int(fs * 60):
        return None
    x = bvp.astype(np.float64) - np.mean(bvp)
    from scipy.signal import welch

    b_lp, a_lp = butter(3, 0.5 / (fs / 2), btype="low")
    riiv = filtfilt(b_lp, a_lp, x)

    xf = bandpass(x, fs, 0.7, 3.0, order=3)
    if xf is None:
        return None

    prom = max(np.std(xf) * 0.6, 1e-6)
    peaks, _ = find_peaks(xf, distance=max(1, int(0.4 * fs)), prominence=prom)
    peaks = peaks if len(peaks) >= 8 else None

    riav, riav_t = None, None
    if peaks is not None:
        w = int(0.25 * fs)
        amps, times = [], []
        for p in peaks:
            lo = max(0, p - w)
            hi = min(len(xf), p + w)
            amps.append(float(xf[p] - np.min(xf[lo:hi])))
            times.append(p / fs)
        if len(amps) >= 8:
            riav   = np.array(amps,  dtype=np.float64)
            riav_t = np.array(times, dtype=np.float64)

    def rr_psd(sig, sig_fs):
        f, Pxx = welch(sig, fs=sig_fs, nperseg=min(1024, len(sig)))
        band = (f >= 0.10) & (f <= 0.50)
        if not np.any(band):
            return None, 0.0
        p = Pxx[band]
        k = int(np.argmax(p))
        return float(f[band][k] * 60.0), float(np.max(p) / (np.median(p) + 1e-12))

    rr1, c1 = rr_psd(riiv, fs)
    rr2, c2 = (None, 0.0)
    if riav is not None and riav_t is not None:
        fs2 = 4.0
        t_u = np.arange(riav_t[0], riav_t[-1], 1.0 / fs2)
        if t_u.size >= 32:
            riav_u = np.interp(t_u, riav_t, riav) - np.mean(riav)
            rr2, c2 = rr_psd(riav_u, fs2)

    if rr1 is None and rr2 is None:
        return None
    if rr1 is not None and rr2 is not None:
        rr   = 0.5 * (rr1 + rr2) if abs(rr1 - rr2) <= 2.0 else (rr1 if c1 >= c2 else rr2)
        conf = max(c1, c2)
    else:
        rr   = rr1 if rr1 is not None else rr2
        conf = c1  if rr1 is not None else c2

    return float(rr) if conf >= 1.4 else None


def motion_ok_from_acc() -> bool:
    win = int(FS_ACC * 2.0)
    if len(acc_mag) < win:
        return True
    a = np.asarray(list(acc_mag)[-win:], dtype=np.float64)
    a = a[np.isfinite(a)]
    return len(a) < win // 2 or np.std(a) < 2000.0


def eda_std_recent() -> float | None:
    if len(eda_proxy_buf) < 2:
        return None
    n = min(EDA_STD_POINTS, len(eda_proxy_buf))
    x = np.asarray(list(eda_proxy_buf)[-n:], dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if x.size >= 2 else None


def contact_from_eda_proxy(prev: bool | None, proxy: float) -> bool:
    if prev is None:
        prev = False
    if prev:
        return not (proxy >= EDA_OFF_MIN)
    stdv = eda_std_recent()
    if stdv is None:
        return False
    return (proxy <= EDA_ON_MAX) and (stdv >= EDA_STD_MIN)


# -------------------- BLE controller thread --------------------
class BLEController(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.loop = None
        self.client: BleakClient | None = None
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

    def connect(self):
        if not self.loop:
            return
        return asyncio.run_coroutine_threadsafe(self._connect_flow(), self.loop)

    def disconnect(self):
        if not self.loop:
            return
        return asyncio.run_coroutine_threadsafe(self._disconnect_flow(), self.loop)

    async def _disconnect_flow(self):
        latest["status"] = "disconnecting"
        try:
            if self.client and self.client.is_connected:
                for char in (BVP_CHAR, ACC_CHAR, TEMP_CHAR, EDA_CHAR, BUTTON_CHAR):
                    try:
                        await self.client.stop_notify(char)
                    except Exception:
                        pass
                await self.client.disconnect()
        finally:
            self.client = None
            latest["status"] = "idle"
            latest["device"] = None
            latest["addr"] = None

    async def _connect_flow(self):
        if self.client and self.client.is_connected:
            return

        latest["status"] = "scanning"
        devices = await BleakScanner.discover(timeout=6.0)

        picked = None
        for d in devices:
            if "empatica" in (d.name or "").lower() or "e4" in (d.name or "").lower():
                picked = d
                break
        if picked is None and devices:
            picked = devices[0]
        if picked is None:
            latest["status"] = "no device found"
            return

        latest["device"] = picked.name
        latest["addr"]   = picked.address
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
                if len(data) != 20:
                    return
                try:
                    v10 = decode_bvp_u16x10(bytes(data))
                    if v10.size != 10:
                        return
                    samples = clean_s9_nibble(v10)
                except Exception:
                    return

                # Update dynamic fs estimate from packet cadence
                global _latest_bvp_pkt_t, _bvp_fs_est
                t_now = time.time()
                if _latest_bvp_pkt_t is not None:
                    dt_pkt = t_now - _latest_bvp_pkt_t
                    if 0.02 < dt_pkt < 2.0:
                        fs_pkt = BVP_SAMPLES_PER_PACKET / dt_pkt
                        _bvp_fs_est = (1.0 - BVP_FS_EMA_ALPHA) * _bvp_fs_est + BVP_FS_EMA_ALPHA * fs_pkt
                _latest_bvp_pkt_t = t_now
                latest["bvp_fs"] = float(_bvp_fs_est)

                # Enforce uniform 64 Hz time base to avoid packet-jitter artifacts
                global _bvp_time_next
                dt = 1.0 / FS_BVP
                if _bvp_time_next is None:
                    _bvp_time_next = t_now - (samples.size - 1) * dt
                for i in range(samples.size):
                    bvp_buf.append(float(samples[i]))
                    t_bvp.append(_bvp_time_next)
                    _bvp_time_next += dt

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
                        t_acc.append(t_now - (triples.shape[0] - 1 - i) * dt)
                        acc_mag.append(mag)

            def on_temp(_sender, data: bytearray):
                dec = decode_u16x8_u32(bytes(data))
                if dec is None:
                    return
                u16s, _ = dec
                t_now = time.time()
                dt = 2.0 / TEMP_SAMPLES_PER_PACKET   # ~2 s packet cadence
                for i in range(TEMP_SAMPLES_PER_PACKET):
                    t_temp.append(t_now - (TEMP_SAMPLES_PER_PACKET - 1 - i) * dt)
                    temp_raw_buf.append(float(u16s[i]))
                latest["temp_raw"] = float(np.median(u16s))

            def on_eda(_sender, data: bytearray):
                proxy = eda_proxy_avg6_rot12(bytes(data))
                if proxy is None:
                    return
                t_now = time.time()
                eda_proxy_buf.append(float(proxy))
                t_eda.append(t_now)
                latest["eda_proxy"] = float(proxy)
                latest["eda_proxy_hex"] = f"0x{int(proxy) & 0xFFFFFF:06x}"
                latest["contact"] = contact_from_eda_proxy(latest["contact"], float(proxy))

            def on_button(_sender, data: bytearray):
                bb = bytes(data)
                if len(bb) != 20 or bb[0] != 1:
                    return
                t_now = time.time()
                if t_now - float(latest.get("button_last_epoch", 0.0)) < 0.3:
                    return   # debounce
                latest["button_last_epoch"] = t_now
                latest["button_count"] = int(latest.get("button_count", 0)) + 1
                print(f"Button pressed: {latest['button_count']}")

            # ---- Subscribe ----
            await self.client.start_notify(BVP_CHAR,    on_bvp)
            await self.client.start_notify(ACC_CHAR,    on_acc)
            await self.client.start_notify(TEMP_CHAR,   on_temp)
            await self.client.start_notify(EDA_CHAR,    on_eda)
            await self.client.start_notify(BUTTON_CHAR, on_button)

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
            latest["addr"]   = None


# -------------------- UI / Plotting --------------------
PLOT_WINDOW_S = 30.0


def main():
    ctrl = BLEController()
    ctrl.start()

    fig = plt.figure(figsize=(12, 9))
    gs  = fig.add_gridspec(4, 1, hspace=0.35)

    ax_bvp  = fig.add_subplot(gs[0, 0])
    ax_acc  = fig.add_subplot(gs[1, 0], sharex=ax_bvp)
    ax_temp = fig.add_subplot(gs[2, 0], sharex=ax_bvp)
    ax_eda  = fig.add_subplot(gs[3, 0], sharex=ax_bvp)

    (line_bvp,)  = ax_bvp.plot([],  [], linestyle="-")
    (line_acc,)  = ax_acc.plot([],  [], linestyle="-")
    (line_temp,) = ax_temp.plot([], [], linestyle="-")
    (line_eda,)  = ax_eda.plot([],  [], linestyle="-")

    ax_bvp.set_title("Empatica E4 — BVP / ACC / TEMP / EDA")
    ax_bvp.set_ylabel("BVP (bandpassed)")
    ax_acc.set_ylabel("|acc|")
    ax_temp.set_ylabel("Temp (raw / T*)")
    ax_eda.set_ylabel("EDA proxy (avg6 rot12)")
    ax_eda.set_xlabel("Time (local)")

    for ax in (ax_bvp, ax_acc, ax_temp, ax_eda):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax_eda.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _pos: f"0x{int(x) & 0xFFFFFF:06x}")
    )
    fig.autofmt_xdate()

    txt      = ax_bvp.text(0.01, 0.95, "", transform=ax_bvp.transAxes, va="top")
    txt_temp = ax_temp.text(0.01, 0.90, "", transform=ax_temp.transAxes, va="top")

    # ---- Buttons ----
    ax_btn_conn = fig.add_axes([0.73, 0.93, 0.08, 0.05])
    ax_btn_disc = fig.add_axes([0.82, 0.93, 0.08, 0.05])
    ax_btn_save = fig.add_axes([0.91, 0.93, 0.08, 0.05])

    btn_conn = Button(ax_btn_conn, "Connect")
    btn_disc = Button(ax_btn_disc, "Disconnect")
    btn_save = Button(ax_btn_save, "Save CSV")

    def on_connect(_evt):
        ctrl.connect()

    def on_disconnect(_evt):
        ctrl.disconnect()

    def on_save(_evt):
        """Snapshot all raw buffers to a timestamped CSV file."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname  = f"e4_data_{ts_str}.csv"

        # Take thread-safe snapshots of all deques
        rows = []
        for t, v in zip(list(t_bvp), list(bvp_buf)):
            rows.append((t, datetime.fromtimestamp(t, tz=timezone.utc)
                         .isoformat(timespec="milliseconds"), "bvp", v))
        for t, v in zip(list(t_acc), list(acc_mag)):
            rows.append((t, datetime.fromtimestamp(t, tz=timezone.utc)
                         .isoformat(timespec="milliseconds"), "acc_mag", v))
        for t, v in zip(list(t_temp), list(temp_raw_buf)):
            rows.append((t, datetime.fromtimestamp(t, tz=timezone.utc)
                         .isoformat(timespec="milliseconds"), "temp_raw", v))
        for t, v in zip(list(t_eda), list(eda_proxy_buf)):
            rows.append((t, datetime.fromtimestamp(t, tz=timezone.utc)
                         .isoformat(timespec="milliseconds"), "eda_proxy", v))

        rows.sort(key=lambda r: r[0])   # chronological order

        try:
            with open(fname, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_epoch_s", "timestamp_utc", "stream", "value"])
                w.writerows(rows)
            msg = f"Saved {len(rows)} rows → {fname}"
        except Exception as exc:
            msg = f"Save failed: {exc}"

        print(msg)
        latest["_save_msg"] = msg

    btn_conn.on_clicked(on_connect)
    btn_disc.on_clicked(on_disconnect)
    btn_save.on_clicked(on_save)

    # ---- Animation update ----
    def update(_):
        if ctrl.last_error is not None:
            txt.set_text(f"BLE controller crashed: {ctrl.last_error}")
            return (line_bvp, line_acc, line_temp, line_eda, txt, txt_temp)

        latest["motion_ok"] = motion_ok_from_acc()
        t_now       = time.time()
        x_min_epoch = t_now - PLOT_WINDOW_S

        # BVP
        if len(t_bvp) >= 2 and len(bvp_buf) >= 2:
            t_np = np.asarray(t_bvp, dtype=np.float64)
            y_np = np.asarray(bvp_buf, dtype=np.float64)
            mask = t_np >= x_min_epoch
            if np.count_nonzero(mask) >= int(FS_BVP * 3):
                t_dt = [datetime.fromtimestamp(tt).astimezone() for tt in t_np[mask]]
                y = y_np[mask] - np.mean(y_np[mask])
                _yf = bandpass(y, FS_BVP, 0.7, 3.5, order=3)
                yf = _yf if _yf is not None else y
                line_bvp.set_data(t_dt, yf)
                ax_bvp.relim()
                ax_bvp.autoscale_view(scalex=False, scaley=True)

        # ACC
        if len(t_acc) >= 2 and len(acc_mag) >= 2:
            ta = np.asarray(t_acc, dtype=np.float64)
            aa = np.asarray(acc_mag, dtype=np.float64)
            mask = ta >= x_min_epoch
            if np.count_nonzero(mask) >= int(FS_ACC):
                line_acc.set_data([datetime.fromtimestamp(tt).astimezone() for tt in ta[mask]], aa[mask])
                ax_acc.relim()
                ax_acc.autoscale_view(scalex=False, scaley=True)

        # TEMP
        if len(t_temp) >= 2 and len(temp_raw_buf) >= 2:
            tl = np.asarray(t_temp, dtype=np.float64)
            yl = np.asarray(temp_raw_buf, dtype=np.float64)
            mask = tl >= x_min_epoch
            if np.any(mask):
                line_temp.set_data([datetime.fromtimestamp(tt).astimezone() for tt in tl[mask]], yl[mask])
                ax_temp.relim()
                ax_temp.autoscale_view(scalex=False, scaley=True)

        # EDA
        if len(t_eda) >= 2 and len(eda_proxy_buf) >= 2:
            tl = np.asarray(t_eda, dtype=np.float64)
            yl = np.asarray(eda_proxy_buf, dtype=np.float64)
            mask = tl >= x_min_epoch
            if np.any(mask):
                line_eda.set_data([datetime.fromtimestamp(tt).astimezone() for tt in tl[mask]], yl[mask])
                ax_eda.relim()
                ax_eda.autoscale_view(scalex=False, scaley=True)

        # Shared X limits
        try:
            ax_bvp.set_xlim(
                datetime.fromtimestamp(x_min_epoch).astimezone(),
                datetime.fromtimestamp(t_now).astimezone(),
            )
        except Exception:
            pass

        # HR / RR estimation
        hr  = latest["hr_bpm"]
        rr  = latest["rr_brpm"]
        fs_live = float(latest.get("bvp_fs") or FS_BVP)

        if len(bvp_buf) > int(fs_live * 10):
            bvp_np = np.asarray(bvp_buf, dtype=np.float64)

            if (t_now - float(latest.get("_t_hr", 0.0))) >= HR_UPDATE_S:
                hr_new = estimate_hr_from_bvp(bvp_np[-int(fs_live * 30):], fs_live)
                if hr_new is not None:
                    if hr is not None and abs(hr_new - hr) > MAX_HR_JUMP_BPM:
                        hr_new = hr + np.sign(hr_new - hr) * MAX_HR_JUMP_BPM
                    latest["hr_bpm"] = hr = float(hr_new)
                latest["_t_hr"] = float(t_now)

            if (t_now - float(latest.get("_t_rr", 0.0))) >= RR_UPDATE_S:
                rr_new = estimate_rr_from_bvp(bvp_np[-int(fs_live * 60):], fs_live)
                if rr_new is not None:
                    if rr is not None and abs(rr_new - rr) > MAX_RR_JUMP_BRPM:
                        rr_new = rr + np.sign(rr_new - rr) * MAX_RR_JUMP_BRPM
                    latest["rr_brpm"] = rr = float(rr_new)
                latest["_t_rr"] = float(t_now)

        contact = latest["contact"]
        mot     = latest["motion_ok"]
        status  = latest["status"]
        dev     = latest["device"]
        addr    = latest["addr"]

        hr_s      = f"HR: {hr:.1f} bpm"   if hr is not None else "HR: (warming up)"
        rr_s      = f"RR: {rr:.1f} brpm"  if rr is not None else "RR: (need ~60s)"
        contact_s = ("contact: YES" if contact else
                     ("contact: NO" if contact is not None else "contact: (n/a)"))
        motion_s  = "motion: OK" if mot else "motion: HIGH"
        dev_s     = f"dev: {dev} @ {addr}" if addr else "dev: (not connected)"

        stdv  = eda_std_recent()
        eda_s = (
            f"EDA proxy: {latest.get('eda_proxy_hex', '(n/a)')} ({latest['eda_proxy']:.0f})"
            f" | std({min(EDA_STD_POINTS, len(eda_proxy_buf))}): {stdv:.1f}"
            if (latest["eda_proxy"] is not None and stdv is not None)
            else f"EDA proxy: {latest.get('eda_proxy_hex', '(n/a)')}"
        )
        temp_s  = ("T raw: (n/a)" if latest["temp_raw"] is None
                   else f"T raw: {latest['temp_raw']:.0f} | T*: {latest['temp_raw'] * TEMP_SCALE + TEMP_OFFSET:.2f}")
        button_s = f"Button: {int(latest.get('button_count', 0))}"
        save_s   = latest.get("_save_msg", "")

        txt_temp.set_text(temp_s)
        txt.set_text(
            f"{dev_s}\n"
            f"status: {status} | fs~{fs_live:.1f} Hz\n"
            f"{contact_s} | {motion_s}\n"
            f"{button_s}\n"
            f"{eda_s}\n"
            f"{hr_s} | {rr_s}\n"
            f"UTC: {iso_utc()}"
            + (f"\n{save_s}" if save_s else "")
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
