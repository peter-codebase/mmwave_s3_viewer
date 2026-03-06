#!/usr/bin/env python3
"""
e4_decoder_plot.py — Plot-only viewer for Empatica E4 *_physical.csv captures

Input CSV format:
  epoch_s, iso_utc, stream, payload_len, payload_hex

Behavior (per your request):
- BVP uses 20 bytes as 10 * uint16 little-endian per packet, with Clean-S9 masking (S9_H[3:0] masked).
- epoch_s is BLE notify receive time.
- Do NOT write decoded CSV outputs; just plot time-aligned signals.
- Skip the first 10 seconds of plot due to initialization artifacts (configurable).
- TEMP: plot raw temperature (u16) and computed temp_c_est = raw * TEMP_SCALE + TEMP_OFFSET.

Decoding logic is based on the structure used in your e4_viewer.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_CSV = "peter_20260224_003455_physical.csv"
TEMP_SCALE = 0.02864
TEMP_OFFSET = -400.9

def decode_bvp_u16x10_cleanS9(payload20: bytes) -> np.ndarray:
    """20 bytes -> 10 samples, uint16 little-endian, with Clean-S9 masking.

    Observation: the low nibble of S9 high byte (S9_H[3:0]) behaves like a 4-bit counter.
    We mask it out and keep all 10 samples.
    """
    if len(payload20) != 20:
        return np.empty(0, dtype=np.float64)
    v = np.frombuffer(payload20, dtype="<u2", count=10).astype(np.int32)
    # Clean S9: mask low nibble of high byte
    s9 = int(v[9])
    s9_l = s9 & 0x00FF
    s9_h = (s9 >> 8) & 0x00FF
    s9_h_clean = s9_h & 0xF0
    v[9] = s9_l + (s9_h_clean << 8)
    return v.astype(np.float64)


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 3) -> Optional[np.ndarray]:
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        return filtfilt(b, a, x)
    except Exception:
        return None


def estimate_hr_psd(bvp: np.ndarray, fs: float) -> Optional[float]:
    """HR from Welch PSD peak in 0.8–2.5 Hz band (48–150 bpm) with a simple confidence gate."""
    if bvp.size < int(fs * 12):
        return None
    x = bvp.astype(np.float64)
    x = x - np.mean(x)
    xf = bandpass(x, fs, 0.7, 3.0, order=3)
    if xf is None:
        xf = x
    try:
        from scipy.signal import welch
        f, Pxx = welch(xf, fs=fs, nperseg=min(2048, len(xf)))
        band = (f >= 0.8) & (f <= 2.5)
        if not np.any(band):
            return None
        p = Pxx[band]
        fb = f[band]
        k = int(np.argmax(p))
        bpm = float(fb[k] * 60.0)
        prom = float(np.max(p) / (np.median(p) + 1e-12))
        if prom < 1.6:
            return None
        return bpm
    except Exception:
        return None


def estimate_rr_psd_from_bvp(bvp: np.ndarray, fs: float) -> Optional[float]:
    """RR estimate (breaths/min) from BVP using baseline modulation (RIIV) + Welch PSD.

    Windowed estimate: works best with >= 45–60s of relatively still data.
    Returns None if confidence is low.
    """
    if bvp.size < int(fs * 45):
        return None
    x = bvp.astype(np.float64)
    x = x - np.mean(x)
    try:
        from scipy.signal import butter, filtfilt, welch
        # RIIV: baseline modulation (<~0.5 Hz)
        b_lp, a_lp = butter(3, 0.5 / (fs / 2), btype="low")
        riiv = filtfilt(b_lp, a_lp, x)

        f, Pxx = welch(riiv, fs=fs, nperseg=min(2048, len(riiv)))
        band = (f >= 0.10) & (f <= 0.50)  # 6–30 brpm
        if not np.any(band):
            return None
        p = Pxx[band]
        fb = f[band]
        k = int(np.argmax(p))
        rr = float(fb[k] * 60.0)

        prom = float(np.max(p) / (np.median(p) + 1e-12))
        if prom < 1.4:
            return None
        return rr
    except Exception:
        return None

def decode_acc_triples(payload18: bytes) -> np.ndarray:
    if len(payload18) != 18:
        return np.empty((0, 3), dtype=np.float64)
    v = np.frombuffer(payload18, dtype="<i2").astype(np.float64)
    if v.size != 9:
        return np.empty((0, 3), dtype=np.float64)
    return v.reshape(3, 3)


def decode_u16x8_u32(payload20: bytes) -> Optional[Tuple[np.ndarray, int]]:
    if len(payload20) != 20:
        return None
    s = np.frombuffer(payload20[:16], dtype="<u2").astype(np.float64)
    c = int.from_bytes(payload20[16:20], "little", signed=False)
    return s, c


def rot24_nibbles_12_from3(b3: bytes) -> int:
    if len(b3) < 3:
        return 0
    v = int.from_bytes(b3[:3], "big", signed=False) & 0xFFFFFF
    return ((v << 12) & 0xFFFFFF) | (v >> 12)


def eda_proxy_avg6_rot12(payload: bytes) -> Optional[float]:
    if len(payload) < 18:
        return None
    vals = [rot24_nibbles_12_from3(payload[i:i + 3]) for i in range(0, 18, 3)]
    return float(int(round(sum(vals) / 6.0)))


def expand_packet_samples(
    pkt_times_s: np.ndarray,
    samples_list: List[np.ndarray],
    fs_nominal: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(pkt_times_s) != len(samples_list):
        raise ValueError("pkt_times_s and samples_list length mismatch")

    out_t: List[float] = []
    out_y: List[float] = []

    prev_t: Optional[float] = None
    for t_i, y in zip(pkt_times_s, samples_list):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(y.size)
        if n == 0 or not np.isfinite(t_i):
            prev_t = float(t_i) if np.isfinite(t_i) else prev_t
            continue

        if prev_t is None or (not np.isfinite(prev_t)) or (t_i <= prev_t):
            dt_pkt = n / float(fs_nominal)
        else:
            dt_pkt = float(t_i - prev_t)

        dt_s = dt_pkt / n
        start = float(t_i - (n - 1) * dt_s)
        for k in range(n):
            out_t.append(start + k * dt_s)
            out_y.append(float(y[k]))

        prev_t = float(t_i)

    return np.asarray(out_t, dtype=np.float64), np.asarray(out_y, dtype=np.float64)


def plot_aligned(
    bvp_t: Optional[np.ndarray], bvp_y: Optional[np.ndarray],
    rr_t: Optional[np.ndarray], rr_y: Optional[np.ndarray],
    acc_t: Optional[np.ndarray], acc_mag: Optional[np.ndarray],
    temp_t: Optional[np.ndarray], temp_raw: Optional[np.ndarray], temp_c: Optional[np.ndarray],
    eda_t: Optional[np.ndarray], eda_y: Optional[np.ndarray],
    skip_first_s: float,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    panels = []

    def add_panel(name: str, t: Optional[np.ndarray], y: Optional[np.ndarray]):
        if t is None or y is None:
            return
        if len(t) == 0 or len(y) == 0:
            return
        panels.append((name, t, y))

    add_panel("HR estimate (bpm)", bvp_t, bvp_y)
    add_panel("Respiration estimate (brpm)", rr_t, rr_y)
    add_panel("ACC magnitude", acc_t, acc_mag)
    add_panel("TEMP c_est", temp_t, temp_c)
    add_panel("EDA proxy (packet-level)", eda_t, eda_y)

    if not panels:
        raise RuntimeError("No streams found to plot.")

    xmin = min(float(np.nanmin(t)) for _, t, _ in panels)
    xmax = max(float(np.nanmax(t)) for _, t, _ in panels)
    xmin_plot = xmin + float(skip_first_s)

    n = len(panels)
    fig_h = max(6, 2.2 * n)
    fig, axes = plt.subplots(n, 1, figsize=(12, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, t, y) in zip(axes, panels):
        mask = (t >= xmin_plot) & np.isfinite(t) & np.isfinite(y)
        ax.plot(t[mask], y[mask])
        ax.set_title(name)
        ax.grid(True)

    axes[-1].set_xlim([xmin_plot, xmax])
    axes[-1].set_xlabel("epoch_s (aligned)")

    try:
        dt0 = datetime.fromtimestamp(xmin_plot, tz=timezone.utc)
        fig.suptitle(f"{title_prefix} (plot start: {dt0.isoformat(timespec='seconds')}, skip_first_s={skip_first_s})")
        fig.subplots_adjust(top=0.93)
    except Exception:
        fig.suptitle(f"{title_prefix} (skip_first_s={skip_first_s})")
        fig.subplots_adjust(top=0.93)

    plt.tight_layout()
    plt.show()


@dataclass
class Options:
    fs_bvp_nominal: float = 64.0
    fs_acc_nominal: float = 32.0
    temp_scale: float = 1.0
    temp_offset: float = 0.0
    skip_first_s: float = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv",nargs="?",default=DEFAULT_CSV,help="Path to *_physical.csv",)
    ap.add_argument("--temp-scale", type=float, default=TEMP_SCALE, help="temp_c_est = raw * scale + offset")
    ap.add_argument("--temp-offset", type=float, default=TEMP_OFFSET)

    ap.add_argument("--skip-first-s", type=float, default=10.0, help="Skip the first N seconds due to init artifacts")
    ap.add_argument("--title", default="E4 physical capture", help="Figure title prefix")
    args = ap.parse_args()

    opts = Options(temp_scale=args.temp_scale, temp_offset=args.temp_offset, skip_first_s=args.skip_first_s)

    df = pd.read_csv(args.in_csv)
    required = {"epoch_s", "iso_utc", "stream", "payload_len", "payload_hex"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")

    df = df.copy()
    df["epoch_s"] = pd.to_numeric(df["epoch_s"], errors="coerce")
    df = df[np.isfinite(df["epoch_s"])]
    df["stream"] = df["stream"].astype(str).str.lower()
    df["payload_hex"] = df["payload_hex"].astype(str)

    def hx_to_bytes(hx: str) -> bytes:
        hx = hx.strip()
        if hx.startswith(("0x", "0X")):
            hx = hx[2:]
        return bytes.fromhex(hx)

    df["payload_bytes"] = df["payload_hex"].map(hx_to_bytes)

    bvp_t = bvp_y = None
    rr_t = rr_y = None
    acc_t = acc_mag = None
    temp_t = temp_raw = temp_c = None
    eda_t = eda_y = None

    bvp_df = df[df["stream"] == "bvp"].sort_values("epoch_s")
    if not bvp_df.empty:
        pkt_t = bvp_df["epoch_s"].to_numpy(dtype=np.float64)
        samples = [decode_bvp_u16x10_cleanS9(bb) for bb in bvp_df["payload_bytes"].tolist()]
        bvp_t_s, bvp_y_raw = expand_packet_samples(pkt_t, samples, fs_nominal=opts.fs_bvp_nominal)

        # Convert raw BVP into an HR time series (bpm) using a sliding Welch-PSD estimate.
        # Window: 30s; step: 2s (offline plot).
        fs = float(opts.fs_bvp_nominal)
        win_s = 30.0
        step_s = 2.0
        n_win = int(win_s * fs)
        n_step = max(1, int(step_s * fs))

        hr_t = []
        hr_y = []

        if bvp_t_s.size >= n_win:
            for end in range(n_win, bvp_y_raw.size + 1, n_step):
                seg = bvp_y_raw[end - n_win:end]
                bpm = estimate_hr_psd(seg, fs)
                hr_t.append(float(bvp_t_s[end - 1]))
                hr_y.append(np.nan if bpm is None else float(bpm))

        # RR estimate (brpm) from BVP using RIIV+PSD
        win_rr_s = 60.0
        step_rr_s = 5.0
        n_win_rr = int(win_rr_s * fs)
        n_step_rr = max(1, int(step_rr_s * fs))

        rr_tt = []
        rr_yy = []
        if bvp_t_s.size >= n_win_rr:
            for end in range(n_win_rr, bvp_y_raw.size + 1, n_step_rr):
                seg = bvp_y_raw[end - n_win_rr:end]
                brpm = estimate_rr_psd_from_bvp(seg, fs)
                rr_tt.append(float(bvp_t_s[end - 1]))
                rr_yy.append(np.nan if brpm is None else float(brpm))

        bvp_t = np.asarray(hr_t, dtype=np.float64)
        bvp_y = np.asarray(hr_y, dtype=np.float64)
        rr_t = np.asarray(rr_tt, dtype=np.float64)
        rr_y = np.asarray(rr_yy, dtype=np.float64)

    acc_df = df[df["stream"] == "acc"].sort_values("epoch_s")
    if not acc_df.empty:
        pkt_t = acc_df["epoch_s"].to_numpy(dtype=np.float64)
        triples_list = [decode_acc_triples(bb[:18]) for bb in acc_df["payload_bytes"].tolist()]
        mag_list = []
        for tri in triples_list:
            if tri.size == 0:
                mag_list.append(np.empty(0))
            else:
                mag_list.append(np.sqrt((tri[:, 0] ** 2) + (tri[:, 1] ** 2) + (tri[:, 2] ** 2)))
        acc_t, acc_mag = expand_packet_samples(pkt_t, mag_list, fs_nominal=opts.fs_acc_nominal)

    temp_df = df[df["stream"] == "temp"].sort_values("epoch_s")
    if not temp_df.empty:
        pkt_t = temp_df["epoch_s"].to_numpy(dtype=np.float64)
        samples_list: List[np.ndarray] = []
        for bb in temp_df["payload_bytes"].tolist():
            dec = decode_u16x8_u32(bb)
            if dec is None:
                samples_list.append(np.empty(0))
            else:
                u16s, _ctr = dec
                samples_list.append(u16s)

        temp_t, temp_raw = expand_packet_samples(pkt_t, samples_list, fs_nominal=4.0)
        temp_c = temp_raw * float(opts.temp_scale) + float(opts.temp_offset)

    eda_df = df[df["stream"] == "eda"].sort_values("epoch_s")
    if not eda_df.empty:
        eda_t = eda_df["epoch_s"].to_numpy(dtype=np.float64)
        eda_vals = []
        for bb in eda_df["payload_bytes"].tolist():
            p = eda_proxy_avg6_rot12(bb)
            eda_vals.append(np.nan if p is None else float(p))
        eda_y = np.asarray(eda_vals, dtype=np.float64)

    plot_aligned(
        bvp_t=bvp_t, bvp_y=bvp_y,
        rr_t=rr_t, rr_y=rr_y,
        acc_t=acc_t, acc_mag=acc_mag,
        temp_t=temp_t, temp_raw=temp_raw, temp_c=temp_c,
        eda_t=eda_t, eda_y=eda_y,
        skip_first_s=opts.skip_first_s,
        title_prefix=args.title,
    )


if __name__ == "__main__":
    main()
