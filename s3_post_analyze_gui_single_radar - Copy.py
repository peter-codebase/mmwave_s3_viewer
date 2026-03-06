#!/usr/bin/env python3
"""
s3_post_analyze_gui_voice_drift_latency.py

GUI goals (Phase 1):
- Connect to S3 using AWS creds from a chosen .env (default: ./.env)
- Browse IDs and dates under an S3 dataset organized as:
    (variant A) s3://mmwave-raw-data/{id}-YYYYMMDD/voice_drift/
    (variant B) s3://<bucket>/mmwave-raw-data/{id}-YYYYMMDD/voice_drift/
- For a selected (ID, date), analyze voice_drift/:
    * read the single _DONE.json for date/time metadata
    * for each .wav in voice_drift/, compute:
        filename, date, time, duration_s, latency_s
      where latency is computed from the wav itself:
        latency = speech_onset_time - beep_end_time
      (beep is recorded at the beginning of each wav)

Dependencies:
  pip install boto3 python-dotenv numpy

Notes:
- Date and time are sourced from voice_drift/_DONE.json:
    date = day (YYYYMMDD)
    time = local time derived from created_at_unix
- This implementation is Windows-friendly (Tkinter).
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import tkinter as tk

# Local helper for consistent S3 key layout
from s3_layout import S3Layout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import boto3
import numpy as np

# ---- Radar config (from raw_collection_ui_s3.py) ----
RANGE_RESOLUTION_M = 0.15  # meters per range bin
MAX_RANGE_M = 4.8          # meters
MAX_SPEED_M_S = 2.45       # m/s
SPEED_RESOLUTION_M_S = 0.2 # m/s
CENTER_FREQUENCY_HZ = 60_750_000_000


# ---- Optional deps for tagging viewer ----
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageTk, ImageDraw  # type: ignore
except Exception:
    Image = None
    ImageTk = None

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # type: ignore
except Exception:
    plt = None
    FigureCanvasTkAgg = None

try:
    import pygame  # type: ignore
except Exception:
    pygame = None

import ast
import tempfile


def _read_wav_as_pcm16_bytes(wav_path: Path):
    """Read a WAV file and return (channels, sampwidth_bytes=2, samplerate, pcm16_bytes).

    Supports:
      - PCM integer WAVs readable by the standard wave module
      - IEEE float 32-bit WAV (format tag 3)
    Additionally applies **auto-normalization** to raise quiet recordings while avoiding clipping.

    This is intentionally dependency-light (no ffmpeg/soundfile required).
    """
    import struct
    import wave

    def _auto_normalize_int16(a16: np.ndarray) -> np.ndarray:
        """Scale int16 audio so peak is ~95% full-scale (if non-silent)."""
        if a16.size == 0:
            return a16.astype("<i2", copy=False)
        peak = float(np.max(np.abs(a16.astype(np.int32))))
        if peak <= 0.0:
            return a16.astype("<i2", copy=False)
        target = 0.95 * 32767.0
        scale = target / peak
        y = np.clip(a16.astype(np.float32) * scale, -32768.0, 32767.0).astype("<i2")
        return y

    # First try the standard wave module (works for PCM integer)
    try:
        with wave.open(str(wav_path), "rb") as wf:
            ch = int(wf.getnchannels())
            sw = int(wf.getsampwidth())
            sr = int(wf.getframerate())
            nframes = int(wf.getnframes())
            raw = wf.readframes(nframes)

        # Convert to 16-bit if needed, then normalize
        if sw == 2:
            a16 = np.frombuffer(raw, dtype="<i2")
            a16 = _auto_normalize_int16(a16)
            return ch, 2, sr, a16.tobytes()

        if sw == 1:
            # unsigned 8-bit PCM -> signed 16-bit
            a = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
            a = (a - 128) << 8
            a16 = _auto_normalize_int16(a.astype("<i2"))
            return ch, 2, sr, a16.tobytes()

        if sw == 3:
            # 24-bit little-endian -> int32 -> int16
            b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            x = (b[:, 0].astype(np.int32) |
                 (b[:, 1].astype(np.int32) << 8) |
                 (b[:, 2].astype(np.int32) << 16))
            # sign extend
            sign = x & 0x800000
            x = x - (sign << 1)
            x16 = np.clip(x >> 8, -32768, 32767).astype("<i2")
            x16 = _auto_normalize_int16(x16)
            return ch, 2, sr, x16.tobytes()

        if sw == 4:
            # assume signed 32-bit PCM
            x = np.frombuffer(raw, dtype="<i4")
            x16 = np.clip(x >> 16, -32768, 32767).astype("<i2")
            x16 = _auto_normalize_int16(x16)
            return ch, 2, sr, x16.tobytes()
    except Exception:
        pass

    # Fallback: parse RIFF and support IEEE float 32-bit WAV (format tag 3)
    data = wav_path.read_bytes()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV (RIFF/WAVE) file")

    pos = 12
    fmt = None
    pcm = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        csz = struct.unpack_from("<I", data, pos + 4)[0]
        cstart = pos + 8
        cend = cstart + csz
        if cend > len(data):
            break

        if cid == b"fmt ":
            audio_format, ch, sr = struct.unpack_from("<HHI", data, cstart)
            bits_per_sample = struct.unpack_from("<H", data, cstart + 14)[0]
            fmt = (int(audio_format), int(ch), int(sr), int(bits_per_sample))
        elif cid == b"data":
            pcm = data[cstart:cend]

        # chunks are padded to even sizes
        pos = cend + (csz % 2)

    if fmt is None or pcm is None:
        raise ValueError("WAV missing fmt or data chunk")

    audio_format, ch, sr, bps = fmt

    if audio_format == 3 and bps == 32:
        # IEEE float32 in [-1, 1] nominally, but often very quiet.
        x = np.frombuffer(pcm, dtype="<f4")
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0.0:
            x = (x / peak) * 0.95  # auto-normalize to 95% full scale
        x = np.clip(x, -1.0, 1.0)
        x16 = (x * 32767.0).astype("<i2")
        # (x already normalized, but keep consistent)
        x16 = _auto_normalize_int16(x16)
        return int(ch), 2, int(sr), x16.tobytes()

    raise ValueError(f"Unsupported WAV format tag={audio_format}, bits={bps}")
    # ---- Optional deps for stronger latency detection (same approach as voice_latency_postproc.py) ----

try:
    import soundfile as sf  # type: ignore
except Exception:
    sf = None

try:
    import webrtcvad  # type: ignore
except Exception:
    webrtcvad = None

try:
    from scipy.signal import resample_poly  # type: ignore
except Exception:
    resample_poly = None


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)


def _frame_audio(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, int, int]:
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop_len = int(round(sr * hop_ms / 1000.0))
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame/hop too small")
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size), mode="constant")
    n_frames = 1 + (x.size - frame_len) // hop_len
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0]),
        writeable=False,
    ).copy()
    return frames, frame_len, hop_len


def _beep_band_energy_ratio(frames: np.ndarray, sr: int, beep_freq_hz: float, band_hz: float) -> tuple[
    np.ndarray, np.ndarray]:
    n = frames.shape[1]
    win = _hann(n)
    X = np.fft.rfft(frames * win[None, :], axis=1)
    P = (np.abs(X) ** 2).astype(np.float64)

    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    f0 = float(beep_freq_hz)
    bw = float(band_hz)
    lo = max(0.0, f0 - bw)
    hi = f0 + bw

    band_mask = (freqs >= lo) & (freqs <= hi)
    band_E = P[:, band_mask].sum(axis=1) + 1e-12
    tot_E = P.sum(axis=1) + 1e-12
    ratio = band_E / tot_E
    return ratio, tot_E


def _detect_beep_end(
        x: np.ndarray,
        sr: int,
        beep_freq_hz: float = 1500.0,
        band_hz: float = 150.0,
        frame_ms: float = 20.0,
        hop_ms: float = 10.0,
        search_s: float = 1.5,
        ratio_th: float = 0.35,
        energy_th_db: float = -35.0,
) -> tuple[float, int, str]:
    frames, frame_len, hop_len = _frame_audio(x, sr, frame_ms, hop_ms)
    n_search_frames = int(max(1, round((search_s * sr - frame_len) / hop_len))) + 1
    frames_s = frames[: min(frames.shape[0], n_search_frames)]

    ratio, totE = _beep_band_energy_ratio(frames_s, sr, beep_freq_hz, band_hz)
    totE_db = 10.0 * np.log10(totE + 1e-12)
    totE_db_rel = totE_db - float(np.max(totE_db))

    beep_mask = (ratio >= float(ratio_th)) & (totE_db_rel >= float(energy_th_db))
    if not np.any(beep_mask):
        return 0.0, 0, "beep_not_found"

    last = int(np.where(beep_mask)[0][-1])
    beep_end_s = (last * hop_len + frame_len) / float(sr)
    return float(beep_end_s), 1, "ok"


def _rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)


def _detect_speech_onset_rms(
        x: np.ndarray,
        sr: int,
        start_s: float,
        frame_ms: float = 20.0,
        hop_ms: float = 10.0,
        noise_est_s: float = 0.6,
        consec: int = 3,
        snr_mult: float = 3.0,
        abs_floor: float = 2e-4,
) -> tuple[float, int, str]:
    start_i = int(max(0, round(start_s * sr)))
    x2 = x[start_i:] if start_i < x.size else np.array([], dtype=np.float32)
    if x2.size == 0:
        return float(start_s), 0, "no_audio_after_start"

    frames, frame_len, hop_len = _frame_audio(x2, sr, frame_ms, hop_ms)
    r = _rms(frames)

    n_noise_frames = int(max(1, round((noise_est_s * 1000.0 - frame_ms) / hop_ms))) + 1
    r_noise = r[: min(len(r), n_noise_frames)]
    if r_noise.size == 0:
        noise = float(np.median(r))
    else:
        cut = np.quantile(r_noise, 0.80)
        noise = float(np.median(r_noise[r_noise <= cut])) if np.any(r_noise <= cut) else float(np.median(r_noise))

    thr = max(float(abs_floor), float(noise) * float(snr_mult))

    above = r > thr
    if not np.any(above):
        return float(start_s), 0, f"speech_not_found(thr={thr:.6f},noise={noise:.6f})"

    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= int(consec):
            onset_frame = i - (int(consec) - 1)
            onset_s = start_s + (onset_frame * hop_len) / float(sr)
            return float(onset_s), 1, f"ok(thr={thr:.6f},noise={noise:.6f})"

    return float(start_s), 0, "speech_not_found_consec"


def _pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


def _detect_speech_onset_vad(
        x: np.ndarray,
        sr: int,
        start_s: float,
        vad_mode: int = 2,
        frame_ms: int = 20,
        hop_ms: int = 10,
        consec: int = 3,
) -> tuple[float, int, str]:
    if webrtcvad is None:
        return float(start_s), 0, "webrtcvad_not_available"
    if sr not in (8000, 16000, 32000, 48000):
        return float(start_s), 0, f"vad_bad_sr({sr})"
    if frame_ms not in (10, 20, 30):
        return float(start_s), 0, f"vad_bad_frame_ms({frame_ms})"

    vad = webrtcvad.Vad(int(vad_mode))

    start_i = int(max(0, round(start_s * sr)))
    x2 = x[start_i:] if start_i < x.size else np.array([], dtype=np.float32)
    if x2.size == 0:
        return float(start_s), 0, "no_audio_after_start"

    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    if x2.size < frame_len:
        return float(start_s), 0, "too_short_for_vad"

    run = 0
    i = 0
    while i + frame_len <= x2.size:
        frame = x2[i:i + frame_len]
        voiced = vad.is_speech(_pcm16(frame), sr)
        run = run + 1 if voiced else 0
        if run >= int(consec):
            onset_s = start_s + (i - (consec - 1) * hop_len) / float(sr)
            return float(max(start_s, onset_s)), 1, "ok_vad"
        i += hop_len

    return float(start_s), 0, "speech_not_found_vad"


def _maybe_resample(x: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int, str]:
    if target_sr <= 0 or target_sr == sr:
        return x, sr, "no_resample"
    if resample_poly is None:
        return x, sr, "scipy_missing_no_resample"
    from math import gcd
    g = gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    y = resample_poly(x, up, down).astype(np.float32)
    return y, target_sr, f"resampled({sr}->{target_sr})"


def _detect_speech_onset_bandflux(
        x: np.ndarray,
        sr: int,
        start_s: float,
        frame_ms: float = 20.0,
        hop_ms: float = 10.0,
        band_lo: float = 250.0,
        band_hi: float = 3500.0,
        flux_z: float = 3.5,
        consec: int = 3,
) -> tuple[float, int, str]:
    """Fallback onset: spectral flux in speech band. Useful when AGC flattens amplitude."""
    start_i = int(max(0, round(start_s * sr)))
    x2 = x[start_i:] if start_i < x.size else np.array([], dtype=np.float32)
    if x2.size == 0:
        return float(start_s), 0, "no_audio_after_start"

    frames, frame_len, hop_len = _frame_audio(x2, sr, frame_ms, hop_ms)
    win = _hann(frame_len)
    X = np.fft.rfft(frames * win[None, :], axis=1)
    Mag = np.abs(X).astype(np.float64)

    freqs = np.fft.rfftfreq(frame_len, d=1.0 / float(sr))
    band = (freqs >= band_lo) & (freqs <= band_hi)
    Mb = Mag[:, band]

    # spectral flux (positive changes)
    d = np.diff(Mb, axis=0)
    d[d < 0] = 0
    flux = np.sqrt(np.sum(d * d, axis=1) + 1e-12)

    # robust stats from first ~0.8s
    n0 = min(len(flux), int(round(0.8 / (hop_ms / 1000.0))))
    base = flux[:n0] if n0 > 10 else flux
    mu = float(np.median(base))
    mad = float(np.median(np.abs(base - mu))) + 1e-12
    z = (flux - mu) / (1.4826 * mad)

    above = z > float(flux_z)
    if not np.any(above):
        return float(start_s), 0, f"flux_not_found(z_th={flux_z:.2f})"

    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= int(consec):
            onset_frame = i - (int(consec) - 1) + 1  # +1 because flux is diffed (N-1)
            onset_s = start_s + (onset_frame * hop_len) / float(sr)
            return float(onset_s), 1, "ok_flux"

    return float(start_s), 0, "flux_not_found_consec"


def _estimate_beep_freq(
        x: np.ndarray,
        sr: int,
        probe_s: float = 0.35,
        f_lo: float = 300.0,
        f_hi: float = 4000.0,
) -> float | None:
    """Estimate dominant tone frequency in the initial probe window (for beep auto-detect)."""
    n = int(max(256, round(probe_s * sr)))
    n = min(n, x.size)
    if n <= 256:
        return None
    seg = x[:n].astype(np.float32)
    win = _hann(seg.size)
    X = np.fft.rfft(seg * win)
    P = (np.abs(X) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / float(sr))
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return None
    f = freqs[mask]
    p = P[mask]
    if p.size == 0:
        return None
    idx = int(np.argmax(p))
    return float(f[idx])


def estimate_latency_beep_to_speech(x: np.ndarray, sr: int) -> tuple[float | None, float | None, float | None, str]:
    """
    Robust latency estimator (beep-in-wav) for voice_drift recordings.

    Steps:
      1) Resample to 16 kHz when possible (for VAD + consistency).
      2) Detect beep end using narrowband energy ratio. If not found at 1500 Hz,
         auto-estimate the dominant tone in the first 0.35s and retry with looser thresholds.
      3) Detect speech onset after beep end using:
           WebRTC VAD -> RMS threshold -> spectral flux fallback.
    Returns: (latency_s, beep_end_s, speech_onset_s, notes)
    """
    if sr <= 0 or x.size == 0:
        return None, None, None, "empty_audio"

    x = np.clip(x.astype(np.float32), -1.0, 1.0)

    # Resample for VAD + consistent analysis
    target_sr = 16000
    x2, sr2, res_note = _maybe_resample(x, sr, target_sr)

    # Beep end detection: default @ 1500 Hz
    beep_end_s, beep_conf, beep_note = _detect_beep_end(
        x2, sr2,
        beep_freq_hz=1500.0,
        band_hz=150.0,
        frame_ms=20.0,
        hop_ms=10.0,
        search_s=1.5,
        ratio_th=0.35,
        energy_th_db=-35.0,
    )

    auto_note = "auto_beep_skip"
    if beep_conf == 0:
        f0 = _estimate_beep_freq(x2, sr2, probe_s=0.35, f_lo=300.0, f_hi=4000.0)
        if f0 is None:
            auto_note = "auto_beep_no_f0"
        else:
            auto_note = f"auto_beep_f0={f0:.1f}"
            beep_end_s2, beep_conf2, beep_note2 = _detect_beep_end(
                x2, sr2,
                beep_freq_hz=float(f0),
                band_hz=250.0,
                frame_ms=20.0,
                hop_ms=10.0,
                search_s=1.5,
                ratio_th=0.20,
                energy_th_db=-45.0,
            )
            if beep_conf2 == 1:
                beep_end_s, beep_conf, beep_note = float(beep_end_s2), 1, "ok_auto"
            else:
                beep_note = beep_note2 or "beep_not_found_auto"

    # If still not found, fall back to treating start of WAV as time-zero (no beep reference).
    # This matches your request: assume beginning of the wav as the "prompt end" and find speech via VAD/RMS/flux.
    if beep_conf == 0:
        beep_end_s = 0.0
        beep_note = "beep_missing_use_wav_start"

    # Speech onset detection
    onset_s, speech_conf, speech_note = _detect_speech_onset_vad(
        x2, sr2,
        start_s=float(beep_end_s),
        vad_mode=2,
        frame_ms=20,
        hop_ms=10,
        consec=3,
    )

    if speech_conf == 0:
        onset_s, speech_conf, speech_note = _detect_speech_onset_rms(
            x2, sr2,
            start_s=float(beep_end_s),
            frame_ms=20.0,
            hop_ms=10.0,
            noise_est_s=0.6,
            consec=3,
            snr_mult=2.2,
            abs_floor=1.5e-4,
        )

    if speech_conf == 0:
        onset_s, speech_conf, speech_note = _detect_speech_onset_bandflux(
            x2, sr2,
            start_s=float(beep_end_s),
            frame_ms=20.0,
            hop_ms=10.0,
            band_lo=250.0,
            band_hi=3500.0,
            flux_z=3.5,
            consec=3,
        )

    notes = ";".join([res_note, beep_note, auto_note, speech_note])

    # Special case: some recordings start with immediate speech and no beep.
    # If all detectors fail but there is clear energy at t=0, treat onset as 0.0 and latency as 0.0.
    if speech_conf == 0 and beep_note == "beep_missing_use_wav_start":
        n0 = int(0.10 * sr2)
        if n0 > 0:
            rms0 = float(np.sqrt(np.mean(x2[:n0] * x2[:n0])))
            if rms0 > 1.5e-4:
                onset_s, speech_conf, speech_note = 0.0, 1, "speech_from_start_assumed"
                notes = ";".join([res_note, beep_note, auto_note, speech_note])

    if speech_conf == 0:
        return None, float(beep_end_s), None, notes

    latency_s = float(onset_s - float(beep_end_s))
    return float(max(0.0, latency_s)), float(beep_end_s), float(onset_s), notes


from botocore.exceptions import BotoCoreError, ClientError
from dotenv import dotenv_values

ROOT_PREFIX_DEFAULT = "sleep/"
# ---------------- AWS helpers ----------------

@dataclass
class AwsConfig:
    access_key: str | None = None
    secret_key: str | None = None
    session_token: str | None = None
    region: str | None = None
    bucket: str | None = None


def load_env(env_path: Path) -> AwsConfig:
    vals = dotenv_values(str(env_path)) if env_path.exists() else {}

    def pick(*keys: str):
        # env vars override .env
        for k in keys:
            v = os.environ.get(k)
            if v:
                return v
        for k in keys:
            v = vals.get(k)
            if v:
                return str(v)
        return None

    return AwsConfig(
        access_key=pick("AWS_ACCESS_KEY_ID"),
        secret_key=pick("AWS_SECRET_ACCESS_KEY"),
        session_token=pick("AWS_SESSION_TOKEN"),
        region=pick("AWS_DEFAULT_REGION", "AWS_REGION"),
        bucket=pick("S3_BUCKET", "AWS_S3_BUCKET"),
    )


def make_s3_client(cfg: AwsConfig):
    session = boto3.session.Session(region_name=cfg.region)
    kwargs = {}
    if cfg.access_key and cfg.secret_key:
        kwargs["aws_access_key_id"] = cfg.access_key
        kwargs["aws_secret_access_key"] = cfg.secret_key
        if cfg.session_token:
            kwargs["aws_session_token"] = cfg.session_token
    return session.client("s3", **kwargs)


def list_common_prefixes(s3, bucket: str, prefix: str) -> list[str]:
    """List 'folders' directly under prefix using Delimiter='/'."""
    prefixes: list[str] = []
    token = None
    while True:
        args = dict(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1000)
        if token:
            args["ContinuationToken"] = token
        resp = s3.list_objects_v2(**args)
        for cp in resp.get("CommonPrefixes", []) or []:
            p = cp.get("Prefix")
            if p:
                prefixes.append(p)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return prefixes


def list_objects(s3, bucket: str, prefix: str, max_items: int = 5000) -> list[dict]:
    """List all objects under prefix (recursive), up to max_items."""
    objs: list[dict] = []
    token = None
    while True:
        args = dict(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        if token:
            args["ContinuationToken"] = token
        resp = s3.list_objects_v2(**args)
        for o in resp.get("Contents", []) or []:
            k = o.get("Key")
            if k and k != prefix:
                objs.append(o)
                if len(objs) >= max_items:
                    return objs
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return objs


def detect_root_prefix(s3, bucket: str) -> str:
    """Detect a reasonable dataset root prefix inside the bucket.

    Newer datasets (sleep) are typically organized as:
        s3://<bucket>/sleep/<user>/<user>_YYYYMMDD/<session>/

    Older datasets may have:
        s3://<bucket>/mmwave-raw-data/<user>/<user>-YYYYMMDD/...

    We prefer 'sleep/' when present.
    """
    tops = list_common_prefixes(s3, bucket, "")
    for cand in ("sleep/", "mmwave-raw-data/", ""):
        if cand == "":
            return ""
        if cand in tops:
            return cand
    return ""





def prefix_exists(s3, bucket: str, prefix: str) -> bool:
    """Return True if any object exists under prefix (or prefix itself as a key)."""
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return bool(resp.get("KeyCount", 0)) or bool(resp.get("Contents"))
    except Exception:
        return False
def object_exists(s3, bucket: str, key: str) -> bool:
    """Return True if the exact object key exists (HEAD request)."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        return False
    except Exception:
        return False


def download_text_object(s3, bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8", errors="replace")


def upload_text_object(s3, bucket: str, key: str, text: str, content_type: str = "text/csv") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)


def list_user_ids(layout: S3Layout, s3, bucket: str) -> list[str]:
    """List user IDs under the canonical layout: <root>/<user>/..."""
    prefixes = list_common_prefixes(s3, bucket, layout.root_prefix or "")
    users: list[str] = []
    for p in prefixes:
        name = p.rstrip("/").split("/")[-1]
        # Skip day-folders like <user>-YYYYMMDD
        if re.match(r".*[-_]\d{8}$", name):
            continue
        if not name:
            continue
        # Skip obvious non-user folders
        if name.lower() in ("voice_drift",):
            continue
        users.append(name)
    return sorted({u for u in users})

def list_dates_for_user(layout: S3Layout, s3, bucket: str, user_id: str) -> list[str]:
    """List YYYYMMDD for a user under canonical <root>/<user>/<user>-YYYYMMDD/."""
    dates: set[str] = set()
    user_root = layout.user_root(user_id)
    prefixes = list_common_prefixes(s3, bucket, user_root)
    pat = re.compile(rf"^{re.escape(user_id)}[-_](\d{{8}})$")
    for p in prefixes:
        name = p.rstrip("/").split("/")[-1]
        m = pat.match(name)
        if m:
            dates.add(m.group(1))
    return sorted(dates)

def pick_day_prefix(layout: S3Layout, s3, bucket: str, user_id: str, day: str) -> str | None:
    """Resolve the day folder prefix for a given (user, day).

    Supports both:
      - <root>/<user>/<user>-YYYYMMDD/
      - <root>/<user>/<user>_YYYYMMDD/
    """
    root = layout.root_prefix or ""
    # Ensure root ends with '/' when non-empty
    if root and not root.endswith("/"):
        root += "/"
    # Canonical under user folder
    candidates = [
        f"{root}{user_id}/{user_id}-{day}/",
        f"{root}{user_id}/{user_id}_{day}/",
        # legacy flat (if ever used)
        f"{root}{user_id}-{day}/",
        f"{root}{user_id}_{day}/",
    ]
    # Also include any candidates provided by S3Layout (if present)
    try:
        for cand in layout.day_prefix_candidates(user_id, day):
            if cand not in candidates:
                candidates.append(cand)
    except Exception:
        pass

    for cand in candidates:
        if prefix_exists(s3, bucket, cand):
            return cand
    return None




def pick_voice_drift_prefix(layout: S3Layout, s3, bucket: str, user_id: str, day: str) -> str | None:
    for cand in layout.voice_drift_prefix_candidates(user_id, day):
        if prefix_exists(s3, bucket, cand):
            return cand
    return None

def parse_ids_from_prefixes(prefixes: list[str], root_prefix: str) -> list[str]:
    ids = set()
    pat = re.compile(rf"^{re.escape(root_prefix)}([A-Za-z0-9_\-]+)-(\d{{8}})/$")
    for p in prefixes:
        m = pat.match(p)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


def parse_dates_for_id(prefixes: list[str], root_prefix: str, user_id: str) -> list[str]:
    dates = set()
    pat = re.compile(rf"^{re.escape(root_prefix)}{re.escape(user_id)}-(\d{{8}})/$")
    for p in prefixes:
        m = pat.match(p)
        if m:
            dates.add(m.group(1))
    return sorted(dates)


def session_has_any_txt(s3, bucket: str, session_prefix: str) -> bool:
    """Return True if any .txt exists under the session prefix (treat as tagged)."""
    token = None
    while True:
        args = dict(Bucket=bucket, Prefix=session_prefix, MaxKeys=200)
        if token:
            args["ContinuationToken"] = token
        resp = s3.list_objects_v2(**args)
        for obj in resp.get("Contents", []):
            key = obj.get("Key", "")
            if key.endswith(".txt") and not key.endswith("/"):
                return True
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
            continue
        return False


# ---------------- WAV + latency helpers ----------------

def _read_wav_mono_float32(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    """
    Read WAV bytes into (sr, mono float32 [-1,1]).
    Prefer soundfile (handles more WAV variants); fall back to stdlib wave.
    """
    if sf is not None:
        x, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if isinstance(x, np.ndarray) and x.ndim > 1:
            x = np.mean(x, axis=1).astype(np.float32)
        return int(sr), x.astype(np.float32)

    import wave
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x_i32 = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        if np.max(np.abs(x_i32)) > 1.5:
            x = x_i32 / 2147483648.0
        else:
            x = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
    elif sampwidth == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if nchan > 1:
        x = x.reshape(-1, nchan).mean(axis=1)
    return int(sr), x.astype(np.float32)


# ---------------- GUI ----------------


class S3FolderPicker(tk.Toplevel):
    """A lightweight S3 'folder' picker (prefix picker) using CommonPrefixes.

    - Double-click a folder to enter it.
    - 'Up' navigates one level up.
    - 'Select this folder' returns the current prefix.
    """

    def __init__(self, parent, s3, bucket: str, start_prefix: str, on_select):
        super().__init__(parent)
        self.parent = parent
        self.s3 = s3
        self.bucket = bucket
        self.on_select = on_select

        p = (start_prefix or "").replace('\\', '/')
        if p and not p.endswith('/'):
            p += '/'
        self.current_prefix = p

        self.title("Browse S3 folders")
        self.geometry("560x420")

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="Current prefix:").pack(anchor="w")
        self.lbl_prefix = ttk.Label(top, text=self.current_prefix or "/", width=80)
        self.lbl_prefix.pack(anchor="w", pady=(2, 6))

        btns = ttk.Frame(top)
        btns.pack(fill="x")

        ttk.Button(btns, text="Up", command=self._go_up).pack(side="left")
        ttk.Button(btns, text="Refresh", command=self._refresh).pack(side="left", padx=6)
        ttk.Button(btns, text="Select this folder", command=self._select_current).pack(side="right")

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=10, pady=8)

        self.lst = tk.Listbox(mid, height=16)
        self.lst.pack(side="left", fill="both", expand=True)
        self.lst.bind("<Double-Button-1>", lambda _e: self._enter_selected())

        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.lst.yview)
        vsb.pack(side="right", fill="y")
        self.lst.configure(yscrollcommand=vsb.set)

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right")

        self._refresh()

    def _refresh(self):
        try:
            self.lbl_prefix.config(text=self.current_prefix or "/")
            self.lst.delete(0, tk.END)
            prefixes = list_common_prefixes(self.s3, self.bucket, self.current_prefix)
            # display only the folder name segment
            names = []
            for p in prefixes:
                name = p.rstrip("/").split("/")[-1]
                if name:
                    names.append(name)
            for name in sorted(names):
                self.lst.insert(tk.END, name)
        except Exception as e:
            messagebox.showerror("Browse error", str(e))

    def _go_up(self):
        if not self.current_prefix:
            return
        p = self.current_prefix.rstrip("/")
        if "/" not in p:
            self.current_prefix = ""
        else:
            self.current_prefix = p.rsplit("/", 1)[0] + "/"
        self._refresh()

    def _enter_selected(self):
        sel = self.lst.curselection()
        if not sel:
            return
        name = self.lst.get(sel[0]).strip()
        if not name:
            return
        self.current_prefix = f"{self.current_prefix}{name}/"
        self._refresh()

    def _select_current(self):
        try:
            if callable(self.on_select):
                self.on_select(self.current_prefix)
        finally:
            self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("S3 Post-Analyze (Tagger)")
        self.geometry("1100x520")

        self.env_path = Path.cwd() / ".env"
        self.s3 = None
        self.bucket: str | None = None
        self.root_prefix: str = ""

        self.layout: S3Layout | None = None
        self._selected_day_prefix: str | None = None
        # Session dropdown (display -> raw session folder name)
        self._session_display_to_raw: dict[str, str] = {}

        self._build_ui()
        self._set_conn_indicator("disconnected")
        self._set_status("Choose .env (optional) and click Connect.")

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm_env = ttk.Frame(self)
        frm_env.pack(fill="x", **pad)
        ttk.Label(frm_env, text=".env file:").pack(side="left")
        self.lbl_env = ttk.Label(frm_env, text=str(self.env_path), width=70)
        self.lbl_env.pack(side="left", padx=8)
        ttk.Button(frm_env, text="Choose .env…", command=self.on_choose_env).pack(side="right")
        ttk.Button(frm_env, text="Connect / Refresh", command=self.on_connect).pack(side="right", padx=8)
        self.lbl_conn = tk.Label(frm_env, text="●", fg="gray", font=("TkDefaultFont", 18, "bold"))
        self.lbl_conn.pack(side="right", padx=(0, 6))

        frm_bucket = ttk.Frame(self)
        frm_bucket.pack(fill="x", **pad)
        ttk.Label(frm_bucket, text="S3 bucket:").pack(side="left")
        self.ent_bucket = ttk.Entry(frm_bucket, width=32)
        self.ent_bucket.pack(side="left", padx=8)
        self.ent_bucket.insert(0, "mmwave-raw-data")
        self.lbl_root = ttk.Label(frm_bucket, text="Root prefix: (auto)")
        self.lbl_root.pack(side="left", padx=10)

        ttk.Button(frm_bucket, text="Browse S3…", command=self.on_browse_s3).pack(side="left", padx=8)

        frm_sel = ttk.Frame(self)
        frm_sel.pack(fill="x", **pad)

        ttk.Label(frm_sel, text="User ID:").grid(row=0, column=0, sticky="w")
        self.cmb_id = ttk.Combobox(frm_sel, state="readonly", width=22, values=[])
        self.cmb_id.grid(row=0, column=1, sticky="w", padx=8)
        self.cmb_id.bind("<<ComboboxSelected>>", self.on_id_selected)

        ttk.Label(frm_sel, text="Date (YYYYMMDD):").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.cmb_date = ttk.Combobox(frm_sel, state="readonly", width=18, values=[])
        self.cmb_date.grid(row=0, column=3, sticky="w", padx=8)
        self.cmb_date.bind("<<ComboboxSelected>>", self.on_date_selected)

        ttk.Label(frm_sel, text="Session:").grid(row=0, column=4, sticky="w", padx=(20, 0))
        self.cmb_session = ttk.Combobox(frm_sel, state="readonly", width=40, values=[])
        self.cmb_session.grid(row=0, column=5, sticky="w", padx=8)
        ttk.Button(frm_sel, text="Tag data", command=self.on_open_tagger).grid(
            row=0, column=7, padx=(10, 0), sticky="w"
        )

        # Results table
        frm_tbl = ttk.Frame(self)
        frm_tbl.pack(fill="both", expand=True, **pad)

        cols = ("filename", "date", "time", "duration_s", "latency_s", "beep_end_s", "speech_onset_s", "notes")
        self.tree = ttk.Treeview(frm_tbl, columns=cols, show="headings", height=16)
        self.tree.heading("filename", text="filename")
        self.tree.heading("date", text="date")
        self.tree.heading("time", text="time")
        self.tree.heading("duration_s", text="duration (s)")
        self.tree.heading("latency_s", text="latency (s)")
        self.tree.heading("beep_end_s", text="beep_end (s)")
        self.tree.heading("speech_onset_s", text="speech_onset (s)")
        self.tree.heading("notes", text="notes")

        self.tree.column("filename", width=320, anchor="w")
        self.tree.column("date", width=110, anchor="center")
        self.tree.column("time", width=110, anchor="center")
        self.tree.column("duration_s", width=120, anchor="e")
        self.tree.column("latency_s", width=120, anchor="e")
        self.tree.column("beep_end_s", width=120, anchor="e")
        self.tree.column("speech_onset_s", width=140, anchor="e")
        self.tree.column("notes", width=340, anchor="w")

        # Scrollbars (vertical + horizontal)
        vsb = ttk.Scrollbar(frm_tbl, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frm_tbl, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Use grid so both scrollbars work correctly
        frm_tbl.rowconfigure(0, weight=1)
        frm_tbl.columnconfigure(0, weight=1)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Status
        frm_status = ttk.Frame(self)
        frm_status.pack(fill="x", **pad)
        ttk.Label(frm_status, text="Status:").pack(side="left")
        self.lbl_status = ttk.Label(frm_status, text="", width=160)
        self.lbl_status.pack(side="left", padx=8)

    def _set_status(self, msg: str):
        self.lbl_status.config(text=msg)
        self.update_idletasks()

    def _set_conn_indicator(self, state: str):
        """state: 'disconnected' | 'connecting' | 'connected' | 'error'"""
        if not hasattr(self, "lbl_conn") or self.lbl_conn is None:
            return
        color = {
            "disconnected": "gray",
            "connecting": "goldenrod",
            "connected": "green",
            "error": "red",
        }.get(state, "gray")
        try:
            self.lbl_conn.config(fg=color)
        except Exception:
            pass

    def on_choose_env(self):
        p = filedialog.askopenfilename(
            title="Choose .env file",
            filetypes=[("dotenv files", ".env"), ("All files", "*.*")],
            initialdir=str(Path.cwd()),
        )
        if p:
            self.env_path = Path(p)
            self.lbl_env.config(text=str(self.env_path))
            self._set_status("Selected .env. Click Connect / Refresh.")

    def on_connect(self):
        self._set_status("Connecting to S3 and listing IDs…")
        self._set_conn_indicator("connecting")
        self.cmb_id.set("")
        self.cmb_date.set("")
        if hasattr(self, "cmb_session"):
            self.cmb_session.set("")
            self.cmb_session["values"] = []
            self._session_display_to_raw = {}
        self.cmb_id["values"] = []
        self.cmb_date["values"] = []
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        bucket_ui = self.ent_bucket.get().strip()
        if not bucket_ui:
            messagebox.showerror("Missing bucket", "Please enter the S3 bucket name.")
            return

        def worker():
            try:
                cfg = load_env(self.env_path)
                s3 = make_s3_client(cfg)
                bucket = (cfg.bucket or bucket_ui).strip()
                root_prefix = detect_root_prefix(s3, bucket)

                layout = S3Layout(root_prefix)
                ids = list_user_ids(layout, s3, bucket)

                def ui_update():
                    self.s3 = s3
                    self.bucket = bucket
                    self.root_prefix = root_prefix
                    self.layout = layout
                    self._selected_day_prefix = None
                    self.lbl_root.config(text=f"Root prefix: {root_prefix or '/'}")
                    self.cmb_id["values"] = ids
                    self._set_conn_indicator("connected")
                    self._set_status(
                        f"Loaded {len(ids)} IDs. Select an ID."
                        if ids
                        else "No IDs found. Expected canonical folders: <root>/<user>/<user>-YYYYMMDD/"
                    )

                self.after(0, ui_update)

            except (BotoCoreError, ClientError) as e:
                self.after(0, lambda: self._set_conn_indicator("error"))
                self.after(0, lambda: self._set_status(f"S3 error: {e}"))
                self.after(0, lambda: messagebox.showerror("S3 error", str(e)))
            except Exception as e:
                self.after(0, lambda: self._set_conn_indicator("error"))
                self.after(0, lambda: self._set_status(f"Error: {e}"))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def on_id_selected(self, _evt=None):
        user_id = self.cmb_id.get().strip()
        if not user_id or not self.s3 or not self.bucket:
            return

        self._set_status(f"Listing dates for {user_id}…")
        self.cmb_date.set("")
        self.cmb_date["values"] = []

        def worker():
            try:
                layout = self.layout or S3Layout(self.root_prefix)
                dates = list_dates_for_user(layout, self.s3, self.bucket, user_id)

                def ui_update():
                    self.cmb_date["values"] = dates
                    if dates:
                        self.cmb_date.current(0)
                        self._set_status(f"Loaded {len(dates)} dates for {user_id}.")
                        self.on_date_selected()
                    else:
                        self._set_status(f"No dates found for {user_id}.")

                self.after(0, ui_update)

            except (BotoCoreError, ClientError) as e:
                self.after(0, lambda: self._set_conn_indicator("error"))
                self.after(0, lambda: self._set_status(f"S3 error: {e}"))
                self.after(0, lambda: messagebox.showerror("S3 error", str(e)))
            except Exception as e:
                self.after(0, lambda: self._set_conn_indicator("error"))
                self.after(0, lambda: self._set_status(f"Error: {e}"))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def on_date_selected(self, _evt=None):
        """Populate session dropdown for the selected (ID, date)."""
        user_id = self.cmb_id.get().strip()
        day = self.cmb_date.get().strip()
        if not user_id or not day or not self.s3 or not self.bucket:
            # Clear sessions if incomplete selection
            if hasattr(self, "cmb_session"):
                self.cmb_session.set("")
                self.cmb_session["values"] = []
                self._session_display_to_raw = {}
            return

        layout = self.layout or S3Layout(self.root_prefix)
        day_prefix = pick_day_prefix(layout, self.s3, self.bucket, user_id, day)
        if not day_prefix:
            self._set_status(f"No day folder found for {user_id}-{day} (checked canonical + legacy).")
            if hasattr(self, "cmb_session"):
                self.cmb_session.set("")
                self.cmb_session["values"] = []
                self._session_display_to_raw = {}
            return
        self._selected_day_prefix = day_prefix
        self._set_status(f"Listing sessions under {user_id}-{day}…")
        self.cmb_session.set("")
        self.cmb_session["values"] = []
        self._session_display_to_raw = {}

        def worker():
            try:
                sub_prefixes = list_common_prefixes(self.s3, self.bucket, day_prefix)

                # New layout support: sessions may be **CSV files** directly under day_prefix, e.g.
                #   sleep/<user>/<user_yyyymmdd>/<user_yyyymmdd_HHMMSS>.csv
                # Legacy layout: sessions may be **subfolders** under day_prefix.
                objs = list_objects(self.s3, self.bucket, day_prefix, max_items=5000)
                keys = [o.get("Key", "") for o in objs if o.get("Key")]

                csv_keys = [k for k in keys if k.lower().endswith(".csv") and k.startswith(day_prefix)]
                txt_keys = set([k for k in keys if k.lower().endswith(".txt") and k.startswith(day_prefix)])

                sessions: list[str] = []
                session_to_csv: dict[str, str] = {}

                # Prefer CSV-file sessions if present
                for k in csv_keys:
                    rel = k[len(day_prefix):]
                    # only treat as "session file" if it's directly under day_prefix
                    if "/" in rel:
                        continue
                    stem = Path(rel).stem
                    sessions.append(stem)
                    session_to_csv[stem] = k

                # If no direct CSV sessions, fall back to folder sessions
                if not sessions:
                    for p in sub_prefixes:
                        name = p.rstrip("/").split("/")[-1]
                        if name == "voice_drift":
                            continue
                        sessions.append(name)

                display_items: list[str] = []
                display_to_raw: dict[str, str] = {}

                for s in sorted(set(sessions)):
                    tagged = False
                    if s in session_to_csv:
                        # Tagged if a same-stem txt exists alongside the csv
                        tagged = f"{day_prefix}{s}.txt" in txt_keys
                    else:
                        # Folder session: tagged if any .txt exists inside the folder
                        sprefix = f"{day_prefix}{s}/"
                        tagged = session_has_any_txt(self.s3, self.bucket, sprefix)

                    disp = f"{s} #TAGGED" if tagged else s
                    display_items.append(disp)
                    display_to_raw[disp] = s

                def ui_update():
                    self._session_display_to_raw = display_to_raw
                    self.cmb_session["values"] = display_items
                    if display_items:
                        self.cmb_session.current(0)
                        self._set_status(f"Loaded {len(display_items)} sessions for {user_id}-{day}.")
                    else:
                        self._set_status(f"No sessions found for {user_id}-{day}.")

                self.after(0, ui_update)

            except (BotoCoreError, ClientError) as e:
                self.after(0, lambda: self._set_conn_indicator("error"))
                self.after(0, lambda: self._set_status(f"S3 error: {e}"))
                self.after(0, lambda: messagebox.showerror("S3 error", str(e)))
            except Exception as e:
                self.after(0, lambda: self._set_conn_indicator("error"))
                self.after(0, lambda: self._set_status(f"Error: {e}"))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _get_selected_session_raw(self) -> str | None:
        if not hasattr(self, "cmb_session"):
            return None
        disp = self.cmb_session.get().strip()
        if not disp:
            return None
        return self._session_display_to_raw.get(disp, disp.split(" #TAGGED")[0].strip() if " #TAGGED" in disp else disp)

    def on_open_tagger(self):
        if not self.s3 or not self.bucket:
            messagebox.showinfo("Not connected", "Click Connect / Refresh first.")
            return
        user_id = self.cmb_id.get().strip()
        day = self.cmb_date.get().strip()
        if not user_id or not day:
            messagebox.showinfo("Missing selection", "Select a User ID and Date first.")
            return
        if pd is None:
            messagebox.showerror("Missing dependency", "pandas is required for tagging. pip install pandas")
            return
        if plt is None or FigureCanvasTkAgg is None:
            messagebox.showerror("Missing dependency",
                                 "matplotlib is required for tagging plots. pip install matplotlib")
            return
        if cv2 is None or Image is None or ImageTk is None:
            messagebox.showwarning("Video dependency missing",
                                   "OpenCV/Pillow not available; video preview will be disabled.")
        S3TaggerWindow(self, self.s3, self.bucket, self.root_prefix, user_id, day,
                       session=self._get_selected_session_raw(), day_prefix=self._selected_day_prefix)


    def on_browse_s3(self):
        if not self.s3 or not self.bucket:
            messagebox.showinfo("Not connected", "Click Connect / Refresh first.")
            return

        # Start browsing at the current root_prefix if set, else at bucket root.
        start_prefix = (self.root_prefix or "")
        S3FolderPicker(
            parent=self,
            s3=self.s3,
            bucket=self.bucket,
            start_prefix=start_prefix,
            on_select=self._apply_browsed_prefix,
        )

    def _apply_browsed_prefix(self, selected_prefix: str):
        """Apply a browsed prefix and populate (User ID, Date, Session) accordingly.

        We support selecting any of:
          - dataset root (e.g., 'sleep/')
          - user folder (e.g., 'sleep/<user>/')
          - day folder (e.g., 'sleep/<user>/<user>_YYYYMMDD/')
          - session folder (e.g., 'sleep/<user>/<user>_YYYYMMDD/<session>/')
        """
        if not selected_prefix:
            selected_prefix = ""
        selected_prefix = selected_prefix.replace("\\", "/")
        if selected_prefix and not selected_prefix.endswith("/"):
            selected_prefix += "/"

        parts = [p for p in selected_prefix.strip("/").split("/") if p]
        root_prefix = ""
        inferred_user = None
        inferred_day = None
        inferred_session = None

        if parts:
            # By convention, treat the first path segment as the dataset root (e.g., 'sleep/').
            root_prefix = parts[0] + "/"
            if len(parts) >= 2:
                inferred_user = parts[1]
            if len(parts) >= 3 and inferred_user:
                day_folder = parts[2]
                mday = re.match(rf"^{re.escape(inferred_user)}[-_](\d{{8}})$", day_folder)
                if mday:
                    inferred_day = mday.group(1)
            if len(parts) >= 4:
                inferred_session = parts[3]

        self.root_prefix = root_prefix
        self.layout = S3Layout(self.root_prefix)
        self.lbl_root.config(text=f"Root prefix: {self.root_prefix or '/'}")

        # Refresh ID list, then apply inferred selections.
        self._set_status(f"Browsing applied. Loading IDs under {self.root_prefix or '/'}…")
        self.cmb_id.set("")
        self.cmb_date.set("")
        self.cmb_session.set("")
        self.cmb_id["values"] = []
        self.cmb_date["values"] = []
        self.cmb_session["values"] = []
        self._session_display_to_raw = {}

        def worker():
            try:
                ids = list_user_ids(self.layout, self.s3, self.bucket)

                def ui_update():
                    self.cmb_id["values"] = ids
                    if inferred_user and inferred_user in ids:
                        self.cmb_id.set(inferred_user)
                        self.on_id_selected()
                    else:
                        self._set_status(f"Loaded {len(ids)} IDs. Select an ID.")
                    # If we inferred user/day/session, apply them after the dependent dropdowns load.
                    if inferred_user and inferred_day:
                        # Wait a tick for dates to populate
                        def apply_day():
                            try:
                                if inferred_day in self.cmb_date["values"]:
                                    self.cmb_date.set(inferred_day)
                                    self.on_date_selected()
                                    if inferred_session:
                                        def apply_session():
                                            # Session dropdown values are display strings; match raw.
                                            raw = inferred_session
                                            for disp in list(self._session_display_to_raw.keys()):
                                                if self._session_display_to_raw.get(disp) == raw:
                                                    self.cmb_session.set(disp)
                                                    return
                                            # Fallback: direct set if present
                                            vals = list(self.cmb_session["values"])
                                            if raw in vals:
                                                self.cmb_session.set(raw)
                                        self.after(150, apply_session)
                            except Exception:
                                pass
                        self.after(150, apply_day)

                self.after(0, ui_update)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Browse apply error", str(e)))
                self.after(0, lambda: self._set_status(f"Browse apply error: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        return


        def _render_rows(rows: list[tuple]):
            for r in rows:
                self.tree.insert("", tk.END, values=r)

        def _rows_from_dataframe(df):
            want = ["filename", "date", "time", "duration_s", "latency_s", "beep_end_s", "speech_onset_s", "notes"]
            for c in want:
                if c not in df.columns:
                    df[c] = ""
            rows = []
            for _, row in df.iterrows():
                rows.append((
                    str(row.get("filename", "")),
                    str(row.get("date", "")),
                    str(row.get("time", "")),
                    str(row.get("duration_s", "")),
                    str(row.get("latency_s", "")),
                    str(row.get("beep_end_s", "")),
                    str(row.get("speech_onset_s", "")),
                    str(row.get("notes", "")),
                ))
            return rows

        def _plot_dataframe(df):
            if plt is None or FigureCanvasTkAgg is None:
                return

            if "filename" not in df.columns:
                return

            def to_num_series(series):
                return pd.to_numeric(series, errors="coerce")

            fn = df["filename"].astype(str)

            def _clip_type(name: str) -> str:
                base = name.rsplit("/", 1)[-1]
                base = base.rsplit(".", 1)[0]
                if base.startswith("self_feel_"):
                    return "self_feel"
                for sep in ("_", "-", " "):
                    if sep in base:
                        return base.split(sep, 1)[0]
                return base

            def _self_feel_score(name: str):
                base = name.rsplit("/", 1)[-1]
                base = base.rsplit(".", 1)[0]
                if not base.startswith("self_feel_"):
                    return None
                try:
                    return float(base[len("self_feel_"):])
                except Exception:
                    return None

            df = df.copy()
            df["clip_type"] = [_clip_type(x) for x in fn.tolist()]
            df["self_feel_score"] = [_self_feel_score(x) for x in fn.tolist()]

            # ---- Date-only x axis (voice drift measured once/day) ----
            df = df.copy()
            df["date_dt"] = pd.to_datetime(
                df["date"].astype(str),
                format="%Y%m%d",
                errors="coerce"
            )
            df = df[df["date_dt"].notna()].sort_values("date_dt")
            if len(df) == 0:
                return
            # Use python date objects (no timezone shift)
            x = df["date_dt"].dt.date

            dur = to_num_series(df.get("duration_s"))
            lat = to_num_series(df.get("latency_s"))
            scores = to_num_series(df.get("self_feel_score"))

            win = tk.Toplevel(self)
            win.title(f"Voice Drift — {user_id} (overall)")
            win.geometry("1100x800")

            fig = plt.Figure(figsize=(10.6, 7.4), dpi=100)

            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            types = sorted([t for t in df["clip_type"].unique() if t])

            # -------- Duration (LINE ONLY) --------
            for t in types:
                if t == "self_feel":
                    continue
                mask = df["clip_type"] == t
                ax1.plot(x[mask], dur[mask], label=t)

            ax1.set_title("Duration per clip (s) by type")
            ax1.set_xlabel("Clip index")
            ax1.set_ylabel("Duration (s)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best", fontsize=8)

            # -------- Latency (LINE ONLY) --------
            for t in types:
                if t == "self_feel":
                    continue
                mask = df["clip_type"] == t
                ax2.plot(x[mask], lat[mask], label=t)

            ax2.set_title("Latency per clip (s) by type")
            ax2.set_xlabel("Clip index")
            ax2.set_ylabel("Latency (s)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best", fontsize=8)

            # -------- Self-feel Score (LINE ONLY) --------
            mask_sf = df["clip_type"] == "self_feel"
            if mask_sf.any():
                ax3.plot(x[mask_sf], scores[mask_sf])
                ax3.set_title("Self-feel score over time")
            else:
                ax3.set_title("Self-feel score (none found)")

            ax3.set_xlabel("Clip index")
            ax3.set_ylabel("Score")
            ax3.grid(True, alpha=0.3)

            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

        def worker():
            try:
                if object_exists(self.s3, self.bucket, csv_key):
                    csv_text = download_text_object(self.s3, self.bucket, csv_key)

                    if pd is None:
                        raise RuntimeError("pandas is required to parse the cached CSV (pip install pandas)")

                    df_old = pd.read_csv(io.StringIO(csv_text))
                    if "s3_key" not in df_old.columns:
                        df_old["s3_key"] = ""

                    existing_keys = set(str(x) for x in df_old["s3_key"].dropna().astype(str).tolist() if str(x).strip())

                    # Scan for new voice_drift wavs not yet in the cached CSV.
                    self.after(0, lambda: self._set_status("Cached CSV found; scanning for new voice_drift wavs…"))

                    dates = list_dates_for_user(layout, self.s3, self.bucket, user_id)
                    new_records = []

                    for day in dates:
                        vd_prefix = pick_voice_drift_prefix(layout, self.s3, self.bucket, user_id, day)
                        if not vd_prefix:
                            continue

                        objs = list_objects(self.s3, self.bucket, vd_prefix, max_items=5000)
                        keys = [o.get("Key") for o in objs if o.get("Key")]
                        done_key = next((k for k in keys if k.lower().endswith("_done.json")), None)
                        wav_keys = sorted([k for k in keys if k.lower().endswith(".wav")])

                        if not wav_keys:
                            continue

                        time_str = "--:--:--"
                        date_str = day
                        if done_key:
                            try:
                                done = json.loads(download_text_object(self.s3, self.bucket, done_key))
                                date_str = str(done.get("day", day))
                                ts = done.get("created_at_unix")
                                if ts is not None:
                                    dt = datetime.fromtimestamp(float(ts))
                                    time_str = dt.strftime("%H:%M:%S")
                            except Exception:
                                pass

                        for k in wav_keys:
                            if k in existing_keys:
                                continue

                            fname = k.split("/")[-1]
                            w = self.s3.get_object(Bucket=self.bucket, Key=k)
                            wav_bytes = w["Body"].read()

                            sr, x = _read_wav_mono_float32(wav_bytes)
                            duration_s = float(x.size) / float(sr) if sr > 0 else 0.0

                            latency_s, beep_end_s, speech_onset_s, notes = estimate_latency_beep_to_speech(x, sr)

                            new_records.append({
                                "filename": fname,
                                "date": date_str,
                                "time": time_str,
                                "duration_s": float(f"{duration_s:.2f}"),
                                "latency_s": (None if latency_s is None else float(f"{latency_s:.3f}")),
                                "beep_end_s": (None if beep_end_s is None else float(f"{beep_end_s:.3f}")),
                                "speech_onset_s": (None if speech_onset_s is None else float(f"{speech_onset_s:.3f}")),
                                "notes": notes,
                                "s3_key": k,
                            })

                    if not new_records:
                        rows = _rows_from_dataframe(df_old)

                        def ui_update_cached_only():
                            _render_rows(rows)
                            self._set_status(f"Loaded cached voice drift CSV ({len(rows)} rows). No new clips found.")
                            _plot_dataframe(df_old)

                        self.after(0, ui_update_cached_only)
                        return

                    df_new = pd.DataFrame(new_records)
                    df_all = pd.concat([df_old, df_new], ignore_index=True)

                    for col in ("date", "time", "filename"):
                        if col not in df_all.columns:
                            df_all[col] = ""
                    df_all = df_all.sort_values(["date", "time", "filename"], kind="mergesort").reset_index(drop=True)

                    buf = io.StringIO()
                    df_all.to_csv(buf, index=False)
                    upload_text_object(self.s3, self.bucket, csv_key, buf.getvalue(), content_type="text/csv")

                    rows = _rows_from_dataframe(df_all)

                    def ui_update_appended():
                        _render_rows(rows)
                        self._set_status(
                            f"Appended {len(df_new)} new clips; updated cached CSV now has {len(df_all)} rows."
                        )
                        _plot_dataframe(df_all)

                    self.after(0, ui_update_appended)
                    return

                self.after(0, lambda: self._set_status("No cached CSV found; computing voice drift across all days…"))

                dates = list_dates_for_user(layout, self.s3, self.bucket, user_id)
                out_records = []

                for day in dates:
                    vd_prefix = pick_voice_drift_prefix(layout, self.s3, self.bucket, user_id, day)
                    if not vd_prefix:
                        continue

                    objs = list_objects(self.s3, self.bucket, vd_prefix, max_items=5000)
                    keys = [o.get("Key") for o in objs if o.get("Key")]
                    done_key = next((k for k in keys if k.lower().endswith("_done.json")), None)
                    wav_keys = sorted([k for k in keys if k.lower().endswith(".wav")])

                    if not wav_keys:
                        continue

                    time_str = "--:--:--"
                    date_str = day
                    if done_key:
                        try:
                            done = json.loads(download_text_object(self.s3, self.bucket, done_key))
                            date_str = str(done.get("day", day))
                            ts = done.get("created_at_unix")
                            if ts is not None:
                                dt = datetime.fromtimestamp(float(ts))
                                time_str = dt.strftime("%H:%M:%S")
                        except Exception:
                            pass

                    for k in wav_keys:
                        fname = k.split("/")[-1]
                        w = self.s3.get_object(Bucket=self.bucket, Key=k)
                        wav_bytes = w["Body"].read()

                        sr, x = _read_wav_mono_float32(wav_bytes)
                        duration_s = float(x.size) / float(sr) if sr > 0 else 0.0

                        latency_s, beep_end_s, speech_onset_s, notes = estimate_latency_beep_to_speech(x, sr)

                        out_records.append({
                            "filename": fname,
                            "date": date_str,
                            "time": time_str,
                            "duration_s": float(f"{duration_s:.2f}"),
                            "latency_s": (None if latency_s is None else float(f"{latency_s:.3f}")),
                            "beep_end_s": (None if beep_end_s is None else float(f"{beep_end_s:.3f}")),
                            "speech_onset_s": (None if speech_onset_s is None else float(f"{speech_onset_s:.3f}")),
                            "notes": notes,
                            "s3_key": k,
                        })

                if not out_records:
                    self.after(0, lambda: self._set_status("No voice_drift wavs found for this user."))
                    self.after(0, lambda: messagebox.showinfo("No data", f"No voice_drift data found under {user_root}"))
                    return

                if pd is None:
                    raise RuntimeError("pandas is required to write the aggregated CSV (pip install pandas)")

                df = pd.DataFrame(out_records)
                for col in ("date", "time", "filename"):
                    if col not in df.columns:
                        df[col] = ""
                df = df.sort_values(["date", "time", "filename"], kind="mergesort").reset_index(drop=True)

                buf = io.StringIO()
                df.to_csv(buf, index=False)
                upload_text_object(self.s3, self.bucket, csv_key, buf.getvalue(), content_type="text/csv")

                rows = _rows_from_dataframe(df)

                def ui_update2():
                    _render_rows(rows)
                    self._set_status(f"Computed {len(rows)} clips and uploaded cached CSV to {csv_key}.")
                    _plot_dataframe(df)

                self.after(0, ui_update2)

            except (BotoCoreError, ClientError) as e:
                self.after(0, lambda: self._set_status(f"S3 error: {e}"))
                self.after(0, lambda: messagebox.showerror("S3 error", str(e)))
            except Exception as e:
                self.after(0, lambda: self._set_status(f"Error: {e}"))
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=worker, daemon=True).start()



def main():
    App().mainloop()


# ---------------- Tagging Viewer (S3-backed) ----------------

TAG_ACTIONS_DEFAULT = [
    "no_motion",
    "walk_close",
    "walk_away",
    "standing",
    "sitting",
    "sit_to_stand",
    "stand_to_sit",
    "hand_washing",
    "tooth_brushing",
    "drinking",
    "lying",
]


def _s3_download_to_file(s3, bucket: str, key: str, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    dst_path.write_bytes(body)


def _s3_upload_file(s3, bucket: str, key: str, src_path: Path) -> None:
    s3.upload_file(str(src_path), bucket, key)


def _safe_literal_array(cell) -> np.ndarray:
    try:
        return np.asarray(ast.literal_eval(cell), dtype=np.float32)
    except Exception:
        return np.zeros((0, 0), dtype=np.float32)


def _compute_range_axis_m(num_samples: int, f_start_hz: float, f_end_hz: float) -> np.ndarray:
    """Range axis in meters for half-FFT bins.

    Uses the collector's configured range resolution (RANGE_RESOLUTION_M) instead of inferring from BW.
    """
    n_bins = max(1, int(num_samples // 2))
    return (np.arange(n_bins, dtype=np.float32) * float(RANGE_RESOLUTION_M)).astype(np.float32)


def _compute_distance_mag(chirp_mat: np.ndarray, window: np.ndarray) -> np.ndarray:
    """
    chirp_mat: (num_chirps, num_samples)
    returns: (num_samples//2,) mean magnitude over chirps for positive-range bins.
    """
    if chirp_mat.size == 0:
        return np.zeros((window.size // 2,), dtype=np.float32)
    if chirp_mat.ndim != 2:
        chirp_mat = np.asarray(chirp_mat).reshape(-1, window.size)
    xw = chirp_mat * window.reshape(1, -1)
    X = np.fft.fft(xw, axis=1)
    mag = np.abs(X).mean(axis=0).astype(np.float32)
    return mag[: mag.size // 2]


def _compute_range_doppler_mag(
    chirp_mat: np.ndarray,
    win_rng: np.ndarray,
    prf_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple range–Doppler magnitude map for one channel.

    Args:
        chirp_mat: (num_chirps, num_samples)
        win_rng:   range FFT window (num_samples,)
        prf_hz:    chirp repetition frequency in Hz (approx; see notes below)

    Returns:
        rd_mag:    (num_doppler_bins, num_range_bins) magnitude
        vel_axis:  (num_doppler_bins,) velocity axis in m/s (approx)

    Notes:
        The CSV stores chirp matrices but usually not chirp timing, so velocity is
        approximate unless you set a correct PRF (chirps per second).
    """

    if chirp_mat.size == 0:
        n_rng = max(1, int(win_rng.size // 2))
        return np.zeros((1, n_rng), dtype=np.float32), np.zeros((1,), dtype=np.float32)

    if chirp_mat.ndim != 2:
        chirp_mat = np.asarray(chirp_mat).reshape(-1, win_rng.size)

    num_chirps, num_samples = chirp_mat.shape
    if num_samples != win_rng.size:
        win_rng = np.hanning(num_samples).astype(np.float32)

    # Range FFT per chirp (fast-time)
    xw = chirp_mat.astype(np.float32) * win_rng.reshape(1, -1)
    Xr = np.fft.fft(xw, axis=1)[:, : max(1, num_samples // 2)]  # (num_chirps, rng_bins)

    # Doppler FFT across chirps (slow-time) per range bin
    win_dop = np.hanning(max(1, num_chirps)).astype(np.float32).reshape(-1, 1)
    Xd = np.fft.fftshift(np.fft.fft(Xr * win_dop, axis=0), axes=0)  # (dop_bins, rng_bins)

    rd_mag = np.abs(Xd).astype(np.float32)

    # Velocity axis: v = f_d * lambda / 2
    if prf_hz <= 0:
        prf_hz = 2000.0
    fd = np.fft.fftshift(np.fft.fftfreq(num_chirps, d=1.0 / float(prf_hz))).astype(np.float32)

    # Use the same band as range axis for lambda
    fc_hz = 0.5 * (60250345952.0 + 61249654048.0)
    c = 299_792_458.0
    lam = float(c / fc_hz)
    vel = (fd * lam / 2.0).astype(np.float32)

    return rd_mag, vel



class S3TaggerWindow:
    """
    A minimal merge of:
      - mmwave_data_viewer: distance-vs-amplitude (2 radars)
      - mmwave_raw_tagger_gui: range tagging + overlap checks + tagged range buttons
    Backed by S3: downloads CSV/MP4/WAV to a local cache folder, and uploads tags (.txt) back to S3.

    Synchronization:
      - Radar index i -> radar timestamp t = ts[i] - ts[0]
      - Video frame idx = round(t * video_fps)
      - Audio playback: starts at t=0 when you hit "Sync Play" (best-effort).
    """

    def __init__(self, parent: tk.Tk, s3, bucket: str, root_prefix: str, user_id: str, day: str,
                 session: str | None = None, day_prefix: str | None = None):
        self.parent = parent
        self.s3 = s3
        self.bucket = bucket
        self.root_prefix = root_prefix
        self.user_id = user_id
        self.day = day
        self.preselect_session = session
        self.day_prefix = day_prefix  # resolved day prefix (new or legacy layout)

        self.ui_font = ("Arial", 9)

        self.win = tk.Toplevel(parent)
        self.win.title(f"Tagger — {user_id}-{day}")
        self.win.geometry("1200x720")

        # session selection
        self.session_var = tk.StringVar(value="")
        self.session_keys = {}  # stem -> csv key

        # local cache folder
        self.cache_root = Path(tempfile.gettempdir()) / "mmwave_s3_cache" / f"{user_id}-{day}"
        self.cache_root.mkdir(parents=True, exist_ok=True)

        # data holders
        self.df = None
        self.N = 0
        self.ts0 = 0.0
        self.timestamps = None  # np.ndarray
        self.range_axis = None  # np.ndarray
        self.dev0_mag = None  # (N, nbins)
        self.dev1_mag = None  # (N, nbins)

        # video/audio
        self.video_cap = None
        self.video_fps = 0.0
        self.video_total = 0
        self.video_loaded = False
        self.preview_image = None

        # animation
        self.current_frame = 0
        self.animation_running = False

        # tagging state
        self.actions = TAG_ACTIONS_DEFAULT.copy()
        self.action_var = tk.StringVar(value=self.actions[0])
        self.tagged_ranges_frame = None


        # plot mode (Range FFT vs Doppler FFT)
        self.plot_mode_var = tk.StringVar(value="Range FFT")
        self._last_plot_mode = None

        # Doppler settings (velocity axis is approximate unless PRF is correct)
        self.doppler_prf_hz = 2000.0

        # tagging window (inclusive frame range length)
        self.window_size = 16
        self.window_size_var = tk.StringVar(value="16")
        self._build_ui()
        self._load_sessions_async()

        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI ----------
    def _build_ui(self):
        top = tk.Frame(self.win)
        top.pack(fill="x", padx=10, pady=8)

        tk.Label(top, text="Session:", font=self.ui_font).pack(side=tk.LEFT)
        self.cmb_session = ttk.Combobox(top, textvariable=self.session_var, width=42, state="readonly")
        self.cmb_session.pack(side=tk.LEFT, padx=8)
        self.cmb_session.bind("<<ComboboxSelected>>", lambda e: self._load_selected_session_async())

        tk.Button(top, text="Reload sessions", command=self._load_sessions_async, font=self.ui_font).pack(side=tk.LEFT,
                                                                                                          padx=4)

        # action / range / save
        mid = tk.Frame(self.win)
        mid.pack(fill="x", padx=10, pady=4)

        tk.Label(mid, text="Action:", font=self.ui_font).pack(side=tk.LEFT)
        self.action_dropdown = ttk.Combobox(mid, textvariable=self.action_var, width=18, state="readonly",
                                            font=self.ui_font)
        self.action_dropdown["values"] = self.actions
        self.action_dropdown.pack(side=tk.LEFT, padx=6)

        tk.Label(mid, text="Frame range (1-index):", font=self.ui_font).pack(side=tk.LEFT, padx=(14, 2))
        self.range_entry = tk.Entry(mid, width=18, font=self.ui_font)
        self.range_entry.pack(side=tk.LEFT, padx=6)

        tk.Button(mid, text="Save tag", command=self.save_range, font=self.ui_font).pack(side=tk.LEFT, padx=6)

        tk.Label(mid, text="Window Size (Frame):", font=self.ui_font).pack(side=tk.LEFT, padx=(14, 2))
        self.window_entry = tk.Entry(mid, width=8, font=self.ui_font, textvariable=self.window_size_var)
        self.window_entry.pack(side=tk.LEFT, padx=6)
        tk.Button(mid, text="Update", command=self.on_update_window_size, font=self.ui_font).pack(side=tk.LEFT, padx=6)


        # (plot selector moved below the scrubber to sit above the plots)


        # controls
        ctrl = tk.Frame(self.win)
        ctrl.pack(fill="x", padx=10, pady=4)

        for label, cmd in [
            ("Play", self.start_synced_play),
            ("Pause", self.pause),
            ("Prev", self.prev_frame),
            ("Next", self.next_frame),
            ("Stop", self.stop),
        ]:
            tk.Button(ctrl, text=label, command=cmd, font=self.ui_font).pack(side=tk.LEFT, padx=4)

        self.lbl_info = tk.Label(ctrl, text="Frame: --/--", font=self.ui_font)
        self.lbl_info.pack(side=tk.LEFT, padx=(10, 0))

        # frame scrubber
        self._scale_job = None
        self._scale_updating = False
        self.frame_scale = tk.Scale(
            self.win,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            length=720,
            showvalue=0,
            resolution=1,
            command=self._on_scale_move,
        )
        self.frame_scale.pack(fill="x", padx=10, pady=(2, 6))
        self.frame_scale.bind("<ButtonRelease-1>", lambda e: self._on_scale_release())

        # plot selector (below scrubber, above plots)
        plotbar = tk.Frame(self.win)
        plotbar.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(plotbar, text="Plot:", font=self.ui_font).pack(side=tk.LEFT)
        self.plot_mode_dropdown = ttk.Combobox(
            plotbar,
            textvariable=self.plot_mode_var,
            width=14,
            state="readonly",
            font=self.ui_font,
            values=["Range FFT", "Doppler FFT"],
        )
        self.plot_mode_dropdown.pack(side=tk.LEFT, padx=8)
        self.plot_mode_dropdown.bind("<<ComboboxSelected>>", lambda _e: self._update_view(self.current_frame))

        # main content: left plots, right video + tags
        body = tk.Frame(self.win)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        left = tk.Frame(body)
        left.pack(side=tk.LEFT, fill="both", expand=True)

        right = tk.Frame(body)
        right.pack(side=tk.LEFT, fill="y", padx=(10, 0))

        # matplotlib plot
        self.fig = None
        self.ax0 = None
        self.ax1 = None
        self.line0 = None
        self.line1 = None
        self.im0 = None
        self.im1 = None
        # Phase / vital-sign plot artists (ax1)
        self.phase_line = None
        self.resp_line = None
        self.heart_line = None
        self.phase_cursor = None  # vertical cursor synced to current frame
        self.phase_t = None
        self.phase_unwrap = None
        self.phase_resp = None
        self.phase_heart = None
        self.est_rr_bpm = None
        self.est_hr_bpm = None
        self.phase_bin_m = None
        self.canvas = None
        if plt is None or FigureCanvasTkAgg is None:
            tk.Label(left, text="matplotlib not available; install matplotlib to see plots.", fg="red").pack()
        else:
            self.fig = plt.Figure(figsize=(7.0, 5.4), dpi=100)
            self.ax0 = self.fig.add_subplot(211)
            self.ax1 = self.fig.add_subplot(212)

            # Increase vertical separation to avoid title/xlabel overlap
            self.fig.subplots_adjust(hspace=0.65, left=0.10, right=0.97, top=0.94, bottom=0.11)

            # Artists (lines / heatmaps) are configured lazily in _update_view()
            self.line0 = None
            self.line1 = None
            self.im0 = None
            self.im1 = None

            self.canvas = FigureCanvasTkAgg(self.fig, master=left)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # video preview
        tk.Label(right, text="Video", font=("Arial", 10, "bold")).pack(anchor="w")
        self.video_label = tk.Label(right)
        self.video_label.pack()

        # tagged ranges display
        tk.Label(right, text="Tagged ranges", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.tagged_ranges_container = tk.Frame(right)
        self.tagged_ranges_container.pack(fill="both", expand=True)

        # make tagged ranges scrollable vertically
        self.tags_canvas = tk.Canvas(self.tagged_ranges_container, width=420, height=280)
        self.tags_scroll = ttk.Scrollbar(self.tagged_ranges_container, orient="vertical",
                                         command=self.tags_canvas.yview)
        self.tags_canvas.configure(yscrollcommand=self.tags_scroll.set)
        self.tags_scroll.pack(side="right", fill="y")
        self.tags_canvas.pack(side="left", fill="both", expand=True)

        self.tagged_ranges_frame = tk.Frame(self.tags_canvas)
        self.tags_canvas_window = self.tags_canvas.create_window((0, 0), window=self.tagged_ranges_frame, anchor="nw")
        self.tagged_ranges_frame.bind("<Configure>",
                                      lambda e: self.tags_canvas.configure(scrollregion=self.tags_canvas.bbox("all")))
        self.tags_canvas.bind("<Configure>",
                              lambda e: self.tags_canvas.itemconfig(self.tags_canvas_window, width=e.width))

    # ---------- S3 session discovery ----------

    # ---------- frame scrubber ----------
    def _on_scale_move(self, val):
        """Throttle updates while dragging the scale."""
        if getattr(self, "_scale_updating", False):
            return
        try:
            disp = int(float(val))
            idx = disp - 1
        except Exception:
            return
        self._pending_scale_idx = idx
        if self._scale_job is not None:
            try:
                self.win.after_cancel(self._scale_job)
            except Exception:
                pass
        self._scale_job = self.win.after(25, self._apply_pending_scale)

    def _apply_pending_scale(self):
        self._scale_job = None
        if getattr(self, "N", 0) <= 0:
            return
        idx = int(max(0, min(int(getattr(self, "_pending_scale_idx", 0)), self.N - 1)))

        # scrubbing: stop/pause playback and sync audio offset to the new frame
        if self.animation_running:
            self._audio_stop(reset=False)
        self.animation_running = False

        self.current_frame = idx
        self._audio_seek_to_frame(idx)  # updates offset (won't play since animation_running=False)
        self._update_view(idx)

    def _on_scale_release(self):
        # ensure final position is applied immediately
        if self._scale_job is not None:
            try:
                self.win.after_cancel(self._scale_job)
            except Exception:
                pass
            self._scale_job = None
        self._apply_pending_scale()

    def _load_sessions_async(self):
        if self.s3 is None:
            messagebox.showerror("Not connected", "S3 client not available.")
            return

        self.cmb_session["values"] = []
        self.session_var.set("")
        self.session_keys = {}
        self._set_info("Listing sessions…")

        if self.day_prefix:
            base_prefix = self.day_prefix
        else:
            layout = S3Layout(self.root_prefix)
            base_prefix = pick_day_prefix(layout, self.s3, self.bucket, self.user_id, self.day) or f"{self.root_prefix}{self.user_id}-{self.day}/"

        def worker():
            try:
                objs = list_objects(self.s3, self.bucket, base_prefix, max_items=5000)
                keys = [o["Key"] for o in objs if "Key" in o]
                csv_keys = [k for k in keys if k.lower().endswith(".csv") and "/voice_drift/" not in k]
                stems = []
                for k in csv_keys:
                    stem = Path(k).name[:-4]  # strip .csv
                    stems.append(stem)
                    self.session_keys[stem] = k
                stems = sorted(set(stems))

                def ui_update():
                    self.cmb_session["values"] = stems
                    if stems:
                        chosen = stems[0]
                        if self.preselect_session:
                            for s in stems:
                                if s == self.preselect_session or s.startswith(self.preselect_session):
                                    chosen = s
                                    break
                        self.session_var.set(chosen)
                        self._load_selected_session_async()
                    else:
                        self._set_info("No CSV sessions found for this day.")

                self.parent.after(0, ui_update)
            except Exception as e:
                self.parent.after(0, lambda: messagebox.showerror("Session list error", str(e)))
                self.parent.after(0, lambda: self._set_info(f"Session list error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Load selected session ----------
    def _load_selected_session_async(self):
        stem = self.session_var.get().strip()
        if not stem:
            return
        if stem not in self.session_keys:
            messagebox.showerror("Missing session", f"Session '{stem}' not found.")
            return

        csv_key = self.session_keys[stem]
        base_key_dir = str(Path(csv_key).parent).replace("\\", "/")
        mp4_key = f"{base_key_dir}/{stem}.mp4"
        wav_key = f"{base_key_dir}/{stem}.wav"
        # Tag file (TXT): action|radar: [(start,end), ...]
        txt_key = f"{base_key_dir}/{stem}.txt"

        # local paths
        sess_dir = self.cache_root / stem
        csv_path = sess_dir / f"{stem}.csv"
        mp4_path = sess_dir / f"{stem}.mp4"
        wav_path = sess_dir / f"{stem}.wav"
        txt_path = sess_dir / f"{stem}.txt"
        sess_dir.mkdir(parents=True, exist_ok=True)

        self._set_info("Downloading session assets…")

        def worker():
            try:
                _s3_download_to_file(self.s3, self.bucket, csv_key, csv_path)
                # mp4/wav/txt are optional
                try:
                    _s3_download_to_file(self.s3, self.bucket, mp4_key, mp4_path)
                except Exception:
                    if mp4_path.exists():
                        mp4_path.unlink(missing_ok=True)
                try:
                    _s3_download_to_file(self.s3, self.bucket, wav_key, wav_path)
                except Exception:
                    if wav_path.exists():
                        wav_path.unlink(missing_ok=True)
                try:
                    _s3_download_to_file(self.s3, self.bucket, txt_key, txt_path)
                except Exception:
                    # ok: no existing tags yet
                    if txt_path.exists():
                        txt_path.unlink(missing_ok=True)

                def ui_update():
                    self._load_local_session(
                        stem,
                        csv_path,
                        mp4_path if mp4_path.exists() else None,
                        wav_path if wav_path.exists() else None,
                        txt_path if txt_path.exists() else None,
                    )
                    self._set_info("Loaded.")

                self.parent.after(0, ui_update)
            except Exception as e:
                self.parent.after(0, lambda: messagebox.showerror("Download error", str(e)))
                self.parent.after(0, lambda: self._set_info(f"Download error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _load_local_session(
            self,
            stem: str,
            csv_path: Path,
            mp4_path: Path | None,
            wav_path: Path | None,
            txt_path: Path | None,
    ):
        if pd is None:
            messagebox.showerror("Missing dependency", "pandas is required for tagging window. pip install pandas")
            return
        self.stop()

        self.stem = stem
        self.csv_path = csv_path
        self.mp4_path = mp4_path
        self.wav_path = wav_path
        # tag file path (TXT)
        self.txt_path = txt_path if txt_path is not None else (csv_path.with_suffix(".txt"))

        self.df = pd.read_csv(str(csv_path))
        if "timestamp" not in self.df.columns:
            raise ValueError("CSV must contain 'timestamp' column for sync.")
        if "frame_number" in self.df.columns:
            self.N = int(len(self.df["frame_number"]))
        else:
            self.N = int(len(self.df))
        # cap window size to available frames
        self.window_size = self._coerce_window_size(self.window_size_var.get())
        self.window_size_var.set(str(self.window_size))
        self.timestamps = self.df["timestamp"].astype(float).to_numpy()
        self.ts0 = float(self.timestamps[0]) if self.N > 0 else 0.0

        # update scrubber range
        if getattr(self, "frame_scale", None) is not None:
            try:
                self.frame_scale.config(from_=1, to=max(1, int(self.N)))
                self.frame_scale.set(1)
            except Exception:
                pass

        # infer sample size from first row ch1
        first = self.df.iloc[0]
        dev0_ch1 = _safe_literal_array(first.get("chirp_data_ch1", "[]"))
        if dev0_ch1.ndim == 2 and dev0_ch1.shape[1] > 0:
            num_samples = int(dev0_ch1.shape[1])
        else:
            num_samples = 64
        self.range_axis = _compute_range_axis_m(num_samples=num_samples, f_start_hz=60250345952.0,
                                                f_end_hz=61249654048.0)

        # precompute distance magnitudes (avg over 3 channels) to keep UI snappy
        win = np.hanning(num_samples).astype(np.float32)

        dev0_mag = np.zeros((self.N, self.range_axis.size), dtype=np.float32)

        for i in range(self.N):
            row = self.df.iloc[i]
            # dev0
            d0 = []
            for ch in ("chirp_data_ch1", "chirp_data_ch2", "chirp_data_ch3"):
                mat = _safe_literal_array(row.get(ch, "[]"))
                if mat.size:
                    d0.append(_compute_distance_mag(mat, win))
            if d0:
                dev0_mag[i, :] = np.mean(np.stack(d0, axis=0), axis=0)

        self.dev0_mag = dev0_mag
        self.dev1_mag = None

        # Compute phase + derived respiration/heart rate from this session (best-effort).
        try:
            self._compute_phase_and_rates()
        except Exception:
            # Keep phase panel empty if anything goes wrong.
            self.phase_t = None
            self.phase_unwrap = None
            self.phase_resp = None
            self.phase_heart = None
            self.est_rr_bpm = None
            self.est_hr_bpm = None
            self.phase_bin_m = None
        # Pre-compute simple motion flags for overlay on video.
        # We detect motion by measuring frame-to-frame change in the range profile magnitude.
        self.radar_motion = [False] * max(0, self.N)
        try:
            if self.N >= 2:
                d0 = np.mean(np.abs(np.diff(self.dev0_mag, axis=0)), axis=1)

                th0 = float(np.percentile(d0, 75)) if d0.size else 0.0

                for i in range(1, self.N):
                    self.radar_motion[i] = bool(d0[i - 1] > th0)
        except Exception:
            # If anything goes wrong, keep flags False.
            self.radar_motion = [False] * max(0, self.N)
        # open video
        self.video_loaded = False
        self.video_cap = None
        self.video_fps = 0.0
        self.video_total = 0
        if mp4_path is not None and cv2 is not None and mp4_path.exists():
            cap = cv2.VideoCapture(str(mp4_path))
            if cap.isOpened():
                self.video_cap = cap
                self.video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self.video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.video_loaded = True
            else:
                try:
                    cap.release()
                except Exception:
                    pass

        # load audio (best-effort, seekable via pygame backend)
        self.audio_params = None  # (channels, sampwidth_bytes=2, samplerate)
        self.audio_frames = None  # raw PCM16 bytes (interleaved)
        self.audio_offset_sec = 0.0
        self.audio_play_t0 = None  # monotonic time when playback started

        if wav_path is not None and wav_path.exists():
            try:
                ch, sw, sr, raw16 = _read_wav_as_pcm16_bytes(wav_path)
                self.audio_params = (int(ch), int(sw), int(sr))
                self.audio_frames = raw16
            except Exception:
                self.audio_params = None
                self.audio_frames = None

        # initialize view
        self.current_frame = 0
        self._update_view(0)
        self.display_tagged_ranges()

    # ---------- Playback / navigation ----------

    # ---------- Audio helpers ----------
    def _pygame_init(self):
        """Initialize pygame mixer for the current WAV parameters."""
        if pygame is None:
            return False
        if self.audio_params is None:
            return False
        ch, sw, sr = self.audio_params
        try:
            if not getattr(self, "_pygame_inited", False):
                # size is in bits; negative => signed
                pygame.mixer.init(frequency=int(sr), size=int(-8 * sw), channels=int(ch))
                self._pygame_inited = True
        except Exception:
            return False
        return True

    def _audio_cleanup_temp(self):
        try:
            p = getattr(self, "_audio_temp_path", None)
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        self._audio_temp_path = None

    def _audio_play_from(self, start_sec: float):
        """Play audio from start_sec (best-effort). Requires pygame + PCM WAV loaded.

        Note: pygame seeking in WAV is not consistently supported across platforms/builds,
        so we generate a temporary WAV that starts at the requested offset and play from t=0.
        """
        if self.audio_params is None or self.audio_frames is None:
            return
        if not self._pygame_init():
            return

        ch, sw, sr = self.audio_params
        bpf = ch * sw
        start_frame = int(max(0.0, start_sec) * sr)
        start_byte = start_frame * bpf
        if start_byte >= len(self.audio_frames):
            return

        # stop existing
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        self.audio_play_t0 = None
        self._audio_cleanup_temp()

        # write a temp wav slice
        try:
            buf = self.audio_frames[start_byte:]
            fd, tmp_path = tempfile.mkstemp(prefix="tagger_audio_", suffix=".wav")
            os.close(fd)
            import wave
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(int(ch))
                wf.setsampwidth(int(sw))
                wf.setframerate(int(sr))
                wf.writeframes(buf)
            self._audio_temp_path = tmp_path
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            self.audio_play_t0 = time.monotonic()
        except Exception:
            self._audio_cleanup_temp()
            self.audio_play_t0 = None

    def _audio_pause(self):
        """Pause audio by stopping and accumulating elapsed time."""
        if self.audio_play_t0 is not None:
            try:
                elapsed = time.monotonic() - self.audio_play_t0
                if elapsed > 0:
                    self.audio_offset_sec += float(elapsed)
            except Exception:
                pass
        try:
            if pygame is not None:
                pygame.mixer.music.stop()
        except Exception:
            pass
        self.audio_play_t0 = None
        self._audio_cleanup_temp()

    def _audio_stop(self, reset: bool = False):
        """Stop audio playback; optionally reset offset to 0."""
        try:
            if pygame is not None:
                pygame.mixer.music.stop()
        except Exception:
            pass
        self.audio_play_t0 = None
        self._audio_cleanup_temp()
        if reset:
            self.audio_offset_sec = 0.0

    def _audio_seek_to_frame(self, frame_idx: int):
        """Seek audio to match a given radar frame timestamp."""
        if self.N <= 0 or frame_idx < 0 or frame_idx >= self.N:
            return
        t = float(self.timestamps[frame_idx] - self.timestamps[0])
        self.audio_offset_sec = max(0.0, t)
        if self.animation_running:
            self._audio_play_from(self.audio_offset_sec)

    def start_synced_play(self):
        if self.N <= 0:
            return

        # Start/resume from the current frame's timestamp (seconds since session start)
        t0 = float(self.timestamps[self.current_frame] - self.timestamps[0]) if self.current_frame < self.N else 0.0
        self.audio_offset_sec = max(0.0, t0)

        self.animation_running = True
        self._audio_play_from(self.audio_offset_sec)

        self._animate_step()

    def _animate_step(self):
        if not self.animation_running:
            return
        if self.current_frame >= self.N:
            self.animation_running = False
            return
        self._update_view(self.current_frame)

        # auto-fill a small suggested tag range (like original tagger)
        start = self.current_frame
        w = int(getattr(self, 'window_size', 16) or 16)
        if w <= 0:
            w = 1
        end = min(self.current_frame + w - 1, self.N - 1)
        self.range_entry.delete(0, tk.END)
        self.range_entry.insert(0, f"({start},{end})")

        self.current_frame += 1

        # pacing: use video fps if available; else derive from timestamps
        delay_ms = 200
        if self.video_loaded and self.video_fps > 0:
            delay_ms = int(1000.0 / self.video_fps)
        elif self.N >= 2:
            dt = float(
                self.timestamps[min(self.current_frame, self.N - 1)] - self.timestamps[max(self.current_frame - 1, 0)])
            if dt > 0:
                delay_ms = max(1, int(1000.0 * dt))
        self.win.after(delay_ms, self._animate_step)


    def _coerce_window_size(self, val: str) -> int:
        try:
            n = int(str(val).strip())
        except Exception:
            n = 16
        if n <= 0:
            n = 1
        # If data loaded, cap to N so (start,end) stays meaningful
        if getattr(self, "N", 0) and n > int(self.N):
            n = int(self.N)
        return n

    def on_update_window_size(self):
        # Commit the window size and refresh the suggested frame range.
        n = self._coerce_window_size(self.window_size_var.get())
        self.window_size = n
        self.window_size_var.set(str(n))
        # Refresh range suggestion for the current frame (without disrupting focus)
        try:
            if getattr(self, "range_entry", None) is not None:
                focused = (self.win.focus_get() == self.range_entry)
            else:
                focused = False
        except Exception:
            focused = False
        if not focused:
            self._update_view(self.current_frame)


    def pause(self):
        # Pause playback (best-effort). For audio we stop and remember the elapsed offset.
        if self.animation_running:
            self._audio_pause()
        self.animation_running = False

    def stop(self):
        self.animation_running = False
        self._audio_stop(reset=True)
        self.current_frame = 0
        self._update_view(0)

    def next_frame(self):
        # step forward one frame; pause playback and keep audio offset synced
        if self.animation_running:
            self._audio_stop(reset=False)
        self.animation_running = False
        if self.current_frame < self.N - 1:
            self.current_frame += 1
        self._audio_seek_to_frame(self.current_frame)  # updates offset (won't play since animation_running=False)
        self._update_view(self.current_frame)

    def prev_frame(self):
        # step back one frame; pause playback and keep audio offset synced
        if self.animation_running:
            self._audio_stop(reset=False)
        self.animation_running = False
        if self.current_frame > 0:
            self.current_frame -= 1
        self._audio_seek_to_frame(self.current_frame)  # updates offset (won't play since animation_running=False)
        self._update_view(self.current_frame)


    # ---------- Plot helpers ----------
    def _ensure_plot_mode(self, mode: str):
        """(Re)configure matplotlib axes for the selected plot mode.

        For single-radar sessions we use:
          - ax0: Range FFT or Doppler FFT for the *current frame*
          - ax1: Phase (unwrapped) across the whole session + derived RR/HR
        """
        if self.canvas is None or self.fig is None or self.ax0 is None or self.ax1 is None:
            return

        mode = (mode or "Range FFT").strip()
        if mode == getattr(self, "_last_plot_mode", None):
            return

        # Reset only ax0 artists; keep ax1 for phase panel.
        try:
            self.ax0.cla()
        except Exception:
            pass
        self.line0 = None
        self.im0 = None

        if mode == "Doppler FFT":
            self.ax0.set_title("Radar — Doppler FFT (velocity vs distance)")
            self.ax0.set_xlabel("Distance (m)")
            self.ax0.set_ylabel("Velocity (m/s)")
        else:
            self.ax0.set_title("Radar — Range FFT (distance vs amplitude)")
            self.ax0.set_xlabel("Distance (m)")
            self.ax0.set_ylabel("Amplitude")
            (self.line0,) = self.ax0.plot([], [])

        self._last_plot_mode = mode

        # (Re)plot phase panel in ax1 if we have it.
        self._plot_phase_panel()

        try:
            self.fig.subplots_adjust(hspace=0.65, left=0.10, right=0.97, top=0.94, bottom=0.11)
        except Exception:
            pass


    # ---------- Phase / vitals ----------
    def _plot_phase_panel(self):
        """Plot unwrapped phase across time plus band-limited respiration/heart components."""
        if self.ax1 is None:
            return

        # ax1.cla() removes all artists from the axes. If we keep references to artists
        # (e.g., self.phase_cursor) across clears, subsequent updates will mutate a
        # *detached* artist that is no longer in the axes, making the cursor appear to
        # "disappear" after mode switches. So, reset per-axes artist handles before
        # clearing/replotting.
        self.phase_line = None
        self.resp_line = None
        self.heart_line = None
        self.phase_cursor = None
        try:
            self.ax1.cla()
        except Exception:
            return

        if self.phase_t is None or self.phase_unwrap is None or len(self.phase_unwrap) < 4:
            self.ax1.set_title("Phase (not available)")
            self.ax1.set_xlabel("Time (s)")
            self.ax1.set_ylabel("Phase (rad)")
            try:
                self.canvas.draw()
            except Exception:
                pass
            return

        t = self.phase_t
        ph = self.phase_unwrap

        # Main phase
        (self.phase_line,) = self.ax1.plot(t, ph, label="phase (unwrap)")

        # Overlay band components if available
        if self.phase_resp is not None and len(self.phase_resp) == len(ph):
            (self.resp_line,) = self.ax1.plot(t, self.phase_resp, label="resp band")
        if self.phase_heart is not None and len(self.phase_heart) == len(ph):
            (self.heart_line,) = self.ax1.plot(t, self.phase_heart, label="heart band")

        # Vertical cursor synced to the current frame (time axis).
        try:
            _i = int(max(0, min(int(getattr(self, "current_frame", 0)), len(t) - 1)))
            cur_t = float(t[_i])
        except Exception:
            cur_t = float(t[0])
        if getattr(self, "phase_cursor", None) is None:
            self.phase_cursor = self.ax1.axvline(cur_t, linestyle="--", linewidth=1.0, alpha=0.9)
        else:
            try:
                self.phase_cursor.set_xdata([cur_t, cur_t])
            except Exception:
                try:
                    self.phase_cursor.remove()
                except Exception:
                    pass
                self.phase_cursor = self.ax1.axvline(cur_t, linestyle="--", linewidth=1.0, alpha=0.9)


        rr = self.est_rr_bpm
        hr = self.est_hr_bpm
        bin_m = self.phase_bin_m
        parts = []
        if bin_m is not None:
            parts.append(f"bin≈{bin_m:.2f} m")
        if rr is not None:
            parts.append(f"RR≈{rr:.1f} bpm")
        if hr is not None:
            parts.append(f"HR≈{hr:.1f} bpm")
        suffix = (" — " + ", ".join(parts)) if parts else ""
        self.ax1.set_title("Phase / vitals" + suffix)

        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Phase (rad)")
        try:
            self.ax1.legend(loc="upper right", fontsize=8)
        except Exception:
            pass

        try:
            self.canvas.draw()
        except Exception:
            pass
    def _update_phase_cursor(self, idx: int):
        """Update (or create) the vertical cursor line on the phase plot (ax1)."""
        if self.ax1 is None or self.canvas is None:
            return
        if self.phase_t is None or len(self.phase_t) == 0:
            return
        try:
            i = int(max(0, min(int(idx), len(self.phase_t) - 1)))
            cur_t = float(self.phase_t[i])
        except Exception:
            return

        if getattr(self, "phase_cursor", None) is None:
            try:
                self.phase_cursor = self.ax1.axvline(cur_t, linestyle="--", linewidth=1.0, alpha=0.9)
            except Exception:
                return
        else:
            try:
                self.phase_cursor.set_xdata([cur_t, cur_t])
            except Exception:
                try:
                    self.phase_cursor.remove()
                except Exception:
                    pass
                try:
                    self.phase_cursor = self.ax1.axvline(cur_t, linestyle="--", linewidth=1.0, alpha=0.9)
                except Exception:
                    return

        try:
            self.canvas.draw_idle()
        except Exception:
            try:
                self.canvas.draw()
            except Exception:
                pass



    def _bandpass_via_fft(self, x: np.ndarray, fs_hz: float, f_lo: float, f_hi: float) -> np.ndarray:
        """Simple bandpass using FFT masking (no scipy)."""
        n = int(x.size)
        if n < 4 or fs_hz <= 0:
            return np.zeros_like(x)
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
        mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
        Xf = X * mask
        y = np.fft.irfft(Xf, n=n)
        return y.astype(np.float32)

    def _estimate_peak_bpm(self, x: np.ndarray, fs_hz: float, f_lo: float, f_hi: float) -> float | None:
        """Estimate dominant frequency in [f_lo,f_hi] and return bpm."""
        n = int(x.size)
        if n < 8 or fs_hz <= 0:
            return None
        win = np.hanning(n).astype(np.float32)
        X = np.fft.rfft((x - float(np.mean(x))) * win)
        P = (np.abs(X) ** 2).astype(np.float64)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
        band = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
        if not np.any(band):
            return None
        idx = int(np.argmax(P[band]))
        f_peak = float(freqs[band][idx])
        if not np.isfinite(f_peak) or f_peak <= 0:
            return None
        return float(f_peak * 60.0)

    def _compute_phase_and_rates(self):
        """Compute unwrapped phase across frames + RR/HR estimates from phase."""
        self.phase_t = None
        self.phase_unwrap = None
        self.phase_resp = None
        self.phase_heart = None
        self.est_rr_bpm = None
        self.est_hr_bpm = None
        self.phase_bin_m = None

        if self.df is None or self.timestamps is None or self.range_axis is None or self.N <= 3:
            return

        # Time axis (seconds since start), using absolute unix timestamps.
        t = (self.timestamps.astype(np.float64) - float(self.timestamps[0])).astype(np.float64)
        # Robust sampling rate estimate
        dt = np.diff(t)
        med_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
        fs = (1.0 / med_dt) if med_dt > 0 else 0.0
        if fs <= 0:
            return

        # Pick a range bin with high energy in a plausible human range (0.3m..2.5m).
        mag_mean = np.mean(self.dev0_mag, axis=0) if getattr(self, "dev0_mag", None) is not None else None
        if mag_mean is None or mag_mean.size != self.range_axis.size:
            return
        mask = (self.range_axis >= 0.30) & (self.range_axis <= 2.50)
        if not np.any(mask):
            mask = np.ones_like(self.range_axis, dtype=bool)
        k = int(np.argmax(mag_mean[mask]))
        k = int(np.where(mask)[0][k])
        self.phase_bin_m = float(self.range_axis[k])

        # Compute complex range FFT value at bin k for each frame (avg over chirps + channels).
        first = self.df.iloc[0]
        mat0 = _safe_literal_array(first.get("chirp_data_ch1", "[]"))
        num_samples = int(mat0.shape[1]) if mat0.ndim == 2 and mat0.shape[1] > 0 else 64
        win = np.hanning(num_samples).astype(np.float32)

        z = np.zeros((self.N,), dtype=np.complex64)
        for i in range(self.N):
            row = self.df.iloc[i]
            zs = []
            for ch in ("chirp_data_ch1", "chirp_data_ch2", "chirp_data_ch3"):
                mat = _safe_literal_array(row.get(ch, "[]"))
                if mat.size == 0:
                    continue
                if mat.ndim != 2:
                    mat = np.asarray(mat).reshape(-1, num_samples)
                if mat.shape[1] != num_samples:
                    # reshape best-effort
                    mat = mat[:, :num_samples] if mat.shape[1] > num_samples else np.pad(mat, ((0,0),(0,num_samples-mat.shape[1])), mode="constant")
                X = np.fft.fft(mat.astype(np.float32) * win.reshape(1, -1), axis=1)  # (chirps, samples)
                X = X[:, : max(1, num_samples // 2)]
                v = np.mean(X[:, k]) if k < X.shape[1] else np.mean(X[:, -1])
                zs.append(v)
            if zs:
                z[i] = np.mean(np.asarray(zs, dtype=np.complex64))
            else:
                z[i] = 0.0 + 0.0j

        ph = np.unwrap(np.angle(z.astype(np.complex128))).astype(np.float32)

        # Detrend (remove linear drift)
        try:
            p = np.polyfit(t, ph.astype(np.float64), deg=1)
            ph_d = (ph - (p[0] * t + p[1]).astype(np.float32)).astype(np.float32)
        except Exception:
            ph_d = (ph - float(np.mean(ph))).astype(np.float32)

        self.phase_t = t.astype(np.float32)
        self.phase_unwrap = ph_d

        # Band components
        self.phase_resp = self._bandpass_via_fft(ph_d, fs_hz=fs, f_lo=0.10, f_hi=0.60)
        self.phase_heart = self._bandpass_via_fft(ph_d, fs_hz=fs, f_lo=0.80, f_hi=2.50)

        # Rate estimates (bpm)
        self.est_rr_bpm = self._estimate_peak_bpm(ph_d, fs_hz=fs, f_lo=0.10, f_hi=0.60)
        self.est_hr_bpm = self._estimate_peak_bpm(ph_d, fs_hz=fs, f_lo=0.80, f_hi=2.50)


    # ---------- View update ----------
    def _update_view(self, idx: int):
        idx = int(max(0, min(idx, max(0, self.N - 1))))
        self.lbl_info.config(text=f"Frame: {idx + 1}/{max(1, int(self.N))}")

        # keep scrubber in sync (avoid recursion while dragging)
        # NOTE: Tk Scale 'command' callbacks may run after .set() returns, so we keep the guard
        # flag true until the next idle cycle to prevent playback from being paused by the slider callback.
        if getattr(self, "frame_scale", None) is not None and not getattr(self, "_scale_updating", False):
            self._scale_updating = True
            try:
                self.frame_scale.set(idx + 1)
            finally:
                self.win.after_idle(lambda: setattr(self, "_scale_updating", False))

        # keep the frame range entry in sync when navigating/scrubbing
        # (don't overwrite while the user is actively typing in the entry)
        try:
            focused = (getattr(self, "range_entry", None) is not None and self.win.focus_get() == self.range_entry)
        except Exception:
            focused = False
        if not focused and getattr(self, "range_entry", None) is not None and self.N > 0:
            start_disp = idx + 1
            end_disp = min(start_disp + max(1, int(getattr(self, 'window_size', 16))) - 1, int(self.N))
            suggested = f"({start_disp},{end_disp})"
            try:
                if self.range_entry.get().strip() != suggested:
                    self.range_entry.delete(0, tk.END)
                    self.range_entry.insert(0, suggested)
            except Exception:
                pass

        # plot
        if self.canvas is not None and self.range_axis is not None and self.df is not None:
            mode = (self.plot_mode_var.get() or "Range FFT").strip()
            self._ensure_plot_mode(mode)
            self._update_phase_cursor(idx)


            if mode == "Doppler FFT":
                # Compute Doppler map on-demand for this frame. Average magnitude across 3 channels.
                row = self.df.iloc[idx]

                def rd_for_device(prefix: str):
                    rd_list = []
                    vel_axis = None
                    for ch in ("ch1", "ch2", "ch3"):
                        key = f"{prefix}_{ch}" if prefix else f"chirp_data_{ch}"
                        mat = _safe_literal_array(row.get(key, "[]"))
                        if mat.size:
                            rd_mag, vel = _compute_range_doppler_mag(
                                mat, np.hanning(max(1, mat.shape[1])).astype(np.float32), float(getattr(self, "doppler_prf_hz", 2000.0))
                            )
                            rd_list.append(rd_mag)
                            if vel_axis is None:
                                vel_axis = vel
                    if not rd_list:
                        return np.zeros((1, max(1, int(self.range_axis.size))), dtype=np.float32), np.zeros((1,), dtype=np.float32)
                    rd = np.mean(np.stack(rd_list, axis=0), axis=0).astype(np.float32)
                    return rd, (vel_axis if vel_axis is not None else np.zeros((rd.shape[0],), dtype=np.float32))

                rd0, vel0 = rd_for_device("")

                # Log scale for visibility
                eps = 1e-6
                rd0_db = 20.0 * np.log10(np.maximum(rd0, eps))

                # Plot as an image (velocity vs distance)
                x0 = float(self.range_axis[0]) if self.range_axis.size else 0.0
                x1 = float(self.range_axis[-1]) if self.range_axis.size else 1.0
                y0 = float(vel0[0]) if vel0.size else -1.0
                y1 = float(vel0[-1]) if vel0.size else 1.0
                extent = [x0, x1, y0, y1]

                if self.im0 is None:
                    self.im0 = self.ax0.imshow(
                        rd0_db,
                        aspect="auto",
                        origin="lower",
                        extent=extent,
                    )
                else:
                    try:
                        self.im0.set_data(rd0_db)
                        self.im0.set_extent(extent)
                    except Exception:
                        # If artist is incompatible (e.g., after backend reset), recreate
                        self.ax0.cla()
                        self.im0 = self.ax0.imshow(
                            rd0_db,
                            aspect="auto",
                            origin="lower",
                            extent=extent,
                        )

                # Autoscale color limits each frame for now (robust: ignore NaNs)
                try:
                    vmin = float(np.nanpercentile(rd0_db, 5))
                    vmax = float(np.nanpercentile(rd0_db, 99))
                    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                        self.im0.set_clim(vmin, vmax)
                except Exception:
                    pass

                self.ax0.set_title("Radar — Doppler FFT (velocity vs distance)")
                self.ax0.set_xlabel("Distance (m)")
                self.ax0.set_ylabel("Velocity (m/s)")

                self.canvas.draw()
            else:
                # Range FFT (precomputed)
                if self.dev0_mag is not None:
                    y0 = self.dev0_mag[idx, :]
                    if self.line0 is None:
                        # In case mode was switched from Doppler -> Range FFT
                        self._ensure_plot_mode("Range FFT")
                    if self.line0 is not None:
                        self.line0.set_data(self.range_axis, y0)
                        self.ax0.relim()
                        self.ax0.autoscale_view()
                    self.canvas.draw()

        # video
        if self.video_loaded and self.video_cap is not None and cv2 is not None and Image is not None and ImageTk is not None:
            t = float(self.timestamps[idx] - self.ts0) if self.timestamps is not None else 0.0
            v_idx = 0
            if self.video_fps > 0:
                v_idx = int(round(t * self.video_fps))
            v_idx = int(max(0, min(v_idx, max(0, self.video_total - 1))))
            try:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, v_idx)
                ok, frame = self.video_cap.read()
                if ok and frame is not None:
                    frame = cv2.resize(frame, (420, 315))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)

                    # Overlay motion status on the video (single radar)
                    try:
                        d = ImageDraw.Draw(img)
                        txt = f"radar: {'Motion' if (hasattr(self, 'radar_motion') and idx < len(self.radar_motion) and self.radar_motion[idx]) else 'No Motion'}"
                        x1, y1 = 8, img.height - 22
                        d.text((x1, y1), txt, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                    except Exception:
                        pass

                    imgtk = ImageTk.PhotoImage(image=img)
                    self.preview_image = imgtk
                    self.video_label.config(image=imgtk)
            except Exception:
                pass

    def _set_info(self, msg: str):
        self.lbl_info.config(text=msg)
        self.win.update_idletasks()

    # ---------- Tag persistence ----------
    def _load_tag_store(self) -> list[dict]:
        """Load tags from TXT at <stem>.txt.

        Format:
          action: [(start,end), (start,end)]
          action|anything: [(start,end), ...]   (accepted for backward compatibility; suffix ignored)

        Returns: list of segments dicts with keys: action, start, end (all ints; 1-index inclusive).
        """
        segs: list[dict] = []
        if getattr(self, "txt_path", None) is None or not self.txt_path.exists():
            return segs
        try:
            lines = self.txt_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return segs

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                left, ranges_str = line.split(":", 1)
                left = left.strip()
                # allow action|suffix, keep only action
                act = left.split("|", 1)[0].strip()
                ranges = ast.literal_eval(ranges_str.strip())
                if isinstance(ranges, tuple):
                    ranges = [ranges]
                if not isinstance(ranges, list):
                    ranges = [ranges]
                for s, e in ranges:
                    segs.append({"action": str(act).strip(), "start": int(s), "end": int(e)})
            except Exception:
                continue
        return segs



        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                left, ranges_str = line.split(":", 1)
                left = left.strip()
                ranges = ast.literal_eval(ranges_str.strip())
                if isinstance(ranges, tuple):
                    ranges = [ranges]
                if not isinstance(ranges, list):
                    ranges = [ranges]

                radar = "both"
                act = left
                if "|" in left:
                    act, radar = [x.strip() for x in left.split("|", 1)]
                    radar = radar.lower()
                    if radar not in {"dev0", "dev1", "both"}:
                        radar = "both"

                for s, e in ranges:
                    segs.append({"action": str(act).strip(), "radar": radar, "start": int(s), "end": int(e)})
            except Exception:
                continue
        return segs

    def _write_tag_store(self, segments: list[dict]):
        """Write tags to TXT (human-readable and S3-uploaded)."""
        # normalize + sort
        clean: list[dict] = []
        for it in segments:
            if not isinstance(it, dict):
                continue
            act = str(it.get("action", "")).strip()
            try:
                s = int(it.get("start"))
                e = int(it.get("end"))
            except Exception:
                continue
            if not act:
                continue
            clean.append({"action": act, "start": s, "end": e})
        clean.sort(key=lambda x: (x["start"], x["end"], x["action"]))

        out_dir = self.txt_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        grouped: dict[str, list[tuple[int, int]]] = {}
        for it in clean:
            grouped.setdefault(it["action"], []).append((int(it["start"]), int(it["end"])))

        try:
            with open(self.txt_path, "w", encoding="utf-8") as f:
                for act in sorted(grouped.keys()):
                    f.write(f"{act}: {sorted(grouped[act])}\n")
        except Exception:
            pass



    def save_range(self):
        if self.N <= 0:
            return
        action = self.action_var.get().strip()
        if not action:
            messagebox.showerror("Error", "No action selected.")
            return

        frame_range_str = self.range_entry.get().strip()
        try:
            new_range = tuple(ast.literal_eval(frame_range_str))
            assert isinstance(new_range, tuple) and len(new_range) == 2
            new_start, new_end = int(new_range[0]), int(new_range[1])
            if new_start > new_end:
                raise ValueError
            if new_start < 1 or new_end > int(self.N):
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid Input", f"Enter a valid frame range like (1, 10) within [1, {int(self.N)}].")
            return

        segments = self._load_tag_store()

        # overlap check (across ALL tags regardless of radar)
        overlaps = []
        for it in segments:
            try:
                s, e = int(it["start"]), int(it["end"])
                if not (new_end < s or new_start > e):
                    overlaps.append((str(it.get("action", "")), (s, e)))
            except Exception:
                continue
        if overlaps:
            msg = "Overlap detected with existing ranges:\n" + "\n".join([f"- {a}: {r}" for a, r in overlaps])
            messagebox.showerror("Overlap Error", msg)
            return

        segments.append({"action": action, "start": new_start, "end": new_end})
        self._write_tag_store(segments)

        # upload to S3 next to csv
        try:
            csv_key = self.session_keys[self.stem]
            base_key_dir = str(Path(csv_key).parent).replace("\\", "/")
            txt_key = f"{base_key_dir}/{self.stem}.txt"
            if getattr(self, "txt_path", None) is not None and self.txt_path.exists():
                _s3_upload_file(self.s3, self.bucket, txt_key, self.txt_path)
        except Exception as e:
            messagebox.showwarning("Upload warning", f"Saved locally, but failed to upload tags to S3:\n{e}")

        self.display_tagged_ranges()
        messagebox.showinfo("Saved", f"Range ({new_start},{new_end}) saved for '{action}'.")

    def delete_frame_range(self, act: str, rng: tuple[int, int]):
        segments = self._load_tag_store()
        act = str(act).strip()
        s0, e0 = int(rng[0]), int(rng[1])
        segments = [
            it for it in segments
            if not (
                str(it.get("action", "")).strip() == act
                and int(it.get("start", -1)) == s0
                and int(it.get("end", -1)) == e0
            )
        ]
        self._write_tag_store(segments)

        try:
            csv_key = self.session_keys[self.stem]
            base_key_dir = str(Path(csv_key).parent).replace("\\", "/")
            txt_key = f"{base_key_dir}/{self.stem}.txt"
            if getattr(self, "txt_path", None) is not None and self.txt_path.exists():
                _s3_upload_file(self.s3, self.bucket, txt_key, self.txt_path)
        except Exception:
            pass

        self.display_tagged_ranges()

    def jump_to_frame(self, frame_num: int):
        # frame_num is 1-indexed (UI / tags)
        idx = int(frame_num) - 1
        self.current_frame = int(max(0, min(idx, self.N - 1)))
        self.animation_running = False
        self._update_view(self.current_frame)

    def display_tagged_ranges(self):
        # clear
        for w in self.tagged_ranges_frame.winfo_children():
            w.destroy()

        segments = self._load_tag_store()
        if not segments:
            tk.Label(self.tagged_ranges_frame, text="(no tags yet)", font=self.ui_font).pack(anchor="w")
            return

        grouped: dict[str, list[tuple[int, int]]] = {}
        for it in segments:
            try:
                act = str(it.get("action", "")).strip()
                s, e = int(it.get("start")), int(it.get("end"))
            except Exception:
                continue
            if not act:
                continue
            grouped.setdefault(act, []).append((s, e))

        for act in sorted(grouped.keys()):
            ranges = sorted(grouped[act])
            row = tk.Frame(self.tagged_ranges_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{act} :", font=self.ui_font).pack(anchor="w")

            canvas = tk.Canvas(row, height=34)
            hsb = ttk.Scrollbar(row, orient="horizontal", command=canvas.xview)
            btn_frame = tk.Frame(canvas)
            btn_frame.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))
            canvas.create_window((0, 0), window=btn_frame, anchor="nw")
            canvas.configure(xscrollcommand=hsb.set)

            canvas.pack(fill="x", expand=True)
            hsb.pack(fill="x")

            for r in ranges:
                chip = tk.Frame(btn_frame, bd=1, relief=tk.SOLID, padx=2)
                chip.pack(side=tk.LEFT, padx=2)
                tk.Button(chip, text=str(r), font=self.ui_font, command=lambda s=r[0]: self.jump_to_frame(s)).pack(side=tk.LEFT)
                tk.Button(
                    chip,
                    text="×",
                    font=("Arial", 8, "bold"),
                    fg="red",
                    command=lambda a=act, rr=r: self.delete_frame_range(a, rr),
                ).pack(side=tk.LEFT)

    # ---------- lifecycle ----------

    def on_close(self):
        self.animation_running = False
        try:
            if pygame is not None:
                pygame.mixer.music.stop()
        except Exception:
            pass
        try:
            if self.video_cap is not None:
                self.video_cap.release()
        except Exception:
            pass
        self.win.destroy()


if __name__ == "__main__":
    main()
