#!/usr/bin/env python3
"""
ADL Real-time Classification GUI
==================================
Dual-radar real-time activity classification with live video and Doppler heatmaps.

GUI layout:
  ┌──────────────────────────┬────────────────┐
  │  Camera feed             │  Dev0 Doppler  │
  │  [time]     [elapsed]    │                │
  │                          ├────────────────┤
  │                          │  Dev1 Doppler  │
  ├─────────────────┬────────┴────────────────┤
  │  DEV 1 result   │        DEV 0 result     │
  └─────────────────┴─────────────────────────┘

  • Current datetime (yyyy-mm-dd hh:mm:ss) — upper-left of camera feed
  • Elapsed time (hh:mm:ss)                — upper-right of camera feed
  • Classification label + probability     — bottom-left (dev1), bottom-right (dev0)
  • Live Doppler heatmap (RX0 magnitude)   — right column, one per device

Radar processing matches adl_feature_extraction.py exactly:
  - DopplerAlgo (Infineon SDK) with persistent MTI filter
  - log1p magnitude
  - 3-channel features for 2D model (rx0/1/2 log-magnitude RD maps); 6-channel for 1D
  - Normalization from training checkpoint (norm_mean / norm_std fields)
  - Sliding-window inference: 64 frames window, new inference every 32 frames

Usage:
  python adl_realtime_classification.py --model 2d
  python adl_realtime_classification.py --model 1d
  python adl_realtime_classification.py --model 2d --cam-index 0

Dependencies (Raspberry Pi):
  sudo apt install -y python3-picamera2 python3-opencv python3-pil python3-tk
  pip install torch numpy
  # Infineon SDK: ifxradarsdk must be installed
"""

import sys
import os
import time
import threading
import argparse
import collections
from pathlib import Path

import numpy as np
import cv2

import tkinter as tk
from PIL import Image, ImageTk

import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
# This file lives in pi-2radar/; helpers/ and adl_model.py are one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from helpers.DopplerAlgo import DopplerAlgo
from adl_model import ADLClassifier, ADLClassifier1D

# ── Optional Picamera2 ────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    _PICAM2_OK = True
except Exception:
    Picamera2 = None
    _PICAM2_OK = False

# ── Optional Infineon SDK ─────────────────────────────────────────────────────
try:
    from ifxradarsdk.fmcw import DeviceFmcw
    from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics
    _RADAR_OK = True
except ImportError:
    DeviceFmcw = None
    FmcwSimpleSequenceConfig = FmcwMetrics = None
    _RADAR_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_CHIRPS    = 32
NUM_SAMPLES   = 64
NUM_RX        = 3

WINDOW_FRAMES = 50    # frames per inference window  (~12.8 s at 5 Hz)
STRIDE_FRAMES = 25    # run inference every N new frames
DEFAULT_FRATE = 5     # Hz

# Camera capture resolution (Picamera2 requests this from hardware)
CAM_W, CAM_H     = 1024, 768
# Doppler heatmap panel (right column, fixed size)
DOPPLER_W        = 240
DOPPLER_H        = 178
# Angle-map panels (third column, half-height so all 4 fit)
ANGLE_W          = 240
ANGLE_H          = 90
# Number of frames to average for the Capon display (single frames are too noisy)
CAPON_AVG_FRAMES = 15

# Colours (dark theme)
BG_DARK   = "#0d0d1a"
BG_MID    = "#1a1a2e"
COL_TIME  = "#00ff88"
COL_ELAPS = "#ffdd00"
COL_LABEL = "#00ff88"
COL_PROB  = "#88ffcc"
COL_HEAD  = "#aaaaff"
COL_WARM  = "#888888"


# ─────────────────────────────────────────────────────────────────────────────
# Radar configuration  (identical to raw_collection_ui_dual_with_audio.py)
# ─────────────────────────────────────────────────────────────────────────────

def configure_device(device, frate: int):
    """Configure an FMCW device with the project's standard metrics."""
    info = device.get_sensor_information()
    num_rx = info.get("num_rx_antennas", NUM_RX)
    metrics = FmcwMetrics(
        range_resolution_m=0.15,
        max_range_m=4.8,
        max_speed_m_s=2.45,
        speed_resolution_m_s=0.2,
        center_frequency_Hz=60_750_000_000,
    )
    seq = device.create_simple_sequence(FmcwSimpleSequenceConfig())
    seq.loop.repetition_time_s = 1.0 / frate
    chirp_loop = seq.loop.sub_sequence.contents
    device.sequence_from_metrics(metrics, chirp_loop)
    chirp = chirp_loop.loop.sub_sequence.contents.chirp
    chirp.sample_rate_Hz   = 1_000_000
    chirp.rx_mask          = (1 << num_rx) - 1
    chirp.tx_mask          = 1
    chirp.tx_power_level   = 31
    chirp.if_gain_dB       = 33
    chirp.lp_cutoff_Hz     = 500_000
    chirp.hp_cutoff_Hz     = 80_000
    device.set_acquisition_sequence(seq)
    return num_rx


def open_two_devices():
    """Discover and open exactly two FMCW devices. Returns (dev0, dev1)."""
    if not _RADAR_OK:
        raise RuntimeError("ifxradarsdk is not installed.")

    # Build candidate list
    cands = []
    for getter in ("get_list", "scan"):
        fn = getattr(DeviceFmcw, getter, None)
        if fn is None:
            continue
        try:
            for item in (fn() or []):
                if isinstance(item, str):
                    cands.append(item)
                elif isinstance(item, dict):
                    for k in ("uri", "id", "uuid", "serial", "device_id"):
                        if k in item and item[k]:
                            cands.append(str(item[k]))
                            break
                else:
                    try:
                        cands.append(str(item))
                    except Exception:
                        pass
        except Exception:
            pass
    cands += [f"usb:{i}" for i in range(4)]

    seen, unique = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    opened = []
    for hint in unique:
        if len(opened) >= 2:
            break
        for trial in [hint, f"usb:{hint}", f"uuid:{hint}"]:
            try:
                dev = DeviceFmcw(trial)
                dev.get_sensor_information()
                opened.append(dev)
                break
            except Exception:
                pass

    if len(opened) < 2:
        raise RuntimeError(
            f"Need 2 FMCW devices; only {len(opened)} found. "
            f"Candidates tried: {unique}"
        )
    return opened[0], opened[1]


# ─────────────────────────────────────────────────────────────────────────────
# Model + normalization loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_type: str, torch_device: torch.device):
    """
    Load model checkpoint from models/ (relative to the script's directory).
    Normalization stats (norm_mean, norm_std) and label_map are stored inside
    the checkpoint by adl_train.py — no separate sidecar needed.

    Returns (model, norm_mean, norm_std, label_names, window_frames).
    """
    models_dir = Path(__file__).resolve().parent / "models"
    stem       = "best_model_1d" if model_type == "1d" else "best_model"
    ckpt_path  = models_dir / f"{stem}.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Copy the trained .pt file to {models_dir}/"
        )

    ckpt = torch.load(str(ckpt_path), map_location=torch_device, weights_only=False)

    label_map     = ckpt["label_map"]           # {name: idx}
    label_names   = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    num_classes   = len(label_names)
    in_channels   = ckpt.get("in_channels", 6)
    window_frames = ckpt.get("window_frames", WINDOW_FRAMES)
    norm_mean     = ckpt["norm_mean"]            # list (see adl_train.py)
    norm_std      = ckpt["norm_std"]             # list

    if model_type == "1d":
        model = ADLClassifier1D(
            num_classes=num_classes,
            cnn_out_dim=64, gru_hidden=64, gru_layers=2,
            dropout=0.4, bidirectional=True, in_channels=in_channels,
        )
    else:
        model = ADLClassifier(
            num_classes=num_classes,
            cnn_out_dim=64, gru_hidden=64, gru_layers=2,
            dropout=0.4, bidirectional=True, in_channels=in_channels,
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(torch_device)
    model.eval()

    print(f"Loaded {model_type.upper()} model: {num_classes} classes, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    print(f"Classes: {label_names}")
    return model, norm_mean, norm_std, label_names, window_frames


# ─────────────────────────────────────────────────────────────────────────────
# Normalization  (mirrors adl_train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_window(window_np: np.ndarray, norm_mean, norm_std,
                     is_1d: bool) -> np.ndarray:
    """Apply training-time normalization to a raw feature window.

    New checkpoints (norm_mean == 'per_window'): per-window z-score per channel.
    Old checkpoints: legacy global stats path preserved for backward compatibility.
    """
    x = window_np.astype(np.float32).copy()

    if norm_mean in ("per_window", "per_session") or norm_std in ("per_window", "per_session"):
        # Per-window z-score: best available approximation of per-session
        # normalization at inference time (no full session history available).
        for c in range(x.shape[-1]):
            ch = x[..., c]
            m = float(ch.mean())
            s = float(ch.std()) + 1e-8
            x[..., c] = (ch - m) / s
    elif is_1d:
        # Legacy: per-channel z-score using stored global stats
        for c in range(x.shape[-1]):
            x[..., c] = (x[..., c] - norm_mean[c]) / norm_std[c]
    else:
        # Legacy: ch 0-2 max-scale, ch 3+ z-score
        mag_max = norm_std[0]
        x[..., :3] = x[..., :3] / max(float(mag_max), 1e-6)
        for i, c in enumerate(range(3, x.shape[-1])):
            x[..., c] = (x[..., c] - norm_mean[1 + i]) / norm_std[1 + i]

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Capon (MVDR) range-angle map
# ─────────────────────────────────────────────────────────────────────────────

def _capon_angle_map(rd_a: np.ndarray, rd_b: np.ndarray,
                     d_lambda: float = 0.5, n_angles: int = 64) -> np.ndarray:
    """Vectorised Capon range-angle map from two complex RD maps (n_range, n_doppler).
    Uses 64 Doppler bins as snapshots to estimate the 2×2 spatial covariance per range bin.
    Returns float32 (n_range, n_angles), log1p-normalised.
    """
    n_doppler = rd_a.shape[1]
    X   = np.stack([rd_a, rd_b], axis=1).astype(np.complex64)       # (R, 2, D)
    R   = (X @ X.conj().transpose(0, 2, 1)) / n_doppler             # (R, 2, 2)
    # Adaptive diagonal loading: 1% of per-range-bin signal power.
    # Fixed 1e-6 loading would dominate when RD map values are small (as on this radar),
    # making R^{-1} ≈ 1e6*I everywhere and erasing all spatial information.
    diag_load = np.real(np.diagonal(R, axis1=1, axis2=2)).mean(axis=1)  # (n_range,)
    R += np.maximum(diag_load * 0.01, 1e-10)[:, None, None] * np.eye(2, dtype=np.complex64)[None]
    R_inv = np.linalg.inv(R)                                         # (R, 2, 2)
    angles = np.linspace(-np.pi / 2, np.pi / 2, n_angles)
    A   = np.stack([np.ones(n_angles, dtype=np.complex64),
                    np.exp(1j * 2 * np.pi * d_lambda * np.sin(angles)).astype(np.complex64)],
                   axis=-1)                                           # (A, 2)
    AR  = np.einsum('ai,rij->raj', A.conj(), R_inv)                  # (A, R, 2)
    den = np.einsum('raj,aj->ra',  AR, A).real                       # (R, A)
    return np.log1p(1.0 / (np.abs(den) + 1e-10)).astype(np.float32)    # (R, A)


def _phase_diff_angle_map(rd_a: np.ndarray, rd_b: np.ndarray,
                           n_angles: int = 64) -> np.ndarray:
    """Simple phase-difference range-angle map for a 2-element array (d = λ/2).

    For each (range, Doppler) bin, Δφ = angle(rd_a · conj(rd_b)) and
    sin(θ) = Δφ / π.  Power is scatter-accumulated into a (n_range, n_angles)
    grid.  Returns float32 log1p-normalised map — same shape as _capon_angle_map.
    """
    cross     = rd_a * np.conj(rd_b)
    delta_phi = np.angle(cross)                              # (R, D)  in [-π, π]
    sin_theta = np.clip(delta_phi / np.pi, -1.0, 1.0)       # d=λ/2 → sin(θ)=Δφ/π
    power     = (np.abs(rd_a) * np.abs(rd_b)).astype(np.float32)

    n_range, n_doppler = rd_a.shape
    angle_edges = np.linspace(-1.0, 1.0, n_angles + 1)
    angle_idx   = np.clip(np.digitize(sin_theta, angle_edges) - 1, 0, n_angles - 1)

    r_idx  = np.repeat(np.arange(n_range), n_doppler)
    ra_map = np.zeros((n_range, n_angles), dtype=np.float32)
    np.add.at(ra_map, (r_idx, angle_idx.ravel()), power.ravel())
    return np.log1p(ra_map)


# ─────────────────────────────────────────────────────────────────────────────
# Per-device state  (shared between radar thread and GUI thread)
# ─────────────────────────────────────────────────────────────────────────────

class DeviceState:
    """
    Holds the rolling feature buffer and latest inference result for one radar.

    Thread safety:
      - frame_buf, algo, frames_since_infer  → only accessed from radar thread
      - _rdmap, _label, _prob, _ready, _buf_len → protected by _lock;
        written by radar thread, read by GUI thread via get_display_data()
    """

    def __init__(self, dev_id: int, is_1d: bool):
        self.dev_id = dev_id
        self.is_1d  = is_1d

        # Radar-thread-only state
        self.algo               = DopplerAlgo(NUM_SAMPLES, NUM_CHIRPS, NUM_RX)
        self.frame_buf          = collections.deque(maxlen=WINDOW_FRAMES)
        self.frames_since_infer = 0
        self._az_buf            = collections.deque(maxlen=CAPON_AVG_FRAMES)
        self._el_buf            = collections.deque(maxlen=CAPON_AVG_FRAMES)
        # Angle-map method: "capon" | "phasediff"  (written by GUI thread, read here)
        self.angle_method       = "capon"

        # Shared state (GUI reads these)
        self._lock    = threading.Lock()
        self._rdmap   = None    # (64, 64) float32 — RX0 magnitude for display
        self._az_map  = None    # (64, 64) float32 — Capon range-azimuth
        self._el_map  = None    # (64, 64) float32 — Capon range-elevation
        self._label   = "—"
        self._prob    = 0.0
        self._ready   = False
        self._buf_len = 0
        self._frames_since_infer = 0

    # ── Called from radar thread ───────────────────────────────────────────────

    def push_frame(self, rx0_chirp: np.ndarray,
                   rx1_chirp: np.ndarray,
                   rx2_chirp: np.ndarray) -> bool:
        """
        Process one radar frame: compute features, push to buffer.
        Returns True if an inference should be triggered now.
        Note: DopplerAlgo is called once per channel — MTI state advances once.
        """
        # Compute Range-Doppler maps (complex) for all 3 Rx channels
        rd0 = self.algo.compute_doppler_map(rx0_chirp, 0)   # (64, 64) complex
        rd1 = self.algo.compute_doppler_map(rx1_chirp, 1)
        rd2 = self.algo.compute_doppler_map(rx2_chirp, 2)

        # Log-magnitude
        mag0 = np.log1p(np.abs(rd0)).astype(np.float32)     # (64, 64)
        mag1 = np.log1p(np.abs(rd1)).astype(np.float32)
        mag2 = np.log1p(np.abs(rd2)).astype(np.float32)

        # Build feature tensor
        if self.is_1d:
            # Range projections (sum over Doppler) + Doppler projections (sum over range)
            feat = np.stack(
                [mag0.sum(1), mag1.sum(1), mag2.sum(1),   # range proj  (64,) each
                 mag0.sum(0), mag1.sum(0), mag2.sum(0)],  # Doppler proj (64,) each
                axis=-1
            )  # (64, 6)
        else:
            # 3-channel RD maps only (matches best_model.pt trained with --channel_indices 0,1,2)
            feat = np.stack([mag0, mag1, mag2], axis=-1)  # (64, 64, 3)

        self.frame_buf.append(feat)
        self.frames_since_infer += 1
        buf_len = len(self.frame_buf)

        # Range-angle maps — averaged over last CAPON_AVG_FRAMES for noise reduction
        _angle_fn = (_phase_diff_angle_map if self.angle_method == "phasediff"
                     else _capon_angle_map)
        self._az_buf.append(_angle_fn(rd0, rd2))
        self._el_buf.append(_angle_fn(rd1, rd2))
        az_avg = np.mean(self._az_buf, axis=0)
        el_avg = np.mean(self._el_buf, axis=0)

        # Update shared display data under lock
        with self._lock:
            self._rdmap   = mag0.copy()
            self._az_map  = az_avg
            self._el_map  = el_avg
            self._ready   = (buf_len >= WINDOW_FRAMES)
            self._buf_len = buf_len
            self._frames_since_infer = self.frames_since_infer

        return self._ready and (self.frames_since_infer >= STRIDE_FRAMES)

    def get_window(self) -> np.ndarray:
        """Return current window as array and reset stride counter. Radar thread only."""
        self.frames_since_infer = 0
        with self._lock:
            self._frames_since_infer = 0
        return np.stack(list(self.frame_buf), axis=0)  # (64, ..., 6)

    def update_result(self, label: str, prob: float) -> None:
        """Store latest inference result. Called from radar thread."""
        with self._lock:
            self._label = label
            self._prob  = prob

    # ── Called from GUI thread ────────────────────────────────────────────────

    def get_display_data(self):
        """Return (rdmap, az_map, el_map, label, prob, ready, buf_len, frames_since_infer) atomically."""
        with self._lock:
            return (self._rdmap, self._az_map, self._el_map,
                    self._label, self._prob,
                    self._ready, self._buf_len, self._frames_since_infer)


# ─────────────────────────────────────────────────────────────────────────────
# Radar acquisition + inference thread
# ─────────────────────────────────────────────────────────────────────────────

class RadarThread(threading.Thread):
    """
    Continuously acquires frames from both devices, computes features,
    and runs inference when STRIDE_FRAMES new frames have been collected.
    """

    def __init__(self, dev0, dev1, state0: DeviceState, state1: DeviceState,
                 model, norm_mean, norm_std, is_1d: bool,
                 torch_device: torch.device, label_names: list,
                 stop_evt: threading.Event):
        super().__init__(daemon=True, name="RadarThread")
        self.dev0         = dev0
        self.dev1         = dev1
        self.state0       = state0
        self.state1       = state1
        self.model        = model
        self.norm_mean    = norm_mean
        self.norm_std     = norm_std
        self.is_1d        = is_1d
        self.torch_device = torch_device
        self.label_names  = label_names
        self.stop_evt     = stop_evt

    def _infer(self, state: DeviceState) -> None:
        """Run one inference pass on the current window in state."""
        window_np = state.get_window()                           # (64, ..., 6)
        x_norm    = normalize_window(window_np, self.norm_mean,
                                     self.norm_std, self.is_1d)
        x_t = torch.from_numpy(x_norm).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            logits = self.model(x_t)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        top_idx   = int(probs.argmax())
        top_prob  = float(probs[top_idx])
        top_label = self.label_names[top_idx]
        state.update_result(top_label, top_prob)

    def run(self) -> None:
        while not self.stop_evt.is_set():
            try:
                fc0 = self.dev0.get_next_frame()
                fc1 = self.dev1.get_next_frame()
                data0 = fc0[0]   # shape: (num_rx, num_chirps, num_samples)
                data1 = fc1[0]

                need0 = self.state0.push_frame(
                    data0[0, :, :], data0[1, :, :], data0[2, :, :])
                need1 = self.state1.push_frame(
                    data1[0, :, :], data1[1, :, :], data1[2, :, :])

                if need0:
                    self._infer(self.state0)
                if need1:
                    self._infer(self.state1)

            except Exception as exc:
                if not self.stop_evt.is_set():
                    print(f"[RadarThread] {exc}", flush=True)
                    time.sleep(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Camera thread
# ─────────────────────────────────────────────────────────────────────────────

class CameraThread(threading.Thread):
    """
    Captures frames from Picamera2 (preferred on Pi) or a USB webcam.
    The latest BGR frame is always accessible via get_frame().
    """

    def __init__(self, stop_evt: threading.Event, cam_index=None):
        super().__init__(daemon=True, name="CameraThread")
        self.stop_evt  = stop_evt
        self.cam_index = cam_index
        self._lock     = threading.Lock()
        self._frame    = None   # latest BGR frame (numpy)

    def get_frame(self):
        with self._lock:
            return self._frame

    def run(self) -> None:
        if _PICAM2_OK:
            self._run_picam2()
        else:
            self._run_opencv()

    def _run_picam2(self) -> None:
        picam2 = None
        try:
            picam2 = Picamera2()
            cfg    = picam2.create_video_configuration(
                main={"size": (CAM_W, CAM_H), "format": "RGB888"})
            picam2.configure(cfg)
            picam2.start()
            time.sleep(0.5)           # longer warmup; USB cameras need it
            fail_count = 0
            while not self.stop_evt.is_set():
                try:
                    rgb = picam2.capture_array("main")
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    with self._lock:
                        self._frame = bgr
                    fail_count = 0
                except Exception:
                    fail_count += 1
                    if fail_count >= 5:
                        raise RuntimeError(f"{fail_count} consecutive frame errors")
                    time.sleep(0.05)
            picam2.stop()
        except Exception as exc:
            print(f"[Camera] Picamera2 failed ({exc}), falling back to OpenCV.")
            # Release the device so OpenCV can open it
            if picam2 is not None:
                for _method in ("stop", "close"):
                    try:
                        getattr(picam2, _method)()
                    except Exception:
                        pass
            time.sleep(1.0)           # wait for OS to release /dev/videoN
            self._run_opencv()

    def _run_opencv(self) -> None:
        indices = ([self.cam_index] if self.cam_index is not None
                   else list(range(4)))
        cap = None
        for idx in indices:
            try:
                c = cv2.VideoCapture(idx)
                if c.isOpened():
                    ok, _ = c.read()
                    if ok:
                        cap = c
                        break
                c.release()
            except Exception:
                pass
        if cap is None:
            print("[Camera] No camera found — video panel will be blank.")
            return
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierGUI:
    """
    Tkinter-based display.

    Panels:
      Left  : camera feed (CAM_W × CAM_H) with text overlays
      Right : two Doppler heatmaps stacked vertically (dev0 top, dev1 bottom)
      Bottom: two result cards side-by-side (dev1 left, dev0 right)
    """

    def __init__(self, root: tk.Tk, state0: DeviceState, state1: DeviceState,
                 cam_thread: CameraThread, label_names: list,
                 start_time: float, model_type: str, frate: int):
        self.root        = root
        self.state0      = state0
        self.state1      = state1
        self.cam_thread  = cam_thread
        self.label_names = label_names
        self.start_time  = start_time
        self.frate       = frate

        root.title(f"ADL Real-time Classifier  [{model_type.upper()} model]")
        root.configure(bg=BG_MID)
        root.resizable(False, False)

        # Compute camera DISPLAY size to fit the screen.
        # CAM_W/CAM_H is the capture resolution; we scale down for display so
        # the result bar always stays on screen regardless of resolution.
        _screen_h = root.winfo_screenheight()
        _avail_h  = max(240, _screen_h - 150)   # reserve ~150 px for result bar + chrome
        self._cam_h = min(CAM_H, _avail_h)
        self._cam_w = int(self._cam_h * CAM_W / CAM_H)

        self._font_hd, self._font_sm = self._init_fonts()

        # ── Top frame: camera + Doppler panels ───────────────────────────────
        top = tk.Frame(root, bg=BG_MID)
        top.pack(padx=6, pady=6)

        # Camera canvas
        self.cam_canvas = tk.Canvas(
            top, width=self._cam_w, height=self._cam_h,
            bg=BG_DARK, highlightthickness=0)
        self.cam_canvas.grid(row=0, column=0, rowspan=2, padx=(0, 6))

        # Dev0 Doppler panel (top-right)
        f0 = tk.Frame(top, bg=BG_MID)
        f0.grid(row=0, column=1, sticky="n", pady=(0, 3))
        tk.Label(f0, text="Dev 0  –  Doppler  (RX0)",
                 font=("Courier", 9), fg=COL_HEAD, bg=BG_MID).pack()
        self.dop0_canvas = tk.Canvas(
            f0, width=DOPPLER_W, height=DOPPLER_H,
            bg=BG_DARK, highlightthickness=1, highlightbackground="#334466")
        self.dop0_canvas.pack()

        # Dev1 Doppler panel (bottom-right)
        f1 = tk.Frame(top, bg=BG_MID)
        f1.grid(row=1, column=1, sticky="n", pady=(3, 0))
        tk.Label(f1, text="Dev 1  –  Doppler  (RX0)",
                 font=("Courier", 9), fg=COL_HEAD, bg=BG_MID).pack()
        self.dop1_canvas = tk.Canvas(
            f1, width=DOPPLER_W, height=DOPPLER_H,
            bg=BG_DARK, highlightthickness=1, highlightbackground="#334466")
        self.dop1_canvas.pack()

        # ── Angle-map column (col 2): 4 panels — az0, el0, az1, el1 ─────────
        f_angle = tk.Frame(top, bg=BG_MID)
        f_angle.grid(row=0, column=2, rowspan=2, sticky="n", padx=(6, 0))

        # Method selector (Capon / Phase-diff)
        self._angle_method_var = tk.StringVar(value="capon")

        def _on_method_change():
            m = self._angle_method_var.get()
            state0.angle_method = m
            state1.angle_method = m

        _sel_row = tk.Frame(f_angle, bg=BG_MID)
        _sel_row.pack(pady=(0, 4))
        tk.Label(_sel_row, text="Map:", font=("Courier", 8),
                 fg=COL_HEAD, bg=BG_MID).pack(side="left")
        for _txt, _val in [("Capon", "capon"), ("Phase-diff", "phasediff")]:
            tk.Radiobutton(
                _sel_row, text=_txt, variable=self._angle_method_var,
                value=_val, command=_on_method_change,
                font=("Courier", 8), fg=COL_HEAD, bg=BG_MID,
                selectcolor=BG_DARK, activebackground=BG_MID,
            ).pack(side="left", padx=2)

        def _angle_panel(parent, title):
            tk.Label(parent, text=title, font=("Courier", 9),
                     fg=COL_HEAD, bg=BG_MID).pack()
            c = tk.Canvas(parent, width=ANGLE_W, height=ANGLE_H,
                          bg=BG_DARK, highlightthickness=1,
                          highlightbackground="#334466")
            c.pack(pady=(0, 4))
            blank = Image.new("RGB", (ANGLE_W, ANGLE_H), (13, 13, 26))
            photo = ImageTk.PhotoImage(blank)
            c.create_image(0, 0, anchor="nw", image=photo, tags="img")
            return c, photo

        self.az0_canvas, self._az0_photo = _angle_panel(f_angle, "Dev0  Range-Azimuth")
        self.el0_canvas, self._el0_photo = _angle_panel(f_angle, "Dev0  Range-Elevation")
        self.az1_canvas, self._az1_photo = _angle_panel(f_angle, "Dev1  Range-Azimuth")
        self.el1_canvas, self._el1_photo = _angle_panel(f_angle, "Dev1  Range-Elevation")

        # ── Bottom frame: result cards (dev1 left, dev0 right) ───────────────
        bot = tk.Frame(root, bg=BG_MID)
        bot.pack(padx=6, pady=(0, 6), fill="x")

        f_res1 = tk.Frame(bot, bg=BG_DARK, bd=1, relief="solid")
        f_res1.pack(side="left", expand=True, fill="both", padx=(0, 3))
        tk.Label(f_res1, text="DEV 1", font=self._font_hd,
                 fg=COL_HEAD, bg=BG_DARK).pack(pady=(4, 0))
        self._lbl1_action = tk.Label(f_res1, text="—",
                                     font=self._font_hd, fg=COL_LABEL, bg=BG_DARK)
        self._lbl1_action.pack()
        self._lbl1_prob = tk.Label(f_res1, text="",
                                   font=self._font_sm, fg=COL_PROB, bg=BG_DARK)
        self._lbl1_prob.pack(pady=(0, 4))

        f_res0 = tk.Frame(bot, bg=BG_DARK, bd=1, relief="solid")
        f_res0.pack(side="left", expand=True, fill="both", padx=(3, 0))
        tk.Label(f_res0, text="DEV 0", font=self._font_hd,
                 fg=COL_HEAD, bg=BG_DARK).pack(pady=(4, 0))
        self._lbl0_action = tk.Label(f_res0, text="—",
                                     font=self._font_hd, fg=COL_LABEL, bg=BG_DARK)
        self._lbl0_action.pack()
        self._lbl0_prob = tk.Label(f_res0, text="",
                                   font=self._font_sm, fg=COL_PROB, bg=BG_DARK)
        self._lbl0_prob.pack(pady=(0, 4))

        # ── Pre-render blank images so canvas items exist from the start ──────
        blank_cam = Image.new("RGB", (self._cam_w, self._cam_h), (13, 13, 26))
        blank_dop = Image.new("RGB", (DOPPLER_W, DOPPLER_H), (13, 13, 26))

        self._cam_photo  = ImageTk.PhotoImage(blank_cam)
        self._dop0_photo = ImageTk.PhotoImage(blank_dop)
        self._dop1_photo = ImageTk.PhotoImage(blank_dop)

        self.cam_canvas.create_image(0, 0, anchor="nw",
                                     image=self._cam_photo, tags="cam")
        self.dop0_canvas.create_image(0, 0, anchor="nw",
                                      image=self._dop0_photo, tags="dop")
        self.dop1_canvas.create_image(0, 0, anchor="nw",
                                      image=self._dop1_photo, tags="dop")
        # Note: angle-map canvases already have their blank images from _angle_panel()

        # Text overlays on camera (created once, text updated each tick)
        _tf = ("Courier", 12, "bold")
        self._tid_time    = self.cam_canvas.create_text(
            10, 10, anchor="nw", fill=COL_TIME,  font=_tf, text="")
        self._tid_elapsed = self.cam_canvas.create_text(
            self._cam_w - 10, 10, anchor="ne", fill=COL_ELAPS, font=_tf, text="")

        # Cache latest raw angle maps for saving
        self._last_az0 = self._last_el0 = None
        self._last_az1 = self._last_el1 = None

        # Press 's' to snapshot angle maps to data/aoa/
        root.bind("<s>", self._save_angle_maps)
        root.bind("<S>", self._save_angle_maps)

        # Kick off update loop
        self.root.after(50, self._tick)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _init_fonts():
        try:
            import tkinter.font as tkfont
            hd = tkfont.Font(family="Courier", size=14, weight="bold")
            sm = tkfont.Font(family="Courier", size=11)
            return hd, sm
        except Exception:
            return (("Courier", 14, "bold"), ("Courier", 11))

    @staticmethod
    def _rdmap_to_photo(rdmap: np.ndarray, w: int, h: int) -> ImageTk.PhotoImage:
        """
        Convert a (64, 64) float32 magnitude map to a colour heatmap PhotoImage.
        Range axis is flipped so near-range targets appear at the bottom.
        """
        disp = np.flipud(rdmap)                # near-range at bottom
        vmin, vmax = disp.min(), disp.max()
        if vmax <= vmin:
            img8 = np.zeros((64, 64), dtype=np.uint8)
        else:
            img8 = ((disp - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        # Plasma colormap via OpenCV
        colored_bgr = cv2.applyColorMap(img8, cv2.COLORMAP_PLASMA)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(colored_rgb).resize((w, h), Image.NEAREST)
        return ImageTk.PhotoImage(pil)

    def _save_angle_maps(self, _event=None) -> None:
        """Save current angle maps to data/aoa/ as a combined PNG and raw .npy files.
        Triggered by pressing 's' while the GUI window is focused.
        """
        entries = [
            ("dev0_azimuth",   self._last_az0),
            ("dev0_elevation", self._last_el0),
            ("dev1_azimuth",   self._last_az1),
            ("dev1_elevation", self._last_el1),
        ]
        if all(arr is None for _, arr in entries):
            print("[Save] No angle maps yet — wait for the radar to produce frames.")
            return

        out_dir = Path("data") / "aoa"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        SAVE_W, SAVE_H = 320, 240   # each panel in the saved PNG (larger than display)

        panels = []
        for name, arr in entries:
            src = arr if arr is not None else np.zeros((64, 64), np.float32)

            # Save raw numpy array for programmatic analysis
            if arr is not None:
                np.save(str(out_dir / f"{ts}_{name}.npy"), arr)

            # Build colour panel
            disp = np.flipud(src)
            mn, mx = float(disp.min()), float(disp.max())
            img8 = (np.zeros((64, 64), np.uint8) if mx <= mn
                    else ((disp - mn) / (mx - mn) * 255).astype(np.uint8))
            panel = cv2.applyColorMap(img8, cv2.COLORMAP_PLASMA)
            panel = cv2.resize(panel, (SAVE_W, SAVE_H), interpolation=cv2.INTER_NEAREST)
            cv2.putText(panel, name.replace("_", " "),
                        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            panels.append(panel)

        grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
        method   = getattr(self, "_angle_method_var", None)
        method   = method.get() if method is not None else "capon"
        png_path = str(out_dir / f"{ts}_angle_maps_{method}.png")
        cv2.imwrite(png_path, grid)
        print(f"[Save] Angle maps saved → {png_path}  (+4 .npy files)")

    # ── Main update tick (50 ms) ──────────────────────────────────────────────

    def _tick(self) -> None:
        now     = time.time()
        elapsed = int(now - self.start_time)
        h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

        # ── Camera ───────────────────────────────────────────────────────────
        bgr = self.cam_thread.get_frame()
        if bgr is not None:
            bgr_rs = cv2.resize(bgr, (self._cam_w, self._cam_h))
            rgb    = cv2.cvtColor(bgr_rs, cv2.COLOR_BGR2RGB)
            pil    = Image.fromarray(rgb)
        else:
            pil = Image.new("RGB", (self._cam_w, self._cam_h), (20, 20, 40))

        # Push camera image to canvas
        self._cam_photo = ImageTk.PhotoImage(pil)
        self.cam_canvas.itemconfigure("cam", image=self._cam_photo)

        # Time/elapsed text on top of camera image (Tkinter canvas items)
        self.cam_canvas.itemconfigure(
            self._tid_time,
            text=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.cam_canvas.itemconfigure(
            self._tid_elapsed,
            text=f"{h:02d}:{m:02d}:{s:02d}")

        # ── Result cards ─────────────────────────────────────────────────────
        rdmap0, az0, el0, lbl0, prob0, ready0, blen0, fsince0 = self.state0.get_display_data()
        rdmap1, az1, el1, lbl1, prob1, ready1, blen1, fsince1 = self.state1.get_display_data()

        if ready1:
            self._lbl1_action.config(text=lbl1.replace("_", " "), fg=COL_LABEL)
            cd1 = max(0.0, (STRIDE_FRAMES - fsince1) / self.frate)
            self._lbl1_prob.config(text=f"{prob1 * 100:.1f}%   ↺ {cd1:.1f}s")
        else:
            self._lbl1_action.config(
                text=f"Warming… {blen1}/{WINDOW_FRAMES}", fg=COL_WARM)
            self._lbl1_prob.config(text="")

        if ready0:
            self._lbl0_action.config(text=lbl0.replace("_", " "), fg=COL_LABEL)
            cd0 = max(0.0, (STRIDE_FRAMES - fsince0) / self.frate)
            self._lbl0_prob.config(text=f"{prob0 * 100:.1f}%   ↺ {cd0:.1f}s")
        else:
            self._lbl0_action.config(
                text=f"Warming… {blen0}/{WINDOW_FRAMES}", fg=COL_WARM)
            self._lbl0_prob.config(text="")

        # ── Doppler heatmaps ──────────────────────────────────────────────────
        if rdmap0 is not None:
            self._dop0_photo = self._rdmap_to_photo(rdmap0, DOPPLER_W, DOPPLER_H)
            self.dop0_canvas.itemconfigure("dop", image=self._dop0_photo)

        if rdmap1 is not None:
            self._dop1_photo = self._rdmap_to_photo(rdmap1, DOPPLER_W, DOPPLER_H)
            self.dop1_canvas.itemconfigure("dop", image=self._dop1_photo)

        # ── Angle maps ────────────────────────────────────────────────────────
        if az0 is not None:
            self._last_az0 = az0
            self._az0_photo = self._rdmap_to_photo(az0, ANGLE_W, ANGLE_H)
            self.az0_canvas.itemconfigure("img", image=self._az0_photo)
        if el0 is not None:
            self._last_el0 = el0
            self._el0_photo = self._rdmap_to_photo(el0, ANGLE_W, ANGLE_H)
            self.el0_canvas.itemconfigure("img", image=self._el0_photo)
        if az1 is not None:
            self._last_az1 = az1
            self._az1_photo = self._rdmap_to_photo(az1, ANGLE_W, ANGLE_H)
            self.az1_canvas.itemconfigure("img", image=self._az1_photo)
        if el1 is not None:
            self._last_el1 = el1
            self._el1_photo = self._rdmap_to_photo(el1, ANGLE_W, ANGLE_H)
            self.el1_canvas.itemconfigure("img", image=self._el1_photo)

        self.root.after(50, self._tick)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADL real-time classifier GUI — dual radar + camera."
    )
    parser.add_argument("--model", choices=["1d", "2d"], default="1d",
                        help="Model type to use for inference (default: 1d)")
    parser.add_argument("--frate", type=int, default=DEFAULT_FRATE,
                        help=f"Radar frame rate in Hz (default: {DEFAULT_FRATE})")
    parser.add_argument("--cam-index", type=int, default=None,
                        help="USB camera index (0, 1, …). Ignored if Picamera2 works.")
    args = parser.parse_args()

    is_1d = (args.model == "1d")

    # Load model from checkpoint
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch device: {torch_device}")
    model, norm_mean, norm_std, label_names, window_frames = \
        load_model(args.model, torch_device)

    # Open both radars
    print("Opening radar devices…")
    dev0, dev1 = open_two_devices()

    with dev0, dev1:
        rx0_count = configure_device(dev0, args.frate)
        rx1_count = configure_device(dev1, args.frate)
        print(f"Dev0: {rx0_count} Rx  |  Dev1: {rx1_count} Rx  "
              f"@ {args.frate} Hz")

        stop_evt   = threading.Event()
        start_time = time.time()

        state0 = DeviceState(0, is_1d)
        state1 = DeviceState(1, is_1d)

        # Background threads
        radar_thread = RadarThread(
            dev0, dev1, state0, state1,
            model, norm_mean, norm_std, is_1d,
            torch_device, label_names, stop_evt)
        cam_thread = CameraThread(stop_evt, args.cam_index)

        radar_thread.start()
        cam_thread.start()

        # GUI (runs on main thread)
        root = tk.Tk()
        _gui = ClassifierGUI(
            root, state0, state1, cam_thread,
            label_names, start_time, args.model, args.frate)

        def _on_close():
            stop_evt.set()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _on_close)
        root.mainloop()

        stop_evt.set()
        print("Stopped.")


if __name__ == "__main__":
    main()
