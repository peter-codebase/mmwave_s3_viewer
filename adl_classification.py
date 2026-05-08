#!/usr/bin/env python3
"""
adl_classification.py

Browse S3 sessions and classify ADL activities using a trained model.

Main window
  .env / Connect → User ID → Date → Session dropdowns
  Features folder + model selector
  "ADL Classification" button → opens ClassificationWindow

ClassificationWindow
  Session dropdown for the selected date
  Range Map / Doppler Map viewer for dev0 + dev1
  Video player (cv2/PIL) + frame number
  Audio playback (pygame)
  Play / Pause / Prev / Next / Stop + scrubber progress bar
  Activity tag buttons (from .txt if present) — click to jump
  "Classify Session" button → runs trained model → activity table
    Double-click a row to jump to that frame
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Shared S3 / signal helpers from the existing GUI ─────────────────────────
# (imports execute module-level code in s3_post_analyze_gui.py, which sets the
#  matplotlib backend to TkAgg and makes boto3/pygame available)
from s3_post_analyze_gui import (
    load_env,
    make_s3_client,
    detect_root_prefix,
    list_common_prefixes,
    list_objects,
    prefix_exists,
    list_user_ids,
    list_dates_for_user,
    pick_day_prefix,
    session_has_any_txt,
    _s3_download_to_file,
    _parse_adl_tags,
    _safe_literal_array,
    _compute_range_axis_m,
    _compute_distance_mag,
    _compute_range_doppler_mag,
    _read_wav_as_pcm16_bytes,
)
from s3_layout import S3Layout
from botocore.exceptions import BotoCoreError, ClientError

# ML helpers
sys.path.insert(0, str(Path(__file__).parent))
from helpers.DopplerAlgo import DopplerAlgo
from adl_model import ADLClassifier, ADLClassifier1D

# Optional GUI deps (backend already set by s3_post_analyze_gui import above)
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    Figure = FigureCanvasTkAgg = None  # type: ignore

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTk  # type: ignore
except ImportError:
    Image = ImageTk = None  # type: ignore

try:
    import pygame  # type: ignore
except ImportError:
    pygame = None

try:
    from scipy.ndimage import median_filter as _scipy_median

    def _smooth_labels(arr: np.ndarray, k: int = 5) -> np.ndarray:
        return _scipy_median(arr.astype(np.float64), size=k).round().astype(int)

except ImportError:
    def _smooth_labels(arr: np.ndarray, k: int = 5) -> np.ndarray:
        n = len(arr)
        h = k // 2
        out = arr.copy()
        for i in range(n):
            lo, hi = max(0, i - h), min(n, i + h + 1)
            vals, cnt = np.unique(arr[lo:hi], return_counts=True)
            out[i] = int(vals[cnt.argmax()])
        return out

# ── Radar constants ───────────────────────────────────────────────────────────
_NUM_CHIRPS  = 32
_NUM_SAMPLES = 64
_NUM_CH      = 3
_FC_START    = 60_250_345_952.0
_FC_END      = 61_249_654_048.0
_DOPPLER_PRF = 2000.0


# ── Feature extraction ────────────────────────────────────────────────────────

def _has_device_data(df: pd.DataFrame, prefix: str) -> bool:
    cols = [f"{prefix}_ch1", f"{prefix}_ch2", f"{prefix}_ch3"]
    if not all(c in df.columns for c in cols):
        return False
    try:
        first = str(df[cols[0]].iloc[0]).strip()
        if first in ("[]", "", "nan", "None"):
            return False
        ast.literal_eval(first)
        return True
    except Exception:
        return False


def _extract_features(
    df: pd.DataFrame, prefix: str, is_1d: bool, log_fn=print
) -> np.ndarray | None:
    """
    Extract per-frame feature tensor for one device.
    2D mode → (N, 64, 64, 3)  [log-mag RD maps, 3 Rx channels]
    1D mode → (N, 64, 6)      [range+Doppler projections, 6 channels]
    Returns None if the device columns are absent or empty.
    """
    if not _has_device_data(df, prefix):
        return None

    algo = DopplerAlgo(_NUM_SAMPLES, _NUM_CHIRPS, _NUM_CH)
    features = []

    for _, row in df.iterrows():
        try:
            ch1 = np.asarray(
                ast.literal_eval(str(row[f"{prefix}_ch1"])), dtype=np.float32
            ).reshape(_NUM_CHIRPS, _NUM_SAMPLES)
            ch2 = np.asarray(
                ast.literal_eval(str(row[f"{prefix}_ch2"])), dtype=np.float32
            ).reshape(_NUM_CHIRPS, _NUM_SAMPLES)
            ch3 = np.asarray(
                ast.literal_eval(str(row[f"{prefix}_ch3"])), dtype=np.float32
            ).reshape(_NUM_CHIRPS, _NUM_SAMPLES)
        except Exception:
            zero = np.zeros((64, 6) if is_1d else (64, 64, 3), dtype=np.float32)
            features.append(zero)
            continue

        rd0 = algo.compute_doppler_map(ch1, 0)
        rd1 = algo.compute_doppler_map(ch2, 1)
        rd2 = algo.compute_doppler_map(ch3, 2)

        mag0 = np.log1p(np.abs(rd0)).astype(np.float32)
        mag1 = np.log1p(np.abs(rd1)).astype(np.float32)
        mag2 = np.log1p(np.abs(rd2)).astype(np.float32)

        if is_1d:
            frame = np.stack([
                mag0.sum(axis=1), mag1.sum(axis=1), mag2.sum(axis=1),
                mag0.sum(axis=0), mag1.sum(axis=0), mag2.sum(axis=0),
            ], axis=-1)  # (64, 6)
        else:
            frame = np.stack([mag0, mag1, mag2], axis=-1)  # (64, 64, 3)

        features.append(frame)

    return np.array(features, dtype=np.float32)


# ── Sliding-window inference ──────────────────────────────────────────────────

def _infer_device(
    features: np.ndarray,
    model: torch.nn.Module,
    window: int,
    stride: int,
    in_channels: int,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Sliding-window inference with per-window z-score normalization.
    Returns per-frame softmax probabilities (N, num_classes).
    """
    N = features.shape[0]

    # Build windows and track centre frames
    windows: list[np.ndarray] = []
    centers: list[int] = []
    start = 0
    while start + window <= N:
        windows.append(features[start : start + window])
        centers.append(start + window // 2)
        start += stride
    # Non-redundant tail window
    tail_start = N - window
    if N >= window and tail_start > (start - stride if start > stride else -1):
        windows.append(features[tail_start:])
        centers.append(tail_start + window // 2)
    # Short session: pad with last frame
    if not windows and N > 0:
        last = features[-1:]
        pad  = np.repeat(last, window - N, axis=0)
        windows = [np.concatenate([features, pad], axis=0)]
        centers = [N // 2]
    if not windows:
        return np.full((N, num_classes), 1.0 / num_classes, dtype=np.float32)

    # Per-window z-score (matches adl_train.py normalization)
    X = np.array(windows, dtype=np.float32)                  # (W, T, ...)
    Xf = X.reshape(len(X), -1, in_channels)                  # (W, spatial, C)
    m  = Xf.mean(axis=1, keepdims=True)
    s  = Xf.std( axis=1, keepdims=True) + 1e-8
    X  = ((Xf - m) / s).reshape(X.shape)

    # Batch forward pass
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch  = torch.from_numpy(X[i : i + batch_size]).to(device)
            logits = model(batch)
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    all_probs_arr = np.concatenate(all_probs, axis=0)         # (W, num_classes)

    # Average overlapping window predictions onto per-frame probs
    frame_probs  = np.zeros((N, num_classes), dtype=np.float64)
    frame_counts = np.zeros(N, dtype=np.int32)
    half = window // 2
    for wi, center in enumerate(centers):
        lo = max(0, center - half)
        hi = min(N, center + half + 1)
        frame_probs[lo:hi]  += all_probs_arr[wi]
        frame_counts[lo:hi] += 1
    uncovered = frame_counts == 0
    if uncovered.any():
        frame_probs[uncovered]  = 1.0 / num_classes
        frame_counts[uncovered] = 1
    return (frame_probs / frame_counts[:, None]).astype(np.float32)


# ── Segment merging ───────────────────────────────────────────────────────────

def _merge_segments(
    labels: np.ndarray,
    probs:  np.ndarray,
    timestamps: np.ndarray,
    label_names: list[str],
    min_frames: int,
) -> list[dict]:
    """Merge consecutive same-label frames into activity segments."""
    if len(labels) == 0:
        return []
    t0   = float(timestamps[0])
    segs = []

    def _emit(start: int, end: int, lab: int) -> None:
        if end - start < min_frames:
            return
        conf   = float(probs[start:end, lab].mean()) * 100.0
        ts_s   = float(timestamps[start]) - t0
        ts_e   = float(timestamps[end - 1]) - t0
        segs.append({
            "activity":    label_names[lab],
            "start_frame": start + 1,       # 1-indexed for UI
            "end_frame":   end,
            "start_sec":   round(ts_s, 1),
            "end_sec":     round(ts_e, 1),
            "duration_sec": round(ts_e - ts_s, 1),
            "confidence":  round(conf, 1),
        })

    cur = int(labels[0])
    seg_start = 0
    for i in range(1, len(labels)):
        if int(labels[i]) != cur:
            _emit(seg_start, i, cur)
            seg_start = i
            cur = int(labels[i])
    _emit(seg_start, len(labels), cur)
    return segs


# ── Top-level classification pipeline ─────────────────────────────────────────

def run_classification(
    df: pd.DataFrame,
    model_path: Path,
    feature_type: str,
    log_fn=print,
    stride: int = 5,
    batch_size: int = 32,
    min_seg_frames: int = 10,
) -> tuple:
    """
    Run the trained ADL model on both radar devices for one session.

    Returns (probs_dev0, probs_dev1, label_names, segments).
    probs_dev* are (N, num_classes) float32 arrays.
    segments is a list of dicts: activity / start_frame / end_frame /
      start_sec / end_sec / duration_sec / confidence.
    """
    torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt        = torch.load(str(model_path), map_location=torch_dev, weights_only=False)
    label_map   = ckpt["label_map"]
    in_ch       = int(ckpt.get("in_channels", 3))
    win         = int(ckpt.get("window_frames", 50))
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    n_cls       = len(label_names)
    is_1d       = (feature_type == "1d_profile")

    log_fn(f"Model       : {model_path.name}")
    log_fn(f"Window      : {win} frames  |  Channels: {in_ch}  |  Classes: {n_cls}")
    log_fn(f"Device      : {torch_dev}  |  Stride: {stride} frames\n")

    if is_1d:
        model = ADLClassifier1D(num_classes=n_cls, in_channels=in_ch)
    else:
        model = ADLClassifier(num_classes=n_cls, in_channels=in_ch)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(torch_dev)

    N  = len(df)
    ts = (
        df["timestamp"].astype(float).to_numpy()
        if "timestamp" in df.columns
        else np.arange(N, dtype=float) / 5.0
    )

    uniform = np.full((N, n_cls), 1.0 / n_cls, dtype=np.float32)
    probs: dict[str, np.ndarray | None] = {}

    for dev in ("dev0", "dev1"):
        log_fn(f"Processing {dev}…")
        feats = _extract_features(df, dev, is_1d, log_fn)
        if feats is None:
            log_fn(f"  {dev}: columns absent or empty — skipped\n")
            probs[dev] = None
            continue
        n_wins = max(0, (len(feats) - win) // stride + 1)
        log_fn(f"  {len(feats)} frames  →  ~{n_wins} windows")
        probs[dev] = _infer_device(feats, model, win, stride, in_ch, n_cls, batch_size, torch_dev)
        log_fn(f"  done.\n")

    has_dev0 = probs["dev0"] is not None
    has_dev1 = probs["dev1"] is not None
    p0, p1   = probs["dev0"], probs["dev1"]

    if has_dev0 and has_dev1:
        combined = (p0 + p1) / 2.0
    elif has_dev0:
        combined = p0;  p1 = uniform
    elif has_dev1:
        combined = p1;  p0 = uniform
    else:
        log_fn("No device data found — cannot classify.")
        return None, None, label_names, []

    per_frame = combined.argmax(axis=1).astype(int)
    smoothed  = _smooth_labels(per_frame, k=5)
    segments  = _merge_segments(smoothed, combined, ts, label_names, min_seg_frames)

    # Annotate each segment with per-device dominant label
    for seg in segments:
        lo = seg["start_frame"] - 1   # 1-indexed → 0-indexed
        hi = seg["end_frame"]
        if has_dev0:
            votes = p0[lo:hi].argmax(axis=1)
            seg["dev0_activity"] = label_names[int(np.bincount(votes, minlength=n_cls).argmax())]
        else:
            seg["dev0_activity"] = "—"
        if has_dev1:
            votes = p1[lo:hi].argmax(axis=1)
            seg["dev1_activity"] = label_names[int(np.bincount(votes, minlength=n_cls).argmax())]
        else:
            seg["dev1_activity"] = "—"

    log_fn(f"Detected {len(segments)} activity segment(s).")
    return p0, p1, label_names, segments


# ── Main App window ───────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ADL Classification")
        self.resizable(True, False)
        self.minsize(880, 300)

        self.env_path    = Path.cwd() / ".env"
        self.s3          = None
        self.bucket: str | None = None
        self.layout: S3Layout | None = None
        self.root_prefix = ""
        self._day_prefix: str | None = None
        self._sess_display_to_raw: dict[str, str] = {}

        self._build_ui()
        self._set_status("Choose .env (optional) then click Connect.")
        self._set_conn("disconnected")

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # Row 0: .env + connect
        r0 = ttk.Frame(self);  r0.pack(fill="x", **pad)
        ttk.Label(r0, text=".env:").pack(side="left")
        self._lbl_env = ttk.Label(r0, text=str(self.env_path), width=55, anchor="w")
        self._lbl_env.pack(side="left", padx=6)
        ttk.Button(r0, text="Choose .env…", command=self._choose_env).pack(side="right")
        ttk.Button(r0, text="Connect / Refresh", command=self._connect).pack(side="right", padx=6)
        self._lbl_conn = tk.Label(r0, text="●", fg="gray", font=("TkDefaultFont", 16, "bold"))
        self._lbl_conn.pack(side="right", padx=4)

        # Row 1: bucket
        r1 = ttk.Frame(self);  r1.pack(fill="x", **pad)
        ttk.Label(r1, text="S3 bucket:").pack(side="left")
        self._ent_bucket = ttk.Entry(r1, width=26)
        self._ent_bucket.insert(0, "mmwave-raw-data")
        self._ent_bucket.pack(side="left", padx=6)
        self._lbl_root = ttk.Label(r1, text="Root prefix: (auto)")
        self._lbl_root.pack(side="left", padx=10)

        # Row 2: User / Date / Session
        r2 = ttk.Frame(self);  r2.pack(fill="x", **pad)
        ttk.Label(r2, text="User ID:").pack(side="left")
        self._cmb_user = ttk.Combobox(r2, state="readonly", width=18)
        self._cmb_user.pack(side="left", padx=6)
        self._cmb_user.bind("<<ComboboxSelected>>", self._on_user)

        ttk.Label(r2, text="Date:").pack(side="left", padx=(16, 0))
        self._cmb_date = ttk.Combobox(r2, state="readonly", width=12)
        self._cmb_date.pack(side="left", padx=6)
        self._cmb_date.bind("<<ComboboxSelected>>", self._on_date)

        ttk.Label(r2, text="Session:").pack(side="left", padx=(16, 0))
        self._cmb_sess = ttk.Combobox(r2, state="readonly", width=40)
        self._cmb_sess.pack(side="left", padx=6)

        # Row 3: features folder + model
        r3 = ttk.Frame(self);  r3.pack(fill="x", **pad)
        ttk.Label(r3, text="Features folder:").pack(side="left")
        self._feat_var = tk.StringVar(value=str(Path("models")))
        ttk.Entry(r3, textvariable=self._feat_var, width=34).pack(side="left", padx=6)
        ttk.Button(r3, text="Browse…", command=self._browse_feat).pack(side="left")
        ttk.Button(r3, text="↺", command=self._refresh_models, width=3).pack(side="left", padx=4)

        ttk.Label(r3, text="Model:").pack(side="left", padx=(16, 0))
        self._model_var = tk.StringVar()
        self._cmb_model = ttk.Combobox(r3, textvariable=self._model_var, state="readonly", width=42)
        self._cmb_model.pack(side="left", padx=6)
        self._feat_var.trace_add("write", lambda *_: self.after(100, self._refresh_models))

        # Row 4: action button
        r4 = ttk.Frame(self);  r4.pack(fill="x", **pad)
        ttk.Button(r4, text="ADL Classification  →", command=self._open_window).pack(side="left")

        # Row 5: status
        r5 = ttk.Frame(self);  r5.pack(fill="x", **pad)
        ttk.Label(r5, text="Status:").pack(side="left")
        self._lbl_status = ttk.Label(r5, text="", anchor="w")
        self._lbl_status.pack(side="left", padx=6, fill="x", expand=True)

        self._refresh_models()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        self._lbl_status.config(text=msg)
        self.update_idletasks()

    def _set_conn(self, state: str):
        color = {"disconnected": "gray", "connecting": "goldenrod",
                 "connected": "green", "error": "red"}.get(state, "gray")
        self._lbl_conn.config(fg=color)

    def _choose_env(self):
        p = filedialog.askopenfilename(
            title="Choose .env file",
            filetypes=[("dotenv", "*.env"), ("All", "*.*")],
        )
        if p:
            self.env_path = Path(p)
            self._lbl_env.config(text=str(self.env_path))

    def _connect(self):
        self._set_status("Connecting…")
        self._set_conn("connecting")
        bucket_ui = self._ent_bucket.get().strip()
        if not bucket_ui:
            messagebox.showerror("Missing", "Enter the S3 bucket name.")
            return

        def worker():
            try:
                cfg    = load_env(self.env_path)
                s3     = make_s3_client(cfg)
                bucket = (cfg.bucket or bucket_ui).strip()
                root   = detect_root_prefix(s3, bucket)
                layout = S3Layout(root)
                users  = list_user_ids(layout, s3, bucket)

                def ui():
                    self.s3 = s3;  self.bucket = bucket
                    self.root_prefix = root;  self.layout = layout
                    self._lbl_root.config(text=f"Root prefix: {root or '/'}")
                    self._cmb_user["values"] = users
                    self._set_conn("connected")
                    self._set_status(f"{len(users)} user(s) loaded.")
                self.after(0, ui)
            except (BotoCoreError, ClientError) as e:
                self.after(0, lambda: (self._set_conn("error"), self._set_status(f"S3 error: {e}")))
            except Exception as e:
                self.after(0, lambda: (self._set_conn("error"), self._set_status(f"Error: {e}")))

        threading.Thread(target=worker, daemon=True).start()

    def _on_user(self, _=None):
        user = self._cmb_user.get().strip()
        if not user or not self.s3:
            return
        self._cmb_date.set(""); self._cmb_date["values"] = []
        self._cmb_sess.set(""); self._cmb_sess["values"] = []
        self._set_status(f"Loading dates for {user}…")

        def worker():
            try:
                layout = self.layout or S3Layout(self.root_prefix)
                dates  = list_dates_for_user(layout, self.s3, self.bucket, user)

                def ui():
                    self._cmb_date["values"] = dates
                    if dates:
                        self._cmb_date.current(0)
                        self._on_date()
                    self._set_status(f"{len(dates)} date(s) for {user}.")
                self.after(0, ui)
            except Exception as e:
                self.after(0, lambda: self._set_status(f"Error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_date(self, _=None):
        user = self._cmb_user.get().strip()
        day  = self._cmb_date.get().strip()
        if not user or not day or not self.s3:
            return
        self._cmb_sess.set(""); self._cmb_sess["values"] = []
        self._sess_display_to_raw = {}
        layout     = self.layout or S3Layout(self.root_prefix)
        day_prefix = pick_day_prefix(layout, self.s3, self.bucket, user, day)
        if not day_prefix:
            self._set_status(f"No folder for {user}-{day}.")
            return
        self._day_prefix = day_prefix
        self._set_status(f"Loading sessions for {user}-{day}…")

        def worker():
            try:
                subs = list_common_prefixes(self.s3, self.bucket, day_prefix)
                d2r: dict[str, str] = {}
                items: list[str] = []
                for p in sorted(subs):
                    name = p.rstrip("/").split("/")[-1]
                    if name == "voice_drift":
                        continue
                    if not name.startswith(f"{user}-{day}-"):
                        continue
                    sp     = f"{day_prefix}{name}/"
                    tagged = session_has_any_txt(self.s3, self.bucket, sp)
                    disp   = f"{name} #TAGGED" if tagged else name
                    items.append(disp);  d2r[disp] = name

                def ui():
                    self._sess_display_to_raw = d2r
                    self._cmb_sess["values"]  = items
                    if items:
                        self._cmb_sess.current(0)
                    self._set_status(f"{len(items)} session(s) loaded.")
                self.after(0, ui)
            except Exception as e:
                self.after(0, lambda: self._set_status(f"Error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _get_session_raw(self) -> str | None:
        disp = self._cmb_sess.get().strip()
        if not disp:
            return None
        return self._sess_display_to_raw.get(
            disp, disp.split(" #TAGGED")[0].strip()
        )

    def _browse_feat(self):
        d = filedialog.askdirectory(
            title="Select features folder", initialdir=self._feat_var.get()
        )
        if d:
            self._feat_var.set(d)

    def _refresh_models(self):
        feat_dir = Path(self._feat_var.get())
        if not feat_dir.is_dir():
            self._cmb_model["values"] = []
            return
        models = sorted(feat_dir.glob("best_model_*.pt"))
        names  = [m.name for m in models]
        self._cmb_model["values"] = names
        if names:
            cur = self._model_var.get()
            self._model_var.set(cur if cur in names else names[-1])
        else:
            self._model_var.set("")

    def _open_window(self):
        if not self.s3 or not self.bucket:
            messagebox.showinfo("Not connected", "Connect to S3 first.")
            return
        user = self._cmb_user.get().strip()
        day  = self._cmb_date.get().strip()
        if not user or not day:
            messagebox.showinfo("Missing selection", "Select User ID and Date.")
            return
        model_name = self._model_var.get().strip()
        if not model_name:
            messagebox.showinfo("No model", "Select a trained model.")
            return
        feat_dir   = Path(self._feat_var.get())
        model_path = feat_dir / model_name
        if not model_path.exists():
            messagebox.showerror("Not found", f"Model file not found:\n{model_path}")
            return
        # Infer feature type from model filename convention
        feature_type = "1d_profile" if "_1d_" in model_name else "rdmap"

        ClassificationWindow(
            parent       = self,
            s3           = self.s3,
            bucket       = self.bucket,
            root_prefix  = self.root_prefix,
            user_id      = user,
            day          = day,
            day_prefix   = self._day_prefix,
            model_path   = model_path,
            feature_type = feature_type,
            preselect    = self._get_session_raw(),
        )


# ── Classification Window ─────────────────────────────────────────────────────

class ClassificationWindow:
    """
    Session viewer + ADL classifier.
    Downloads CSV / mp4 / wav / txt from S3 to a local cache on first open,
    then reuses the cached files on subsequent opens.
    """

    def __init__(
        self, parent, s3, bucket, root_prefix,
        user_id, day, day_prefix,
        model_path: Path, feature_type: str,
        preselect: str | None = None,
    ):
        self.parent      = parent
        self.s3          = s3
        self.bucket      = bucket
        self.root_prefix = root_prefix
        self.user_id     = user_id
        self.day         = day
        self.day_prefix  = day_prefix
        self.model_path  = model_path
        self.feature_type = feature_type
        self.preselect   = preselect

        self.win = tk.Toplevel(parent)
        self.win.title(
            f"ADL Classification — {user_id}-{day}  |  {model_path.name}"
        )
        self.win.geometry("1300x860")

        # S3 session index
        self.session_keys: dict[str, str] = {}   # stem → S3 csv key

        # Data
        self.df: pd.DataFrame | None = None
        self.N  = 0
        self.timestamps: np.ndarray | None = None
        self.ts0 = 0.0
        self.range_axis: np.ndarray | None = None
        self.dev0_mag: np.ndarray | None = None  # (N, range_bins)
        self.dev1_mag: np.ndarray | None = None

        # Video
        self.video_cap: "cv2.VideoCapture | None" = None
        self.video_fps   = 0.0
        self.video_total = 0
        self.video_loaded = False
        self.preview_image = None

        # Audio
        self.audio_params: tuple | None = None   # (channels, sampwidth, samplerate)
        self.audio_frames: bytes | None = None   # PCM-16 bytes
        self.audio_offset_sec = 0.0
        self.audio_play_t0: float | None = None
        self._pygame_inited = False
        self._pygame_mixer_params: tuple | None = None
        self._audio_temp_path: str | None = None

        # Playback
        self.current_frame    = 0
        self.animation_running = False
        self._scale_updating  = False
        self._scale_job: str | None = None
        self._pending_idx     = 0

        # Classification background thread
        self._classify_thread: threading.Thread | None = None

        self.cache_root = (
            Path(tempfile.gettempdir()) / "mmwave_s3_cache" / f"{user_id}-{day}"
        )
        self.cache_root.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._load_sessions_async()
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        uf = ("Arial", 9)

        # ── Session selector row ──────────────────────────────────────────────
        top = tk.Frame(self.win)
        top.pack(fill="x", padx=10, pady=6)
        tk.Label(top, text="Session:", font=uf).pack(side="left")
        self._sess_var = tk.StringVar()
        self._cmb_sess = ttk.Combobox(
            top, textvariable=self._sess_var, width=52, state="readonly"
        )
        self._cmb_sess.pack(side="left", padx=8)
        self._cmb_sess.bind("<<ComboboxSelected>>", lambda _: self._load_session_async())
        self._lbl_info = tk.Label(top, text="Frame: --/--", font=uf)
        self._lbl_info.pack(side="left", padx=(20, 0))

        # ── Plot mode selector ────────────────────────────────────────────────
        plotbar = tk.Frame(self.win)
        plotbar.pack(fill="x", padx=10, pady=(0, 2))
        tk.Label(plotbar, text="Plot:", font=uf).pack(side="left")
        self._plot_mode_var = tk.StringVar(value="Range Map")
        self._cmb_plot = ttk.Combobox(
            plotbar, textvariable=self._plot_mode_var, state="readonly",
            width=14, font=uf, values=["Range Map", "Doppler Map"],
        )
        self._cmb_plot.pack(side="left", padx=6)
        self._cmb_plot.bind(
            "<<ComboboxSelected>>",
            lambda _: self._update_view(self.current_frame),
        )
        self._last_plot_mode: str | None = None

        # ── Main body: plots left, video+tags right ───────────────────────────
        body = tk.Frame(self.win)
        body.pack(fill="both", expand=True, padx=10, pady=4)

        left = tk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(body, width=440)
        right.pack(side="left", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        # Matplotlib: 2 rows (dev0 top, dev1 bottom)
        self.fig = self.canvas = None
        self.ax0 = self.ax1 = None
        self.line0 = self.line1 = None
        self.im0 = self.im1 = None
        if Figure is not None and FigureCanvasTkAgg is not None:
            self.fig = Figure(figsize=(8.0, 4.6), dpi=100)
            self.ax0 = self.fig.add_subplot(211)
            self.ax1 = self.fig.add_subplot(212)
            self.fig.tight_layout(pad=2.2)
            self.canvas = FigureCanvasTkAgg(self.fig, master=left)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            tk.Label(left, text="matplotlib not available — install it", fg="red").pack()

        # Video
        tk.Label(right, text="Video", font=("Arial", 10, "bold")).pack(anchor="w")
        self._video_label = tk.Label(right, bg="black", width=420, height=315)
        self._video_label.pack()

        # Activity tag buttons
        tk.Label(
            right, text="Activity tags  (click to jump):",
            font=("Arial", 9, "bold"),
        ).pack(anchor="w", pady=(8, 2))
        tags_outer = tk.Frame(right)
        tags_outer.pack(fill="both", expand=True)
        tags_cv = tk.Canvas(tags_outer, height=160, highlightthickness=0)
        tags_sb = ttk.Scrollbar(tags_outer, orient="vertical", command=tags_cv.yview)
        tags_cv.configure(yscrollcommand=tags_sb.set)
        tags_sb.pack(side="right", fill="y")
        tags_cv.pack(side="left", fill="both", expand=True)
        self._tags_frame = tk.Frame(tags_cv)
        _cw = tags_cv.create_window((0, 0), window=self._tags_frame, anchor="nw")
        self._tags_frame.bind(
            "<Configure>", lambda e: tags_cv.configure(scrollregion=tags_cv.bbox("all"))
        )
        tags_cv.bind("<Configure>", lambda e: tags_cv.itemconfig(_cw, width=e.width))

        # ── Playback controls ─────────────────────────────────────────────────
        ctrl = tk.Frame(self.win)
        ctrl.pack(fill="x", padx=10, pady=4)
        for lbl, cmd in [
            ("Play",  self._play),  ("Pause", self._pause),
            ("Prev",  self._prev),  ("Next",  self._next),
            ("Stop",  self._stop),
        ]:
            tk.Button(ctrl, text=lbl, command=cmd, font=uf, width=7).pack(
                side="left", padx=3
            )

        # ── Frame scrubber ────────────────────────────────────────────────────
        self._scale = tk.Scale(
            self.win, from_=1, to=1, orient=tk.HORIZONTAL,
            showvalue=0, resolution=1, command=self._on_scale,
        )
        self._scale.pack(fill="x", padx=10, pady=(0, 4))
        self._scale.bind("<ButtonRelease-1>", lambda _: self._on_scale_release())

        # ── Classification results ─────────────────────────────────────────────
        cls_frm = ttk.LabelFrame(self.win, text="Classification", padding=6)
        cls_frm.pack(fill="both", expand=True, padx=10, pady=4)

        cls_top = tk.Frame(cls_frm)
        cls_top.pack(fill="x", pady=(0, 4))
        self._cls_btn = ttk.Button(
            cls_top, text="▶  Classify Session", command=self._run_classification
        )
        self._cls_btn.pack(side="left", padx=4)
        self._cls_prog = ttk.Progressbar(cls_top, mode="indeterminate", length=180)
        self._cls_prog.pack(side="left", padx=8)
        self._cls_status = ttk.Label(cls_top, text="", anchor="w")
        self._cls_status.pack(side="left", padx=4, fill="x", expand=True)

        # Results table
        cols = (
            "Activity",
            "Start (frame)", "End (frame)",
            "Start (s)",     "End (s)",
            "Duration (s)",  "Confidence (%)",
        )
        self._tree = ttk.Treeview(cls_frm, columns=cols, show="headings", height=6)
        col_widths = [260, 100, 100, 80, 80, 100, 110]
        for col, w in zip(cols, col_widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="center")
        self._tree.column("Activity", anchor="w")
        tree_vsb = ttk.Scrollbar(cls_frm, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        tree_vsb.pack(side="left", fill="y")
        # Double-click a row → jump to start frame
        self._tree.bind("<Double-1>", self._on_row_dblclick)

    # ── S3 session discovery ──────────────────────────────────────────────────

    def _load_sessions_async(self):
        base = self.day_prefix or (
            f"{self.root_prefix}{self.user_id}-{self.day}/"
        )

        def worker():
            try:
                objs = list_objects(self.s3, self.bucket, base, max_items=5000)
                keys = [o["Key"] for o in objs if "Key" in o]
                csv_keys = [
                    k for k in keys
                    if k.lower().endswith(".csv") and "/voice_drift/" not in k
                ]
                sess_keys: dict[str, str] = {}
                for k in csv_keys:
                    stem = Path(k).stem
                    sess_keys[stem] = k
                stems = sorted(sess_keys)
                self.session_keys = sess_keys

                def ui():
                    self._cmb_sess["values"] = stems
                    if stems:
                        chosen = stems[0]
                        if self.preselect:
                            for s in stems:
                                if s == self.preselect or s.startswith(self.preselect):
                                    chosen = s
                                    break
                        self._sess_var.set(chosen)
                        self._load_session_async()
                    else:
                        self._set_info("No sessions found for this day.")
                self.win.after(0, ui)
            except Exception as e:
                self.win.after(0, lambda: self._set_info(f"Session list error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    # ── Session download + load ───────────────────────────────────────────────

    def _load_session_async(self):
        stem = self._sess_var.get().strip()
        if not stem or stem not in self.session_keys:
            return
        csv_key  = self.session_keys[stem]
        base_dir = str(Path(csv_key).parent).replace("\\", "/")

        mp4_key = f"{base_dir}/{stem}.mp4"
        wav_key = f"{base_dir}/{stem}.wav"
        txt_key = f"{base_dir}/{stem}.txt"

        sess_dir = self.cache_root / stem
        sess_dir.mkdir(parents=True, exist_ok=True)
        csv_path = sess_dir / f"{stem}.csv"
        mp4_path = sess_dir / f"{stem}.mp4"
        wav_path = sess_dir / f"{stem}.wav"
        txt_path = sess_dir / f"{stem}.txt"

        self._set_info("Downloading…")

        def worker():
            try:
                # CSV is mandatory
                if not csv_path.exists():
                    _s3_download_to_file(self.s3, self.bucket, csv_key, csv_path)
                # Optional assets: cached files are reused
                for key, path in [
                    (mp4_key, mp4_path),
                    (wav_key, wav_path),
                    (txt_key, txt_path),
                ]:
                    if not path.exists():
                        try:
                            _s3_download_to_file(self.s3, self.bucket, key, path)
                        except Exception:
                            if path.exists():
                                path.unlink(missing_ok=True)

                def ui():
                    self._load_local(
                        stem, csv_path,
                        mp4_path if mp4_path.exists() else None,
                        wav_path if wav_path.exists() else None,
                        txt_path if txt_path.exists() else None,
                    )
                self.win.after(0, ui)
            except Exception as e:
                self.win.after(0, lambda: self._set_info(f"Download error: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _load_local(self, stem, csv_path, mp4_path, wav_path, txt_path):
        self._stop()

        self.df = pd.read_csv(str(csv_path))
        self.N  = len(self.df)

        if "timestamp" in self.df.columns:
            self.timestamps = self.df["timestamp"].astype(float).to_numpy()
        else:
            self.timestamps = np.arange(self.N, dtype=float) / 5.0
        self.ts0 = float(self.timestamps[0]) if self.N > 0 else 0.0

        self._scale.config(from_=1, to=max(1, self.N))
        self._scale.set(1)

        # Range axis
        first      = self.df.iloc[0]
        ch1_arr    = _safe_literal_array(first.get("dev0_ch1", "[]"))
        num_samp   = int(ch1_arr.shape[1]) if ch1_arr.ndim == 2 else _NUM_SAMPLES
        self.range_axis = _compute_range_axis_m(num_samp, _FC_START, _FC_END)
        win_rng    = np.hanning(num_samp).astype(np.float32)

        # Pre-compute averaged range profiles for fast Range Map display
        dev0_mag = np.zeros((self.N, self.range_axis.size), dtype=np.float32)
        dev1_mag = np.zeros((self.N, self.range_axis.size), dtype=np.float32)
        for i in range(self.N):
            row = self.df.iloc[i]
            for mag_arr, pfx in [(dev0_mag, "dev0"), (dev1_mag, "dev1")]:
                chs = []
                for c in ("ch1", "ch2", "ch3"):
                    mat = _safe_literal_array(row.get(f"{pfx}_{c}", "[]"))
                    if mat.size:
                        chs.append(_compute_distance_mag(mat, win_rng))
                if chs:
                    mag_arr[i] = np.mean(np.stack(chs), axis=0)
        self.dev0_mag = dev0_mag
        self.dev1_mag = dev1_mag

        # Video
        self.video_loaded = False
        if self.video_cap:
            try:
                self.video_cap.release()
            except Exception:
                pass
            self.video_cap = None
        self.video_fps = 0.0;  self.video_total = 0
        if mp4_path and cv2 and mp4_path.exists():
            cap = cv2.VideoCapture(str(mp4_path))
            if cap.isOpened():
                self.video_cap   = cap
                self.video_fps   = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self.video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.video_loaded = True

        # Audio
        self.audio_params = None;  self.audio_frames = None
        self.audio_offset_sec = 0.0
        if wav_path and wav_path.exists():
            try:
                ch, sw, sr, raw16 = _read_wav_as_pcm16_bytes(wav_path)
                self.audio_params = (int(ch), int(sw), int(sr))
                self.audio_frames = raw16
            except Exception:
                pass

        # Tag buttons
        self._rebuild_tag_buttons(txt_path)

        # Reset classification panel
        self._tree.delete(*self._tree.get_children())
        self._cls_status.config(text="")
        self._last_plot_mode = None   # force axis rebuild on first view

        self.current_frame = 0
        self._update_view(0)
        self._set_info(f"Loaded — {self.N} frames")

    # ── Tag buttons ───────────────────────────────────────────────────────────

    def _rebuild_tag_buttons(self, txt_path):
        for w in self._tags_frame.winfo_children():
            w.destroy()
        if txt_path is None or not txt_path.exists():
            return
        try:
            text = txt_path.read_text(encoding="utf-8")
        except Exception:
            return
        segs = _parse_adl_tags(text)
        if not segs:
            return
        for seg in segs:
            act   = seg["action"]
            dev   = seg["radar"]
            start = int(seg["start"])
            end   = int(seg["end"])
            lbl   = f"{act}  |  {dev}  [{start}–{end}]"
            tk.Button(
                self._tags_frame, text=lbl, anchor="w", font=("Arial", 8),
                command=lambda s=start: self._jump(s),
            ).pack(fill="x", pady=1)

    # ── Playback ──────────────────────────────────────────────────────────────

    def _set_info(self, msg: str):
        self._lbl_info.config(text=msg)
        self.win.update_idletasks()

    def _play(self):
        if self.N <= 0:
            return
        t0 = float(self.timestamps[self.current_frame] - self.timestamps[0])
        self.audio_offset_sec  = max(0.0, t0)
        self.animation_running = True
        self._audio_play_from(self.audio_offset_sec)
        self._animate()

    def _pause(self):
        if self.animation_running:
            self._audio_pause()
        self.animation_running = False

    def _stop(self):
        self.animation_running = False
        self._audio_stop(reset=True)
        self.current_frame = 0
        if self.N > 0:
            self._update_view(0)

    def _next(self):
        self.animation_running = False
        self._audio_stop(reset=False)
        if self.current_frame < self.N - 1:
            self.current_frame += 1
        self._audio_seek(self.current_frame)
        self._update_view(self.current_frame)

    def _prev(self):
        self.animation_running = False
        self._audio_stop(reset=False)
        if self.current_frame > 0:
            self.current_frame -= 1
        self._audio_seek(self.current_frame)
        self._update_view(self.current_frame)

    def _jump(self, frame_1idx: int):
        """Jump to a 1-indexed frame number."""
        idx = max(0, min(int(frame_1idx) - 1, self.N - 1))
        self.animation_running = False
        self._audio_stop(reset=False)
        self.current_frame = idx
        self._audio_seek(idx)
        self._update_view(idx)

    def _animate(self):
        if not self.animation_running:
            return
        if self.current_frame >= self.N:
            self.animation_running = False
            return
        self._update_view(self.current_frame)
        self.current_frame += 1

        delay_ms = 200
        if self.video_loaded and self.video_fps > 0:
            delay_ms = int(1000.0 / self.video_fps)
        elif self.N >= 2 and self.timestamps is not None:
            i  = min(self.current_frame, self.N - 1)
            dt = float(self.timestamps[i] - self.timestamps[max(0, i - 1)])
            if dt > 0:
                delay_ms = max(1, int(1000.0 * dt))
        self.win.after(delay_ms, self._animate)

    # ── Scrubber ──────────────────────────────────────────────────────────────

    def _on_scale(self, val):
        if self._scale_updating:
            return
        try:
            self._pending_idx = int(float(val)) - 1
        except Exception:
            return
        if self._scale_job:
            try:
                self.win.after_cancel(self._scale_job)
            except Exception:
                pass
        self._scale_job = self.win.after(25, self._apply_scale)

    def _apply_scale(self):
        self._scale_job = None
        if self.N <= 0:
            return
        idx = max(0, min(self._pending_idx, self.N - 1))
        if self.animation_running:
            self._audio_stop(reset=False)
        self.animation_running = False
        self.current_frame = idx
        self._audio_seek(idx)
        self._update_view(idx)

    def _on_scale_release(self):
        if self._scale_job:
            try:
                self.win.after_cancel(self._scale_job)
            except Exception:
                pass
            self._scale_job = None
        self._apply_scale()

    # ── Audio helpers (mirrors S3TaggerWindow approach) ───────────────────────

    def _pygame_init(self) -> bool:
        if pygame is None or self.audio_params is None:
            return False
        ch, sw, sr = self.audio_params
        params = (int(sr), int(-8 * sw), int(ch))
        try:
            if not self._pygame_inited or self._pygame_mixer_params != params:
                if self._pygame_inited:
                    try:
                        pygame.mixer.quit()
                    except Exception:
                        pass
                pygame.mixer.init(
                    frequency=params[0], size=params[1], channels=params[2]
                )
                self._pygame_inited = True
                self._pygame_mixer_params = params
        except Exception:
            return False
        return True

    def _audio_cleanup_temp(self):
        try:
            p = self._audio_temp_path
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
        self._audio_temp_path = None

    def _audio_play_from(self, start_sec: float):
        if self.audio_params is None or self.audio_frames is None:
            return
        if not self._pygame_init():
            return
        ch, sw, sr = self.audio_params
        bpf        = ch * sw
        start_byte = int(max(0.0, start_sec) * sr) * bpf
        if start_byte >= len(self.audio_frames):
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        self.audio_play_t0 = None
        self._audio_cleanup_temp()
        try:
            import wave
            buf = self.audio_frames[start_byte:]
            fd, tmp = tempfile.mkstemp(prefix="adlcls_", suffix=".wav")
            os.close(fd)
            with wave.open(tmp, "wb") as wf:
                wf.setnchannels(int(ch))
                wf.setsampwidth(int(sw))
                wf.setframerate(int(sr))
                wf.writeframes(buf)
            self._audio_temp_path = tmp
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            self.audio_play_t0 = time.monotonic()
        except Exception:
            self._audio_cleanup_temp()

    def _audio_pause(self):
        if self.audio_play_t0 is not None:
            try:
                self.audio_offset_sec += time.monotonic() - self.audio_play_t0
            except Exception:
                pass
        try:
            if pygame:
                pygame.mixer.music.stop()
        except Exception:
            pass
        self.audio_play_t0 = None
        self._audio_cleanup_temp()

    def _audio_stop(self, reset: bool = False):
        try:
            if pygame:
                pygame.mixer.music.stop()
        except Exception:
            pass
        self.audio_play_t0 = None
        self._audio_cleanup_temp()
        if reset:
            self.audio_offset_sec = 0.0

    def _audio_seek(self, frame_idx: int):
        if self.N <= 0 or self.timestamps is None:
            return
        t = float(self.timestamps[frame_idx] - self.timestamps[0])
        self.audio_offset_sec = max(0.0, t)
        if self.animation_running:
            self._audio_play_from(self.audio_offset_sec)

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _ensure_plot_mode(self, mode: str):
        """Rebuild matplotlib axes when the plot mode changes."""
        if self.canvas is None or mode == self._last_plot_mode:
            return
        for ax in (self.ax0, self.ax1):
            ax.cla()
        self.line0 = self.line1 = None
        self.im0   = self.im1   = None

        if mode == "Doppler Map":
            self.ax0.set_title("dev0 — Doppler Map", fontsize=9)
            self.ax1.set_title("dev1 — Doppler Map", fontsize=9)
            self.ax0.set_ylabel("Velocity (m/s)", fontsize=8)
            self.ax1.set_ylabel("Velocity (m/s)", fontsize=8)
            self.ax1.set_xlabel("Distance (m)",   fontsize=8)
        else:
            self.ax0.set_title("dev0 — Range Map", fontsize=9)
            self.ax1.set_title("dev1 — Range Map", fontsize=9)
            self.ax0.set_ylabel("Amplitude", fontsize=8)
            self.ax1.set_ylabel("Amplitude", fontsize=8)
            self.ax1.set_xlabel("Distance (m)", fontsize=8)
            (self.line0,) = self.ax0.plot([], [], lw=1.2)
            (self.line1,) = self.ax1.plot([], [], lw=1.2)
        self._last_plot_mode = mode

    def _update_view(self, idx: int):
        if self.N <= 0:
            return
        idx = max(0, min(idx, self.N - 1))
        self._lbl_info.config(text=f"Frame: {idx + 1}/{self.N}")

        # Sync scrubber (guard against recursive callback)
        if not self._scale_updating:
            self._scale_updating = True
            try:
                self._scale.set(idx + 1)
            finally:
                self.win.after_idle(
                    lambda: setattr(self, "_scale_updating", False)
                )

        # ── Radar plots ───────────────────────────────────────────────────────
        if self.canvas is not None and self.range_axis is not None and self.df is not None:
            mode = self._plot_mode_var.get()
            self._ensure_plot_mode(mode)
            row     = self.df.iloc[idx]
            win_rng = np.hanning(_NUM_SAMPLES).astype(np.float32)

            if mode == "Doppler Map":
                def _rd_for(pfx: str):
                    rd_list, vel = [], None
                    for c in ("ch1", "ch2", "ch3"):
                        mat = _safe_literal_array(row.get(f"{pfx}_{c}", "[]"))
                        if mat.size:
                            rd_mag, v = _compute_range_doppler_mag(
                                mat, win_rng, _DOPPLER_PRF
                            )
                            rd_list.append(rd_mag)
                            if vel is None:
                                vel = v
                    if not rd_list:
                        return (
                            np.zeros((1, max(1, self.range_axis.size)), dtype=np.float32),
                            np.zeros(1, dtype=np.float32),
                        )
                    return np.mean(np.stack(rd_list), axis=0), vel

                rd0, v0 = _rd_for("dev0")
                rd1, v1 = _rd_for("dev1")
                eps  = 1e-6
                db0  = 20.0 * np.log10(np.maximum(rd0, eps))
                db1  = 20.0 * np.log10(np.maximum(rd1, eps))
                x0   = float(self.range_axis[0])
                x1   = float(self.range_axis[-1])
                ext0 = (x0, x1, float(v0[0]), float(v0[-1]))
                ext1 = (x0, x1, float(v1[0]), float(v1[-1]))
                if self.im0 is None:
                    self.im0 = self.ax0.imshow(
                        db0, origin="lower", aspect="auto", extent=ext0, cmap="inferno"
                    )
                else:
                    self.im0.set_data(db0);  self.im0.set_extent(ext0)
                if self.im1 is None:
                    self.im1 = self.ax1.imshow(
                        db1, origin="lower", aspect="auto", extent=ext1, cmap="inferno"
                    )
                else:
                    self.im1.set_data(db1);  self.im1.set_extent(ext1)
            else:
                # Range Map (pre-computed)
                if self.dev0_mag is not None and self.line0 is not None:
                    self.line0.set_data(self.range_axis, self.dev0_mag[idx])
                    self.ax0.relim();  self.ax0.autoscale_view()
                if self.dev1_mag is not None and self.line1 is not None:
                    self.line1.set_data(self.range_axis, self.dev1_mag[idx])
                    self.ax1.relim();  self.ax1.autoscale_view()

            self.canvas.draw()

        # ── Video frame ───────────────────────────────────────────────────────
        if (
            self.video_loaded and self.video_cap is not None
            and cv2 is not None and Image is not None and ImageTk is not None
            and self.timestamps is not None
        ):
            t     = float(self.timestamps[idx] - self.ts0)
            v_idx = int(round(t * self.video_fps)) if self.video_fps > 0 else 0
            v_idx = max(0, min(v_idx, self.video_total - 1))
            try:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, v_idx)
                ok, frame = self.video_cap.read()
                if ok and frame is not None:
                    frame   = cv2.resize(frame, (420, 315))
                    frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imgtk   = ImageTk.PhotoImage(image=Image.fromarray(frame))
                    self.preview_image = imgtk          # keep reference
                    self._video_label.config(image=imgtk)
            except Exception:
                pass

    # ── Classification ────────────────────────────────────────────────────────

    def _run_classification(self):
        if self.df is None or self.N == 0:
            messagebox.showinfo("No data", "Load a session first.")
            return
        if self._classify_thread and self._classify_thread.is_alive():
            messagebox.showinfo("Busy", "Classification is already running.")
            return

        self._cls_btn.configure(state="disabled")
        self._cls_prog.start(10)
        self._cls_status.config(text="Running…")
        self._tree.delete(*self._tree.get_children())

        df_snap      = self.df.copy()
        model_path   = self.model_path
        feature_type = self.feature_type

        def log_fn(msg: str):
            # Show last status line in the label (safe from any thread via .after)
            self.win.after(0, lambda m=str(msg): self._cls_status.config(text=m[:90]))

        def worker():
            try:
                _p0, _p1, label_names, segments = run_classification(
                    df_snap, model_path, feature_type,
                    log_fn=log_fn,
                )

                def ui():
                    self._cls_prog.stop()
                    self._cls_btn.configure(state="normal")
                    if segments is None:
                        self._cls_status.config(text="Error: no device data.")
                        return
                    self._cls_status.config(
                        text=f"{len(segments)} segment(s) detected.  "
                             f"Double-click a row to jump to it."
                    )
                    for seg in segments:
                        act = (
                            f"[dev0] {seg['dev0_activity']}"
                            f"  [dev1] {seg['dev1_activity']}"
                        )
                        self._tree.insert("", "end", values=(
                            act,
                            seg["start_frame"],
                            seg["end_frame"],
                            seg["start_sec"],
                            seg["end_sec"],
                            seg["duration_sec"],
                            f"{seg['confidence']:.1f}%",
                        ))
                self.win.after(0, ui)

            except Exception as e:
                import traceback
                tb = traceback.format_exc()

                def ui_err():
                    self._cls_prog.stop()
                    self._cls_btn.configure(state="normal")
                    self._cls_status.config(text=f"Error: {e}")
                    messagebox.showerror(
                        "Classification error", f"{e}\n\n{tb[:600]}"
                    )
                self.win.after(0, ui_err)

        self._classify_thread = threading.Thread(target=worker, daemon=True)
        self._classify_thread.start()

    def _on_row_dblclick(self, _):
        sel = self._tree.selection()
        if not sel:
            return
        vals = self._tree.item(sel[0])["values"]
        try:
            self._jump(int(vals[1]))   # vals[1] = start_frame (1-indexed)
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop()
        if self.video_cap:
            try:
                self.video_cap.release()
            except Exception:
                pass
        self.win.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
