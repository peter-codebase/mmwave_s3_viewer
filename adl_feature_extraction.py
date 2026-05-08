#!/usr/bin/env python3
"""
ADL Feature Extraction Pipeline
================================
Converts raw chirp data stored in ADL CSVs into tensors ready for ML training.

Two feature types are supported (--feature_type):

  2D Range-Doppler maps  (default, 'rdmap')
  ─────────────────────────────────────────
  Processing chain per frame:
    raw chirp (32 chirps × 64 samples × 3 Rx channels)
      → DopplerAlgo  (range FFT + MTI + Doppler FFT) → complex (64×64) per channel
      → log1p magnitude  (64 range bins × 64 Doppler bins)  — ch0-2
          ch0: rx0_mag, ch1: rx1_mag, ch2: rx2_mag
  Each window shape: (T, 64, 64, 3)   e.g. (50, 64, 64, 3) at window=50

  1D Doppler/range projection  ('1d_profile')
  ─────────────────────────────────────────────
  Processing chain per frame:
    same radar processing up to the Range-Doppler map, then:
      → range profile  : sum over Doppler axis  → (64,) per Rx  (ch 0-2)
      → Doppler profile: sum over range  axis   → (64,) per Rx  (ch 3-5)
      → 6 channels: rx0_range, rx1_range, rx2_range,
                    doppler_proj_rx0, doppler_proj_rx1, doppler_proj_rx2
  Each window shape: (T, 64, 6)   e.g. (64, 64, 6) at window=64
  Much smaller footprint; feeds ADLClassifier1D (CNN1D + BiGRU).

Outputs saved to  data/adl_features/{user}/ :
  adl_features.npz        — 2D features  (X, y, label_map, meta)
  adl_features_1d.npz     — 1D features  (same structure)
  adl_features_<tag>.npz  — subset features when --tag is used

Usage (GUI — default):
  python adl_feature_extraction.py

Usage (CLI — 2D, all activities):
  python adl_feature_extraction.py --nogui

Usage (CLI — 1D, all activities):
  python adl_feature_extraction.py --nogui --model 1d

Usage (CLI — bathroom subset, both models):
  python adl_feature_extraction.py --nogui --model 2d \\
      --activities "face_washing,hair_washing,hand_washing,tooth_brushing,drinking" --tag bathroom
  python adl_feature_extraction.py --nogui --model 1d \\
      --activities "face_washing,hair_washing,hand_washing,tooth_brushing,drinking" --tag bathroom
"""

import ast
import json
import sys
import argparse
import threading
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from helpers.DopplerAlgo import DopplerAlgo

# ── Radar hardware constants ───────────────────────────────────────────────────
NUM_CHIRPS   = 32
NUM_SAMPLES  = 64
NUM_CHANNELS = 3

LENGTH_WINDOW = 50
LENGTH_STRIDE = 25

# ── Chirp parsing ─────────────────────────────────────────────────────────────

def _parse_chirp_cell(cell) -> np.ndarray:
    arr = np.asarray(ast.literal_eval(cell), dtype=np.float32)
    return arr.reshape(NUM_CHIRPS, NUM_SAMPLES)


# ── Range-Doppler computation ─────────────────────────────────────────────────

def _segment_to_rdmaps(frames_data: list[tuple]) -> np.ndarray:
    """
    Compute per-frame RD magnitude maps for a segment.
    A fresh DopplerAlgo is created per segment so MTI history does not
    bleed between unrelated segments.

    Returns ndarray shape (N_frames, 64, 64, 3).
    Channel layout:
      0  rx0_mag — log-magnitude RD map, RX channel 0
      1  rx1_mag — log-magnitude RD map, RX channel 1
      2  rx2_mag — log-magnitude RD map, RX channel 2
    """
    algo = DopplerAlgo(NUM_SAMPLES, NUM_CHIRPS, NUM_CHANNELS)
    rd_maps = []
    for ch1, ch2, ch3 in frames_data:
        rd0 = algo.compute_doppler_map(ch1, 0)   # (64, 64) complex
        rd1 = algo.compute_doppler_map(ch2, 1)
        rd2 = algo.compute_doppler_map(ch3, 2)

        mag0 = np.log1p(np.abs(rd0)).astype(np.float32)
        mag1 = np.log1p(np.abs(rd1)).astype(np.float32)
        mag2 = np.log1p(np.abs(rd2)).astype(np.float32)

        frame_3ch = np.stack([mag0, mag1, mag2], axis=-1)    # (64, 64, 3)
        rd_maps.append(frame_3ch)
    return np.array(rd_maps, dtype=np.float32)               # (N, 64, 64, 3)


def _capon_angle_map(rd_a: np.ndarray, rd_b: np.ndarray, n_angles: int = 64) -> np.ndarray:
    """Capon MVDR beamformer: two complex RD maps → (rng_bins, n_angles) float32."""
    n_range, n_doppler = rd_a.shape
    X = np.stack([rd_a, rd_b], axis=1).astype(np.complex64)
    R = (X @ X.conj().transpose(0, 2, 1)) / n_doppler
    diag_load = np.real(np.diagonal(R, axis1=1, axis2=2)).mean(axis=1)
    R += np.maximum(diag_load * 0.01, 1e-10)[:, None, None] * np.eye(2, dtype=np.complex64)[None]
    R_inv = np.linalg.inv(R)
    angles = np.linspace(-np.pi / 2, np.pi / 2, n_angles)
    A = np.stack([np.ones(n_angles, dtype=np.complex64),
                  np.exp(1j * 2 * np.pi * 0.5 * np.sin(angles)).astype(np.complex64)], axis=-1)
    AR  = np.einsum('ai,rij->raj', A.conj(), R_inv)
    den = np.einsum('raj,aj->ra', AR, A).real                 # (rng, A)
    return np.log1p(1.0 / (np.abs(den) + 1e-10)).astype(np.float32)  # (rng, A)


def _segment_to_rdmaps_ramap(frames_data: list[tuple]) -> np.ndarray:
    """
    Compute per-frame 5-channel feature maps (3 RD + Range-Azimuth + Range-Elevation).

    Returns ndarray shape (N_frames, 64, 64, 5).
    Channel layout:
      0  rx0_mag — log-magnitude RD map, Rx0
      1  rx1_mag — log-magnitude RD map, Rx1
      2  rx2_mag — log-magnitude RD map, Rx2 (corner reference)
      3  ra_map  — Capon Range-Azimuth  (Rx0 + Rx2 pair)
      4  re_map  — Capon Range-Elevation (Rx1 + Rx2 pair)
    """
    algo = DopplerAlgo(NUM_SAMPLES, NUM_CHIRPS, NUM_CHANNELS)
    frames = []
    for ch1, ch2, ch3 in frames_data:
        rd0 = algo.compute_doppler_map(ch1, 0)   # complex (64, 64)
        rd1 = algo.compute_doppler_map(ch2, 1)
        rd2 = algo.compute_doppler_map(ch3, 2)

        mag0 = np.log1p(np.abs(rd0)).astype(np.float32)
        mag1 = np.log1p(np.abs(rd1)).astype(np.float32)
        mag2 = np.log1p(np.abs(rd2)).astype(np.float32)

        az_map = _capon_angle_map(rd0, rd2)   # (64, 64)
        el_map = _capon_angle_map(rd1, rd2)   # (64, 64)

        frames.append(np.stack([mag0, mag1, mag2, az_map, el_map], axis=-1))
    return np.array(frames, dtype=np.float32)   # (N, 64, 64, 5)


def _segment_to_profiles1d(frames_data: list[tuple]) -> np.ndarray:
    """
    Extract 1D amplitude profiles per frame: range projection + Doppler projection
    per Rx channel.  No spatial 2D grid — each channel is a plain (64,) vector.

    Returns ndarray shape (N_frames, 64, 6).
    Channel layout (all indexed over 64 bins):
      0-2  range_proj_rx0/1/2   — log-mag RD map summed over Doppler bins → WHERE things are
      3-5  doppler_proj_rx0/1/2 — log-mag RD map summed over range  bins  → HOW FAST things move
    """
    algo = DopplerAlgo(NUM_SAMPLES, NUM_CHIRPS, NUM_CHANNELS)
    profiles = []
    for ch1, ch2, ch3 in frames_data:
        rd0 = algo.compute_doppler_map(ch1, 0)
        rd1 = algo.compute_doppler_map(ch2, 1)
        rd2 = algo.compute_doppler_map(ch3, 2)

        mag0 = np.log1p(np.abs(rd0)).astype(np.float32)  # (64, 64)
        mag1 = np.log1p(np.abs(rd1)).astype(np.float32)
        mag2 = np.log1p(np.abs(rd2)).astype(np.float32)

        rp0 = mag0.sum(axis=1)   # sum over Doppler → (64,) range profile
        rp1 = mag1.sum(axis=1)
        rp2 = mag2.sum(axis=1)

        dp0 = mag0.sum(axis=0)   # sum over range  → (64,) Doppler profile
        dp1 = mag1.sum(axis=0)
        dp2 = mag2.sum(axis=0)

        frame_6ch = np.stack([rp0, rp1, rp2, dp0, dp1, dp2], axis=-1)  # (64, 6)
        profiles.append(frame_6ch)
    return np.array(profiles, dtype=np.float32)              # (N, 64, 6)


# ── Sliding window ────────────────────────────────────────────────────────────

def _sliding_windows(rd_maps: np.ndarray, window: int, stride: int) -> list[np.ndarray]:
    """
    Slide a fixed-length window across the frame axis.
    Short segments are zero-padded; a non-redundant tail window is added
    when the last full window does not reach the end.

    3 channels: rx0_mag, rx1_mag, rx2_mag.
    No temporal_max — it encodes absolute range position, hurting generalization across
    different recording distances/angles.
    """
    N = rd_maps.shape[0]
    wins = []
    if N < window:
        # Repeat-pad with the last real frame so padded frames resemble actual
        # quiet radar frames.  Zero-padding creates artificially low std which
        # amplifies motion frames to very large z-scores — a pattern the model
        # will never see at inference time (where quiet frames have real noise).
        last = rd_maps[-1:]                                   # (1, H, W, C)
        pad  = np.repeat(last, window - N, axis=0)
        wins.append(np.concatenate([rd_maps, pad], axis=0).astype(np.float32))
    else:
        start = 0
        while start + window <= N:
            wins.append(rd_maps[start : start + window].astype(np.float32))
            start += stride
        tail_start = N - window
        prev_start = start - stride
        if tail_start > prev_start:
            wins.append(rd_maps[tail_start:].astype(np.float32))
    return wins


def _sliding_windows_1d(profiles: np.ndarray, window: int, stride: int) -> list[np.ndarray]:
    """
    Sliding window for 1D profiles.

    profiles: (N_frames, bins=64, channels=6)
    Returns list of windows, each shape (window, 64, 6).
    Short segments are zero-padded; non-redundant tail window added when needed.
    No temporal_max channel (not meaningful for 1D projections).
    """
    N = profiles.shape[0]
    wins = []
    if N < window:
        last = profiles[-1:]                                  # (1, bins, C)
        pad  = np.repeat(last, window - N, axis=0)
        wins.append(np.concatenate([profiles, pad], axis=0))
    else:
        start = 0
        while start + window <= N:
            wins.append(profiles[start : start + window].astype(np.float32))
            start += stride
        tail_start = N - window
        prev_start = start - stride
        if tail_start > prev_start:
            wins.append(profiles[tail_start:].astype(np.float32))
    return wins


# ── Core extraction ───────────────────────────────────────────────────────────

def _extract_sessions(csv_files, sessions_to_extract, label_map,
                      allowed_sessions, window, stride, feature_type, log_fn):
    """
    Extract windows for the given (label, source_file) pairs.
    sessions_to_extract: set of (label, source_file) or None to extract all.
    Returns (X_list, y_list, meta_list) with per-activity count logging.
    """
    all_X, all_y, all_meta = [], [], []

    for csv_path in csv_files:
        label = csv_path.stem
        if label not in label_map:
            continue
        y_val = label_map[label]
        df = pd.read_csv(str(csv_path), on_bad_lines='skip')

        n_segs = df.groupby(["source_file", "segment_idx"]).ngroups
        n_wins = 0
        n_skipped = 0

        for (src, seg_idx), grp in df.groupby(["source_file", "segment_idx"]):
            if allowed_sessions is not None and src not in allowed_sessions:
                continue
            if sessions_to_extract is not None and (label, src) not in sessions_to_extract:
                continue
            grp = grp.sort_values("frame_number").reset_index(drop=True)
            frames_data = []
            for _, row in grp.iterrows():
                try:
                    ch1 = _parse_chirp_cell(row["dev0_ch1"])
                    ch2 = _parse_chirp_cell(row["dev0_ch2"])
                    ch3 = _parse_chirp_cell(row["dev0_ch3"])
                    frames_data.append((ch1, ch2, ch3))
                except Exception as e:
                    n_skipped += 1
                    log_fn(f"  [skip frame] {src} seg={seg_idx}: {e}")
                    continue

            if not frames_data:
                continue

            if feature_type == '1d_profile':
                maps = _segment_to_profiles1d(frames_data)
                wins = _sliding_windows_1d(maps, window, stride)
            elif feature_type == 'rdmap_ramap':
                maps = _segment_to_rdmaps_ramap(frames_data)
                wins = _sliding_windows(maps, window, stride)
            else:
                maps = _segment_to_rdmaps(frames_data)
                wins = _sliding_windows(maps, window, stride)

            for w_idx, w in enumerate(wins):
                all_X.append(w)
                all_y.append(y_val)
                all_meta.append((label, src, int(seg_idx), w_idx))
            n_wins += len(wins)

        if sessions_to_extract is None:
            skip_note = f"  ({n_skipped} frames skipped)" if n_skipped else ""
            log_fn(f"  {label:22s}: {n_segs:3d} segments → {n_wins:4d} windows{skip_note}")

    return all_X, all_y, all_meta


def _try_incremental(out_npz, meta_path, label_path, params_path,
                     csv_files, allowed_sessions, window, stride,
                     feature_type, log_fn):
    """
    Attempt an incremental update of an existing .npz.

    Returns (X, y, meta_df, label_map, changed) where changed=False means
    the dataset was already up to date and saving can be skipped.
    Returns None if a full re-extraction is required.
    """
    if not all(p.exists() for p in [out_npz, meta_path, label_path, params_path]):
        return None

    with open(str(params_path)) as f:
        params = json.load(f)

    if (params['window'] != window or params['stride'] != stride or
            params['feature_type'] != feature_type):
        log_fn(f"  Parameters changed (window/stride/type) → full re-extraction.\n")
        return None

    log_fn("  Existing .npz found — checking for new/removed sessions…")

    data     = np.load(str(out_npz))
    X_old    = data['X']
    y_old    = data['y']
    meta_old = pd.read_csv(str(meta_path))
    with open(str(label_path)) as f:
        label_map_old = json.load(f)

    # ── Determine current (label, source_file) pairs ──────────────────────────
    current_sessions: set[tuple] = set()
    for csv_path in csv_files:
        label = csv_path.stem
        df_src = pd.read_csv(str(csv_path), usecols=["source_file"])
        for src in df_src["source_file"].unique():
            if allowed_sessions is None or src in allowed_sessions:
                current_sessions.add((label, src))

    existing_sessions = set(zip(meta_old["label"], meta_old["source_file"]))
    new_sessions      = current_sessions - existing_sessions
    removed_sessions  = existing_sessions - current_sessions

    if not new_sessions and not removed_sessions:
        log_fn("  No changes detected — .npz is already up to date.\n")
        return X_old, y_old, meta_old, label_map_old, False

    log_fn(f"  +{len(new_sessions)} session(s) to add   "
           f"  -{len(removed_sessions)} session(s) to remove\n")

    # ── Drop removed sessions ─────────────────────────────────────────────────
    if removed_sessions:
        keep = ~pd.Series(
            list(zip(meta_old["label"], meta_old["source_file"]))
        ).isin(removed_sessions).values
        X_kept    = X_old[keep]
        y_kept    = y_old[keep]
        meta_kept = meta_old[keep].reset_index(drop=True)
        for lbl, src in sorted(removed_sessions):
            log_fn(f"  - removed: {lbl} / {src}")
    else:
        X_kept, y_kept, meta_kept = X_old, y_old, meta_old.copy()

    # ── Rebuild label_map ─────────────────────────────────────────────────────
    label_map = dict(label_map_old)
    current_labels = {lbl for lbl, _ in current_sessions}

    # Add new labels at the end
    for lbl in sorted(current_labels):
        if lbl not in label_map:
            label_map[lbl] = max(label_map.values()) + 1 if label_map else 0
            log_fn(f"  + new activity added: {lbl} → class {label_map[lbl]}")

    # Remove labels with no remaining windows and no incoming new sessions
    remaining_labels = set(meta_kept["label"].unique()) | {lbl for lbl, _ in new_sessions}
    dropped_labels   = set(label_map.keys()) - remaining_labels
    if dropped_labels:
        for lbl in dropped_labels:
            del label_map[lbl]
            log_fn(f"  - activity fully removed: {lbl}")
        # Remap integers to stay contiguous
        sorted_lbls = sorted(label_map, key=lambda k: label_map[k])
        old_to_new  = {label_map_old[lbl]: new_i
                       for new_i, lbl in enumerate(sorted_lbls)
                       if lbl in label_map_old}
        label_map   = {lbl: new_i for new_i, lbl in enumerate(sorted_lbls)}
        y_kept      = np.array([old_to_new[yi] for yi in y_kept], dtype=np.int32)

    # ── Extract new sessions ──────────────────────────────────────────────────
    if new_sessions:
        log_fn(f"  Extracting {len(new_sessions)} new session(s)…")
        new_X, new_y, new_meta = _extract_sessions(
            csv_files, new_sessions, label_map,
            allowed_sessions, window, stride, feature_type, log_fn)
        for lbl, src in sorted(new_sessions):
            n = sum(1 for m in new_meta if m[0] == lbl and m[1] == src)
            log_fn(f"  + added:   {lbl} / {src}  ({n} windows)")
        if new_X:
            X_new    = np.array(new_X, dtype=np.float32)
            y_new    = np.array(new_y, dtype=np.int32)
            meta_new = pd.DataFrame(new_meta,
                                    columns=["label", "source_file",
                                             "segment_idx", "window_idx"])
            X_final    = np.concatenate([X_kept, X_new], axis=0)
            y_final    = np.concatenate([y_kept, y_new], axis=0)
            meta_final = pd.concat([meta_kept, meta_new], ignore_index=True)
        else:
            X_final, y_final, meta_final = X_kept, y_kept, meta_kept
    else:
        X_final, y_final, meta_final = X_kept, y_kept, meta_kept

    return X_final, y_final, meta_final, label_map, True


def run(adl_dir: Path, output_dir: Path,
        selected_labels: list[str] | None,
        window: int, stride: int,
        allowed_sessions: set[str] | None = None,
        feature_type: str = 'rdmap',
        tag: str = '',
        force: bool = False,
        log_fn=print) -> None:
    """
    Extract features for the given selected_labels (or all CSVs if None).

    Incremental mode (default): if an existing .npz is found with matching
    window/stride/feature_type, only new sessions are extracted and removed
    sessions are dropped.  Pass force=True to always do a full re-extraction.

    allowed_sessions: if provided, only segments whose source_file is in this
                      set are included.  Pass None to include all sessions.
    feature_type: 'rdmap'      — 2D Range-Doppler maps (default, 6ch, saves adl_features.npz)
                  '1d_profile' — 1D range+Doppler projections (6ch, saves adl_features_1d.npz)
    tag: optional string appended to output filenames, e.g. 'bathroom' →
         adl_features_bathroom.npz  (or adl_features_1d_bathroom.npz)
    log_fn: callable used for all status messages — print() for CLI,
            a GUI text-widget writer for the GUI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(adl_dir.glob("*.csv"))
    if not csv_files:
        log_fn(f"No CSVs found in {adl_dir}")
        return

    if selected_labels is not None:
        csv_files = [f for f in csv_files if f.stem in selected_labels]
        if not csv_files:
            log_fn("No matching CSVs for the selected activities.")
            return

    labels = [f.stem for f in csv_files]
    log_fn(f"Activities selected: {labels}")
    log_fn(f"Window={window} frames, stride={stride} frames  "
           f"(≈ {window/5:.1f}s at 5 Hz)\n")

    if feature_type == '1d_profile':
        model_type = "1d";   ch_count = 6
    elif feature_type == 'rdmap_ramap':
        model_type = "2d_ra"; ch_count = 5
    else:
        model_type = "2d";   ch_count = 3
    suffix      = f"_{model_type}_w{window}_ch{ch_count}"
    if tag:
        suffix  = f"{suffix}_{tag}"
    out_npz     = output_dir / f"adl_features{suffix}.npz"
    meta_path   = output_dir / f"meta{suffix}.csv"
    label_path  = output_dir / f"label_map{suffix}.json"
    params_path = output_dir / f"adl_params{suffix}.json"

    # ── Try incremental update ────────────────────────────────────────────────
    incremental = None
    if not force:
        incremental = _try_incremental(
            out_npz, meta_path, label_path, params_path,
            csv_files, allowed_sessions, window, stride, feature_type, log_fn)

    if incremental is not None:
        X, y, meta_df, label_map, changed = incremental
        if not changed:
            # Up to date — just print summary and return without re-saving
            _log_summary(X, y, label_map, out_npz, feature_type, log_fn)
            return
    else:
        # ── Full extraction ───────────────────────────────────────────────────
        if force:
            log_fn("  Force re-extraction requested.\n")
        label_map = {name: idx for idx, name in enumerate(labels)}
        all_X, all_y, all_meta = _extract_sessions(
            csv_files, None, label_map,
            allowed_sessions, window, stride, feature_type, log_fn)
        X       = np.array(all_X, dtype=np.float32)
        y       = np.array(all_y,  dtype=np.int32)
        meta_df = pd.DataFrame(all_meta,
                               columns=["label", "source_file",
                                        "segment_idx", "window_idx"])

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez_compressed(str(out_npz), X=X, y=y)
    with open(str(label_path), "w") as f:
        json.dump(label_map, f, indent=2)
    meta_df.to_csv(str(meta_path), index=False)
    with open(str(params_path), "w") as f:
        json.dump({"window": window, "stride": stride,
                   "feature_type": feature_type}, f, indent=2)

    _log_summary(X, y, label_map, out_npz, feature_type, log_fn)


def _log_summary(X, y, label_map, out_npz, feature_type, log_fn):
    log_fn(f"\n{'─'*50}")
    log_fn(f"X shape : {X.shape}")
    if feature_type == '1d_profile':
        log_fn(f"          └─ {X.shape[0]} windows  ×  {X.shape[1]} frames  ×  "
               f"{X.shape[2]} bins  ×  {X.shape[3]} channels  "
               f"(ch0-2: range_proj, ch3-5: doppler_proj)")
    elif feature_type == 'rdmap_ramap':
        log_fn(f"          └─ {X.shape[0]} windows  ×  {X.shape[1]} frames  ×  "
               f"{X.shape[2]} range bins  ×  {X.shape[3]} angle/Doppler bins  ×  {X.shape[4]} channels  "
               f"(ch0-2: RD mag rx0/1/2, ch3: range-azimuth, ch4: range-elevation)")
    else:
        log_fn(f"          └─ {X.shape[0]} windows  ×  {X.shape[1]} frames  ×  "
               f"{X.shape[2]} range bins  ×  {X.shape[3]} Doppler bins  ×  {X.shape[4]} channels  "
               f"(ch0-2: RD mag — rx0_mag, rx1_mag, rx2_mag)")
    log_fn(f"y shape : {y.shape}")
    log_fn(f"Size    : {X.nbytes / 1e6:.1f} MB uncompressed")
    log_fn(f"Saved   → {out_npz}")
    log_fn(f"\nWindows per class:")
    unique, counts = np.unique(y, return_counts=True)
    label_rev = {v: k for k, v in label_map.items()}
    for u, c in zip(unique, counts):
        bar = "█" * (c // 5)
        log_fn(f"  {label_rev[u]:22s}: {c:4d}  {bar}")


# ── GUI ───────────────────────────────────────────────────────────────────────

def launch_gui() -> None:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    class _TextWriter:
        """Redirect print() output into a tk.Text widget, thread-safely."""
        def __init__(self, widget: tk.Text):
            self._w = widget

        def write(self, s: str):
            self._w.after(0, self._append, s)

        def _append(self, s: str):
            self._w.configure(state="normal")
            self._w.insert("end", s)
            self._w.see("end")
            self._w.configure(state="disabled")

        def flush(self):
            pass

        def __call__(self, *args, sep=" ", end="\n"):
            self.write(sep.join(str(a) for a in args) + end)

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("ADL Feature Extraction")
            self.resizable(True, True)
            self.minsize(680, 620)

            self._activity_vars: dict[str, tk.BooleanVar] = {}
            self._activity_info: dict[str, str] = {}   # label → info string
            self._session_vars:  dict[str, tk.BooleanVar] = {}
            self._src_stats: dict[str, tuple] = {}     # label → (n_sessions, seg_frame_series)
            self._running = False

            self._build_ui()
            # Try to auto-scan the default folder on startup
            self.after(200, self._scan_folder)

        # ── UI construction ────────────────────────────────────────────────────

        def _build_ui(self):
            # ── Horizontal split: controls left, log right ─────────────────────
            body = ttk.Frame(self)
            body.pack(fill="both", expand=True)

            left = ttk.Frame(body)
            left.pack(side="left", fill="both", padx=(8, 4), pady=4)

            frm_log = ttk.LabelFrame(body, text="Output log", padding=8)
            frm_log.pack(side="right", fill="both", expand=True, padx=(4, 8), pady=4)
            frm_log.columnconfigure(0, weight=1)
            frm_log.rowconfigure(0, weight=1)

            self._log = tk.Text(frm_log, width=50, font=("Consolas", 9),
                                state="disabled", wrap="none",
                                bg="#1e1e1e", fg="#d4d4d4",
                                insertbackground="white")
            _hsb = ttk.Scrollbar(frm_log, orient="horizontal",
                                  command=self._log.xview)
            _vsb = ttk.Scrollbar(frm_log, orient="vertical",
                                  command=self._log.yview)
            self._log.configure(xscrollcommand=_hsb.set, yscrollcommand=_vsb.set)
            self._log.grid(row=0, column=0, sticky="nsew")
            _vsb.grid(row=0, column=1, sticky="ns")
            _hsb.grid(row=1, column=0, sticky="ew")

            self._writer = _TextWriter(self._log)

            # ── Folder frame ──────────────────────────────────────────────────
            frm_folder = ttk.LabelFrame(left, text="Folders", padding=8)
            frm_folder.pack(fill="x", pady=4)
            frm_folder.columnconfigure(1, weight=1)

            ttk.Label(frm_folder, text="Source (data/adl/…):").grid(
                row=0, column=0, sticky="w", pady=2)
            self._src_var = tk.StringVar(value=str(Path("data/adl/peter")))
            src_entry = ttk.Entry(frm_folder, textvariable=self._src_var)
            src_entry.grid(row=0, column=1, sticky="ew", padx=4)
            ttk.Button(frm_folder, text="Browse…",
                       command=self._browse_src).grid(row=0, column=2, padx=2)
            ttk.Button(frm_folder, text="Scan ↺",
                       command=self._scan_folder).grid(row=0, column=3, padx=2)

            ttk.Label(frm_folder, text="Output folder:").grid(
                row=1, column=0, sticky="w", pady=2)
            self._out_var = tk.StringVar(value=str(Path("data/adl_features/peter")))
            ttk.Entry(frm_folder, textvariable=self._out_var).grid(
                row=1, column=1, sticky="ew", padx=4)
            ttk.Button(frm_folder, text="Browse…",
                       command=self._browse_out).grid(row=1, column=2, padx=2)

            # Auto-fill output when source changes
            self._src_var.trace_add("write", self._auto_fill_output)

            # ── Notebook: Activities + Session Filter ──────────────────────────
            nb = ttk.Notebook(left)
            nb.pack(fill="both", expand=True, pady=4)

            # — Activities tab —
            tab_act = ttk.Frame(nb, padding=8)
            nb.add(tab_act, text="Activities")
            tab_act.columnconfigure(0, weight=1)
            tab_act.rowconfigure(1, weight=1)

            btn_bar = ttk.Frame(tab_act)
            btn_bar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
            ttk.Button(btn_bar, text="Select All",
                       command=self._select_all).pack(side="left", padx=2)
            ttk.Button(btn_bar, text="Deselect All",
                       command=self._deselect_all).pack(side="left", padx=2)
            self._scan_lbl = ttk.Label(btn_bar, text="(click Scan ↺ to load activities)",
                                       foreground="grey")
            self._scan_lbl.pack(side="right")

            outer = ttk.Frame(tab_act)
            outer.grid(row=1, column=0, sticky="nsew")
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(0, weight=1)

            self._canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
            vsb = ttk.Scrollbar(outer, orient="vertical",
                                 command=self._canvas.yview)
            self._canvas.configure(yscrollcommand=vsb.set)
            self._canvas.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")

            self._chk_frame = ttk.Frame(self._canvas)
            self._canvas_win = self._canvas.create_window(
                (0, 0), window=self._chk_frame, anchor="nw")
            self._chk_frame.bind("<Configure>", self._on_chk_frame_resize)
            self._canvas.bind("<Configure>", self._on_canvas_resize)
            self._canvas.bind_all("<MouseWheel>",
                lambda e: self._canvas.yview_scroll(-1*(e.delta//120), "units"))

            # — Session Filter tab —
            tab_ses = ttk.Frame(nb, padding=8)
            nb.add(tab_ses, text="Session Filter")
            tab_ses.columnconfigure(0, weight=1)
            tab_ses.rowconfigure(1, weight=1)

            ses_btn_bar = ttk.Frame(tab_ses)
            ses_btn_bar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
            ttk.Button(ses_btn_bar, text="Select All",
                       command=self._select_all_sessions).pack(side="left", padx=2)
            ttk.Button(ses_btn_bar, text="Deselect All",
                       command=self._deselect_all_sessions).pack(side="left", padx=2)
            self._ses_lbl = ttk.Label(ses_btn_bar,
                                      text="(scan folder to load sessions)",
                                      foreground="grey")
            self._ses_lbl.pack(side="right")

            ses_outer = ttk.Frame(tab_ses)
            ses_outer.grid(row=1, column=0, sticky="nsew")
            ses_outer.columnconfigure(0, weight=1)
            ses_outer.rowconfigure(0, weight=1)

            self._ses_canvas = tk.Canvas(ses_outer, borderwidth=0, highlightthickness=0)
            ses_vsb = ttk.Scrollbar(ses_outer, orient="vertical",
                                     command=self._ses_canvas.yview)
            self._ses_canvas.configure(yscrollcommand=ses_vsb.set)
            self._ses_canvas.grid(row=0, column=0, sticky="nsew")
            ses_vsb.grid(row=0, column=1, sticky="ns")

            self._ses_chk_frame = ttk.Frame(self._ses_canvas)
            self._ses_canvas_win = self._ses_canvas.create_window(
                (0, 0), window=self._ses_chk_frame, anchor="nw")
            self._ses_chk_frame.bind(
                "<Configure>",
                lambda _: self._ses_canvas.configure(
                    scrollregion=self._ses_canvas.bbox("all")))
            self._ses_canvas.bind(
                "<Configure>",
                lambda e: self._ses_canvas.itemconfig(
                    self._ses_canvas_win, width=e.width))

            # ── Data Summary frame ─────────────────────────────────────────────
            frm_summary = ttk.LabelFrame(left, text="Data Summary  (source folder)", padding=8)
            frm_summary.pack(fill="x", pady=4)
            frm_summary.columnconfigure(0, weight=1)

            sum_top = ttk.Frame(frm_summary)
            sum_top.grid(row=0, column=0, sticky="ew")
            ttk.Button(sum_top, text="⟳ Refresh",
                       command=self._refresh_summary).pack(side="left", padx=2)
            self._sum_lbl = ttk.Label(sum_top,
                                      text="(scan folder or run extraction to load)",
                                      foreground="grey")
            self._sum_lbl.pack(side="left", padx=8)

            cols = ("activity", "sessions", "segments", "windows", "balance")
            self._sum_tree = ttk.Treeview(frm_summary, columns=cols,
                                          show="headings", height=7,
                                          selectmode="none")
            self._sum_tree.heading("activity", text="Activity")
            self._sum_tree.heading("sessions", text="Sessions")
            self._sum_tree.heading("segments", text="Segments")
            self._sum_tree.heading("windows",  text="Est. Windows")
            self._sum_tree.heading("balance",  text="Balance")
            self._sum_tree.column("activity", width=160, anchor="w")
            self._sum_tree.column("sessions", width=65,  anchor="center")
            self._sum_tree.column("segments", width=65,  anchor="center")
            self._sum_tree.column("windows",  width=65,  anchor="center")
            self._sum_tree.column("balance",  width=220, anchor="w", stretch=True)
            self._sum_tree.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            self._sum_tree.tag_configure("low",   foreground="#e06c75")   # red
            self._sum_tree.tag_configure("total", font=("TkDefaultFont", 9, "bold"))

            # ── Parameters frame ───────────────────────────────────────────────
            frm_params = ttk.LabelFrame(left, text="Parameters", padding=8)
            frm_params.pack(fill="x", pady=4)

            ttk.Label(frm_params, text="Window (frames):").grid(
                row=0, column=0, sticky="w")
            self._win_var = tk.IntVar(value=LENGTH_WINDOW)
            ttk.Spinbox(frm_params, from_=4, to=128,
                        textvariable=self._win_var, width=6).grid(
                row=0, column=1, padx=4)

            ttk.Label(frm_params, text="Stride (frames):").grid(
                row=0, column=2, padx=12, sticky="w")
            self._stride_var = tk.IntVar(value=LENGTH_STRIDE)
            ttk.Spinbox(frm_params, from_=1, to=64,
                        textvariable=self._stride_var, width=6).grid(
                row=0, column=3, padx=4)

            # Model type selector
            ttk.Label(frm_params, text="Model:").grid(
                row=0, column=4, padx=(16, 4), sticky="w")
            self._model_var = tk.StringVar(value="2d")
            ttk.Radiobutton(frm_params, text="2D (RD-map)",
                            variable=self._model_var, value="2d").grid(
                row=0, column=5, padx=2)
            ttk.Radiobutton(frm_params, text="1D (projection)",
                            variable=self._model_var, value="1d").grid(
                row=0, column=6, padx=2)

            # Force re-extract checkbox + new 2D (RD+RA+RE) radio on same row
            self._force_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(frm_params, text="Force full re-extract",
                            variable=self._force_var).grid(
                row=1, column=4, columnspan=2, sticky="w", pady=(4, 0))
            ttk.Radiobutton(frm_params, text="2D (RD+RA+RE)",
                            variable=self._model_var, value="2d_ra").grid(
                row=1, column=6, sticky="w", padx=2, pady=(4, 0))

            # Auto-refresh summary when window/stride values change
            self._win_var.trace_add(   "write", lambda *_: self.after(0, self._refresh_summary))
            self._stride_var.trace_add("write", lambda *_: self.after(0, self._refresh_summary))

            self._run_btn = ttk.Button(frm_params, text="▶  Run Extraction",
                                       command=self._run)
            self._run_btn.grid(row=0, column=7, padx=20)


        # ── Canvas resize helpers ──────────────────────────────────────────────

        def _on_chk_frame_resize(self, _):
            self._canvas.configure(
                scrollregion=self._canvas.bbox("all"))

        def _on_canvas_resize(self, event):
            self._canvas.itemconfig(self._canvas_win, width=event.width)

        # ── Browse / auto-fill ─────────────────────────────────────────────────

        def _browse_src(self):
            chosen = filedialog.askdirectory(
                title="Select source folder (data/adl/user)",
                initialdir=self._src_var.get())
            if chosen:
                self._src_var.set(chosen)
                self._scan_folder()

        def _browse_out(self):
            chosen = filedialog.askdirectory(
                title="Select output folder",
                initialdir=self._out_var.get())
            if chosen:
                self._out_var.set(chosen)

        def _auto_fill_output(self, *_):
            src = Path(self._src_var.get())
            # Replace leading data/adl with data/adl_features
            parts = src.parts
            try:
                adl_idx = next(i for i, p in enumerate(parts) if p == "adl")
                new_parts = parts[:adl_idx] + ("adl_features",) + parts[adl_idx + 1:]
                self._out_var.set(str(Path(*new_parts)))
            except StopIteration:
                pass  # non-standard path — leave output as-is

        # ── Scan folder ────────────────────────────────────────────────────────

        def _scan_folder(self):
            src = Path(self._src_var.get())
            if not src.is_dir():
                self._scan_lbl.config(text=f"Folder not found: {src}", foreground="red")
                return

            self._scan_lbl.config(text="Scanning…", foreground="grey")
            self.update_idletasks()

            csv_files = sorted(src.glob("*.csv"))
            if not csv_files:
                self._scan_lbl.config(text="No CSV files found.", foreground="red")
                self._clear_checkboxes()
                return

            # Quick segment count per activity + collect unique sessions
            info: dict[str, str] = {}
            all_sessions: dict[str, int] = {}   # session → segment count
            self._src_stats.clear()
            for f in csv_files:
                try:
                    df = pd.read_csv(str(f), usecols=["source_file", "segment_idx",
                                                       "frame_number"])
                    seg_frames = df.groupby(["source_file", "segment_idx"]).size()
                    n_segs     = len(seg_frames)
                    n_sessions = int(df["source_file"].nunique())
                    n_frames   = len(df)
                    info[f.stem] = f"{n_segs} segments  ·  {n_frames} frames"
                    self._src_stats[f.stem] = (n_sessions, seg_frames)
                    for sess, cnt in df.groupby("source_file")["segment_idx"].nunique().items():
                        all_sessions[sess] = all_sessions.get(sess, 0) + cnt
                except Exception:
                    info[f.stem] = "(could not read)"

            self._activity_info = info
            self._rebuild_checkboxes()
            self._rebuild_session_checkboxes(all_sessions)
            self._scan_lbl.config(
                text=f"{len(csv_files)} activities found", foreground="green")
            self._refresh_summary()

        def _clear_checkboxes(self):
            for w in self._chk_frame.winfo_children():
                w.destroy()
            self._activity_vars.clear()

        def _rebuild_checkboxes(self):
            self._clear_checkboxes()
            for i, (label, info) in enumerate(self._activity_info.items()):
                var = tk.BooleanVar(value=True)
                self._activity_vars[label] = var
                row_frm = ttk.Frame(self._chk_frame)
                row_frm.grid(row=i, column=0, sticky="ew", pady=1)
                ttk.Checkbutton(row_frm, variable=var).pack(side="left")
                ttk.Label(row_frm, text=f"{label}",
                          width=22, anchor="w",
                          font=("TkDefaultFont", 9, "bold")).pack(side="left")
                ttk.Label(row_frm, text=info,
                          foreground="grey").pack(side="left", padx=6)

        def _select_all(self):
            for v in self._activity_vars.values():
                v.set(True)

        def _deselect_all(self):
            for v in self._activity_vars.values():
                v.set(False)

        def _rebuild_session_checkboxes(self, sessions: dict[str, int]):
            for w in self._ses_chk_frame.winfo_children():
                w.destroy()
            self._session_vars.clear()
            for i, (sess, n_segs) in enumerate(sorted(sessions.items())):
                var = tk.BooleanVar(value=True)
                self._session_vars[sess] = var
                row_frm = ttk.Frame(self._ses_chk_frame)
                row_frm.grid(row=i, column=0, sticky="ew", pady=1)
                ttk.Checkbutton(row_frm, variable=var).pack(side="left")
                ttk.Label(row_frm, text=sess, width=28, anchor="w",
                          font=("TkDefaultFont", 9, "bold")).pack(side="left")
                ttk.Label(row_frm, text=f"{n_segs} segments",
                          foreground="grey").pack(side="left", padx=6)
            n = len(sessions)
            self._ses_lbl.config(
                text=f"{n} sessions found", foreground="green" if n else "red")

        def _select_all_sessions(self):
            for v in self._session_vars.values():
                v.set(True)

        def _deselect_all_sessions(self):
            for v in self._session_vars.values():
                v.set(False)

        # ── Data Summary ───────────────────────────────────────────────────────

        def _refresh_summary(self):
            for row in self._sum_tree.get_children():
                self._sum_tree.delete(row)
            if not self._src_stats:
                self._sum_lbl.config(
                    text="(scan folder to load)", foreground="grey")
                return
            try:
                window = self._win_var.get()
                stride = self._stride_var.get()

                def est_wins(n_frames):
                    if n_frames < window:
                        return 1
                    return int((n_frames - window) // stride + 1)

                rows = []
                for label, (n_sess, seg_frames) in self._src_stats.items():
                    n_segs  = len(seg_frames)
                    n_wins  = int(seg_frames.apply(est_wins).sum())
                    rows.append((label, n_sess, n_segs, n_wins))

                rows.sort(key=lambda r: r[3])   # sort by est. windows ascending
                max_w      = max(r[3] for r in rows)
                bar_len    = 18
                low_thresh = max_w * 0.6

                for label, n_sess, n_segs, n_wins in rows:
                    bar = "█" * int(n_wins / max_w * bar_len)
                    tag = "low" if n_wins < low_thresh else ""
                    self._sum_tree.insert("", "end", values=(
                        label, n_sess, n_segs, n_wins, bar,
                    ), tags=(tag,))

                total_wins = sum(r[3] for r in rows)
                total_segs = sum(r[2] for r in rows)
                self._sum_tree.insert("", "end", values=(
                    "TOTAL", "—", total_segs, total_wins, "",
                ), tags=("total",))

                n_low = sum(1 for r in rows if r[3] < low_thresh)
                msg = (f"{len(rows)} activities · {total_wins} est. windows "
                       f"(window={window}, stride={stride})")
                if n_low:
                    msg += f"  ·  {n_low} class(es) below 60% — highlighted red"
                self._sum_lbl.config(text=msg, foreground="green")
            except Exception as e:
                self._sum_lbl.config(text=f"Error: {e}", foreground="red")

        # ── Run extraction ─────────────────────────────────────────────────────

        def _run(self):
            if self._running:
                return

            selected = [lbl for lbl, v in self._activity_vars.items() if v.get()]
            if not selected:
                messagebox.showwarning("No activities selected",
                                       "Please select at least one activity.")
                return

            # None means "all sessions"; a set means "only these"
            if self._session_vars:
                allowed = {s for s, v in self._session_vars.items() if v.get()}
                if not allowed:
                    messagebox.showwarning("No sessions selected",
                                           "Please select at least one session "
                                           "in the Session Filter tab.")
                    return
                # If every session is checked, pass None (no filtering needed)
                allowed_sessions = (None if len(allowed) == len(self._session_vars)
                                    else allowed)
            else:
                allowed_sessions = None

            src          = Path(self._src_var.get())
            outdir       = Path(self._out_var.get())
            window       = self._win_var.get()
            stride       = self._stride_var.get()
            _mv = self._model_var.get()
            feature_type = ("1d_profile" if _mv == "1d"
                            else "rdmap_ramap" if _mv == "2d_ra"
                            else "rdmap")
            force        = self._force_var.get()

            # Clear log
            self._log.configure(state="normal")
            self._log.delete("1.0", "end")
            self._log.configure(state="disabled")

            self._running = True
            self._run_btn.configure(state="disabled", text="Running…")

            def worker():
                try:
                    run(adl_dir=src, output_dir=outdir,
                        selected_labels=selected,
                        window=window, stride=stride,
                        allowed_sessions=allowed_sessions,
                        feature_type=feature_type,
                        force=force,
                        log_fn=self._writer)
                    self._writer("\n✓ Done.")
                except Exception as e:
                    import traceback
                    self._writer(f"\n✗ Error: {e}\n{traceback.format_exc()}")
                finally:
                    self.after(0, self._on_run_done)

            threading.Thread(target=worker, daemon=True).start()

        def _on_run_done(self):
            self._running = False
            self._run_btn.configure(state="normal", text="▶  Run Extraction")
            self._refresh_summary()

    app = App()
    app.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Extract ADL Range-Doppler features for ML training")
    p.add_argument("--nogui",  action="store_true",
                   help="Run in CLI mode (no GUI)")
    p.add_argument("--user",   default="peter",
                   help="User subfolder under data/adl/  (default: peter)")
    p.add_argument("--window", type=int, default=64,
                   help="Sliding window length in frames  (default: 64 = 12.8 s at 5 Hz)")
    p.add_argument("--stride", type=int, default=32,
                   help="Sliding window stride in frames  (default: 32 = 50%% overlap)")
    p.add_argument("--exclude-sessions", default="",
                   help="Comma-separated session names to exclude, e.g. "
                        "peter-20260323-085920,peter-20260324-093312")
    p.add_argument("--model", default="2d", choices=["2d", "2d_ra", "1d"],
                   help="Model type: '2d' (RD maps, 3ch), '2d_ra' (RD+RA+RE maps, 5ch), "
                        "or '1d' (range+Doppler projections, 6ch)")
    p.add_argument("--feature_type", default=None,
                   choices=["rdmap", "1d_profile"],
                   help="Alias for --model: 'rdmap'='2d', '1d_profile'='1d' (legacy)")
    p.add_argument("--activities", default="",
                   help="Comma-separated activity names to include, e.g. "
                        "'face_washing,hand_washing,tooth_brushing' (default: all)")
    p.add_argument("--tag", default="",
                   help="Optional extra tag appended after auto-name, e.g. 'bathroom' "
                        "→ adl_features_2d_w64_ch6_bathroom.npz (default: none)")
    args = p.parse_args()

    if args.nogui:
        adl_dir    = Path("data") / "adl"    / args.user
        output_dir = Path("data") / "adl_features" / args.user

        # Resolve model type: --feature_type overrides --model (legacy compat)
        if args.feature_type is not None:
            feature_type = args.feature_type
        elif args.model == "1d":
            feature_type = "1d_profile"
        elif args.model == "2d_ra":
            feature_type = "rdmap_ramap"
        else:
            feature_type = "rdmap"

        # Build selected_labels from --activities
        selected_labels = [a.strip() for a in args.activities.split(",") if a.strip()] or None

        # Build allowed_sessions from exclusion list
        excluded = {s.strip() for s in args.exclude_sessions.split(",") if s.strip()}
        if excluded:
            # Read all unique sessions first, then subtract excluded
            all_sess = set()
            for csv_path in adl_dir.glob("*.csv"):
                df = pd.read_csv(str(csv_path), usecols=["source_file"])
                all_sess.update(df["source_file"].unique())
            allowed_sessions = all_sess - excluded
            print(f"Excluding {len(excluded)} session(s): {sorted(excluded)}")
            print(f"Keeping {len(allowed_sessions)} session(s).")
        else:
            allowed_sessions = None

        run(adl_dir=adl_dir, output_dir=output_dir,
            selected_labels=selected_labels,
            window=args.window, stride=args.stride,
            allowed_sessions=allowed_sessions,
            feature_type=feature_type,
            tag=args.tag)
    else:
        launch_gui()
