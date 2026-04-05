#!/usr/bin/env python3
"""
ADL Feature Extraction Pipeline
================================
Converts raw chirp data stored in ADL CSVs into Range-Doppler map tensors
ready for ML model training.

Processing chain per frame:
  raw chirp (32 chirps × 64 samples × 3 channels)
    → DopplerAlgo  (range FFT + MTI + Doppler FFT)
    → complex Range-Doppler map  (64 range bins × 64 Doppler bins)
    → log1p magnitude            (compresses dynamic range)
    → (64, 64, 4) float32 array
      channels: rx0_mag, rx1_mag, rx2_mag, doppler_proj

Processing chain per segment:
  N frames of (64, 64, 4)
    → sliding window (default: 16 frames, stride 8)
    → temporal_max channel appended per window
    → each window = one training sample of shape (16, 64, 64, 5)
    → short segments (< 16 frames, e.g. falling) → zero-padded to 16

Outputs saved to  data/adl_features/{user}/ :
  adl_features.npz  — compressed arrays:
                        X: (N_windows, window, 64, 64, 4)  float32
                        y: (N_windows,)                     int32
  label_map.json    — { "activity_name": class_index, ... }
  meta.csv          — traceability: label / source_file / segment_idx / window_idx

Usage (GUI — default):
  python adl_feature_extraction.py

Usage (CLI):
  python adl_feature_extraction.py --nogui --user peter --window 16 --stride 8
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


# ── Chirp parsing ─────────────────────────────────────────────────────────────

def _parse_chirp_cell(cell) -> np.ndarray:
    arr = np.asarray(ast.literal_eval(cell), dtype=np.float32)
    return arr.reshape(NUM_CHIRPS, NUM_SAMPLES)


# ── Range-Doppler computation ─────────────────────────────────────────────────

def _segment_to_rdmaps(frames_data: list[tuple]) -> np.ndarray:
    """
    Compute Range-Doppler feature maps for every frame in a segment.
    A fresh DopplerAlgo is created per segment so the MTI history does not
    bleed between unrelated segments.

    Returns ndarray shape (N_frames, 64, 64, 4).
    Channel layout:
      0-2  rx0_mag, rx1_mag, rx2_mag  — log-magnitude RD maps
      3    doppler_projection          — Doppler energy profile (tiled)
    """
    algo = DopplerAlgo(NUM_SAMPLES, NUM_CHIRPS, NUM_CHANNELS)
    rd_maps = []
    for ch1, ch2, ch3 in frames_data:
        rd0 = algo.compute_doppler_map(ch1, 0)   # (64, 64) complex
        rd1 = algo.compute_doppler_map(ch2, 1)   # (64, 64) complex
        rd2 = algo.compute_doppler_map(ch3, 2)   # (64, 64) complex

        mag0 = np.log1p(np.abs(rd0)).astype(np.float32)
        mag1 = np.log1p(np.abs(rd1)).astype(np.float32)
        mag2 = np.log1p(np.abs(rd2)).astype(np.float32)
        mag3 = np.stack([mag0, mag1, mag2], axis=-1)      # (64, 64, 3)

        # Channel 3: Doppler projection — sum over range bins, average across
        # receive channels → (64,) Doppler energy profile, tiled to (64, 64).
        doppler_proj = mag3.sum(axis=0).mean(axis=-1)     # (64,)
        doppler_ch   = np.tile(doppler_proj[None, :], (64, 1))  # (64, 64)

        frame_4ch = np.concatenate(
            [mag3, doppler_ch[:, :, None]],
            axis=-1
        )                                                 # (64, 64, 4)
        rd_maps.append(frame_4ch)
    return np.array(rd_maps, dtype=np.float32)           # (N, 64, 64, 4)


# ── Sliding window ────────────────────────────────────────────────────────────

def _sliding_windows(rd_maps: np.ndarray, window: int, stride: int) -> list[np.ndarray]:
    """
    Slide a fixed-length window across the frame axis.
    Short segments are zero-padded; a non-redundant tail window is added
    when the last full window does not reach the end.

    A 5th channel (temporal_max) is appended to every window:
      max over the time axis of magnitude channels (0-2), averaged across
      receive channels → (H, W), tiled to (T, H, W, 1).
      Near-zero everywhere for no_motion (empty room); shows a clear peak
      at the person's range bin for standing/other activities.
    """
    N, H, W, C = rd_maps.shape
    wins = []
    if N < window:
        pad = np.zeros((window - N, H, W, C), dtype=np.float32)
        wins.append(np.concatenate([rd_maps, pad], axis=0))
    else:
        start = 0
        while start + window <= N:
            wins.append(rd_maps[start : start + window])
            start += stride
        tail_start = N - window
        prev_start = start - stride
        if tail_start > prev_start:
            wins.append(rd_maps[tail_start:])

    # Append temporal max channel to each window
    result = []
    for w in wins:
        # max over time of magnitude channels (0-2) → (H, W, 1)
        temporal_max = w[:, :, :, :3].max(axis=0).mean(axis=-1, keepdims=True)
        # tile across time → (T, H, W, 1) and concatenate
        max_ch = np.tile(temporal_max[np.newaxis], (w.shape[0], 1, 1, 1))
        result.append(np.concatenate([w, max_ch], axis=-1).astype(np.float32))
    return result


# ── Core extraction ───────────────────────────────────────────────────────────

def run(adl_dir: Path, output_dir: Path,
        selected_labels: list[str] | None,
        window: int, stride: int,
        allowed_sessions: set[str] | None = None,
        log_fn=print) -> None:
    """
    Extract features for the given selected_labels (or all CSVs if None).

    allowed_sessions: if provided, only segments whose source_file is in this
                      set are included.  Pass None to include all sessions.
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

    labels    = [f.stem for f in csv_files]
    label_map = {name: idx for idx, name in enumerate(labels)}

    log_fn(f"Activities selected: {labels}")
    log_fn(f"Window={window} frames, stride={stride} frames  "
           f"(≈ {window/5:.1f}s at 5 Hz)\n")

    all_X:    list[np.ndarray] = []
    all_y:    list[int]        = []
    all_meta: list[tuple]      = []

    for csv_path in csv_files:
        label = csv_path.stem
        y_val = label_map[label]
        df    = pd.read_csv(str(csv_path))

        n_segs    = df.groupby(["source_file", "segment_idx"]).ngroups
        n_wins    = 0
        n_skipped = 0

        for (src, seg_idx), grp in df.groupby(["source_file", "segment_idx"]):
            if allowed_sessions is not None and src not in allowed_sessions:
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

            rd_maps = _segment_to_rdmaps(frames_data)
            wins    = _sliding_windows(rd_maps, window, stride)

            for w_idx, w in enumerate(wins):
                all_X.append(w)
                all_y.append(y_val)
                all_meta.append((label, src, int(seg_idx), w_idx))

            n_wins += len(wins)

        skip_note = f"  ({n_skipped} frames skipped)" if n_skipped else ""
        log_fn(f"  {label:22s}: {n_segs:3d} segments → {n_wins:4d} windows{skip_note}")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y,  dtype=np.int32)

    out_npz = output_dir / "adl_features.npz"
    np.savez_compressed(str(out_npz), X=X, y=y)

    with open(str(output_dir / "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    meta_df = pd.DataFrame(all_meta,
                           columns=["label", "source_file", "segment_idx", "window_idx"])
    meta_df.to_csv(str(output_dir / "meta.csv"), index=False)

    log_fn(f"\n{'─'*50}")
    log_fn(f"X shape : {X.shape}")
    log_fn(f"          └─ {X.shape[0]} windows  ×  {X.shape[1]} frames  ×  "
           f"{X.shape[2]} range bins  ×  {X.shape[3]} Doppler bins  ×  {X.shape[4]} channels")
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
            self._win_var = tk.IntVar(value=32)
            ttk.Spinbox(frm_params, from_=4, to=128,
                        textvariable=self._win_var, width=6).grid(
                row=0, column=1, padx=4)

            ttk.Label(frm_params, text="Stride (frames):").grid(
                row=0, column=2, padx=12, sticky="w")
            self._stride_var = tk.IntVar(value=8)
            ttk.Spinbox(frm_params, from_=1, to=64,
                        textvariable=self._stride_var, width=6).grid(
                row=0, column=3, padx=4)

            # Auto-refresh summary when window/stride values change
            self._win_var.trace_add(   "write", lambda *_: self.after(0, self._refresh_summary))
            self._stride_var.trace_add("write", lambda *_: self.after(0, self._refresh_summary))

            self._run_btn = ttk.Button(frm_params, text="▶  Run Extraction",
                                       command=self._run)
            self._run_btn.grid(row=0, column=4, padx=20)


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

            src    = Path(self._src_var.get())
            outdir = Path(self._out_var.get())
            window = self._win_var.get()
            stride = self._stride_var.get()

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
    p.add_argument("--window", type=int, default=32,
                   help="Sliding window length in frames  (default: 32 = 6.4 s at 5 Hz)")
    p.add_argument("--stride", type=int, default=8,
                   help="Sliding window stride in frames  (default: 8 = 50%% overlap)")
    p.add_argument("--exclude-sessions", default="",
                   help="Comma-separated session names to exclude, e.g. "
                        "peter-20260323-085920,peter-20260324-093312")
    args = p.parse_args()

    if args.nogui:
        adl_dir    = Path("data") / "adl"    / args.user
        output_dir = Path("data") / "adl_features" / args.user

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
            selected_labels=None,
            window=args.window, stride=args.stride,
            allowed_sessions=allowed_sessions)
    else:
        launch_gui()
