"""
ADL Classifier — Training Script
==================================
Trains the CNN+GRU model defined in adl_model.py on features produced by
adl_feature_extraction.py.

Key design decisions:
  - Train/val split is done by SOURCE SESSION (source_file), not by individual
    windows.  This prevents data leakage where adjacent frames from the same
    recording appear in both splits.
  - Class weights compensate for any imbalance between activity counts.
  - Data augmentation during training: Gaussian noise, random time shift,
    random channel dropout.
  - Best model (by val accuracy) is saved as  data/adl_features/{user}/best_model.pt

Usage (GUI — default):
  python adl_train.py

Usage (CLI):
  python adl_train.py --nogui
  python adl_train.py --nogui --user peter --epochs 60 --batch_size 32 --lr 1e-3
"""

import json
import argparse
import random
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from adl_model import ADLClassifier


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ADLDataset(Dataset):
    """
    Wraps the (X, y) arrays produced by adl_feature_extraction.py.
    Applies augmentation on-the-fly during training.

    Augmentations (training only):
      - Gaussian noise:      adds small random noise to every pixel
      - Random time shift:   rolls the frame sequence by ±shift_frames
      - Channel dropout:     zeros out one random channel with probability p
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        noise_std: float = 0.05,
        max_shift: int = 4,
        channel_drop_prob: float = 0.2,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.channel_drop_prob = channel_drop_prob

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()   # (T, H, W, C)

        if self.augment:
            # 1. Gaussian noise
            x += np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
            x = np.clip(x, 0.0, None)

            # 2. Random time shift (roll frames, wraps around)
            if self.max_shift > 0:
                shift = random.randint(-self.max_shift, self.max_shift)
                x = np.roll(x, shift, axis=0)

            # 3. Random channel dropout
            if random.random() < self.channel_drop_prob:
                ch = random.randint(0, x.shape[-1] - 1)
                x[:, :, :, ch] = 0.0

            # 4. Random magnitude scaling — simulates distance/orientation variation
            scale = np.random.uniform(0.7, 1.3)
            x = x * scale

        return torch.from_numpy(x), self.y[idx]


# ── Session-level train/val split ─────────────────────────────────────────────

def session_split(
    meta: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Session-level train/val split with no leakage.

    All windows from a recording session go to exactly one split — train or
    val — regardless of which activity class they belong to.  This prevents
    the same session appearing in both splits (which happened previously when
    the split was done independently per class).

    A deterministic RNG seeded from `seed` shuffles the global session list
    before the ratio cut, so results are reproducible.
    """
    all_sessions = np.array(sorted(meta["source_file"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_sessions)
    n_val = max(1, int(len(all_sessions) * val_ratio))
    val_sessions   = set(all_sessions[:n_val])
    train_sessions = set(all_sessions[n_val:])

    train_idx = meta.index[meta["source_file"].isin(train_sessions)].to_numpy()
    val_idx   = meta.index[meta["source_file"].isin(val_sessions)].to_numpy()

    return train_idx, val_idx


# ── Training helpers ──────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total   = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(dim=1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


# ── Main training function ────────────────────────────────────────────────────

def train(
    args: argparse.Namespace,
    log_fn=print,
    epoch_callback=None,    # (epoch, tr_loss, tr_acc, vl_loss, vl_acc, lr,
                            #  best_val_acc, best_epoch, total_epochs) → None
    results_callback=None,  # (cm, label_names, report_dict, mismatches_df) → None
    stop_event: threading.Event | None = None,
) -> None:
    set_seed(args.seed)

    feature_dir = Path("data") / "adl_features" / args.user
    npz_path    = feature_dir / "adl_features.npz"
    meta_path   = feature_dir / "meta.csv"
    label_path  = feature_dir / "label_map.json"

    if not npz_path.exists():
        log_fn(f"Feature file not found: {npz_path}")
        log_fn("Run adl_feature_extraction.py first.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    data        = np.load(str(npz_path))
    X, y        = data["X"], data["y"]
    meta        = pd.read_csv(str(meta_path))
    label_map   = json.load(open(str(label_path)))
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    num_classes = len(label_names)

    if getattr(args, "channels", 0) > 0:
        X = X[:, :, :, :, :args.channels]

    in_channels = X.shape[-1]
    log_fn(f"Loaded  X: {X.shape}  y: {y.shape}")
    ch_desc = {3: "mag only", 4: "mag + doppler_proj",
               5: "mag + doppler_proj + temporal_max",
               6: "mag + doppler_proj + temporal_max + IPD"}.get(in_channels, f"{in_channels} channels")
    log_fn(f"Channels: {in_channels}  ({ch_desc})")
    log_fn(f"Window:   {X.shape[1]} frames  ({X.shape[1]/5:.1f}s at 5 Hz)")
    log_fn(f"Classes : {label_names}\n")

    # ── Optional class filter ─────────────────────────────────────────────────
    selected = getattr(args, 'selected_classes', None)
    if selected:
        keep_ids  = sorted([label_map[c] for c in selected if c in label_map])
        unknown   = [c for c in selected if c not in label_map]
        if unknown:
            log_fn(f"WARNING: unknown classes ignored: {unknown}")
        mask      = np.isin(y, keep_ids)
        X, y      = X[mask], y[mask]
        meta      = meta[mask].reset_index(drop=True)
        # Remap labels to 0..N-1 contiguously
        id_remap  = {old: new for new, old in enumerate(keep_ids)}
        y         = np.array([id_remap[v] for v in y], dtype=np.int32)
        label_names = [label_names[i] for i in keep_ids]
        num_classes = len(label_names)
        log_fn(f"Filtered to {num_classes} classes: {label_names}")
        log_fn(f"Remaining windows: {len(X)}\n")

    # ── Session-level split ───────────────────────────────────────────────────
    train_idx, val_idx = session_split(meta, val_ratio=args.val_ratio,
                                       seed=args.seed)

    log_fn(f"Train sessions: {len(meta.loc[train_idx, 'source_file'].unique()):3d}  "
           f"({len(train_idx)} windows)")
    log_fn(f"Val   sessions: {len(meta.loc[val_idx,   'source_file'].unique()):3d}  "
           f"({len(val_idx)} windows)\n")

    # ── Normalization ─────────────────────────────────────────────────────────
    # Two-group normalization handles the different value ranges:
    #   Magnitude channels (0-2): log1p of small floats → max-scale to [0, 1]
    #   Angle channels (3-4):     already in [-1, 1]   → mean/std normalize
    # A single global std would be dominated by angle channel statistics,
    # making magnitude channels vanish; per-element std amplifies noise.
    X_train_raw = X[train_idx].astype(np.float32)

    if in_channels > 3:
        mag_max  = float(X_train_raw[:, :, :, :, :3].max())
        mag_max  = max(mag_max, 1e-6)       # guard against all-zero data

        # Normalize each extra channel independently so doppler_proj (large scale,
        # always positive) does not dominate the z-score of the angle maps (small
        # scale, signed).  Each channel c gets its own mean/std computed on train.
        extra_means = []
        extra_stds  = []
        for c in range(3, in_channels):
            ch_data = X_train_raw[:, :, :, :, c]
            m = float(ch_data.mean())
            s = float(ch_data.std()) + 1e-8
            extra_means.append(m)
            extra_stds.append(s)
            log_fn(f"  Extra ch{c}:          mean={m:+.4f}  std={s:.4f}")

        log_fn(f"  Magnitude (ch0-2):   max={mag_max:.6f}  → scaled to [0, 1]")

        X_f    = X.astype(np.float32)
        X_norm = X_f.copy()
        X_norm[:, :, :, :, :3] = X_f[:, :, :, :, :3] / mag_max
        for i, c in enumerate(range(3, in_channels)):
            X_norm[:, :, :, :, c] = (X_f[:, :, :, :, c] - extra_means[i]) / extra_stds[i]

        norm_mean = [0.0] + extra_means   # [mag_offset, ch3_mean, ch4_mean, ...]
        norm_std  = [mag_max] + extra_stds  # [mag_scale,  ch3_std,  ch4_std,  ...]
    else:
        norm_mean_v = float(X_train_raw.mean())
        norm_std_v  = float(X_train_raw.std()) + 1e-8
        log_fn(f"  Global: mean={norm_mean_v:+.4f}  std={norm_std_v:.4f}  "
               f"min={X_train_raw.min():.4f}  max={X_train_raw.max():.4f}")
        X_norm = (X.astype(np.float32) - norm_mean_v) / norm_std_v
        norm_mean = norm_mean_v
        norm_std  = norm_std_v

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_ds = ADLDataset(X_norm[train_idx], y[train_idx], augment=True)
    val_ds   = ADLDataset(X_norm[val_idx],   y[val_idx],   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Model, loss, optimiser ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_fn(f"Device: {device}\n")

    model = ADLClassifier(
        num_classes=num_classes,
        cnn_out_dim=64,
        gru_hidden=64,
        gru_layers=2,
        dropout=getattr(args, 'dropout', 0.4),
        bidirectional=True,
        in_channels=in_channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_fn(f"Model parameters: {n_params:,}\n")

    class_weights = compute_class_weights(y[train_idx], num_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer     = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=getattr(args, 'weight_decay', 5e-4))
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc  = 0.0
    best_epoch    = 0
    patience_left = args.early_stop
    model_path    = feature_dir / "best_model.pt"

    log_fn(f"{'Epoch':>6}  {'Train loss':>10}  {'Train acc':>9}  "
           f"{'Val loss':>8}  {'Val acc':>7}  {'LR':>8}")
    log_fn("─" * 60)

    for epoch in range(1, args.epochs + 1):
        if stop_event is not None and stop_event.is_set():
            log_fn(f"\nStopped by user at epoch {epoch}.")
            break

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion,
                                    None,      device)

        scheduler.step(vl_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if vl_acc > best_val_acc:
            best_val_acc  = vl_acc
            best_epoch    = epoch
            patience_left = args.early_stop
            torch.save({
                "epoch":         epoch,
                "model_state":   model.state_dict(),
                "label_map":     label_map,
                "val_acc":       best_val_acc,
                "norm_mean":     norm_mean,
                "norm_std":      norm_std,
                "in_channels":   in_channels,
                "window_frames": X.shape[1],
            }, str(model_path))
            marker = "  ← best"
        else:
            patience_left -= 1

        log_fn(f"{epoch:6d}  {tr_loss:10.4f}  {tr_acc:8.1%}  "
               f"{vl_loss:8.4f}  {vl_acc:6.1%}  {current_lr:8.1e}{marker}")

        if epoch_callback is not None:
            epoch_callback(epoch, tr_loss, tr_acc, vl_loss, vl_acc,
                           current_lr, best_val_acc, best_epoch, args.epochs)

        if patience_left <= 0:
            log_fn(f"\nEarly stopping at epoch {epoch} "
                   f"(no improvement for {args.early_stop} epochs)")
            break

    log_fn(f"\nBest val accuracy: {best_val_acc:.1%}  at epoch {best_epoch}")
    log_fn(f"Model saved → {model_path}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    checkpoint = torch.load(str(model_path), map_location=device,
                            weights_only=False)
    ckpt_T = checkpoint.get("window_frames")
    if ckpt_T is not None and ckpt_T != X.shape[1]:
        log_fn(f"WARNING: checkpoint was trained with window_frames={ckpt_T} "
               f"but current data has T={X.shape[1]}. "
               f"Re-extract features with --window {ckpt_T} or retrain.")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    log_fn(f"\n{'─'*60}")
    log_fn("Per-class results (validation set):\n")
    log_fn(classification_report(all_true, all_preds,
                                  target_names=label_names, digits=3))

    cm = confusion_matrix(all_true, all_preds)
    col_w = max(len(n) for n in label_names) + 2
    log_fn("Confusion matrix (rows=true, cols=predicted):")
    log_fn(" " * col_w + "".join(f"{n:>{col_w}}" for n in label_names))
    for i, row_name in enumerate(label_names):
        log_fn(f"{row_name:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in cm[i]))

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    fig_cm, ax_cm = plt.subplots(figsize=(max(8, num_classes),
                                          max(6, num_classes)))
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax_cm)
    ax_cm.set_xticks(range(num_classes))
    ax_cm.set_yticks(range(num_classes))
    ax_cm.set_xticklabels(label_names, rotation=45, ha="right", fontsize=9)
    ax_cm.set_yticklabels(label_names, fontsize=9)
    ax_cm.set_xlabel("Predicted", fontsize=11)
    ax_cm.set_ylabel("True", fontsize=11)
    ax_cm.set_title(
        f"Confusion Matrix — val acc {best_val_acc:.1%} (epoch {best_epoch})",
        fontsize=12)
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=9,
                       color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    cm_path = feature_dir / "confusion_matrix.png"
    fig_cm.savefig(str(cm_path), dpi=150)
    plt.close(fig_cm)
    log_fn(f"\nConfusion matrix saved → {cm_path}")

    # ── Mismatch report ───────────────────────────────────────────────────────
    val_meta   = meta.loc[val_idx].reset_index(drop=True)
    mismatches = []
    for i, (pred, true) in enumerate(zip(all_preds, all_true)):
        if pred != true:
            row = val_meta.iloc[i]
            mismatches.append({
                "source_file": row["source_file"],
                "segment_idx": int(row["segment_idx"]),
                "window_idx":  int(row["window_idx"]),
                "true_label":  label_names[true],
                "predicted":   label_names[pred],
            })

    log_fn(f"\n{'─'*60}")
    log_fn(f"MISCLASSIFIED SAMPLES — {len(mismatches)} of {len(val_idx)} val windows\n")
    if mismatches:
        header_cols = ["source_file", "segment_idx", "window_idx",
                       "true_label", "predicted"]
        log_fn("\t".join(header_cols))
        for m in mismatches:
            log_fn("\t".join(str(m[c]) for c in header_cols))

    mismatch_df  = pd.DataFrame(mismatches)
    mismatch_path = feature_dir / "mismatches.tsv"
    mismatch_df.to_csv(str(mismatch_path), sep="\t", index=False)
    log_fn(f"\nMismatch table saved → {mismatch_path}")

    # ── Surface results to GUI if requested ──────────────────────────────────
    if results_callback is not None:
        report_dict = classification_report(
            all_true, all_preds,
            target_names=label_names, digits=3, output_dict=True)
        results_callback(cm, label_names, report_dict, mismatch_df,
                         best_val_acc, best_epoch)


# ── GUI ───────────────────────────────────────────────────────────────────────

def launch_gui() -> None:
    import tkinter as tk
    from tkinter import ttk, filedialog
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    class _TextWriter:
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
            self.title("ADL Classifier Training")
            self.resizable(True, True)
            self.minsize(860, 700)

            self._running    = False
            self._stop_event = threading.Event()

            # Accumulated epoch data for the live plot
            self._epochs:     list[int]   = []
            self._tr_accs:    list[float] = []
            self._vl_accs:    list[float] = []
            self._best_epoch: int         = 0
            self._best_acc:   float       = 0.0

            self._build_ui()

        # ── UI construction ────────────────────────────────────────────────────

        def _build_ui(self):
            pad = {"padx": 8, "pady": 4}

            # ── Config frame ──────────────────────────────────────────────────
            frm_cfg = ttk.LabelFrame(self, text="Configuration", padding=8)
            frm_cfg.pack(fill="x", **pad)
            frm_cfg.columnconfigure(1, weight=1)

            # Features folder
            ttk.Label(frm_cfg, text="Features folder:").grid(
                row=0, column=0, sticky="w", pady=2)
            self._feat_var = tk.StringVar(
                value=str(Path("data/adl_features/peter")))
            ttk.Entry(frm_cfg, textvariable=self._feat_var).grid(
                row=0, column=1, columnspan=5, sticky="ew", padx=4)
            ttk.Button(frm_cfg, text="Browse…",
                       command=self._browse_feat).grid(row=0, column=6, padx=2)

            # Hyperparameters row 1
            def _lbl_spin(parent, row, col, text, var, lo, hi, w=7):
                ttk.Label(parent, text=text).grid(
                    row=row, column=col, sticky="w", padx=(8 if col else 0, 0))
                ttk.Spinbox(parent, from_=lo, to=hi,
                            textvariable=var, width=w).grid(
                    row=row, column=col + 1, padx=4)

            self._epochs_var   = tk.IntVar(value=80)
            self._batch_var    = tk.IntVar(value=32)
            self._lr_var       = tk.DoubleVar(value=1e-3)
            self._val_var      = tk.DoubleVar(value=0.2)
            self._estop_var    = tk.IntVar(value=15)
            self._seed_var     = tk.IntVar(value=42)
            self._dropout_var  = tk.DoubleVar(value=0.4)
            self._wd_var       = tk.DoubleVar(value=5e-4)

            _lbl_spin(frm_cfg, 1, 0, "Epochs:",      self._epochs_var, 1,  500)
            _lbl_spin(frm_cfg, 1, 2, "Batch size:",  self._batch_var,  4,  256)
            ttk.Label(frm_cfg, text="LR:").grid(row=1, column=4, sticky="w", padx=(8, 0))
            ttk.Entry(frm_cfg, textvariable=self._lr_var, width=8).grid(
                row=1, column=5, padx=4)

            _lbl_spin(frm_cfg, 2, 0, "Val ratio:",   self._val_var,    0.05, 0.5,  w=5)
            _lbl_spin(frm_cfg, 2, 2, "Early stop:",  self._estop_var,  1,    100)
            _lbl_spin(frm_cfg, 2, 4, "Seed:",        self._seed_var,   0,    9999)

            ttk.Label(frm_cfg, text="Dropout:").grid(row=3, column=0, sticky="w", padx=(8, 0))
            ttk.Entry(frm_cfg, textvariable=self._dropout_var, width=8).grid(
                row=3, column=1, padx=4)
            ttk.Label(frm_cfg, text="(0.0–0.7 | overfitting → increase)",
                      foreground="#888").grid(row=3, column=2, columnspan=2, sticky="w")
            ttk.Label(frm_cfg, text="Weight decay:").grid(row=3, column=4, sticky="w", padx=(8, 0))
            ttk.Entry(frm_cfg, textvariable=self._wd_var, width=8).grid(
                row=3, column=5, padx=4)
            ttk.Label(frm_cfg, text="(e.g. 5e-4 | overfitting → increase)",
                      foreground="#888").grid(row=3, column=6, sticky="w")

            # Buttons
            btn_row = ttk.Frame(frm_cfg)
            btn_row.grid(row=4, column=0, columnspan=7, sticky="e", pady=(6, 0))
            self._run_btn  = ttk.Button(btn_row, text="▶  Start Training",
                                        command=self._start)
            self._stop_btn = ttk.Button(btn_row, text="■  Stop",
                                        command=self._stop, state="disabled")
            self._overview_btn = ttk.Button(btn_row, text="🔍  Model Overview",
                                            command=self._show_model_overview)
            self._data_btn = ttk.Button(btn_row, text="📊  Data Overview",
                                           command=self._show_data_overview)
            self._run_btn.pack(side="left", padx=4)
            self._stop_btn.pack(side="left", padx=4)
            self._overview_btn.pack(side="left", padx=4)
            self._data_btn.pack(side="left", padx=4)

            # ── Progress bar ──────────────────────────────────────────────────
            frm_prog = ttk.Frame(self)
            frm_prog.pack(fill="x", padx=8, pady=(0, 2))
            self._prog_var = tk.DoubleVar(value=0)
            self._prog     = ttk.Progressbar(frm_prog, variable=self._prog_var,
                                             maximum=100)
            self._prog.pack(fill="x", side="left", expand=True, padx=(0, 8))
            self._status_lbl = ttk.Label(frm_prog, text="Ready", width=36,
                                         anchor="w")
            self._status_lbl.pack(side="left")

            # ── Live training chart ───────────────────────────────────────────
            frm_chart = ttk.LabelFrame(self, text="Training curve", padding=4)
            frm_chart.pack(fill="x", padx=8, pady=2)

            self._fig = Figure(figsize=(8, 2.2), dpi=100)
            self._ax  = self._fig.add_subplot(111)
            self._ax.set_xlabel("Epoch")
            self._ax.set_ylabel("Accuracy")
            self._ax.set_ylim(0, 1)
            self._ax.grid(True, alpha=0.3)
            self._fig.tight_layout(pad=1.2)

            self._chart_canvas = FigureCanvasTkAgg(self._fig, master=frm_chart)
            self._chart_canvas.get_tk_widget().pack(fill="both", expand=True)

            # ── Results notebook ──────────────────────────────────────────────
            frm_nb = ttk.LabelFrame(self, text="Results", padding=4)
            frm_nb.pack(fill="both", expand=True, padx=8, pady=4)
            frm_nb.columnconfigure(0, weight=1)
            frm_nb.rowconfigure(0, weight=1)

            self._nb = ttk.Notebook(frm_nb)
            self._nb.grid(row=0, column=0, sticky="nsew")

            self._build_log_tab()
            self._build_results_tab()
            self._build_cm_tab()
            self._build_mismatch_tab()

        # ── Notebook tabs ──────────────────────────────────────────────────────

        def _build_log_tab(self):
            frm = ttk.Frame(self._nb)
            self._nb.add(frm, text="  Log  ")
            frm.columnconfigure(0, weight=1)
            frm.rowconfigure(0, weight=1)

            self._log = tk.Text(frm, font=("Consolas", 9), state="disabled",
                                wrap="none", bg="#1e1e1e", fg="#d4d4d4")
            hsb = ttk.Scrollbar(frm, orient="horizontal", command=self._log.xview)
            vsb = ttk.Scrollbar(frm, orient="vertical",   command=self._log.yview)
            self._log.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
            self._log.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")

            self._writer = _TextWriter(self._log)

        def _build_results_tab(self):
            frm = ttk.Frame(self._nb)
            self._nb.add(frm, text="  Per-class Results  ")
            frm.columnconfigure(0, weight=1)
            frm.rowconfigure(0, weight=1)

            cols = ("Activity", "Precision", "Recall", "F1-score", "Support")
            self._res_tree = ttk.Treeview(frm, columns=cols,
                                          show="headings", height=12)
            for c in cols:
                w = 80 if c != "Activity" else 180
                self._res_tree.heading(c, text=c)
                self._res_tree.column(c, width=w, anchor="center")
            self._res_tree.column("Activity", anchor="w")

            vsb = ttk.Scrollbar(frm, orient="vertical",
                                 command=self._res_tree.yview)
            self._res_tree.configure(yscrollcommand=vsb.set)
            self._res_tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")

            # Summary label at bottom
            self._res_summary = ttk.Label(frm, text="", foreground="grey")
            self._res_summary.grid(row=1, column=0, columnspan=2,
                                   sticky="w", pady=2, padx=4)

        def _build_cm_tab(self):
            frm = ttk.Frame(self._nb)
            self._nb.add(frm, text="  Confusion Matrix  ")
            frm.columnconfigure(0, weight=1)
            frm.rowconfigure(0, weight=1)

            self._cm_fig    = Figure(dpi=100)
            self._cm_canvas = FigureCanvasTkAgg(self._cm_fig, master=frm)
            self._cm_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            self._cm_placeholder = ttk.Label(frm,
                text="Confusion matrix will appear here after training.",
                foreground="grey")
            self._cm_placeholder.grid(row=0, column=0)

        def _build_mismatch_tab(self):
            frm = ttk.Frame(self._nb)
            self._nb.add(frm, text="  Mismatches  ")
            frm.columnconfigure(0, weight=1)
            frm.rowconfigure(0, weight=1)

            cols = ("source_file", "seg", "win", "true_label", "predicted")
            self._mis_tree = ttk.Treeview(frm, columns=cols,
                                          show="headings", height=12)
            widths = [240, 45, 45, 150, 150]
            for c, w in zip(cols, widths):
                self._mis_tree.heading(c, text=c)
                self._mis_tree.column(c, width=w, anchor="w")

            vsb = ttk.Scrollbar(frm, orient="vertical",
                                 command=self._mis_tree.yview)
            hsb = ttk.Scrollbar(frm, orient="horizontal",
                                 command=self._mis_tree.xview)
            self._mis_tree.configure(yscrollcommand=vsb.set,
                                      xscrollcommand=hsb.set)
            self._mis_tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")

            self._mis_summary = ttk.Label(frm, text="", foreground="grey")
            self._mis_summary.grid(row=2, column=0, columnspan=2,
                                   sticky="w", pady=2, padx=4)

        # ── Browse ─────────────────────────────────────────────────────────────

        def _browse_feat(self):
            chosen = filedialog.askdirectory(
                title="Select features folder (data/adl_features/user)",
                initialdir=self._feat_var.get())
            if chosen:
                self._feat_var.set(chosen)

        # ── Start / Stop ───────────────────────────────────────────────────────

        def _start(self):
            if self._running:
                return

            feat_dir = Path(self._feat_var.get())
            # Derive user from folder name
            user = feat_dir.name

            args = argparse.Namespace(
                user       = user,
                epochs     = self._epochs_var.get(),
                batch_size = self._batch_var.get(),
                lr         = float(self._lr_var.get()),
                val_ratio  = float(self._val_var.get()),
                early_stop   = self._estop_var.get(),
                seed         = self._seed_var.get(),
                dropout      = float(self._dropout_var.get()),
                weight_decay = float(self._wd_var.get()),
            )

            # Override feature_dir resolution: patch the Path inside train()
            # by temporarily symlinking? — simpler: just set cwd or pass directly.
            # We pass args.user and train() builds data/adl_features/{user}.
            # If user chose a custom path, we ensure it resolves correctly by
            # injecting a _feat_dir override via args.
            args._feat_dir = feat_dir  # train() reads this if present

            # Clear previous data
            self._epochs.clear()
            self._tr_accs.clear()
            self._vl_accs.clear()
            self._best_epoch = 0
            self._best_acc   = 0.0
            self._prog_var.set(0)
            self._clear_log()
            self._res_tree.delete(*self._res_tree.get_children())
            self._mis_tree.delete(*self._mis_tree.get_children())
            self._res_summary.config(text="")
            self._mis_summary.config(text="")

            self._stop_event.clear()
            self._running = True
            self._run_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
            self._status_lbl.config(text="Training…")
            self._nb.select(0)   # jump to Log tab

            def worker():
                try:
                    train(args,
                          log_fn          = self._writer,
                          epoch_callback  = self._on_epoch,
                          results_callback= self._on_results,
                          stop_event      = self._stop_event)
                except Exception as e:
                    import traceback
                    self._writer(f"\n✗ Error: {e}\n{traceback.format_exc()}")
                finally:
                    self.after(0, self._on_done)

            threading.Thread(target=worker, daemon=True).start()

        def _stop(self):
            self._stop_event.set()
            self._status_lbl.config(text="Stopping…")

        def _show_model_overview(self):
            import tkinter as tk
            from tkinter import ttk

            # Build a temporary model with current GUI parameters to inspect structure
            num_classes_preview = 11  # placeholder; real count comes from data
            model = ADLClassifier(
                num_classes   = num_classes_preview,
                cnn_out_dim   = 64,
                gru_hidden    = 64,
                gru_layers    = 2,
                dropout       = 0.4,
                bidirectional = True,
            )

            # ── Collect per-layer details ──────────────────────────────────────
            rows = []  # (layer_name, type, output_shape_hint, param_count)

            def count_params(module):
                return sum(p.numel() for p in module.parameters())

            # CNN backbone layers
            h, w = 64, 64
            for name, layer in model.cnn.net.named_children():
                ltype = type(layer).__name__
                p = count_params(layer)
                if hasattr(layer, 'out_channels'):          # Conv2d
                    h2 = h; w2 = w  # padding=1 keeps size
                    shape = f"(batch×T, {layer.out_channels}, {h2}, {w2})"
                elif ltype == 'MaxPool2d':
                    h //= 2; w //= 2
                    shape = f"(batch×T, ?, {h}, {w})"
                elif ltype == 'AdaptiveAvgPool2d':
                    h, w = 1, 1
                    shape = f"(batch×T, ?, 1, 1)"
                elif ltype == 'Flatten':
                    shape = f"(batch×T, 64)"
                elif ltype == 'Linear':
                    shape = f"(batch×T, {layer.out_features})"
                elif ltype == 'ReLU':
                    shape = "— (in-place activation)"
                elif ltype == 'BatchNorm2d':
                    shape = f"(batch×T, {layer.num_features}, {h}, {w})"
                else:
                    shape = "?"
                rows.append((f"  cnn.{name}", ltype, shape, f"{p:,}"))

            # Reshape step (not a module, but important to show)
            rows.append(("  [reshape]", "view", "(batch, T=16, 64)", "—"))

            # GRU
            bidir = model.gru.bidirectional
            gru_p = count_params(model.gru)
            gru_out = model.gru.hidden_size * (2 if bidir else 1)
            rows.append((
                "  gru",
                f"GRU ({'bidir' if bidir else 'unidir'}, {model.gru.num_layers} layers)",
                f"(batch, T, {gru_out})  →  last hidden: (batch, {gru_out})",
                f"{gru_p:,}",
            ))

            # Classifier layers
            for name, layer in model.classifier.named_children():
                ltype = type(layer).__name__
                p = count_params(layer)
                if ltype == 'Linear':
                    shape = f"(batch, {layer.out_features})"
                elif ltype == 'ReLU':
                    shape = "— activation"
                elif ltype == 'Dropout':
                    shape = f"— dropout p={layer.p}"
                else:
                    shape = "?"
                rows.append((f"  classifier.{name}", ltype, shape, f"{p:,}"))

            total_params   = count_params(model)
            trainable      = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # ── Build popup window ─────────────────────────────────────────────
            win = tk.Toplevel(self)
            win.title("Model Overview — ADLClassifier (CNN + Bidirectional GRU)")
            win.resizable(True, True)
            win.geometry("920x560")

            # Summary header
            hdr = (
                f"ADLClassifier   |   Input: (batch, T=16 frames, H=64, W=64, C=3 channels)\n"
                f"Total parameters: {total_params:,}   |   Trainable: {trainable:,}   |   "
                f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n\n"
                f"Processing flow:\n"
                f"  1. Each of the 16 frames is passed through the CNN backbone independently (shared weights)\n"
                f"  2. CNN outputs a 64-dim spatial summary per frame  →  sequence shape (batch, 16, 64)\n"
                f"  3. Bidirectional GRU reads the sequence forward AND backward  →  (batch, 128)\n"
                f"  4. Classifier maps 128-dim vector to {num_classes_preview} class probabilities"
            )
            hdr_lbl = tk.Label(win, text=hdr, justify="left", anchor="w",
                               font=("Consolas", 9), bg="#1e1e2e", fg="#cdd6f4",
                               padx=10, pady=8)
            hdr_lbl.pack(fill="x")

            ttk.Separator(win, orient="horizontal").pack(fill="x", padx=6)

            # Layer table
            frm_tree = ttk.Frame(win)
            frm_tree.pack(fill="both", expand=True, padx=6, pady=4)

            cols = ("Layer", "Type", "Output shape", "Parameters")
            tree = ttk.Treeview(frm_tree, columns=cols, show="headings", height=18)
            col_widths = (170, 200, 350, 100)
            for col, w in zip(cols, col_widths):
                tree.heading(col, text=col)
                tree.column(col, width=w, anchor="w")

            # Section headers + rows
            tree.insert("", "end", values=("── CNN Backbone (shared across all 16 frames) ──", "", "", ""), tags=("section",))
            for row in rows:
                if row[0].startswith("  cnn"):
                    tree.insert("", "end", values=row)
                elif row[0] == "  [reshape]":
                    tree.insert("", "end", values=row, tags=("reshape",))
            tree.insert("", "end", values=("── Temporal GRU ──", "", "", ""), tags=("section",))
            for row in rows:
                if row[0].startswith("  gru"):
                    tree.insert("", "end", values=row)
            tree.insert("", "end", values=("── Classifier Head ──", "", "", ""), tags=("section",))
            for row in rows:
                if row[0].startswith("  classifier"):
                    tree.insert("", "end", values=row)
            tree.insert("", "end", values=("", "", "", ""))
            tree.insert("", "end", values=(
                "TOTAL", "", "", f"{total_params:,}"
            ), tags=("total",))

            tree.tag_configure("section", background="#313244", foreground="#89b4fa", font=("Consolas", 9, "bold"))
            tree.tag_configure("reshape", foreground="#a6e3a1")
            tree.tag_configure("total",   background="#1e1e2e", foreground="#f38ba8", font=("Consolas", 9, "bold"))

            vsb = ttk.Scrollbar(frm_tree, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            tree.pack(side="left", fill="both", expand=True)
            vsb.pack(side="right", fill="y")

            # Parameter breakdown footer
            cnn_p  = count_params(model.cnn)
            gru_p2 = count_params(model.gru)
            cls_p  = count_params(model.classifier)
            footer = (
                f"Parameter breakdown:  "
                f"CNN backbone {cnn_p:,} ({100*cnn_p//total_params}%)  |  "
                f"GRU {gru_p2:,} ({100*gru_p2//total_params}%)  |  "
                f"Classifier {cls_p:,} ({100*cls_p//total_params}%)"
            )
            ttk.Label(win, text=footer, font=("Consolas", 9), foreground="#888").pack(pady=4)
            ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 6))

        def _show_data_overview(self):
            import tkinter as tk
            from tkinter import ttk, messagebox

            feat_dir  = Path(self._feat_var.get())
            meta_path = feat_dir / "meta.csv"
            npz_path  = feat_dir / "adl_features.npz"

            if not meta_path.exists():
                messagebox.showwarning(
                    "Data Overview",
                    f"No meta.csv found in:\n{feat_dir}\n\nRun feature extraction first.")
                return

            meta = pd.read_csv(str(meta_path))
            total_windows  = len(meta)
            num_classes    = meta["label"].nunique()

            # Per-class stats
            class_stats = (
                meta.groupby("label", sort=False)
                .agg(windows=("label", "count"),
                     sessions=("source_file", "nunique"))
                .reset_index()
                .sort_values("windows", ascending=False)
            )

            # Per-session stats: unique (activity, source_file) pairs
            session_stats = (
                meta.groupby(["source_file", "label"], sort=False)
                .agg(windows=("label", "count"))
                .reset_index()
                .sort_values(["label", "source_file"])
            )
            # Total sessions = distinct (activity, file) pairs, not just distinct filenames
            total_sessions = len(session_stats)

            # Check npz shape
            shape_str = ""
            if npz_path.exists():
                try:
                    d = np.load(str(npz_path))
                    shape_str = f"  |  Array shape: {d['X'].shape}"
                except Exception:
                    pass

            # ── Build popup ────────────────────────────────────────────────────
            win = tk.Toplevel(self)
            win.title(f"Data Overview — {feat_dir.name}")
            win.resizable(True, True)
            win.geometry("760x520")

            hdr = (
                f"Folder: {feat_dir}\n"
                f"Total windows: {total_windows}   |   "
                f"Classes: {num_classes}   |   "
                f"Sessions: {total_sessions}{shape_str}"
            )
            hdr_lbl = tk.Label(win, text=hdr, justify="left", anchor="w",
                               font=("Consolas", 9), bg="#1e1e2e", fg="#cdd6f4",
                               padx=10, pady=8)
            hdr_lbl.pack(fill="x")
            ttk.Separator(win, orient="horizontal").pack(fill="x", padx=6)

            nb = ttk.Notebook(win)
            nb.pack(fill="both", expand=True, padx=6, pady=4)

            # ── Per-class tab ──────────────────────────────────────────────────
            frm_cls = ttk.Frame(nb)
            nb.add(frm_cls, text="  Per Class  ")
            frm_cls.columnconfigure(0, weight=1)
            frm_cls.rowconfigure(0, weight=1)

            cls_cols = ("Activity", "Windows", "Sessions", "% of total")
            cls_tree = ttk.Treeview(frm_cls, columns=cls_cols,
                                    show="headings", height=14)
            for col, w in zip(cls_cols, (220, 80, 80, 100)):
                cls_tree.heading(col, text=col)
                cls_tree.column(col, width=w, anchor="center")
            cls_tree.column("Activity", anchor="w")

            for _, row in class_stats.iterrows():
                pct = 100.0 * row["windows"] / total_windows
                cls_tree.insert("", "end", values=(
                    row["label"], int(row["windows"]),
                    int(row["sessions"]), f"{pct:.1f}%"))
            cls_tree.insert("", "end",
                            values=("── TOTAL", total_windows, total_sessions, "100%"),
                            tags=("total",))
            cls_tree.tag_configure("total", foreground="#f38ba8",
                                   font=("Consolas", 9, "bold"))

            vsb_cls = ttk.Scrollbar(frm_cls, orient="vertical",
                                    command=cls_tree.yview)
            cls_tree.configure(yscrollcommand=vsb_cls.set)
            cls_tree.grid(row=0, column=0, sticky="nsew")
            vsb_cls.grid(row=0, column=1, sticky="ns")

            # ── Per-session tab ────────────────────────────────────────────────
            frm_ses = ttk.Frame(nb)
            nb.add(frm_ses, text="  Per Session  ")
            frm_ses.columnconfigure(0, weight=1)
            frm_ses.rowconfigure(0, weight=1)

            ses_cols = ("Activity", "Session file", "Windows")
            ses_tree = ttk.Treeview(frm_ses, columns=ses_cols,
                                    show="headings", height=14)
            for col, w in zip(ses_cols, (160, 340, 80)):
                ses_tree.heading(col, text=col)
                ses_tree.column(col, width=w, anchor="w")
            ses_tree.column("Windows", anchor="center")

            for _, row in session_stats.iterrows():
                ses_tree.insert("", "end", values=(
                    row["label"], row["source_file"], int(row["windows"])))

            vsb_ses = ttk.Scrollbar(frm_ses, orient="vertical",
                                    command=ses_tree.yview)
            hsb_ses = ttk.Scrollbar(frm_ses, orient="horizontal",
                                    command=ses_tree.xview)
            ses_tree.configure(yscrollcommand=vsb_ses.set,
                               xscrollcommand=hsb_ses.set)
            ses_tree.grid(row=0, column=0, sticky="nsew")
            vsb_ses.grid(row=0, column=1, sticky="ns")
            hsb_ses.grid(row=1, column=0, sticky="ew")

            ttk.Button(win, text="Close", command=win.destroy).pack(pady=(2, 6))

        def _on_done(self):
            self._running = False
            self._run_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")
            if self._stop_event.is_set():
                self._status_lbl.config(text="Stopped.")
            else:
                self._status_lbl.config(
                    text=f"Done — best val {self._best_acc:.1%} @ epoch {self._best_epoch}")

        # ── Epoch callback (called from worker thread) ─────────────────────────

        def _on_epoch(self, epoch, _tr_loss, tr_acc, _vl_loss, vl_acc,
                      _lr, best_val_acc, best_epoch, total_epochs):
            # Schedule GUI update on the main thread
            self.after(0, self._update_epoch_ui,
                       epoch, tr_acc, vl_acc, best_val_acc, best_epoch,
                       total_epochs)

        def _update_epoch_ui(self, epoch, tr_acc, vl_acc,
                             best_val_acc, best_epoch, total_epochs):
            self._best_epoch = best_epoch
            self._best_acc   = best_val_acc

            self._epochs.append(epoch)
            self._tr_accs.append(tr_acc)
            self._vl_accs.append(vl_acc)

            # Progress bar
            pct = (epoch / total_epochs) * 100
            self._prog_var.set(pct)
            self._status_lbl.config(
                text=f"Epoch {epoch}/{total_epochs}   best {best_val_acc:.1%} @ {best_epoch}")

            # Redraw training curve
            ax = self._ax
            ax.clear()
            ax.plot(self._epochs, self._tr_accs,
                    color="steelblue", label="Train acc", linewidth=1.5)
            ax.plot(self._epochs, self._vl_accs,
                    color="darkorange", label="Val acc",  linewidth=1.5)
            if best_epoch > 0:
                ax.axvline(best_epoch, color="green", linestyle="--",
                           linewidth=1, alpha=0.7,
                           label=f"Best {best_val_acc:.1%}")
            ax.set_xlim(1, max(total_epochs, len(self._epochs)))
            ax.set_ylim(0, 1)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Accuracy", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)
            self._fig.tight_layout(pad=1.0)
            self._chart_canvas.draw_idle()

        # ── Results callback (called from worker thread after training) ────────

        def _on_results(self, cm, label_names, report_dict,
                        mismatch_df, best_val_acc, best_epoch):
            self.after(0, self._populate_results,
                       cm, label_names, report_dict, mismatch_df,
                       best_val_acc, best_epoch)

        def _populate_results(self, cm, label_names, report_dict,
                              mismatch_df, best_val_acc, best_epoch):
            # ── Per-class results treeview ─────────────────────────────────────
            self._res_tree.delete(*self._res_tree.get_children())
            for lbl in label_names:
                if lbl not in report_dict:
                    continue
                d = report_dict[lbl]
                tag = "good" if d["f1-score"] >= 0.75 else (
                      "warn" if d["f1-score"] >= 0.5 else "bad")
                self._res_tree.insert("", "end", values=(
                    lbl,
                    f"{d['precision']:.3f}",
                    f"{d['recall']:.3f}",
                    f"{d['f1-score']:.3f}",
                    int(d["support"]),
                ), tags=(tag,))

            # Separator + macro/weighted avg
            for key in ("macro avg", "weighted avg"):
                if key in report_dict:
                    d = report_dict[key]
                    self._res_tree.insert("", "end", values=(
                        f"── {key}",
                        f"{d['precision']:.3f}",
                        f"{d['recall']:.3f}",
                        f"{d['f1-score']:.3f}",
                        int(d["support"]),
                    ))

            self._res_tree.tag_configure("good", foreground="#2ecc71")
            self._res_tree.tag_configure("warn", foreground="#e67e22")
            self._res_tree.tag_configure("bad",  foreground="#e74c3c")

            self._res_summary.config(
                text=f"Best val accuracy: {best_val_acc:.1%}  at epoch {best_epoch}")

            # Switch to results tab
            self._nb.select(1)

            # ── Confusion matrix ───────────────────────────────────────────────
            self._cm_placeholder.grid_remove()
            self._cm_fig.clear()
            n = len(label_names)
            self._cm_fig.set_size_inches(max(5, n * 0.8), max(4, n * 0.7))

            ax = self._cm_fig.add_subplot(111)
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            self._cm_fig.colorbar(im, ax=ax)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(label_names, fontsize=8)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.set_title(
                f"Confusion Matrix — {best_val_acc:.1%} (epoch {best_epoch})",
                fontsize=10)
            thresh = cm.max() / 2.0
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=8,
                            color="white" if cm[i, j] > thresh else "black")
            self._cm_fig.tight_layout()
            self._cm_canvas.draw()

            # ── Mismatch treeview ──────────────────────────────────────────────
            self._mis_tree.delete(*self._mis_tree.get_children())
            if not mismatch_df.empty:
                for _, row in mismatch_df.iterrows():
                    self._mis_tree.insert("", "end", values=(
                        row["source_file"],
                        int(row["segment_idx"]),
                        int(row["window_idx"]),
                        row["true_label"],
                        row["predicted"],
                    ))
            self._mis_summary.config(
                text=f"{len(mismatch_df)} misclassified windows  "
                     f"(copy rows: select then Ctrl+C)")

            # Bind Ctrl+C copy for mismatch treeview
            self._mis_tree.bind("<Control-c>", self._copy_mismatch_selection)

        def _copy_mismatch_selection(self, _):
            rows = [self._mis_tree.item(iid)["values"]
                    for iid in self._mis_tree.selection()]
            if not rows:
                return
            text = "\n".join("\t".join(str(v) for v in r) for r in rows)
            self.clipboard_clear()
            self.clipboard_append(text)

        # ── Helpers ────────────────────────────────────────────────────────────

        def _clear_log(self):
            self._log.configure(state="normal")
            self._log.delete("1.0", "end")
            self._log.configure(state="disabled")

    app = App()
    app.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train ADL CNN+GRU classifier")
    p.add_argument("--nogui",      action="store_true",
                   help="Run in CLI mode (no GUI)")
    p.add_argument("--user",       default="peter")
    p.add_argument("--epochs",     type=int,   default=80)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val_ratio",  type=float, default=0.2)
    p.add_argument("--early_stop",   type=int,   default=15)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--dropout",      type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--channels",     type=int,   default=0,
                   help="Use only first N channels (0 = use all)")
    args = p.parse_args()

    if args.nogui:
        train(args)
    else:
        launch_gui()
