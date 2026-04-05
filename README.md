# mmWave Sleep & ADL Monitor

A research toolkit for contactless physiological monitoring and activity recognition using 60 GHz mmWave radar (Infineon BGT60TR13C). Data is stored on AWS S3 and analyzed offline via a Tkinter GUI.

---

## Project Overview

| Goal | Method |
|---|---|
| Sleep vital signs (HR, RR) | Phase extraction from Range-Doppler maps, bandpass filtering, harmonic notching |
| Activity recognition (ADL) | CNN + GRU classifier on sliding-window Range-Doppler maps |
| Ground truth | Empatica E4 wristband (BVP → HR, accelerometer, TEMP, EDA) |

---

## Repository Structure

```
.
├── s3_post_analyze_gui_single_radar.py   # Main GUI — sleep analysis (HR, RR, phase)
├── s3_post_analyze_gui.py                # Multi-radar GUI + ADL Export tool
├── s3_layout.py                          # S3 bucket path helpers
├── e4_viewer.py                          # Empatica E4 wristband data viewer
├── adl_feature_extraction.py             # Step 1: raw CSV → Range-Doppler .npz
├── adl_model.py                          # CNN + GRU model architecture
├── adl_train.py                          # Step 2: train & evaluate the model
├── adl_tune.py                           # Optional: hyperparameter grid search
├── helpers/
│   ├── DopplerAlgo.py                    # Infineon Range-Doppler processor (do not modify)
│   └── fft_spectrum.py                   # Infineon FFT helpers (do not modify)
├── data/                                 # (gitignored — large files)
│   ├── adl/peter/                        # Per-activity raw CSVs (one file per activity)
│   └── adl_features/peter/              # Extracted .npz features + saved model
└── .env                                  # AWS credentials (gitignored)
```

---

## Radar Data Format

Raw recordings are CSV files with columns:

| Column | Description |
|---|---|
| `frame_number` | Frame index (1-based) |
| `timestamp` | Unix epoch (seconds) |
| `dev0_ch1/2/3` | 32 chirps × 64 samples float32, stored as string literal |

Typical frame rate: ~5 Hz. Sessions are 3–10 minutes long.

---

## ADL Classification Pipeline

### Supported Activities (11 classes)

`drinking`, `face_washing`, `hair_washing`, `hand_washing`, `no_motion`,
`sit_to_stand`, `stand_to_sit`, `standing`, `tooth_brushing`, `walk_away`, `walk_close`

### Step 1 — Feature Extraction

```bash
python adl_feature_extraction.py --nogui --window 64 --stride 32
```

For each labeled segment, processes raw chirp data into Range-Doppler maps using a sliding window:

```
raw chirp (32 chirps × 64 samples × 3 Rx channels)
  → DopplerAlgo (range FFT + MTI filter + Doppler FFT)
  → complex Range-Doppler map (64 range bins × 64 Doppler bins)
  → log1p magnitude (dynamic range compression)
  → 4-channel frame: [rx0_mag, rx1_mag, rx2_mag, doppler_proj]
```

**doppler_proj**: the Doppler energy profile (summed over range bins, averaged over channels, tiled spatially). This explicit marginal of the RD map acts as a shortcut feature that helps small CNNs learn Doppler patterns efficiently.

Each window of 64 frames (~12.8 s at 5 Hz) becomes one training sample of shape `(64, 64, 64, 4)`.

Outputs saved to `data/adl_features/peter/`:
- `adl_features.npz` — `X: (N, 64, 64, 64, 4)`, `y: (N,)`
- `label_map.json` — class index mapping
- `meta.csv` — traceability (source file, segment, window index)

### Step 2 — Training

```bash
python adl_train.py --nogui --channels 4 --epochs 80
```

Key training decisions:
- **Session-level train/val split** — windows from the same recording session never appear in both train and val, preventing data leakage
- **Class weights** — compensates for imbalance between activity window counts
- **Data augmentation** — Gaussian noise, random time shift, random channel dropout
- **Best model** saved by validation accuracy to `data/adl_features/peter/best_model.pt`

Current best: **89% validation accuracy** (window=64, stride=32, 4 channels)

---

## Model Architecture — CNN + GRU

The model treats each window of T=64 radar frames as a **video-like sequence**: a 2D CNN extracts spatial features from each frame independently, then a GRU captures the temporal rhythm across the sequence.

```
Input: (batch, T=64 frames, 64 range bins, 64 Doppler bins, 4 channels)
         └─ one 12.8s sliding window of Range-Doppler maps

┌─────────────────────────────────────────────────────────┐
│  CNN Backbone  (shared weights, applied to all T frames) │
│                                                          │
│  Block 1: Conv2d(4→8, 3×3) + BN + ReLU + Dropout2d      │
│           MaxPool2d(2)   →  (8, 32, 32)                  │
│                                                          │
│  Block 2: Conv2d(8→16, 3×3) + BN + ReLU + Dropout2d     │
│           MaxPool2d(2)   →  (16, 16, 16)                 │
│                                                          │
│  Block 3: Conv2d(16→32, 3×3) + BN + ReLU + Dropout2d    │
│           MaxPool2d(2)   →  (32, 8, 8)                   │
│                                                          │
│  AdaptiveAvgPool2d(4)    →  (32, 4, 4) = 512-d flat      │
│  Linear(512 → 64) + ReLU →  64-d frame feature vector   │
└─────────────────────────────────────────────────────────┘
         ↓  repeat for each of T=64 frames  ↓

         sequence of T=64 frame feature vectors (64-d each)

┌─────────────────────────────────────────────────────────┐
│  Bidirectional GRU  (2 layers, hidden=64)               │
│                                                          │
│  Processes the frame sequence in both directions.        │
│  The final hidden states of the forward and backward     │
│  passes are concatenated → 128-d context vector.        │
└─────────────────────────────────────────────────────────┘
         ↓

┌─────────────────────────────────────────────────────────┐
│  MLP Classifier Head                                    │
│                                                          │
│  Linear(128 → 64) + ReLU + Dropout(0.4)                 │
│  Linear(64 → num_classes)                               │
└─────────────────────────────────────────────────────────┘
         ↓
Output: (batch, 11) logits
```

Total parameters: ~172,000

---

## Sleep Vital Sign Analysis

The main GUI (`s3_post_analyze_gui_single_radar.py`) loads a radar session from S3 and extracts:

- **Respiration rate (RR)** — reliable at ~1.2m range; SNR 16–30 dB
- **Heart rate (HR)** — limited by physics at >0.5m; chest displacement (~0.1mm) is ~50× smaller than respiration (~5mm)
- **Phase plot** — 30s sliding-window HR time series overlaid with E4 ground truth

Signal processing chain:
1. Range FFT per chirp frame
2. DC removal (clutter suppression)
3. Phase extraction from peak range bin
4. Zero-phase Butterworth bandpass (order=4) for RR (0.1–0.5 Hz) and HR (0.8–2.5 Hz)
5. RR harmonic subtraction before HR bandpass
6. Adaptive harmonic notching (`_estimate_hr_notched`) — removes n×RR_Hz bins before picking HR peak

---

## Setup

```bash
pip install boto3 pandas numpy scipy torch scikit-learn matplotlib python-dotenv
```

Create `.env` in the project root:
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=...
S3_BUCKET=...
```

---

## ADL Data Collection

Labeled sessions are exported from S3 using the **ADL Export** button in `s3_post_analyze_gui.py`. Tag files (`.txt`) on S3 define activity segments per session in the format:

```
activity_name|dev0: [(start_frame, end_frame), ...]
```

Exported rows are appended to `data/adl/peter/<activity>.csv` (one file per activity). The export is incremental — already-exported `(label, session)` pairs are skipped automatically.
