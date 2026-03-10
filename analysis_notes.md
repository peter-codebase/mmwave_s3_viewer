# Sleep Radar Analysis Notes

## Session: 2026-03-06

---

## Bugs Fixed

| # | Location | Bug | Fix |
|---|---|---|---|
| 1 | Line ~41 | `timedelta` not imported → `NameError` in sound analysis | `from datetime import datetime, timedelta` |
| 2 | `_apply_browsed_prefix` ~line 2081 | ~330 lines of dead code after `return` (unreachable functions + duplicate `threading.Thread` call) | Deleted |
| 3 | `_load_tag_store` ~line 4198 | Duplicate `for line in lines` loop after `return segs` | Deleted |
| 4 | `_compute_phase_and_rates` ~line 3962 | No DC removal before range FFT (unlike Infineon SDK helper) | Added `mat = mat - mat.mean(axis=1, keepdims=True)` |

---

## Signal Processing Improvements

### 1. Butter+filtfilt bandpass (`_bandpass_via_fft`)
Replaced rectangular FFT mask (caused Gibbs ringing) with zero-phase Butterworth order=4 via `scipy.signal.butter` + `filtfilt`. Falls back to FFT mask if scipy unavailable.

### 2. RR harmonic subtraction before HR bandpass
```python
ph_no_resp = ph_d - self.phase_resp   # remove RR fundamental
self.phase_heart = bandpass(ph_no_resp, 0.80, 2.50)  # then HR bandpass
```
2nd harmonic of max RR (0.60 Hz × 2 = 1.20 Hz) falls inside the HR band — this removes it.

### 3. Adaptive RR harmonic notching (`_estimate_hr_notched`)
For each 30s window:
1. Estimate local RR frequency from PSD peak in 0.10–0.60 Hz
2. Zero out PSD bins within ±0.05 Hz of each harmonic n×rr_hz (n=2..24) that falls in HR band
3. Find HR peak in remaining PSD

### 4. 30s sliding-window HR time series
- Stored as `self.radar_hr_t` (epoch_s) and `self.radar_hr_bpm`
- Same x-axis as E4 HR panel for direct visual comparison
- Phase plot (ax1) now shows this time series instead of raw phase

---

## Data Analysis: peter_20260304_004707.csv

**Radar setup**: ~1.2 m range, 5 Hz frame rate, 32 chirps × 64 samples, 3 channels

### RR Detection — RELIABLE
- SNR: **16–30 dB** across all bins
- Detected peak: **0.215 Hz = 12.9 bpm** (physiologically plausible for sleep)

### HR Detection — NOT FEASIBLE AT THIS SETUP
- SNR: only **4–8 dB** (all bins)
- HR band (0.8–2.5 Hz) is entirely covered by RR harmonics:
  - 4th: 0.860 Hz = 51.6 bpm
  - 5th: 1.075 Hz = 64.5 bpm
  - 6th: 1.290 Hz = 77.4 bpm
  - 7th–11th: fill up to 2.36 Hz
- After harmonic notching: best remaining peak = **13.9 dB at 55.2 bpm**
  → This is a sidelobe of the 4th RR harmonic, NOT the true heartbeat
- MAE vs E4: ~30 bpm (all bins) — worse than chance
- **Root cause**: heartbeat chest displacement (~0.1–0.3 mm) vs respiration (~5–10 mm) = 20–50× power gap

### Would 10 Hz frame rate help?
**No.** Nyquist at 5 Hz is already 2.5 Hz = 150 bpm, fully covering the HR band. The problem is SNR, not sampling rate. Doubling samples does not change the heartbeat/respiration power ratio.

---

## Planned Hardware Experiment (Phase 1 SNR Survey)

**Goal**: Find a radar placement that gives HR SNR > 15 dB (minimum for reliable detection).

**Protocol**: Wear E4, lie still in sleep position, 3-minute recording per config.

| Config | Distance | Aim point | Cover |
|---|---|---|---|
| A | 1.2 m | Chest | Normal ← baseline (4–8 dB, fails) |
| B | 0.5 m | Chest | Normal cover |
| C | 0.3 m | Chest | Normal cover |
| D | 0.3 m | Chest | No cover |
| E | 0.5 m | Face/cheek | No cover |
| F | 0.5 m | Upper chest/clavicle | No cover |

**Analysis script**: to be written — reads each CSV, computes HR SNR per range bin, outputs summary table.

**Why face (Config E)?** The face has near-zero respiration displacement (unlike chest/abdomen), so the heartbeat/respiration ratio is fundamentally better. Main risk: head movement during sleep causes frequent signal loss.

---

## Remaining Known Issues (Not Yet Fixed)

| # | Issue | Priority |
|---|---|---|
| 5 | Motion threshold uses 75th percentile → always flags 25% of frames as motion | Medium |
| 6 | E4 subplot x-limits not shared → cursor hidden in top 4 panels | Low |
| 7 | No max-distance guard in `_pick_closest_physical_stem` | Low |
| 8 | Double normalization in float32 WAV path | Low |
| 9 | 150ms race condition timer for S3 session list population | Low |
