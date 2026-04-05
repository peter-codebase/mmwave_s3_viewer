#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sense Demo — dual Infineon FMCW radars + optional camera, with local web UI,
plus motion-triggered speaker identification (SpeechBrain ECAPA) and TTS prompts.

Pipeline:
1) At least one radar detects MOTION
2) Pi speaks: "How are you?"
3) Record voice and identify user from ./voice_id/*.wav (stem = name)
4) If known: "Hi <name>, nice to see you!"
   If unknown: "Hi stranger, may I know your name?" -> enroll user
5) Lock interaction until BOTH radars are NOT in MOTION for unlock window

Enrollment (NEW):
- If UNKNOWN: ask name, record short clip
- If Vosk is available + model path exists: transcribe name (offline) and use it as filename
- Else: fall back to user_YYYYMMDD_HHMMSS
- Save: voice_id/<name>.wav
- Add embedding to in-memory reference list (no restart needed)

Target: Raspberry Pi OS Bookworm.
"""

import os
import sys
os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "*:ERROR")
import time
import threading
import subprocess
import signal

STOP_EVENT = threading.Event()
COLLECTOR_LOCK = threading.Lock()
# Use a normal assignment to keep Python 3.11 compatible across environments.
COLLECTOR_PROC = None  # type: ignore

def _request_stop(signum, frame):
    """Request a graceful shutdown for all threads and any running collector subprocess."""
    STOP_EVENT.set()
    try:
        print(f"[STOP] Signal {signum} received → shutting down...")
    except Exception:
        pass
    # If a collector subprocess is running, ask it to stop cleanly.
    try:
        with COLLECTOR_LOCK:
            proc = COLLECTOR_PROC
        if proc is not None and getattr(proc, "poll", lambda: 0)() is None:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
    except Exception:
        pass

for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    try:
        signal.signal(_sig, _request_stop)
    except Exception:
        pass

import re
import json
import tempfile
import datetime
import shutil
import socket
from typing import Tuple, Optional, Dict, List

import numpy as np

from scipy import signal, constants
from scipy.signal import butter, filtfilt

# Load the .env file
from dotenv import load_dotenv
load_dotenv()

# Audio + embedding
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly
import sounddevice as sd  # pip install sounddevice (requires PortAudio)

# Optional VAD (end-of-speech) and S3 marker check/upload for voice drift.
try:
    import webrtcvad  # type: ignore
    WEBRTCVAD_AVAILABLE = True
except Exception as _e:
    webrtcvad = None  # type: ignore
    WEBRTCVAD_AVAILABLE = False
    _WEBRTCVAD_IMPORT_ERR = _e

try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError  # type: ignore
    BOTO3_AVAILABLE = True
except Exception as _e:
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore
    BOTO3_AVAILABLE = False
    _BOTO3_IMPORT_ERR = _e

from pathlib import Path

# Infineon FMCW SDK
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics

# SpeechBrain ECAPA
import scipy.io.wavfile
from scipy.signal import resample
try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except Exception as _e:
    SpeakerRecognition = None  # type: ignore
    SPEECHBRAIN_AVAILABLE = False
    _SPEECHBRAIN_IMPORT_ERR = _e

# Optional offline STT (name extraction)
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except Exception:
    VoskModel = None  # type: ignore
    KaldiRecognizer = None  # type: ignore
    VOSK_AVAILABLE = False

# ------------------------- CLI -------------------------
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    p = argparse.ArgumentParser(description="Dual-radar live processing (no web UI) + motion-triggered speaker ID")
    p.add_argument('--frate', type=float, default=5.0, help='Radar frame rate (Hz)')
    p.add_argument('--num_chirps', type=int, default=32, help='Chirps per frame (estimate)')
    p.add_argument('--num_samples', type=int, default=64, help='Samples per chirp (estimate)')
    p.add_argument('--phase_radar', choices=['dev0', 'dev1', 'off'], default='dev1',
                   help='Which radar to estimate vitals from (ch1)')
    p.add_argument('--port0', type=str, default=os.getenv('RADAR0_URI'), help='Device URI for radar #0 (e.g., usb:0)')
    p.add_argument('--port1', type=str, default=os.getenv('RADAR1_URI'), help='Device URI for radar #1 (e.g., usb:1)')

    # ------------------------- Motion detection (per radar) -------------------------
    p.add_argument('--motion_gate_min_m', type=float, default=0.30,
                   help='Min range (m) used for motion energy gating')
    p.add_argument('--motion_gate_max_m', type=float, default=3.00,
                   help='Max range (m) used for motion energy gating')
    p.add_argument('--motion_alpha', type=float, default=0.35,
                   help='EMA smoothing alpha for motion score (0..1)')
    p.add_argument('--motion_on', type=float, default=0.00030,
                   help='Motion ON threshold for smoothed energy')
    p.add_argument('--motion_off', type=float, default=0.00025,
                   help='Motion OFF threshold (hysteresis)')
    p.add_argument('--motion_enter_frames', type=int, default=3,
                   help='Frames above ON threshold to enter MOTION')
    p.add_argument('--motion_exit_frames', type=int, default=5,
                   help='Frames below OFF threshold to exit MOTION')
    p.add_argument('--motion_cooldown_s', type=float, default=3.0,
                   help='Cooldown seconds after motion ends')

    # ------------------------- Voice interaction -------------------------
    p.add_argument('--voice_enable', action='store_true', default=True,
                   help='Enable motion-triggered speaker identification')
    p.add_argument('--voice_folder', type=str, default=os.path.join(SCRIPT_DIR, 'voice_id'),
                   help='Folder containing reference wavs (stem = name)')
    p.add_argument('--voice_fs', type=int, default=48000,
                   help='Recording sample rate (Hz)')
    p.add_argument('--voice_seconds', type=float, default=4.0,
                   help='Recording duration after prompt (seconds)')
    p.add_argument('--voice_device', type=str, default=0,
                   help='sounddevice input device (name or index). Default: system default.')
    p.add_argument('--voice_threshold', type=float, default=0.25,
                   help='Accept speaker if cosine similarity >= threshold (hard gate)')
    p.add_argument('--voice_margin', type=float, default=0.05,
                   help='Require best - second_best >= margin (reduces ambiguity)')
    p.add_argument('--voice_post_tts_delay_s', type=float, default=0.35,
                   help='Delay after TTS before recording (reduce self-capture)')
    p.add_argument('--beep_enable', action='store_true', default=True,
                   help='Play a short beep right before starting each voice-drift recording (for offline latency analysis).')

    p.add_argument('--unlock_no_motion_s', type=float, default=10.0,
                   help='Unlock interaction after BOTH radars are not MOTION for this many seconds')
    # Internet / Wi-Fi gating (optional): only proceed with identity/drift/collection when internet is reachable
    p.add_argument('--netcheck_enable', action='store_true', default=True,
                   help='Gate identity/drift/collection on internet reachability (recommended when uploading to S3).')
    p.add_argument('--wifi_iface', type=str, default=os.getenv('WIFI_IFACE', 'wlan0'),
                   help='Wireless interface to check (default: wlan0).')
    p.add_argument('--internet_host', type=str, default=os.getenv('INTERNET_HOST', 's3.amazonaws.com'),
                   help='Host used to test internet reachability (default: s3.amazonaws.com).')
    p.add_argument('--internet_port', type=int, default=int(os.getenv('INTERNET_PORT', '443')),
                   help='Port used to test internet reachability (default: 443).')
    p.add_argument('--netcheck_timeout_s', type=float, default=float(os.getenv('NETCHECK_TIMEOUT_S', '1.0')),
                   help='Timeout (seconds) per reachability attempt.')
    p.add_argument('--netcheck_attempts', type=int, default=int(os.getenv('NETCHECK_ATTEMPTS', '2')),
                   help='Number of reachability attempts before declaring failure.')
    p.add_argument('--offline_cooldown_s', type=float, default=float(os.getenv('OFFLINE_COOLDOWN_S', '60.0')),
                   help='Cooldown (seconds) after an offline prompt before prompting again (motion edge required).')
    # Collection triggering (optional): pause motion detection and run raw collector as a subprocess
    p.add_argument('--collect_enable', action='store_true', default=True,
                   help='After resolving a user name (known or enrolled), pause motion detection and run the raw collection script.')
    p.add_argument('--collect_script', type=str, default=str(Path(__file__).with_name('raw_collection_ui_dual_with_audio.py')),
                   help='Path to the raw collection script (default: raw_collection_ui_dual_with_audio.py next to this file).')
    p.add_argument('--collect_seconds', type=int, default=600,
                   help='Collection duration in seconds (default: 600 = 10 minutes).')
    p.add_argument('--collect_segment_minutes', type=int, default=10,
                   help='Segment minutes passed to collector (default: 10).')
    p.add_argument('--collect_timeout_s', type=int, default=660,
                   help='Timeout for the collector subprocess in seconds (default: 660).')
    p.add_argument('--collect_audio', action='store_true', default=True,
                   help='Pass --audio to collector to record WAV.')
    p.add_argument('--collect_audio_rate', type=int, default=48000,
                   help='Audio sample rate passed to collector (default: 48000).')
    p.add_argument('--collect_audio_channels', type=int, default=2,
                   help='Audio channels passed to collector (default: 2).')
    p.add_argument('--collect_audio_device', type=str, default=None,
                   help='Audio device passed to collector (--audio-device). If omitted, collector uses its default.')
    p.add_argument('--collect_no_video', action='store_true',
                   help='Pass --no-video to collector (force disable video).')
    p.add_argument('--collect_cam_index', type=int, default=None,
                   help='Pass --cam-index to collector (USB cam index).')

    # ------------------------- Enrollment (NEW) -------------------------
    p.add_argument('--enroll_enable', action='store_true', default=True,
                   help='If speaker is unknown, ask for name and enroll new user (default: enabled)')
    p.add_argument('--enroll_seconds', type=float, default=2.0,
                   help='Enrollment recording duration (name clip) in seconds')
    p.add_argument('--enroll_confirm_seconds', type=float, default=2.0,
                   help='Confirmation recording duration (yes/no) in seconds')
    p.add_argument('--enroll_confirm_retries', type=int, default=2,
                   help='How many times to re-ask yes/no before asking for the name again')
    p.add_argument('--enroll_max_name_attempts', type=int, default=3,
                   help='Maximum name re-tries before falling back to a timestamp-based name')
    p.add_argument('--enroll_vosk_model', type=str, default=os.path.join(SCRIPT_DIR, 'vosk-model-small-en-us-0.15'),
                   help='Path to Vosk model directory for offline name transcription (optional)')
    p.add_argument('--enroll_min_name_len', type=int, default=2,
                   help='Minimum sanitized name length. Otherwise use user_<timestamp>.')

    # ------------------------- Voice drift (daily prompted clips) -------------------------
    p.add_argument('--voice_drift_enable', action='store_true', default=True,
                   help='After identifying a user, optionally collect a daily voice-drift prompt set (default: enabled).')
    p.add_argument('--voice_drift_once_per_day', action='store_true', default=True,
                   help='Run voice-drift prompt set at most once per user per day (default: enabled).')
    p.add_argument('--voice_drift_items', type=str, default='ah,read,count,conversation',
                   help='Comma-separated clip keys to collect (default: ah,read,count,conversation).')
    p.add_argument('--voice_drift_max_seconds', type=float, default=20.0,
                   help='Hard cap per prompt recording length (seconds).')
    p.add_argument('--voice_drift_min_seconds', type=float, default=1.0,
                   help='Minimum voiced duration before allowing end-of-speech stop (seconds).')

    # VAD / end-of-speech behavior
    p.add_argument('--vad_enable', action='store_true', default=True,
                   help='Use VAD to stop recording when the user finishes speaking (default: enabled).')
    p.add_argument('--vad_mode', type=int, default=2, choices=[0, 1, 2, 3],
                   help='WebRTC VAD aggressiveness 0..3 (higher = more aggressive). Default: 2.')
    p.add_argument('--vad_silence_ms', type=int, default=1000,
                   help='Stop when this many milliseconds of trailing silence are detected (default: 800).')

    # S3 (used for voice drift marker + uploads). Mirrors raw_collection_ui_dual_with_audio.py
    p.add_argument('--s3-bucket', type=str, default=os.getenv('S3_BUCKET'),
                   help='AWS S3 bucket for voice drift uploads and daily marker checks (optional, but recommended).')
    p.add_argument('--s3-prefix', type=str, default=(os.getenv('S3_PREFIX') or '').strip('/'),
                   help='Optional S3 key prefix (e.g., "runs"). Leave blank for bucket root.')
    p.add_argument('--aws-region', type=str, default=os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION'),
                   help='AWS region for the client (optional).')
    p.add_argument('--aws-profile', type=str, default=os.getenv('AWS_PROFILE'),
                   help='AWS CLI profile name to use (optional).')

    return p.parse_args()

ARGS = parse_args()

# ------------------------- Radar setup -------------------------

def configure_device(device: DeviceFmcw, frate: float) -> int:
    """Configure FMCW device roughly per metrics and return number of RX antennas."""
    info = device.get_sensor_information()
    num_rx_antennas = info.get("num_rx_antennas", 3)

    metrics = FmcwMetrics(
        range_resolution_m=0.15,
        max_range_m=4.8,
        max_speed_m_s=2.45,
        speed_resolution_m_s=0.2,
        center_frequency_Hz=60_750_000_000,
    )
    sequence = device.create_simple_sequence(FmcwSimpleSequenceConfig())
    sequence.loop.repetition_time_s = 1 / float(frate)
    chirp_loop = sequence.loop.sub_sequence.contents
    device.sequence_from_metrics(metrics, chirp_loop)
    chirp = chirp_loop.loop.sub_sequence.contents.chirp
    chirp.sample_rate_Hz = 1_000_000
    chirp.rx_mask = (1 << num_rx_antennas) - 1
    chirp.tx_mask = 1
    chirp.tx_power_level = 31
    chirp.if_gain_dB = 33
    chirp.lp_cutoff_Hz = 500000
    chirp.hp_cutoff_Hz = 80000
    device.set_acquisition_sequence(sequence)
    return int(num_rx_antennas)

def discover_candidates() -> list:
    cands = []
    if ARGS.port0:
        cands.append(ARGS.port0)
    if ARGS.port1:
        cands.append(ARGS.port1)
    for getter_name in ("get_list", "scan"):
        getter = getattr(DeviceFmcw, getter_name, None)
        if getter:
            try:
                items = getter() or []
                for it in items:
                    if isinstance(it, str):
                        cands.append(it)
                    elif isinstance(it, dict):
                        for key in ("uri", "id", "uuid", "serial", "device_id"):
                            if key in it and it[key]:
                                cands.append(str(it[key]))
                                break
                    else:
                        try:
                            cands.append(str(it))
                        except Exception:
                            pass
            except Exception:
                pass
    cands.extend([f"usb:{i}" for i in range(4)])
    cands.extend([str(i) for i in range(4)])
    out, seen = [], set()
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def try_open_single(hint: str) -> Tuple[Optional[DeviceFmcw], Optional[str]]:
    trials = [hint]
    if isinstance(hint, str) and hint.isalnum() and len(hint) >= 4:
        trials += [f"usb:{hint}", f"uuid:{hint}"]
    if hint in ("0", "1", "2", "3"):
        trials += [f"usb:{hint}"]
    for t in trials:
        try:
            dev = DeviceFmcw(t)
            _ = dev.get_sensor_information()
            return dev, t
        except Exception:
            pass
        try:
            os.environ["IFX_DEVICE_URI"] = t
            dev = DeviceFmcw()
            _ = dev.get_sensor_information()
            return dev, t
        except Exception:
            pass
    return None, None

def open_two_devices() -> Tuple[DeviceFmcw, str, DeviceFmcw, str]:
    candidates = discover_candidates()
    opened = []
    tried = []
    for cand in candidates:
        if len(opened) >= 2:
            break
        dev, used = try_open_single(cand)
        tried.append(cand)
        if dev is not None:
            try:
                info = dev.get_sensor_information()
                uid = info.get("uuid") or info.get("serial_number") or info.get("device_id") or used
            except Exception:
                uid = used
            if any(uid == x[2] for x in opened):
                try:
                    dev.close()
                except Exception:
                    pass
                continue
            opened.append((dev, used, uid))
    if len(opened) < 2:
        raise SystemExit(f"ERROR: Need two FMCW devices. Tried: {tried}. Found {len(opened)}.")
    return opened[0][0], opened[0][1], opened[1][0], opened[1][1]

# ------------------------- Processing -------------------------

class RadarProcessor:
    def __init__(self, num_chirps_per_frame: int, num_samples: int,
                 mti_alpha: float = 0.8,
                 f_start: float = 60_250_000_000.0, f_end: float = 61_250_000_000.0):
        self.num_chirps_per_frame = num_chirps_per_frame
        self.num_samples = num_samples
        self.mti_alpha = mti_alpha

        try:
            self.range_window = signal.blackmanharris(self.num_samples).reshape(1, self.num_samples)
        except AttributeError:
            self.range_window = signal.windows.blackmanharris(self.num_samples).reshape(1, self.num_samples)

        bandwidth_hz = abs(f_end - f_start)
        fft_size = self.num_samples * 2
        self.range_bin_length = constants.c / (2 * bandwidth_hz * (fft_size / self.num_samples))

    def _fft_spectrum(self, mat: np.ndarray) -> np.ndarray:
        mat = mat - np.average(mat, axis=1, keepdims=True)
        mat = np.multiply(mat, self.range_window)
        zp = np.pad(mat, ((0, 0), (0, self.num_samples)), mode='constant')
        fft = np.fft.fft(zp, axis=1) / self.num_samples
        return 2 * fft[:, :self.num_samples]

    def compute_distance(self, chirp_data: np.ndarray):
        rf = self._fft_spectrum(chirp_data)
        mag = np.abs(rf)
        phase = np.angle(rf)
        dist = mag.mean(axis=0)
        skip = 8
        peak_bin = int(np.argmax(dist[skip:]) + skip)
        phase_at_peak = float(np.mean(phase[:, peak_bin]))
        return dist, phase_at_peak, peak_bin, rf

def bandpass_filter_1d(sig_in, lowcut, highcut, fs, order=4):
    sig_in = np.asarray(sig_in, dtype=np.float64)
    if np.allclose(sig_in, 0):
        return sig_in
    nyq = 0.5 * float(fs)
    lo = max(1e-6, float(lowcut) / nyq)
    hi = min(0.999, float(highcut) / nyq)
    if hi <= lo:
        return sig_in - np.mean(sig_in)
    try:
        b, a = butter(order, [lo, hi], btype='band')
        if sig_in.size < max(len(a), len(b)) * 3:
            return sig_in - np.mean(sig_in)
        return filtfilt(b, a, sig_in)
    except Exception:
        return sig_in - np.mean(sig_in)

def unwrap_and_filter_phase(phase_series, fs):
    raw = np.unwrap(np.asarray(phase_series, dtype=np.float64))
    raw = raw - np.mean(raw)
    resp = bandpass_filter_1d(raw, 0.1, 0.4, fs=fs, order=4)
    heart = bandpass_filter_1d(raw, 0.8, 2.3, fs=fs, order=4)
    return resp, heart, raw

def estimate_bpm(sig_in, fs):
    sig_in = np.asarray(sig_in, dtype=np.float64)
    if sig_in.size < 4:
        return 0.0
    fft = np.abs(np.fft.fft(sig_in))
    freqs = np.fft.fftfreq(sig_in.size, d=1.0 / float(fs))
    mask = freqs > 0
    if not np.any(mask):
        return 0.0
    peak = freqs[mask][np.argmax(fft[mask])]
    return float(peak) * 60.0

# ------------------------- Collection control (shared across threads) -------------------------
COLLECT_REQUESTED = threading.Event()   # acquisition thread should release radars ASAP
RADARS_RELEASED   = threading.Event()   # acquisition thread confirms radars are closed

# ------------------------- Shared state -------------------------

class SharedState:
    def __init__(self, num_samples: int):
        self.lock = threading.Lock()
        self.range_axis_m = np.arange(num_samples, dtype=np.float32)
        self.dev0 = [np.zeros(num_samples, dtype=np.float32) for _ in range(3)]
        self.dev1 = [np.zeros(num_samples, dtype=np.float32) for _ in range(3)]
        self.vitals = {"resp_bpm": 0.0, "heart_bpm": 0.0}
        self.last_ts = time.time()
        self.frame_idx = 0
        self.overlay_text = ""
        self.overlay_lines = ("", "")

        self.motion = {
            "dev0": {"score": 0.0, "state": "INIT", "on_cnt": 0, "off_cnt": 0, "cooldown_until": 0.0},
            "dev1": {"score": 0.0, "state": "INIT", "on_cnt": 0, "off_cnt": 0, "cooldown_until": 0.0},
        }
        # Discovered radar URIs (for pinning collector subprocess device order)
        self.motion["uri0"] = ""
        self.motion["uri1"] = ""

        self.voice = {
            "enabled": bool(ARGS.voice_enable),
            "session_state": "IDLE",  # IDLE, BUSY, LOCKED, ENROLLING
            "locked": False,
            "last_any_motion_ts": 0.0,
            "last_no_motion_ts": 0.0,
            "last_user": "",
            "last_score": 0.0,
            "last_msg": "",
            "enrolled_last": "",
            "collect_running": False,
            "collect_user": "",
            "collect_last_rc": None,
        }

STATE = SharedState(num_samples=int(ARGS.num_samples))

# ------------------------- Motion detection helpers -------------------------

def _range_gate_indices(range_axis_m: np.ndarray, rmin: float, rmax: float):
    if range_axis_m is None or len(range_axis_m) == 0:
        return 0, 0
    i0 = int(np.searchsorted(range_axis_m, rmin, side="left"))
    i1 = int(np.searchsorted(range_axis_m, rmax, side="right"))
    i0 = max(0, min(i0, len(range_axis_m)))
    i1 = max(i0, min(i1, len(range_axis_m)))
    return i0, i1

def compute_motion_energy(curr_ch, prev_ch, i0: int, i1: int) -> float:
    if prev_ch is None:
        return 0.0
    energies = []
    for k in range(3):
        c = curr_ch[k]
        p = prev_ch[k]
        if c is None or p is None:
            continue
        d = (c[i0:i1] - p[i0:i1]) if (i1 > i0) else (c - p)
        energies.append(float(np.mean(np.abs(d))))
    return float(np.mean(energies)) if energies else 0.0

def update_motion_fsm(motion_rec: dict, score_raw: float, now: float, score_valid: bool):
    if not score_valid:
        motion_rec["score"] = 0.0
        motion_rec["state"] = "INIT"
        motion_rec["on_cnt"] = 0
        motion_rec["off_cnt"] = 0
        motion_rec["cooldown_until"] = 0.0
        return

    if motion_rec.get("state") == "INIT":
        motion_rec["state"] = "QUIET"
        motion_rec["on_cnt"] = 0
        motion_rec["off_cnt"] = 0
        motion_rec["cooldown_until"] = 0.0

    a = float(ARGS.motion_alpha)
    prev = float(motion_rec.get("score", 0.0))
    score = (a * float(score_raw)) + ((1.0 - a) * prev)
    motion_rec["score"] = score

    state = motion_rec.get("state", "QUIET")
    on_thr = float(ARGS.motion_on)
    off_thr = float(ARGS.motion_off)
    enter_n = int(ARGS.motion_enter_frames)
    exit_n = int(ARGS.motion_exit_frames)
    cooldown_s = float(ARGS.motion_cooldown_s)

    if state == "COOLDOWN":
        if now >= float(motion_rec.get("cooldown_until", 0.0)):
            motion_rec["state"] = "QUIET"
        return

    if state not in ("QUIET", "MOTION"):
        motion_rec["state"] = "QUIET"
        state = "QUIET"

    if state == "QUIET":
        motion_rec["on_cnt"] = (int(motion_rec.get("on_cnt", 0)) + 1) if (score >= on_thr) else 0
        if motion_rec["on_cnt"] >= enter_n:
            motion_rec["state"] = "MOTION"
            motion_rec["on_cnt"] = 0
            motion_rec["off_cnt"] = 0

    elif state == "MOTION":
        motion_rec["off_cnt"] = (int(motion_rec.get("off_cnt", 0)) + 1) if (score <= off_thr) else 0
        if motion_rec["off_cnt"] >= exit_n:
            motion_rec["state"] = "COOLDOWN"
            motion_rec["cooldown_until"] = now + cooldown_s
            motion_rec["off_cnt"] = 0
            motion_rec["on_cnt"] = 0

# ------------------------- Voice: enrollment + identification -------------------------

def tts_say(text: str) -> None:
    """Speak text offline.

    Preference order:
      1) Piper (neural TTS, offline) if a model is available
      2) espeak-ng fallback
    Environment overrides:
      - PIPER_BIN: full path to piper executable (optional)
      - PIPER_MODEL: full path to voice .onnx model (expects sidecar .onnx.json)
    Default model discovery:
      - ./piper-model/*.onnx (folder next to this script)
    """
    text = (text or "").strip()
    if not text:
        return

    # --- Try Piper first ---
    try:
        piper_bin = (os.environ.get("PIPER_BIN") or "").strip()

        if not piper_bin:
            # 1) Look for piper on PATH (works when venv is activated)
            piper_bin = shutil.which("piper") or ""

        if not piper_bin:
            # 2) If running via an absolute venv python (without activation),
            #    piper is often in the same bin dir.
            try:
                cand = Path(sys.executable).resolve().parent / "piper"
                if cand.exists():
                    piper_bin = str(cand)
            except Exception:
                pass

        if not piper_bin:
            # 3) Project-local venv: ./.venv/bin/piper (systemd/cron friendly)
            try:
                here = Path(__file__).resolve().parent
                for cand in (
                    here / ".venv" / "bin" / "piper",
                    here.parent / ".venv" / "bin" / "piper",
                ):
                    if cand.exists():
                        piper_bin = str(cand)
                        break
            except Exception:
                pass

        model = (os.environ.get("PIPER_MODEL") or "").strip()

        if not model:
            # Look for a local model folder next to this script: ./piper-model/*.onnx
            model_dir = Path(__file__).with_name("piper-model")
            if model_dir.exists():
                onnx_files = sorted(model_dir.glob("*.onnx"))
                if onnx_files:
                    model = str(onnx_files[0])

        if piper_bin and model:
            model_json = model + ".json"
            if Path(piper_bin).exists() and Path(model).exists() and Path(model_json).exists():
                out_wav = str(
                    Path(tempfile.gettempdir())
                    / f"piper_tts_{os.getpid()}_{int(time.time()*1000)}.wav"
                )
                #cmd = [piper_bin, "--model", model, "--output_file", out_wav]
                cmd = [piper_bin, "--model", model, "--length-scale", "1.2", "--output_file", out_wav]
                cp = subprocess.run(cmd, input=text, text=True, capture_output=True, check=False)

                if cp.returncode == 0 and Path(out_wav).exists():
                    player = shutil.which("aplay") or shutil.which("paplay") or ""
                    if player:
                        subprocess.run([player, out_wav], check=False)
                    else:
                        print(
                            "[TTS][WARN] No audio player found (aplay/paplay). "
                            "Install 'alsa-utils' or configure audio playback."
                        )
                    try:
                        Path(out_wav).unlink(missing_ok=True)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    return
                else:
                    err = (cp.stderr or cp.stdout or "").strip()
                    if err:
                        print(f"[TTS][WARN] Piper failed (rc={cp.returncode}): {err[:200]}")

    except Exception as e:
        print(f"[TTS][WARN] Piper exception: {e}")

    # --- Fallback: espeak-ng ---
    try:
        subprocess.run(["espeak-ng", text], check=False)
    except FileNotFoundError:
        print("[TTS][WARN] espeak-ng not found. Install: sudo apt install -y espeak-ng")

def play_beep() -> None:
    """Play a short, fixed beep on the default output device.

    Purpose: provide an explicit 'start' cue for the user so latency can be
    estimated offline from the recorded WAV files (no VAD latency logic).
    """
    try:
        fs = 48000
        dur_s = 0.15
        freq_hz = 1500.0
        t = np.arange(int(fs * dur_s), dtype=np.float32) / float(fs)
        y = 0.20 * np.sin(2.0 * np.pi * float(freq_hz) * t)
        sd.play(y, samplerate=fs, blocking=True)
    except Exception as e:
        print(f"[BEEP][WARN] Failed to play beep: {e}")

def maybe_beep() -> None:
    if bool(getattr(ARGS, 'beep_enable', False)):
        play_beep()

def _safe_device_arg(dev):
    if dev is None:
        return None
    try:
        s = str(dev).strip()
    except Exception:
        return dev
    if s == "":
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return s
    return s

# ------------------------- Connectivity helpers -------------------------

def _run_cmd(cmd: List[str], timeout_s: float = 1.0) -> Tuple[int, str]:
    """Run a command and return (rc, stdout). Never raises."""
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_s), check=False)
        out = (cp.stdout or "").strip()
        return int(cp.returncode), out
    except Exception:
        return 1, ""

def _wifi_ssid(iface: str, timeout_s: float = 1.0) -> str:
    """Best-effort SSID lookup. Empty string means 'not associated' or unknown."""
    iface = (iface or "").strip() or "wlan0"

    # iwgetid (most common on Pi)
    rc, out = _run_cmd(["iwgetid", "-r", "-i", iface], timeout_s=timeout_s)
    if rc == 0 and out:
        return out.strip()

    # iw (fallback)
    rc, out = _run_cmd(["iw", "dev", iface, "link"], timeout_s=timeout_s)
    if rc == 0 and out:
        # Example: "Connected to xx:xx:xx:xx:xx:xx" and "SSID: MyWifi"
        m = re.search(r"\bSSID:\s*(.+)$", out, flags=re.MULTILINE)
        if m:
            return (m.group(1) or "").strip()

    # nmcli (if NetworkManager is used)
    rc, out = _run_cmd(["nmcli", "-t", "-f", "ACTIVE,SSID,DEVICE", "dev", "wifi"], timeout_s=timeout_s)
    if rc == 0 and out:
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0].strip().lower() == "yes" and parts[2].strip() == iface:
                return parts[1].strip()

    return ""

def _has_ipv4_and_route(iface: str, timeout_s: float = 1.0) -> Tuple[bool, bool]:
    """Return (has_ipv4, has_default_route_for_iface)."""
    iface = (iface or "").strip() or "wlan0"
    rc, out = _run_cmd(["ip", "-4", "addr", "show", "dev", iface], timeout_s=timeout_s)
    has_ip = (rc == 0) and (" inet " in f" {out} ")

    rc, out = _run_cmd(["ip", "route", "show", "default", "dev", iface], timeout_s=timeout_s)
    has_def = (rc == 0) and ("default" in out)
    return bool(has_ip), bool(has_def)

def classify_connectivity(
    iface: str,
    host: str,
    port: int,
    timeout_s: float = 1.0,
    attempts: int = 2,
) -> Tuple[str, Dict[str, str]]:
    """
    Classify connectivity into:
      - WIFI_DOWN: not associated to an AP (or interface down)
      - WIFI_NO_IP: associated but missing IPv4 or default route
      - NO_INTERNET: local network seems up, but cannot reach the internet host:port
      - OK: reachable
    Returns (status, details)
    """
    details: Dict[str, str] = {}
    iface = (iface or "").strip() or "wlan0"
    host = (host or "").strip() or "s3.amazonaws.com"
    port = int(port) if port else 443

    ssid = _wifi_ssid(iface, timeout_s=min(1.0, float(timeout_s)))
    details["iface"] = iface
    details["ssid"] = ssid or ""

    if not ssid:
        return "WIFI_DOWN", details

    has_ip, has_def = _has_ipv4_and_route(iface, timeout_s=min(1.0, float(timeout_s)))
    details["has_ipv4"] = "1" if has_ip else "0"
    details["has_default_route"] = "1" if has_def else "0"

    if (not has_ip) or (not has_def):
        return "WIFI_NO_IP", details

    # Internet reachability: short, bounded TCP connect (includes DNS resolution)
    tmo = max(0.2, float(timeout_s))
    attempts = max(1, int(attempts))
    last_err = ""
    for _ in range(attempts):
        try:
            with socket.create_connection((host, port), timeout=tmo):
                return "OK", details
        except socket.gaierror as e:
            last_err = f"DNS:{e}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.05)

    details["reachability_error"] = last_err
    return "NO_INTERNET", details

def connectivity_voice_message(status: str) -> str:
    status = (status or "").strip().upper()
    if status == "WIFI_DOWN":
        return "WiFi is not connected. Please connect me to your home WiFi."
    if status == "WIFI_NO_IP":
        return "WiFi is connected, but I cannot get a network address. Please check your router or WiFi settings."
    if status == "NO_INTERNET":
        return "I'm connected to WiFi, but there's no internet access. Please check your internet connection."
    return "No internet connection. Please check."

# ------------------------- Voice drift helpers -------------------------

def _yyyymmdd(ts: Optional[float] = None) -> str:
    dt = datetime.datetime.fromtimestamp(ts or time.time())
    return dt.strftime("%Y%m%d")

def _s3_session_and_client():
    """Return (session, s3_client) or (None, None) if boto3 is unavailable."""
    if not BOTO3_AVAILABLE or boto3 is None:
        return None, None
    session_kwargs = {}
    if getattr(ARGS, 'aws_profile', None):
        session_kwargs['profile_name'] = str(ARGS.aws_profile)
    session = boto3.Session(**session_kwargs)
    if getattr(ARGS, 'aws_region', None):
        s3 = session.client('s3', region_name=str(ARGS.aws_region))
    else:
        s3 = session.client('s3')
    return session, s3

def _s3_key(*parts: str) -> str:
    """Join parts into an S3 key, applying --s3-prefix if provided."""
    clean_parts = [str(p).strip('/').replace('\\', '/') for p in parts if p not in (None, "")]
    path = "/".join(clean_parts)
    prefix = (getattr(ARGS, 's3_prefix', None) or '').strip('/')
    return path if not prefix else f"{prefix}/{path}"

def _s3_exists(bucket: str, key: str) -> bool:
    _session, s3 = _s3_session_and_client()
    if s3 is None:
        return False
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False

def voice_drift_marker_key(user_name: str, day: str) -> str:
    user_name = _sanitize_name(user_name) or "stranger"
    return _s3_key(user_name, f"{user_name}-{day}", "voice_drift", "_DONE.json")

def voice_drift_done_today(user_name: str, day: Optional[str] = None) -> bool:
    """Check S3 for a per-user, per-day voice drift marker."""
    if not getattr(ARGS, 'voice_drift_once_per_day', True):
        return False
    bucket = getattr(ARGS, 's3_bucket', None)
    if not bucket:
        # Without S3, fall back to a local marker so we still enforce the daily limit.
        day = day or _yyyymmdd()
        marker = Path(tempfile.gettempdir()) / f"voice_drift_done__{_sanitize_name(user_name)}__{day}.json"
        return marker.exists()
    day = day or _yyyymmdd()
    return _s3_exists(str(bucket), voice_drift_marker_key(user_name, day))

def _write_local_daily_marker(user_name: str, day: str, payload: dict) -> None:
    marker = Path(tempfile.gettempdir()) / f"voice_drift_done__{_sanitize_name(user_name)}__{day}.json"
    try:
        marker.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except Exception:
        pass

def _upload_file_to_s3(local_path: str, key: str, content_type: Optional[str] = None) -> bool:
    bucket = getattr(ARGS, 's3_bucket', None)
    if not bucket:
        return False
    _session, s3 = _s3_session_and_client()
    if s3 is None:
        return False
    extra = {'ContentType': content_type} if content_type else None
    try:
        s3.upload_file(local_path, str(bucket), key, ExtraArgs=extra or {})
        return True
    except Exception as e:
        print(f"[VOICE_DRIFT][WARN] S3 upload failed for {local_path} -> s3://{bucket}/{key}: {e}")
        return False

def voice_id_s3_key(voice_name: str) -> str:
    """S3 key for a reference voice-id wav (flat folder: voice_id/<name>.wav)."""
    clean = _sanitize_name(voice_name) or voice_name
    return _s3_key("voice_id", f"{clean}.wav")

def sync_all_voice_ids_to_s3(local_folder: Optional[str] = None) -> int:
    """Sync all local voice-id wavs to S3 if local copy is newer (or missing on S3).

    - Local folder defaults to --voice-folder
    - Destination is s3://<bucket>/<prefix>/voice_id/<stem>.wav
    """
    bucket = getattr(ARGS, 's3_bucket', None)
    if not bucket:
        return 0

    folder = Path(local_folder or getattr(ARGS, 'voice_folder', '')).expanduser()
    if not folder.exists():
        return 0

    _session, s3 = _s3_session_and_client()
    if s3 is None:
        return 0

    uploaded = 0
    skew = datetime.timedelta(seconds=1)

    for p in sorted(folder.glob("*.wav")):
        if not p.is_file():
            continue

        # Use sanitized stem for S3 object name
        stem = _sanitize_name(p.stem) or p.stem
        key = voice_id_s3_key(stem)

        try:
            local_mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime, tz=datetime.timezone.utc)
        except Exception:
            # If stat fails, best-effort attempt upload
            local_mtime = None

        try:
            head = s3.head_object(Bucket=str(bucket), Key=key)
            s3_mtime = head.get("LastModified", None)
        except ClientError:
            s3_mtime = None
        except Exception:
            # Any other error -> do not block the main flow; skip this file
            continue

        should_upload = False
        if s3_mtime is None:
            should_upload = True
        elif local_mtime is None:
            # Can't compare -> don't overwrite
            should_upload = False
        else:
            # Upload only if local is clearly newer
            try:
                if local_mtime > (s3_mtime + skew):
                    should_upload = True
            except Exception:
                should_upload = False

        if should_upload:
            ok = _upload_file_to_s3(str(p), key, content_type="audio/wav")
            if ok:
                uploaded += 1

    if uploaded:
        print(f"[VOICE_ID] Synced {uploaded} local voice-id file(s) to s3://{bucket}/{_s3_key('voice_id')}")
    return uploaded

def record_audio_until_eos(
    fs: int,
    device,
    max_seconds: float,
    min_seconds: float,
    vad_enable: bool,
    vad_mode: int,
    silence_ms: int,
) -> np.ndarray:
    """Record mono audio until end-of-speech (VAD) or max_seconds."""
    max_seconds = float(max_seconds)
    min_seconds = float(min_seconds)
    if max_seconds <= 0:
        raise ValueError("max_seconds must be > 0")

    dev = _safe_device_arg(device)
    channels = 1
    frame_ms = 20  # VAD frame size (10/20/30 ms)
    frame_len = int(fs * (frame_ms / 1000.0))
    if frame_len <= 0:
        frame_len = max(1, int(fs / 50))

    use_vad = bool(vad_enable) and WEBRTCVAD_AVAILABLE and (webrtcvad is not None)
    vad = webrtcvad.Vad(int(vad_mode)) if use_vad else None

    chunks: List[np.ndarray] = []
    started = False
    voiced_frames = 0
    silence_frames = 0
    max_frames = int(np.ceil(max_seconds * fs / frame_len))
    min_frames = int(np.ceil(min_seconds * fs / frame_len))
    silence_target = int(np.ceil((float(silence_ms) / 1000.0) * fs / frame_len))
    if silence_target < 1:
        silence_target = 1

    def _to_int16_bytes(x: np.ndarray) -> bytes:
        x = np.asarray(x, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        i16 = (x * 32767.0).astype(np.int16)
        return i16.tobytes()

    with sd.InputStream(
        samplerate=int(fs),
        channels=channels,
        dtype='float32',
        blocksize=int(frame_len),
        device=dev,
    ) as stream:
        t0 = time.time()
        for _ in range(max_frames):
            data, overflowed = stream.read(frame_len)
            if overflowed:
                print("[VOICE_DRIFT][WARN] Input overflow")
            frame = np.asarray(data, dtype=np.float32).reshape(-1)
            chunks.append(frame.copy())

            is_speech = False
            if use_vad and vad is not None:
                try:
                    is_speech = bool(vad.is_speech(_to_int16_bytes(frame), int(fs)))
                except Exception:
                    is_speech = False
            else:
                # Energy fallback if VAD is unavailable.
                rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
                is_speech = rms >= 0.01

            if is_speech:
                started = True
                voiced_frames += 1
                silence_frames = 0
            else:
                if started:
                    silence_frames += 1

            if started and voiced_frames >= min_frames and silence_frames >= silence_target:
                break

            if (time.time() - t0) >= max_seconds:
                break

    audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
    return audio

def run_voice_drift_session(user_name: str, transcriber: Optional["NameTranscriber"] = None) -> bool:
    """Collect the daily prompted voice drift clips, upload them, and write a done marker."""
    user_name = _sanitize_name(user_name) or "stranger"
    day = _yyyymmdd()

    if getattr(ARGS, 'voice_drift_once_per_day', True) and voice_drift_done_today(user_name, day=day):
        print(f"[VOICE_DRIFT] Already done today for {user_name} ({day}); skipping.")
        return False

    items = [s.strip() for s in str(getattr(ARGS, 'voice_drift_items', '')).split(',') if s.strip()]
    if not items:
        print("[VOICE_DRIFT][WARN] No --voice_drift_items specified; skipping.")
        return False

    prompts = {
        'ah': "Please say ahhhhhhh for about five seconds.",
        'read': "Please read: The quick brown fox jumps over the lazy dog.",
        'count': "Please count from one to ten.",
        'conversation': "Please describe what you did today in a few sentences.",
    }

    # Local temp staging directory (so uploads are robust even if an utterance is retried).
    tmp_root = Path(tempfile.mkdtemp(prefix=f"voice_drift_{user_name}_{day}_"))
    try:
        uploaded = []
        meta = {
            'user': user_name,
            'day': day,
            'voice_fs': int(ARGS.voice_fs),
            'device': str(getattr(ARGS, 'voice_device', '')),
            'items': [],
            'created_at_unix': time.time(),
        }

        tts_say("Let's do our daily test, shall we?")
        # ------------------------------------------------------------
        # Self-feeling (free response) + 1–5 rating
        # Workflow:
        #   1) Ask "How do you feel today?"
        #   2) Record several-sentence response (audio saved later)
        #   3) Ask for 1–5 rating and parse
        #   4) Save step-2 audio as self_feel_{score}.wav (uploaded with voice drift)
        #   5) Hint: "Next, let's do several speech tests."
        # ------------------------------------------------------------
        try:
            # Step 1: free response prompt
            tts_say("How do you feel today? Please respond after the beep.")
            time.sleep(float(getattr(ARGS, 'voice_post_tts_delay_s', 0.35)))

            feel_text_max_seconds = float(getattr(ARGS, "self_feel_text_max_seconds", 10.0))
            feel_text_min_seconds = float(getattr(ARGS, "self_feel_text_min_seconds", 1.2))

            maybe_beep()

            feel_text_audio = record_audio_until_eos(
                fs=int(ARGS.voice_fs),
                device=ARGS.voice_device,
                max_seconds=float(feel_text_max_seconds),
                min_seconds=float(feel_text_min_seconds),
                vad_enable=bool(getattr(ARGS, 'vad_enable', True)),
                vad_mode=int(getattr(ARGS, 'vad_mode', 2)),
                silence_ms=int(getattr(ARGS, 'vad_silence_ms', 1100)),
            )

            # Step 3: rating prompt (we keep this audio only for parsing; not saved unless you want it later)
            tts_say("How would you rate yourself today from one to five? One means terrible and five means wonderful.")
            time.sleep(float(getattr(ARGS, 'voice_post_tts_delay_s', 0.35)))

            feel_rate_retries = int(getattr(ARGS, "self_feel_rate_retries", 3))
            feel_rate_max_seconds = float(getattr(ARGS, "self_feel_rate_max_seconds", 4.0))
            feel_rate_min_seconds = float(getattr(ARGS, "self_feel_rate_min_seconds", 0.6))

            score: Optional[int] = None
            raw_token_last = ""

            for _ in range(max(1, feel_rate_retries)):
                maybe_beep()

                feel_rate_audio = record_audio_until_eos(
                    fs=int(ARGS.voice_fs),
                    device=ARGS.voice_device,
                    max_seconds=float(feel_rate_max_seconds),
                    min_seconds=float(feel_rate_min_seconds),
                    vad_enable=bool(getattr(ARGS, 'vad_enable', True)),
                    vad_mode=int(getattr(ARGS, 'vad_mode', 2)),
                    silence_ms=int(getattr(ARGS, 'vad_silence_ms', 800)),
                )

                if feel_rate_audio.size == 0:
                    tts_say("Sorry, I did not catch that. Please say a number from one to five.")
                    continue

                if transcriber is not None:
                    raw_token_last = transcriber.transcribe_last_token(feel_rate_audio, int(ARGS.voice_fs))
                else:
                    raw_token_last = ""

                score = _parse_1_to_5(raw_token_last)
                if score is not None and 1 <= score <= 5:
                    break

                tts_say("Sorry, I did not catch the number. Please say one, two, three, four, or five.")
                time.sleep(float(getattr(ARGS, 'voice_post_tts_delay_s', 0.35)))

            # If we still cannot parse, default to 3 but still keep the free-response audio.
            if score is None:
                score = int(getattr(ARGS, "self_feel_default_score", 3))

            # Step 4: save the step-2 free-response audio using the parsed score
            if feel_text_audio is not None and feel_text_audio.size > 0:
                feel_name = f"self_feel_{int(score)}.wav"
                local_wav = tmp_root / feel_name
                sf.write(str(local_wav), np.asarray(feel_text_audio, dtype=np.float32), int(ARGS.voice_fs), subtype='PCM_16')

                s3_wav_key = _s3_key(user_name, f"{user_name}-{day}", "voice_drift", feel_name)
                ok = _upload_file_to_s3(str(local_wav), s3_wav_key, content_type='audio/wav')
                uploaded.append({'item': 'self_feel', 'score': int(score), 'local': str(local_wav), 's3_key': s3_wav_key, 'uploaded': bool(ok)})
                meta['items'].append({
                    'item': 'self_feel',
                    'score': int(score),
                    'samples': int(feel_text_audio.shape[0]),
                    'seconds': float(feel_text_audio.shape[0]) / float(int(ARGS.voice_fs)),
                    's3_key': s3_wav_key,
                    'uploaded': bool(ok),
                    'stt_token': str(raw_token_last),
                })

            # Step 5: hint before the standard drift prompts
            tts_say("Next, let's do several speech tests.")
        except Exception as e:
            print(f"[SELF_FEEL][WARN] self-feel capture failed: {e}")

        
        for item in items:
            key_name = item
            prompt = prompts.get(item, f"Please speak for prompt: {item}.")            
            tts_say(prompt)
            time.sleep(float(getattr(ARGS, 'voice_post_tts_delay_s', 0.35)))

            maybe_beep()

            audio = record_audio_until_eos(
                fs=int(ARGS.voice_fs),
                device=ARGS.voice_device,
                max_seconds=float(ARGS.voice_drift_max_seconds),
                min_seconds=float(ARGS.voice_drift_min_seconds),
                vad_enable=bool(getattr(ARGS, 'vad_enable', True)),
                vad_mode=int(getattr(ARGS, 'vad_mode', 2)),
                silence_ms=int(getattr(ARGS, 'vad_silence_ms', 800)),
            )

            if audio.size == 0:
                print(f"[VOICE_DRIFT][WARN] Empty audio for item={item}; continuing.")
                continue

            local_wav = tmp_root / f"{key_name}.wav"
            sf.write(str(local_wav), audio, int(ARGS.voice_fs), subtype='PCM_16')

            s3_wav_key = _s3_key(user_name, f"{user_name}-{day}", "voice_drift", f"{key_name}.wav")
            ok = _upload_file_to_s3(str(local_wav), s3_wav_key, content_type='audio/wav')
            uploaded.append({'item': item, 'local': str(local_wav), 's3_key': s3_wav_key, 'uploaded': bool(ok)})
            meta['items'].append({
                'item': item,
                'samples': int(audio.shape[0]),
                'seconds': float(audio.shape[0]) / float(int(ARGS.voice_fs)),
                's3_key': s3_wav_key,
                'uploaded': bool(ok),
            })

        tts_say("That's all for today's test, thank you! Have a wonderful day!")
        
        # Done marker.
        marker_payload = {
            'schema': 'voice_drift_done_v1',
            'user': user_name,
            'day': day,
            'time': time.strftime('%H:%M:%S'),
            'items': meta['items'],
            'created_at_unix': time.time(),
        }

        marker_local = tmp_root / "_DONE.json"
        marker_local.write_text(json.dumps(marker_payload, indent=2), encoding='utf-8')
        marker_key = voice_drift_marker_key(user_name, day)
        marker_uploaded = _upload_file_to_s3(str(marker_local), marker_key, content_type='application/json')
        if not marker_uploaded:
            _write_local_daily_marker(user_name, day, marker_payload)

        print(f"[VOICE_DRIFT] Completed for {user_name} ({day}). Marker: {marker_key}")
        return True
    finally:
        # Keep temp dir by default for debugging; delete if you prefer.
        # shutil.rmtree(tmp_root, ignore_errors=True)
        pass

def trigger_collection_for_user(user_name: str):
    global COLLECTOR_PROC

    """
    Run the raw collection script for a bounded duration, after forcing the radar thread to release devices.

    While the collector is running, print a live countdown of remaining time. Collector stdout/stderr is
    streamed to this console as well.
    """
    if not ARGS.collect_enable:
        return

    user_name = _sanitize_name(user_name) or "stranger"

    # Ask the radar acquisition thread to release devices, then wait.
    RADARS_RELEASED.clear()
    COLLECT_REQUESTED.set()

    ok = RADARS_RELEASED.wait(timeout=10.0)
    if not ok:
        print("[COLLECT][WARN] Timed out waiting for radars to be released; proceeding anyway (collector may fail).")

    # Build collector command.
    script_path = str(Path(ARGS.collect_script).expanduser())
    cmd = [
        sys.executable, script_path,
        "--max-run-seconds", str(int(ARGS.collect_seconds)),
        "--segment-minutes", str(int(ARGS.collect_segment_minutes)),
        "--user", str(user_name),
        "--frate", str(int(float(ARGS.frate))),
    ]

    # Forward S3 configuration so the collector and voice-drift uploads share the same destination.
    if getattr(ARGS, 's3_bucket', None):
        cmd += ["--s3-bucket", str(ARGS.s3_bucket)]
    if getattr(ARGS, 's3_prefix', None) not in (None, ""):
        cmd += ["--s3-prefix", str(ARGS.s3_prefix)]
    if getattr(ARGS, 'aws_region', None) not in (None, ""):
        cmd += ["--aws-region", str(ARGS.aws_region)]
    if getattr(ARGS, 'aws_profile', None) not in (None, ""):
        cmd += ["--aws-profile", str(ARGS.aws_profile)]

    # Forward radar URIs discovered by this script (prevents device order swaps).
    with STATE.lock:
        uri0 = str(STATE.motion.get("uri0", "") or "")
        uri1 = str(STATE.motion.get("uri1", "") or "")
    if uri0:
        cmd += ["--port0", uri0]
    if uri1:
        cmd += ["--port1", uri1]

    # Optional: forward audio/video toggles.
    if bool(ARGS.collect_no_video):
        cmd.append("--no-video")
    if ARGS.collect_cam_index is not None:
        cmd += ["--cam-index", str(int(ARGS.collect_cam_index))]
    if bool(ARGS.collect_audio):
        cmd.append("--audio")
        cmd += ["--audio-rate", str(int(ARGS.collect_audio_rate))]
        cmd += ["--audio-channels", str(int(ARGS.collect_audio_channels))]
        if ARGS.collect_audio_device not in (None, ""):
            cmd += ["--audio-device", str(ARGS.collect_audio_device)]

    print("[COLLECT] Launching:", " ".join(cmd))

    with STATE.lock:
        STATE.voice["collect_running"] = True
        STATE.voice["collect_user"] = user_name
        STATE.voice["collect_last_rc"] = None
        STATE.voice["last_msg"] = f"Collecting for {int(ARGS.collect_seconds)}s as '{user_name}'..."

    rc = -1
    start_t = time.time()
    end_t = start_t + float(ARGS.collect_seconds)
    hard_timeout = float(max(int(ARGS.collect_timeout_s), int(ARGS.collect_seconds) + 30))

    try:
        # Collector runs in its own process group so the gpio killpg does not hit it.
        # stdout/stderr inherit the terminal (no pipe) so the collector is not killed
        # by SIGPIPE if this process exits while the collector is still uploading.
        proc = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
        )

        with COLLECTOR_LOCK:
            COLLECTOR_PROC = proc

        last_print = 0.0
        while True:
            if STOP_EVENT.is_set():
                print("[COLLECT] Stop requested → sending SIGINT to collector...")
                try:
                    proc.send_signal(signal.SIGINT)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=120.0)
                except Exception:
                    try:
                        proc.terminate()
                        proc.wait(timeout=10.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                rc = int(proc.returncode or 0)
                break

            if proc.poll() is not None:
                rc = int(proc.returncode or 0)
                break

            now = time.time()
            elapsed = now - start_t
            if elapsed >= hard_timeout:
                print("[COLLECT][ERROR] Hard timeout reached; terminating collector.")
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                rc = -9
                break

            remaining = max(0.0, end_t - now)

            # Print every 10s, and every 1s in last 30s.
            interval = 1.0 if remaining <= 30.0 else 10.0
            if (now - last_print) >= interval:
                mm = int(remaining) // 60
                ss = int(remaining) % 60
                print(f"[COLLECT] Remaining: {mm:02d}:{ss:02d}")
                last_print = now

            time.sleep(0.2)

        print(f"[COLLECT] Collector exited rc={rc}")
        with COLLECTOR_LOCK:

            COLLECTOR_PROC = None

    except FileNotFoundError as e:
        rc = -2
        print(f"[COLLECT][ERROR] Collector script not found: {e}")
    except Exception as e:
        rc = -1
        print(f"[COLLECT][ERROR] Collector launch failed: {e}")

    with STATE.lock:
        STATE.voice["collect_running"] = False
        STATE.voice["collect_last_rc"] = rc
        STATE.voice["last_msg"] = f"Collection finished for '{user_name}' (rc={rc})."

    # Allow radar acquisition thread to resume.
    COLLECT_REQUESTED.clear()

def record_audio(seconds: float, fs: int, device) -> np.ndarray:
    dev = _safe_device_arg(device)
    n = int(max(1, round(float(seconds) * float(fs))))
    rec = sd.rec(frames=n, samplerate=int(fs), channels=1, dtype='float32', device=dev)
    sd.wait()
    return rec.reshape(-1)

def _ts_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _sanitize_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _parse_yes_no(token: str) -> Optional[bool]:
    """Best-effort parsing for a yes/no spoken response."""
    t = (token or "").strip().lower()
    if not t:
        return None
    yes = {"yes", "yeah", "yep", "correct", "right", "sure", "affirmative", "ok", "okay"}
    no = {"no", "nope", "nah", "incorrect", "wrong"}
    if t in yes:
        return True
    if t in no:
        return False
    return None

def _parse_1_to_5(token: str) -> Optional[int]:
    """Best-effort parsing for a 1–5 spoken response (digits or words)."""
    t = (token or "").strip().lower()
    if not t:
        return None
    mapping = {
        "1": 1, "one": 1,
        "2": 2, "two": 2, "to": 2, "too": 2,
        "3": 3, "three": 3,
        "4": 4, "four": 4, "for": 4,
        "5": 5, "five": 5,
    }
    if t in mapping:
        return mapping[t]
    # Fallback: extract a digit if present
    for ch in t:
        if ch in "12345":
            return int(ch)
    return None

class NameTranscriber:
    """Optional offline name transcription (Vosk). If unavailable, returns empty string."""
    def __init__(self, model_path: str):
        self.model = None
        if not VOSK_AVAILABLE:
            return
        p = Path(model_path)
        if not p.exists():
            return
        try:
            self.model = VoskModel(str(p))
            print(f"[ENROLL] Vosk model loaded: {p.resolve()}")
        except Exception as e:
            print(f"[ENROLL][WARN] Failed to load Vosk model: {e}")
            self.model = None

    def transcribe_last_token(self, audio: np.ndarray, sr: int) -> str:
        if self.model is None:
            return ""
        # Vosk expects 16k int16 PCM bytes
        target_sr = 16000
        x = np.asarray(audio, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        if sr != target_sr:
            import math
            g = math.gcd(int(sr), int(target_sr))
            up = int(target_sr // g)
            down = int(sr // g)
            x = resample_poly(x, up=up, down=down).astype(np.float32, copy=False)
        pcm = (x * 32767.0).astype(np.int16).tobytes()

        rec = KaldiRecognizer(self.model, target_sr)
        rec.SetWords(False)
        rec.AcceptWaveform(pcm)
        try:
            j = json.loads(rec.FinalResult())
            text = (j.get("text") or "").strip()
        except Exception:
            text = ""
        toks = [t for t in text.split() if t]
        return toks[-1] if toks else ""

class SpeakerId:
    """SpeechBrain-based speaker identification (cache embeddings once, compare many)."""

    def __init__(self, folder: str, model_dir: str = "speechbrain_pretrained_models/spkrec-ecapa-voxceleb"):
        self.folder = Path(folder)
        self.model_dir = model_dir

        self.ref_paths: List[Path] = []
        self.ref_names: List[str] = []
        self.verification = None
        self.ref_embeds: List[torch.Tensor] = []
        self.device = "cpu"

    def _init_model(self) -> None:
        if not SPEECHBRAIN_AVAILABLE:
            raise RuntimeError(f"SpeechBrain import failed: {_SPEECHBRAIN_IMPORT_ERR}")

        # Best-effort: torchaudio backend preference
        try:
            import torchaudio  # type: ignore
            if hasattr(torchaudio, "set_audio_backend"):
                try:
                    torchaudio.set_audio_backend("soundfile")
                except Exception:
                    pass
        except Exception:
            pass

        self.verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=self.model_dir,
        )
        try:
            self.device = str(getattr(self.verification, "device", "cpu"))
        except Exception:
            self.device = "cpu"

    @staticmethod
    def _to_mono_float32(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 2:
            x = x.mean(axis=1)
        return x.reshape(-1).astype(np.float32)

    @staticmethod
    def _resample_to_16k(x: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        if orig_sr == target_sr or x.size == 0:
            return x.astype(np.float32, copy=False)
        import math
        g = math.gcd(int(orig_sr), int(target_sr))
        up = int(target_sr // g)
        down = int(orig_sr // g)
        y = resample_poly(x, up=up, down=down)
        return y.astype(np.float32, copy=False)

    @staticmethod
    def _normalize_audio(x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        mx = float(np.max(np.abs(x))) + 1e-9
        return (x / mx).astype(np.float32, copy=False)

    def _embed_16k(self, x16k: np.ndarray) -> torch.Tensor:
        if self.verification is None:
            raise RuntimeError("SpeechBrain model not initialized")

        if x16k.size == 0:
            if self.ref_embeds:
                return torch.zeros_like(self.ref_embeds[0])
            return torch.zeros(192, device=self.device)

        wav = torch.from_numpy(x16k).to(torch.float32)
        if self.device != "cpu":
            wav = wav.to(self.device)
        wav = wav.unsqueeze(0)

        with torch.no_grad():
            emb = self.verification.encode_batch(wav)
            emb = torch.as_tensor(emb).squeeze()
            if emb.ndim > 1:
                emb = emb.reshape(-1)
            emb = F.normalize(emb, dim=0)
        return emb

    def load_reference(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)

        wavs = sorted([p for p in self.folder.glob("*.wav") if p.is_file()])
        if not wavs:
            raise FileNotFoundError(f"No .wav files found in {self.folder.resolve()}")

        self.ref_paths = wavs
        self.ref_names = [p.stem for p in wavs]

        self._init_model()

        self.ref_embeds = []
        for p in self.ref_paths:
            try:
                x, sr = sf.read(str(p), dtype="float32", always_2d=False)
                x = self._to_mono_float32(x)
                x = self._resample_to_16k(x, orig_sr=int(sr), target_sr=16000)
                x = self._normalize_audio(x)
                emb = self._embed_16k(x)
                self.ref_embeds.append(emb.detach().cpu())
            except Exception as e:
                print(f"[VOICE][WARN] Failed to embed reference {p.name}: {e}")
                if self.ref_embeds:
                    self.ref_embeds.append(torch.zeros_like(self.ref_embeds[0]).cpu())
                else:
                    self.ref_embeds.append(torch.zeros(192).cpu())

        print(f"[VOICE] Loaded {len(self.ref_names)} speaker(s) from {self.folder.resolve()} (cached embeddings)")

    def add_reference(self, name: str, audio: np.ndarray, sr: int) -> bool:
        """Enroll new speaker: save wav handled outside; here we embed and add/update memory DB."""
        if self.verification is None:
            print("[ENROLL][WARN] Model not initialized; cannot embed enrollment.")
            return False

        clean = _sanitize_name(name)
        if not clean:
            return False

        x = self._to_mono_float32(audio)
        x = self._resample_to_16k(x, orig_sr=int(sr), target_sr=16000)
        x = self._normalize_audio(x)
        try:
            emb = self._embed_16k(x).detach().cpu()
        except Exception as e:
            print(f"[ENROLL][WARN] embedding enrollment failed: {e}")
            return False

        if clean in self.ref_names:
            idx = self.ref_names.index(clean)
            self.ref_embeds[idx] = emb
            self.ref_paths[idx] = (self.folder / f"{clean}.wav")
            print(f"[ENROLL] Updated speaker '{clean}' in memory DB")
        else:
            self.ref_names.append(clean)
            self.ref_embeds.append(emb)
            self.ref_paths.append(self.folder / f"{clean}.wav")
            print(f"[ENROLL] Added speaker '{clean}' in memory DB")
        return True

    def identify(self, orig_rate: int, data: np.ndarray, threshold: float, margin: float, debug_scores: bool = True):
        if self.verification is None or not self.ref_embeds:
            return "unknown", float("-inf"), []

        x = self._to_mono_float32(data)
        x = self._resample_to_16k(x, orig_sr=int(orig_rate), target_sr=16000)
        x = self._normalize_audio(x)

        try:
            emb = self._embed_16k(x).detach().cpu()
        except Exception as e:
            print(f"[VOICE][ERROR] embedding failed: {e}")
            return "unknown", float("-inf"), []

        scores: List[Tuple[str, float]] = []
        best_name = "unknown"
        best = float("-inf")

        emb = F.normalize(emb, dim=0)
        for name, ref_emb in zip(self.ref_names, self.ref_embeds):
            try:
                ref_emb = torch.as_tensor(ref_emb)
                ref_emb = F.normalize(ref_emb, dim=0)
                s = float(F.cosine_similarity(emb, ref_emb, dim=0).item())
            except Exception as e:
                s = float("-inf")
                if debug_scores:
                    print(f"[VOICE][WARN] cosine failed for {name}: {e}")
            scores.append((name, s))
            if s > best:
                best = s
                best_name = name

        scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)

        # Hard gate + margin gate
        accept = False
        if scores_sorted:
            second = scores_sorted[1][1] if len(scores_sorted) > 1 else float("-inf")
            if (best >= float(threshold)) and ((best - second) >= float(margin)):
                accept = True

        if debug_scores and scores_sorted:
            msg = " | ".join([f"{n}:{v:.3f}" for n, v in scores_sorted])
            print(f"[VOICE][SCORES] {msg}")

        if not accept:
            return "unknown", best, scores_sorted
        return best_name, best, scores_sorted

def save_enrollment_wav(folder: str, name: str, audio: np.ndarray, sr: int) -> Path:
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = _sanitize_name(name)
    if not clean:
        clean = f"user_{_ts_id()}"
    out_path = out_dir / f"{clean}.wav"
    sf.write(str(out_path), np.asarray(audio, dtype=np.float32), int(sr))
    print(f"[ENROLL] Saved: {out_path.resolve()}")
    return out_path

# ------------------------- Interaction manager -------------------------

def interaction_manager():
    if not ARGS.voice_enable:
        return

    sid = SpeakerId(folder=ARGS.voice_folder)
    try:
        sid.load_reference()
    except Exception as e:
        print(f"[VOICE][ERROR] {e}")
        with STATE.lock:
            STATE.voice["enabled"] = False
            STATE.voice["last_msg"] = f"Voice disabled: {e}"
        return

    unlock_s = float(ARGS.unlock_no_motion_s)
    transcriber = NameTranscriber(ARGS.enroll_vosk_model) if (ARGS.enroll_enable and VOSK_AVAILABLE) else None
    if ARGS.enroll_enable and not VOSK_AVAILABLE:
        print("[ENROLL] Vosk not installed; enrollment will use timestamp-based names.")

    while True:
        time.sleep(0.05)

        with STATE.lock:
            enabled = bool(STATE.voice.get("enabled", False))
            dev0_state = STATE.motion.get("dev0", {}).get("state", "INIT")
            dev1_state = STATE.motion.get("dev1", {}).get("state", "INIT")
            locked = bool(STATE.voice.get("locked", False))
            session_state = str(STATE.voice.get("session_state", "IDLE"))

        if not enabled:
            continue

        any_motion = (dev0_state == "MOTION") or (dev1_state == "MOTION")
        now = time.time()

        # Motion edge detection (QUIET->MOTION) to avoid repeated triggering while motion persists
        with STATE.lock:
            prev_any_motion = bool(STATE.voice.get("prev_any_motion", False))
            STATE.voice["prev_any_motion"] = bool(any_motion)
            offline_cooldown_until = float(STATE.voice.get("offline_cooldown_until", 0.0))
        motion_edge = bool(any_motion) and (not prev_any_motion)

        # Track last motion / no-motion timestamps
        with STATE.lock:
            if any_motion:
                STATE.voice["last_any_motion_ts"] = now
                STATE.voice["last_no_motion_ts"] = 0.0
            else:
                if float(STATE.voice.get("last_no_motion_ts", 0.0)) == 0.0:
                    STATE.voice["last_no_motion_ts"] = now

        # Unlock condition
        if locked and (not any_motion) and session_state != "COLLECTING":
            with STATE.lock:
                t0 = float(STATE.voice.get("last_no_motion_ts", 0.0))
            if t0 > 0.0 and (now - t0) >= unlock_s:
                with STATE.lock:
                    STATE.voice["locked"] = False
                    STATE.voice["session_state"] = "IDLE"
                    STATE.voice["last_msg"] = f"Unlocked (no motion {unlock_s:.0f}s)"
                    STATE.voice["enrolled_last"] = ""
                print(f"[VOICE] stage=UNLOCKED (no motion {unlock_s:.0f}s)")
            continue

        # Trigger interaction
        if motion_edge and (not locked) and session_state == "IDLE":
            # Internet / Wi-Fi gating: only proceed when internet is reachable
            if getattr(ARGS, "netcheck_enable", True):
                if now < offline_cooldown_until:
                    # Cooldown active: do not re-prompt; wait for next motion edge after cooldown
                    continue

                status, details = classify_connectivity(
                    iface=str(getattr(ARGS, "wifi_iface", "wlan0")),
                    host=str(getattr(ARGS, "internet_host", "s3.amazonaws.com")),
                    port=int(getattr(ARGS, "internet_port", 443)),
                    timeout_s=float(getattr(ARGS, "netcheck_timeout_s", 1.0)),
                    attempts=int(getattr(ARGS, "netcheck_attempts", 2)),
                )
                if status != "OK":
                    msg = connectivity_voice_message(status)
                    print(f"[NET] status={status} details={details}")
                    tts_say(msg)
                    with STATE.lock:
                        STATE.voice["last_msg"] = f"Connectivity gate: {status}"
                        STATE.voice["offline_cooldown_until"] = now + float(getattr(ARGS, "offline_cooldown_s", 60.0))
                    continue

            with STATE.lock:
                STATE.voice["session_state"] = "BUSY"
                STATE.voice["last_msg"] = "Prompting..."
            print("[VOICE] stage=BUSY -> prompting")

            # Prompt
            tts_say("How are you?")
            time.sleep(max(0.0, float(ARGS.voice_post_tts_delay_s)))

            # Record + identify
            name = "unknown"
            best = float("-inf")
            try:
                audio = record_audio(float(ARGS.voice_seconds), int(ARGS.voice_fs), ARGS.voice_device)
                name, best, _scores_sorted = sid.identify(
                    orig_rate=int(ARGS.voice_fs),
                    data=audio,
                    threshold=float(ARGS.voice_threshold),
                    margin=float(ARGS.voice_margin),
                    debug_scores=True,
                )
            except Exception as e:
                print(f"[VOICE][ERROR] record/identify failed: {e}")
                name = "unknown"

            enrolled_last = ""
            skip_post_actions = False  # if True, skip drift + collection for this motion session

            # Known path
            if name != "unknown":
                tts_say(f"Hi {name}, nice to see you!")
                msg = f"Identified {name} ({best:.3f})"

            # Unknown path -> stranger + optional enroll
            else:
                msg = f"Unknown ({best:.3f})"

                # Enrollment consent gate (requested): ask before asking for a name.
                consent = False
                if ARGS.enroll_enable:
                    try:
                        post_delay = max(0.0, float(ARGS.voice_post_tts_delay_s))
                        confirm_retries = max(1, int(ARGS.enroll_confirm_retries))
                        confirm_seconds = float(ARGS.enroll_confirm_seconds)

                        for _c in range(confirm_retries):
                            tts_say("Hi stranger, do you want to be enrolled to our health monitoring system?")
                            time.sleep(post_delay)
                            yn_audio = record_audio(confirm_seconds, int(ARGS.voice_fs), ARGS.voice_device)
                            yn_raw = ""
                            if transcriber is not None:
                                yn_raw = transcriber.transcribe_last_token(yn_audio, int(ARGS.voice_fs))
                            decision = _parse_yes_no(yn_raw)
                            if decision is True:
                                consent = True
                                break
                            if decision is False:
                                consent = False
                                break
                            tts_say("Sorry, I did not catch that. Please say yes or no.")

                    except Exception as e:
                        print(f"[ENROLL][WARN] Consent step failed: {e}")
                        consent = False

                if not consent:
                    # User declined or consent could not be obtained -> skip drift/collection and wait for unlock.
                    tts_say("Understood, have a nice day!")
                    skip_post_actions = True
                else:
                    tts_say("Great. Please say your name clearly.")

                if ARGS.enroll_enable and consent:
                    try:
                        # Brief lock: update state only. All I/O (record_audio, tts_say,
                        # transcription) runs OUTSIDE the lock so the acquisition thread
                        # is not starved during the enrollment conversation.
                        with STATE.lock:
                            STATE.voice["session_state"] = "ENROLLING"
                            STATE.voice["last_msg"] = "Enrolling new speaker..."

                        # Name capture + confirmation loop:
                        # 1) Record name
                        # 2) Pi confirms: "Your name is <name>, yes or no?"
                        # 3) If reply is "no", ask for the name again.
                        post_delay = max(0.0, float(ARGS.voice_post_tts_delay_s))
                        max_name_attempts = int(ARGS.enroll_max_name_attempts)
                        confirm_retries = int(ARGS.enroll_confirm_retries)
                        confirm_seconds = float(ARGS.enroll_confirm_seconds)

                        chosen_clean = ""
                        chosen_audio: Optional[np.ndarray] = None

                        for _name_try in range(max_name_attempts):
                            time.sleep(post_delay)
                            name_audio = record_audio(float(ARGS.enroll_seconds), int(ARGS.voice_fs), ARGS.voice_device)
                            chosen_audio = name_audio

                            raw_name = ""
                            if transcriber is not None:
                                raw_name = transcriber.transcribe_last_token(name_audio, int(ARGS.voice_fs))

                            clean = _sanitize_name(raw_name)
                            if len(clean) < int(ARGS.enroll_min_name_len):
                                clean = f"user_{_ts_id()}"

                            decision: Optional[bool] = None
                            for _c in range(confirm_retries):
                                tts_say(f"Your name is {clean}, yes or no?")
                                time.sleep(post_delay)
                                yn_audio = record_audio(confirm_seconds, int(ARGS.voice_fs), ARGS.voice_device)
                                yn_raw = ""
                                if transcriber is not None:
                                    yn_raw = transcriber.transcribe_last_token(yn_audio, int(ARGS.voice_fs))
                                decision = _parse_yes_no(yn_raw)
                                if decision is not None:
                                    break
                                tts_say("Sorry, I did not catch that. Please say yes or no.")

                            if decision is True:
                                chosen_clean = clean
                                break

                            # decision is False or still None -> re-ask name
                            tts_say("Okay, please say your name again.")

                        if not chosen_clean:
                            # Best-effort fallback to a timestamp-based name if we never got a confirmed name.
                            chosen_clean = f"user_{_ts_id()}"

                        if chosen_audio is None:
                            raise RuntimeError("Enrollment audio capture failed (no audio recorded).")

                        _ = save_enrollment_wav(
                            ARGS.voice_folder,
                            chosen_clean,
                            np.asarray(chosen_audio, dtype=np.float32),
                            int(ARGS.voice_fs),
                        )
                        sid.add_reference(chosen_clean, np.asarray(chosen_audio, dtype=np.float32), int(ARGS.voice_fs))

                        enrolled_last = chosen_clean
                        tts_say(f"Nice to meet you, {chosen_clean}. I will remember your voice.")
                        msg = f"Unknown -> enrolled '{chosen_clean}' (prev best={best:.3f})"

                    except Exception as e:
                        print(f"[ENROLL][ERROR] Enrollment failed: {e}")
                        tts_say("Sorry, I could not save your voice this time.")
                        msg = f"Unknown (enrollment failed) ({best:.3f})"

            with STATE.lock:
                STATE.voice["last_user"] = name if name != "unknown" else "stranger"
                STATE.voice["last_score"] = float(best) if np.isfinite(best) else 0.0
                STATE.voice["last_msg"] = msg
                STATE.voice["locked"] = True
                STATE.voice["session_state"] = "LOCKED"
                STATE.voice["enrolled_last"] = enrolled_last

            # NEW: daily voice-drift prompt set, then raw collection.
            final_user = name if name != "unknown" else (enrolled_last if enrolled_last else "stranger")

            # If the user declined enrollment (or consent was not obtained), skip drift + collection.
            if skip_post_actions:
                print("[VOICE] Post-actions skipped (enrollment declined).")
                print("[VOICE] stage=LOCKED -> waiting for absence to unlock")
                continue

            # Sync local voice-id folder to S3 after successful identification/enrollment.
            try:
                _ = sync_all_voice_ids_to_s3(ARGS.voice_folder)
            except Exception as e:
                print(f"[VOICE_ID][WARN] Sync skipped: {e}")

            if bool(getattr(ARGS, 'voice_drift_enable', True)):
                with STATE.lock:
                    STATE.voice["session_state"] = "VOICE_DRIFT"
                    STATE.voice["locked"] = True
                    STATE.voice["last_msg"] = f"Voice drift check for '{final_user}'..."
                try:
                    _ = run_voice_drift_session(final_user, transcriber=transcriber)
                except Exception as e:
                    print(f"[VOICE_DRIFT][ERROR] Failed: {e}")
                finally:
                    with STATE.lock:
                        STATE.voice["session_state"] = "LOCKED"
                        STATE.voice["locked"] = True

            if ARGS.collect_enable:
                with STATE.lock:
                    # Prevent auto-unlock during collection.
                    STATE.voice["session_state"] = "COLLECTING"
                    STATE.voice["locked"] = True
                    STATE.voice["last_msg"] = f"Collecting radar/video/audio for '{final_user}'..."
                try:
                    trigger_collection_for_user(final_user)
                finally:
                    # Return to LOCKED so the normal no-motion unlock policy still applies after collection.
                    with STATE.lock:
                        STATE.voice["session_state"] = "LOCKED"
                        STATE.voice["locked"] = True

            print("[VOICE] stage=LOCKED -> waiting for absence to unlock")

# ------------------------- Acquisition loop (unchanged radar logic) -------------------------

def acquisition_loop():
    while not STOP_EVENT.is_set():
        d0, uri0, d1, uri1 = open_two_devices()
        print(f"Using devices: {uri0} and {uri1}")
        # Save URIs so the collector subprocess can pin device order.
        with STATE.lock:
            STATE.motion["uri0"] = uri0
            STATE.motion["uri1"] = uri1

        with d0 as dev0, d1 as dev1:
            _ = configure_device(dev0, ARGS.frate)
            _ = configure_device(dev1, ARGS.frate)
            try:
                dev0.start_acquisition()
                dev1.start_acquisition()
            except Exception as e:
                print(f"[ERROR] start_acquisition failed: {e}")
                raise

            RADARS_RELEASED.clear()
            proc = RadarProcessor(ARGS.num_chirps, ARGS.num_samples)
            phase_series = []
            fs_phase = ARGS.frate

            prev_d0_ch = None
            prev_d1_ch = None

            status_last_t = time.time()
            status_last_frame = 0

            while True:
                # Pause condition: release devices so the raw collector can run exclusively.
                if COLLECT_REQUESTED.is_set():
                    with STATE.lock:
                        for dk in ("dev0", "dev1"):
                            STATE.motion[dk]["score"] = 0.0
                            STATE.motion[dk]["state"] = "INIT"
                            STATE.motion[dk]["on_cnt"] = 0
                            STATE.motion[dk]["off_cnt"] = 0
                            STATE.motion[dk]["cooldown_until"] = 0.0
                    break

                now = time.time()
                try:
                    fc0 = dev0.get_next_frame()
                    fc1 = dev1.get_next_frame()
                except Exception as e:
                    print(f"[WARN] get_next_frame failed: {e}")
                    time.sleep(0.1)
                    continue

                data0 = fc0[0]
                data1 = fc1[0]

                chirps0 = [data0[i, :, :] for i in range(min(3, data0.shape[0]))]
                chirps1 = [data1[i, :, :] for i in range(min(3, data1.shape[0]))]

                d0_ch, d1_ch = [], []
                for k in range(3):
                    dist0, _, _, _ = proc.compute_distance(chirps0[k])
                    dist1, _, _, _ = proc.compute_distance(chirps1[k])
                    d0_ch.append(dist0.astype(np.float32))
                    d1_ch.append(dist1.astype(np.float32))

                # Phase vitals
                if ARGS.phase_radar != 'off':
                    pick = chirps1[0] if ARGS.phase_radar == 'dev1' else chirps0[0]
                    _, phase_at_peak, _, _ = proc.compute_distance(pick)
                    phase_series.append(phase_at_peak)
                    max_len = int(60 * fs_phase)
                    if len(phase_series) > max_len:
                        phase_series[:] = phase_series[-max_len:]
                    resp_bpm = heart_bpm = 0.0
                    if len(phase_series) >= max(16, int(5 * fs_phase)):
                        resp, heart, _ = unwrap_and_filter_phase(phase_series, fs=fs_phase)
                        resp_bpm = estimate_bpm(resp, fs=fs_phase)
                        heart_bpm = estimate_bpm(heart, fs=fs_phase)
                else:
                    resp_bpm = heart_bpm = 0.0

                with STATE.lock:
                    STATE.dev0 = d0_ch
                    STATE.dev1 = d1_ch
                    STATE.last_ts = now
                    STATE.frame_idx += 1
                    STATE.vitals = {"resp_bpm": float(resp_bpm), "heart_bpm": float(heart_bpm)}
                    STATE.range_axis_m = np.arange(proc.num_samples) * proc.range_bin_length
                    line1 = f"F{STATE.frame_idx} | {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    line2 = f"Resp {resp_bpm:.1f} | Heart {heart_bpm:.1f}"
                    STATE.overlay_lines = (line1, line2)
                    STATE.overlay_text = f"{line1}  |  {line2}"

                # Motion
                i0, i1 = _range_gate_indices(np.arange(proc.num_samples) * proc.range_bin_length,
                                             ARGS.motion_gate_min_m, ARGS.motion_gate_max_m)
                score0_valid = (prev_d0_ch is not None)
                score1_valid = (prev_d1_ch is not None)
                score0_raw = compute_motion_energy(d0_ch, prev_d0_ch, i0, i1)
                score1_raw = compute_motion_energy(d1_ch, prev_d1_ch, i0, i1)
                prev_d0_ch = [a.copy() for a in d0_ch]
                prev_d1_ch = [a.copy() for a in d1_ch]

                with STATE.lock:
                    update_motion_fsm(STATE.motion["dev0"], score0_raw, now, score0_valid)
                    update_motion_fsm(STATE.motion["dev1"], score1_raw, now, score1_valid)

                # Status print ~1 Hz
                if (now - status_last_t) >= 1.0:
                    with STATE.lock:
                        fcount = int(STATE.frame_idx)
                        m0 = dict(STATE.motion.get('dev0', {}))
                        m1 = dict(STATE.motion.get('dev1', {}))
                        v = dict(STATE.voice)
                    dt = max(1e-6, now - status_last_t)
                    fps_est = (fcount - status_last_frame) / dt
                    print(
                        f"[STAT] frames={fcount} fps~{fps_est:.2f} "
                        f"| dev0={m0.get('state','?')} ΔE={float(m0.get('score',0.0)):.4f} "
                        f"| dev1={m1.get('state','?')} ΔE={float(m1.get('score',0.0)):.4f} "
                        f"| voice={v.get('session_state','OFF')} locked={bool(v.get('locked', False))}"
                    )
                    status_last_t = now
                    status_last_frame = fcount

                time.sleep(max(0.0, (1.0 / ARGS.frate) - 0.001))
            # Reached only when the frame loop breaks (e.g., to run a raw collection session).
            try:
                dev0.stop_acquisition()
                dev1.stop_acquisition()
            except Exception:
                pass

        # Devices are closed when exiting the context manager. Signal release for collector.
        RADARS_RELEASED.set()

        # Wait until collector is done, then resume acquisition by re-opening devices.
        while COLLECT_REQUESTED.is_set():
            if STOP_EVENT.is_set():
                break
            time.sleep(0.1)

# ------------------------- Main -------------------------

if __name__ == '__main__':
    t = threading.Thread(target=acquisition_loop, daemon=True)
    t.start()

    if ARGS.voice_enable:
        vt = threading.Thread(target=interaction_manager, daemon=True)
        vt.start()

    # No web UI. Keep the main thread alive.
    while not STOP_EVENT.is_set():
        time.sleep(1.0)

    print("[STOP] sense_motion_interact exiting.")
