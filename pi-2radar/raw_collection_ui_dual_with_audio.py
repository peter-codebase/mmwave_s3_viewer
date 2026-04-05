#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mmwave_collector_s3.py
Dual-radar raw collector that uploads directly to AWS S3.

- Segments CSV + MP4 + WAV every N minutes (default 10).
- Each segment uses the original timestamped naming:
  s3://<bucket>/<user>-YYYYMMDD/<user>-YYYYMMDD-HHMMSS/<user>-YYYYMMDD-HHMMSS.{csv,mp4,wav,_radar_meta.json,_audio_meta.json}
- If run time < segment window, only one set is created.
- Max run time defaults to 24h (safety cutoff).

Deps:
  pip install boto3 python-dotenv numpy
  # optional for audio/video
  sudo apt install -y python3-picamera2 python3-opencv libportaudio2 libsndfile1 ffmpeg
  pip install sounddevice soundfile opencv-python
"""

import os
import sys
import cv2
import csv
import io
import time
import json
import queue
import shutil
import tempfile
import threading
import signal
import argparse
import numpy as np
from pathlib import Path

# ----------------------- .env BOOTSTRAP (BEFORE argparse) -----------------------
try:
    from dotenv import load_dotenv as _load_dotenv
    def load_dotenv(path=None, override=False):
        if path is None:
            return _load_dotenv(override=override)
        return _load_dotenv(dotenv_path=path, override=override)
except Exception:
    def load_dotenv(path=None, override=False):  # no-op if python-dotenv not installed
        return False

_env_file = None
for i, a in enumerate(sys.argv):
    if a == "--env-file" and i + 1 < len(sys.argv):
        _env_file = sys.argv[i + 1]
        break
if _env_file:
    load_dotenv(_env_file, override=True)
load_dotenv(Path(__file__).with_name(".env"), override=False)
load_dotenv(Path.cwd() / ".env", override=False)
load_dotenv(Path.home() / ".env", override=False)
# -------------------------------------------------------------------------------

# Optional camera (Picamera2)
try:
    from picamera2 import Picamera2
    PICAM2_AVAILABLE = True
except Exception:
    Picamera2 = None  # type: ignore
    PICAM2_AVAILABLE = False

# Optional audio stack
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_STACK_AVAILABLE = True
    AUDIO_IMPORT_ERROR = None
except Exception as _e:
    sd = None
    sf = None
    AUDIO_STACK_AVAILABLE = False
    AUDIO_IMPORT_ERROR = _e

# AWS (required for S3 uploads)
_BOTO3_IMPORT_ERROR = None
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception as _e:
    boto3 = None
    BotoCoreError = ClientError = Exception
    _BOTO3_IMPORT_ERROR = _e

# Infineon radar SDK
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwMetrics

# ------------------------- Defaults -------------------------
FRAME_RATE_DEFAULT = 5
video_record_en = True
DEFAULT_USER = "tester1"

# ------------------------- Argparse -------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Dual-radar → S3 with CSV/MP4/WAV segmentation and organized S3 layout."
    )
    # Acquisition
    parser.add_argument('-f', '--frate', type=int, default=FRAME_RATE_DEFAULT,
                        help=f"frame rate in Hz, default {FRAME_RATE_DEFAULT}")
    parser.add_argument('-n', '--nframes', type=int, default=0,
                        help="number of frames to collect; 0 = unlimited")
    parser.add_argument('--raw-float', choices=['float16', 'float32'], default='float32',
                        help='Float dtype for raw chirp arrays in CSV (default: float32).')
    parser.add_argument('--port0', type=str, default=os.getenv('RADAR0_URI'),
                        help='Device URI/ID for radar #0 (e.g., usb:0)')
    parser.add_argument('--port1', type=str, default=os.getenv('RADAR1_URI'),
                        help='Device URI/ID for radar #1 (e.g., usb:1)')
    parser.add_argument('--swap-sides', action='store_true',
                        help='Swap left/right mapping: write second device as dev0 and first as dev1.')
    parser.add_argument('--user', type=str, default=(os.getenv('USER_ID') or DEFAULT_USER),
                        help=f'User/session owner for S3 folder naming (default: {DEFAULT_USER}).')

    # Video
    parser.add_argument('--cam-index', type=int, default=None,
                        help='USB camera index for OpenCV (e.g., 0). Ignored if Picamera2 works.')
    parser.add_argument('--no-video', action='store_true',
                        help='Force disable video even if a camera is available.')

    # Audio
    parser.add_argument('--audio', action='store_true', default=True, help='Enable audio capture (WAV).')
    parser.add_argument('--audio-rate', type=int, default=48000, help='Audio sample rate (Hz).')
    parser.add_argument('--audio-channels', type=int, default=1, help='Audio channels (1=mono, 2=stereo).')
    parser.add_argument('--audio-device', type=str, default=0, help='Audio input device index or name substring.')
    parser.add_argument('--audio-rms-log', action='store_true', help='Print audio RMS every ~1s.')
    parser.add_argument('--list-audio-devices', action='store_true', help='List audio input devices and exit.')

    # S3
    parser.add_argument('--s3-bucket', type=str, default=os.getenv('S3_BUCKET'),
                        help='AWS S3 bucket to upload results (required).')
    parser.add_argument('--s3-prefix', type=str, default=(os.getenv('S3_PREFIX') or '').strip('/'),
                        help='Optional S3 key prefix (e.g., "runs"). Leave blank for bucket root.')
    parser.add_argument('--aws-region', type=str, default=os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION'),
                        help='AWS region for the client (optional).')
    parser.add_argument('--aws-profile', type=str, default=os.getenv('AWS_PROFILE'),
                        help='AWS CLI profile name to use (optional).')
    parser.add_argument('--keep-local', action='store_true',
                        help='Keep local temp files. Default: delete after uploading to S3.')

    # Segmentation / guard rails
    parser.add_argument('--segment-minutes', type=int, default=int(os.getenv('SEGMENT_MINUTES', '10')),
                        help='Segment length in minutes; 0 disables segmentation (default: 10).')
    parser.add_argument('--max-run-seconds', type=int, default=int(os.getenv('MAX_RUN_SECONDS', str(24*3600))),
                        help='Maximum total run time in seconds (default: 86400 = 24h).')

    # .env visibility
    parser.add_argument('--env-file', type=str, default=_env_file,
                        help='Path to a .env file to load (already applied before argparse).')

    return parser

def _dtype_from_flag(name: str):
    return np.float16 if name == 'float16' else np.float32

# ------------------------- S3 helpers -------------------------
class S3Context:
    def __init__(self, bucket: str, prefix: str, region: str|None, profile: str|None):
        if boto3 is None:
            raise RuntimeError(f"boto3 required for S3 upload but not available: {_BOTO3_IMPORT_ERROR}")
        session_kwargs = {}
        if profile:
            session_kwargs['profile_name'] = profile
        session = boto3.Session(**session_kwargs)
        self.s3 = session.client('s3', region_name=region) if region else session.client('s3')
        self.bucket = bucket
        self.prefix = (prefix or '').strip('/')

    def key(self, *parts: str) -> str:
        parts = [p.strip('/').replace('\\', '/') for p in parts if p]
        path = '/'.join(parts)
        return path if not self.prefix else f"{self.prefix}/{path}"

    def put_json(self, key: str, obj: dict):
        body = json.dumps(obj, indent=2).encode('utf-8')
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType='application/json')

    def upload_file(self, local_path: str, key: str, content_type: str|None=None):
        extra = {'ContentType': content_type} if content_type else None
        self.s3.upload_file(local_path, self.bucket, key, ExtraArgs=extra or {})

class CSVtoS3Multipart:
    """
    Buffer CSV rows into a text buffer and upload to S3 as multipart when >= part_size bytes.
    Ensures each part >= 5 MiB (S3 requirement), except possibly the final part.
    """
    def __init__(self, s3ctx: S3Context, key: str, part_size: int = 5 * 1024 * 1024):
        self.s3ctx = s3ctx
        self.bucket = s3ctx.bucket
        self.key = key
        self.part_size = max(part_size, 5 * 1024 * 1024)
        self.buf = io.StringIO(newline="")
        self.writer = csv.writer(self.buf)
        self.parts = []
        self.part_number = 1
        resp = self.s3ctx.s3.create_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            ContentType='text/csv'
        )
        self.upload_id = resp['UploadId']

    def write_header(self, header):
        self.writer.writerow(header)
        self._maybe_flush()

    def write_row(self, row):
        self.writer.writerow(row)
        self._maybe_flush()

    def _maybe_flush(self):
        data = self.buf.getvalue()
        if len(data.encode('utf-8')) >= self.part_size:
            self._upload_part(data)
            self.buf = io.StringIO(newline="")
            self.writer = csv.writer(self.buf)

    def _upload_part(self, text_chunk: str):
        b = text_chunk.encode('utf-8')
        resp = self.s3ctx.s3.upload_part(
            Bucket=self.bucket,
            Key=self.key,
            PartNumber=self.part_number,
            UploadId=self.upload_id,
            Body=b
        )
        self.parts.append({'ETag': resp['ETag'], 'PartNumber': self.part_number})
        self.part_number += 1

    def close(self):
        tail = self.buf.getvalue()
        if tail:
            self._upload_part(tail)
        self.s3ctx.s3.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=self.upload_id,
            MultipartUpload={'Parts': self.parts}
        )

# ------------------------- Camera helpers -------------------------
def setup_camera(fps, output_video_path, cam_index=None, force_disable=False):
    """Returns (mode, cam_obj, writer, frame_size) with mode in {'picam2','opencv',None}."""
    if force_disable:
        print("Video disabled by --no-video.")
        return None, None, None, None

    env = os.getenv("VIDEO_ENABLED")
    if env is not None and str(env).strip().lower() in ("0", "false", "no"):
        print("Video disabled by env VIDEO_ENABLED.")
        return None, None, None, None

    global video_record_en
    if not video_record_en:
        print("Video disabled by preference (video_record_en=False).")
        return None, None, None, None

    # Picamera2 first
    if PICAM2_AVAILABLE and cam_index is None:
        try:
            picam2 = Picamera2()
            camera_config = picam2.create_video_configuration(main={"size": (1024, 768), "format": "RGB888"})
            picam2.configure(camera_config)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (1024, 768))
            picam2.start()
            time.sleep(0.5)  # warm-up: let the sensor produce valid frames before capture_array is called
            print("Pi camera initialized via Picamera2.")
            return "picam2", picam2, writer, (1024, 768)
        except Exception as e:
            print(f"WARNING: Picamera2 init failed: {e}. Trying USB webcam via OpenCV...")
            try:
                picam2.stop()
            except Exception:
                pass

    # Fallback to OpenCV USB cams
    indices = [cam_index] if cam_index is not None else [0, 1, 2, 3]
    for idx in indices:
        try:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                continue
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                cap.release()
                continue
            print(f"USB camera initialized at index {idx}.")
            return "opencv", cap, writer, (w, h)
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            continue

    print("WARNING: No working camera found. Continuing without video.")
    return None, None, None, None

def rotate_video_writer(mode, cam_obj, old_writer, new_path, fps, frame_size):
    # Stop/release previous writer
    try:
        if old_writer is not None:
            old_writer.release()
    except Exception:
        pass
    if mode is None or cam_obj is None or frame_size is None:
        return None
    w, h = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(new_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print("WARNING: Failed to open new video writer; disabling video for this segment.")
        return None
    return writer

# ------------------------- Audio helpers -------------------------
def list_audio_devices():
    if not AUDIO_STACK_AVAILABLE:
        print(f"Audio stack not available: {AUDIO_IMPORT_ERROR}")
        return
    print("=== Audio Input Devices ===")
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"Failed to query devices: {e}")
        return
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    for idx, d in enumerate(devices):
        mark = " (default)" if default_in == idx else ""
        print(f"[{idx}] {d['name']}  in:{d['max_input_channels']} out:{d['max_output_channels']}{mark}")

def resolve_audio_device(device_arg, needed_channels):
    if not AUDIO_STACK_AVAILABLE:
        return None, None
    try:
        devices = sd.query_devices()
    except Exception:
        return None, None
    if device_arg is not None:
        try:
            idx = int(device_arg)
            if 0 <= idx < len(devices) and devices[idx]['max_input_channels'] >= needed_channels:
                return idx, devices[idx]
        except Exception:
            pass
        name_lower = str(device_arg).strip().lower()
        for idx, d in enumerate(devices):
            if name_lower in d['name'].lower() and d['max_input_channels'] >= needed_channels:
                return idx, d
        return None, None
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            d = sd.query_devices(default_in)
            if d['max_input_channels'] >= needed_channels:
                return default_in, d
    except Exception:
        pass
    for idx, d in enumerate(devices):
        if d['max_input_channels'] >= needed_channels:
            return idx, d
    return None, None

class AudioRecorder:
    """Threaded audio capture to WAV + JSON sidecar. On stop(), uploads to S3 then deletes local (unless keep_local)."""
    def __init__(self, wav_path, samplerate=48000, channels=1, dtype='float32', blocksize=2048,
                 device_arg=None, rms_log=False, s3ctx=None,
                 s3_key_wav: str|None=None, s3_key_meta: str|None=None, keep_local=False):
        self.wav_path = wav_path
        self.meta_path = os.path.splitext(wav_path)[0] + "_audio_meta.json"
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.device_arg = device_arg
        self.rms_log = rms_log
        self._q = queue.Queue(maxsize=64)
        self._thread = None
        self._stop_evt = threading.Event()
        self._sf = None
        self._stream = None
        self.frames_written = 0
        self.start_wall_time_unix_ns = None
        self.start_perf_counter_ns = None
        self.first_adc_time = None
        self.backend = None
        self.device_info = None
        self.device_index = None
        self._last_rms_print = 0.0
        self._ema_rms = None
        self.s3ctx = s3ctx
        self.s3_key_wav = s3_key_wav
        self.s3_key_meta = s3_key_meta
        self.keep_local = keep_local

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        if self.rms_log:
            try:
                rms = float(np.sqrt(np.mean(np.square(indata.astype(np.float32)))) + 1e-20)
                self._ema_rms = rms if self._ema_rms is None else (0.9 * self._ema_rms + 0.1 * rms)
                # optional: print RMS every second
            except Exception:
                pass
        try:
            self._q.put_nowait(indata.copy())
        except queue.Full:
            pass

    def _writer_loop(self):
        """Writer thread: drains the queue and exits only after receiving a sentinel (None).
        This prevents truncated WAVs on stop() under load."""
        try:
            subtype = 'FLOAT' if self.dtype == 'float32' else 'PCM_24'
            self._sf = sf.SoundFile(
                self.wav_path,
                mode='w',
                samplerate=self.samplerate,
                channels=self.channels,
                subtype=subtype,
            )
            while True:
                block = self._q.get()
                if block is None:
                    break
                try:
                    self._sf.write(block)
                    self.frames_written += len(block)
                finally:
                    try:
                        self._q.task_done()
                    except Exception:
                        pass
        finally:
            try:
                if self._sf:
                    self._sf.flush()
                    self._sf.close()
            except Exception:
                pass

    def start(self):
        if not AUDIO_STACK_AVAILABLE:
            raise RuntimeError(f"Audio stack not available: {AUDIO_IMPORT_ERROR}")
        self.device_index, devinfo = resolve_audio_device(self.device_arg, self.channels)
        self.device_info = devinfo
        if self.device_index is None:
            raise RuntimeError("No suitable audio input device found. Try --list-audio-devices and --audio-device.")
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        self.start_wall_time_unix_ns = time.time_ns()
        self.start_perf_counter_ns = time.perf_counter_ns()
        try:
            self.backend = sd.get_portaudio_version()[1] if hasattr(sd, 'get_portaudio_version') else 'PortAudio'
        except Exception:
            self.backend = 'PortAudio'
        self._stream = sd.InputStream(samplerate=self.samplerate, channels=self.channels,
                                      dtype=self.dtype, callback=self._callback,
                                      blocksize=self.blocksize, device=self.device_index)
        self._stream.start()
        try:
            print(f"Audio device: [{self.device_index}] {self.device_info['name']}  "
                  f"in:{self.device_info['max_input_channels']} @ {self.samplerate} Hz")
        except Exception:
            pass

    def stop(self):
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stop_evt.set()
        # Ensure writer drains queued blocks and closes WAV before upload.
        try:
            self._q.put(None, timeout=1.0)  # sentinel
        except Exception:
            pass
        if self._thread:
            try:
                self._thread.join()  # no timeout
            except Exception:
                pass
        meta = {
            "start_wall_time_unix_ns": self.start_wall_time_unix_ns,
            "start_perf_counter_ns": self.start_perf_counter_ns,
            "samplerate": self.samplerate,
            "channels": self.channels,
            "dtype": self.dtype,
            "blocksize": self.blocksize,
            "device_index": self.device_index,
            "device_info": self.device_info,
            "backend": self.backend,
            "frames_written": self.frames_written,
        }
        # write sidecar, upload both, cleanup
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"WARNING: Failed to write audio meta: {e}")

        if self.s3ctx and self.s3_key_wav and os.path.exists(self.wav_path):
            try:
                self.s3ctx.upload_file(self.wav_path, self.s3_key_wav, content_type="audio/wav")
                print(f"Uploaded audio → s3://{self.s3ctx.bucket}/{self.s3_key_wav}")
            except Exception as e:
                print(f"WARNING: Failed to upload WAV to S3: {e}")
        if self.s3ctx and self.s3_key_meta and os.path.exists(self.meta_path):
            try:
                self.s3ctx.upload_file(self.meta_path, self.s3_key_meta, content_type="application/json")
                print(f"Uploaded audio meta → s3://{self.s3ctx.bucket}/{self.s3_key_meta}")
            except Exception as e:
                print(f"WARNING: Failed to upload audio meta to S3: {e}")

        if self.s3ctx and not self.keep_local:
            for p in (self.wav_path, self.meta_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

# ------------------------- Radar utils -------------------------
def configure_device(device, frate):
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
    sequence.loop.repetition_time_s = 1 / frate
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
    return num_rx_antennas

def discover_candidates(args):
    cands = []
    if args.port0:
        cands.append(args.port0)
    if args.port1:
        cands.append(args.port1)
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
    seen = set()
    out = []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out

def try_open_single(hint):
    trials = [hint]
    if isinstance(hint, str) and all(ch.isalnum() for ch in hint) and len(hint) >= 4:
        trials += [f"usb:{hint}", f"uuid:{hint}"]
    if hint in ("0", "1", "2", "3"):
        trials += [f"usb:{hint}"]
    for trial in trials:
        try:
            dev = DeviceFmcw(trial)
            _ = dev.get_sensor_information()
            return dev, trial
        except Exception:
            pass
        try:
            os.environ["IFX_DEVICE_URI"] = trial
            dev = DeviceFmcw()
            _ = dev.get_sensor_information()
            return dev, trial
        except Exception:
            pass
    return None, None

def open_two_devices(args):
    candidates = discover_candidates(args)
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
                try: dev.close()
                except Exception: pass
                continue
            opened.append((dev, used, uid))
    if len(opened) < 2:
        raise SystemExit(f"ERROR: Need two FMCW devices. Tried: {tried}. Found {len(opened)}.")
    return opened[0][0], opened[0][1], opened[1][0], opened[1][1]

# ------------------------- Helpers -------------------------
def dumps3(arrs, dtype=np.float16, target_channels=3):
    out = []
    for k in range(target_channels):
        if k < len(arrs) and arrs[k] is not None:
            f_lists = [np.asarray(a, dtype=dtype).tolist() for a in arrs[k]]
            out.append(json.dumps(f_lists, separators=(',', ':')))
        else:
            out.append(json.dumps(None))
    return out

def mk_names_for_segment(user_id: str, when_unix: float):
    """Return (day_folder, session_stem) for the given time (local).

    New S3 layout: <user>/<user>-YYYYMMDD/<session_stem>/...
    """
    lt = time.localtime(when_unix)
    day_str = time.strftime("%Y%m%d", lt)
    ts_str  = time.strftime("%Y%m%d-%H%M%S", lt)
    day_folder   = f"{user_id}/{user_id}-{day_str}"
    session_stem = f"{user_id}-{ts_str}"
    return day_folder, session_stem

# ------------------------- Main -------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    stop_requested = threading.Event()

    def _request_stop(signum, frame):
        # Graceful stop: break loop, finalize current segment, upload.
        stop_requested.set()
        try:
            print(f"\n[STOP] Signal {signum} received → stopping after final uploads...")
        except Exception:
            pass

    for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_sig, _request_stop)
        except Exception:
            pass

    # Post-parse fallback: pull from env again if any are missing.
    if not args.s3_bucket:
        args.s3_bucket = os.getenv("S3_BUCKET")
    if args.s3_prefix is None or args.s3_prefix == "":
        args.s3_prefix = (os.getenv("S3_PREFIX") or "").strip("/")
    if not args.aws_region:
        args.aws_region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if args.aws_profile in (None, ""):
        args.aws_profile = os.getenv("AWS_PROFILE")

    print("Resolved config →",
          f"bucket={args.s3_bucket!r}, prefix={args.s3_prefix!r}, region={args.aws_region!r}, profile={args.aws_profile!r}")
    print(f"Segmentation → segment_minutes={args.segment_minutes} (0=disabled), max_run_seconds={args.max_run_seconds}")
    print(f"User → {args.user}")

    if not args.s3_bucket:
        raise SystemExit("ERROR: --s3-bucket (or S3_BUCKET in .env) is required.")

    # globals / toggles
    global video_record_en
    if args.no_video:
        video_record_en = False
    else:
        env = os.getenv("VIDEO_ENABLED")
        if env is not None and str(env).strip().lower() in ("0", "false", "no"):
            video_record_en = False

    chosen_dtype = _dtype_from_flag(args.raw_float)
    print(f"RAW_FLOAT={args.raw_float}")

    if args.list_audio_devices:
        list_audio_devices()
        return

    # --- S3 setup ---
    s3ctx = S3Context(bucket=args.s3_bucket, prefix=args.s3_prefix,
                      region=args.aws_region, profile=args.aws_profile)
    print(f"S3 target: s3://{s3ctx.bucket}/{s3ctx.prefix}" if s3ctx.prefix else f"S3 target: s3://{s3ctx.bucket}")
    keep_local = bool(args.keep_local)

    # ---- Open devices once ----
    dev0, uri0, dev1, uri1 = open_two_devices(args)
    print(f"Using devices: {uri0} and {uri1}")

    # ---- Camera once (we will rotate writers only) ----
    # We'll open the camera now; writers rotate per segment.
    workdir = tempfile.mkdtemp(prefix="mmwave_seg_")
    cam_mode, cam_obj, video_writer, frame_size = (None, None, None, None)
    if video_record_en:
        # Temporary path for the first segment (will be set on first rotate)
        cam_mode, cam_obj, video_writer, frame_size = setup_camera(args.frate,
                                                                   os.path.join(workdir, "seed.mp4"),
                                                                   args.cam_index,
                                                                   force_disable=(not video_record_en))
        # Immediately release the seed so first segment can set proper path
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
            video_writer = None

    # ---- Audio recorder per segment (we will restart per segment) ----
    def start_audio_recorder(wav_path, wav_key, meta_key):
        if not args.audio or not AUDIO_STACK_AVAILABLE:
            return None
        rec = AudioRecorder(
            wav_path,
            samplerate=args.audio_rate,
            channels=args.audio_channels,
            dtype='float32',
            blocksize=2048,
            device_arg=args.audio_device,
            rms_log=args.audio_rms_log,
            s3ctx=s3ctx,
            s3_key_wav=wav_key,
            s3_key_meta=meta_key,
            keep_local=keep_local
        )
        try:
            rec.start()
            print(f"Audio recording started for segment: {Path(wav_path).name}")
            return rec
        except Exception as e:
            print(f"WARNING: Failed to start audio: {e}")
            return None

    # ---- CSV writer per segment ----
    def new_csv_writer(csv_key):
        w = CSVtoS3Multipart(s3ctx, csv_key)
        w.write_header([
            "frame_number", "timestamp",
            "dev0_ch1", "dev0_ch2", "dev0_ch3",
            "dev1_ch1", "dev1_ch2", "dev1_ch3",
        ])
        return w

    # ---- Segment lifecycle helpers ----
    def upload_radar_meta(user_id, session_stem, day_folder, raw_dtype, frate):
        # Radar physics derived from the fixed FmcwMetrics in configure_device().
        # Storing these here means analysis tools can read the correct PRF and
        # center frequency rather than hardcoding assumed values.
        _c        = 299_792_458.0
        _fc_hz    = 60_750_000_000.0   # center_frequency_Hz in configure_device()
        _max_spd  = 2.45               # max_speed_m_s
        _spd_res  = 0.2                # speed_resolution_m_s
        _rng_res  = 0.15               # range_resolution_m
        _max_rng  = 4.8                # max_range_m
        _lam      = _c / _fc_hz        # wavelength (m)
        # PRF = chirp repetition frequency within a frame (~1985 Hz, SDK rounds to ~2000)
        _prf_hz   = 4.0 * _max_spd / _lam
        # Nominal chirp count (SDK may round to power-of-2; actual count readable from CSV)
        _num_chirps_nominal = int(round(2.0 * _max_spd / _spd_res))  # = 25

        radar_meta = {
            "user_id": user_id,
            "session_date": session_stem.split('-')[1],
            "session_id": session_stem,
            "radar_raw_dtype": raw_dtype,
            "storage": "csv_json_arrays",
            "columns": ["frame_number","timestamp",
                        "dev0_ch1","dev0_ch2","dev0_ch3",
                        "dev1_ch1","dev1_ch2","dev1_ch3"],
            "created_unix": time.time(),
            # Radar physics — used by analysis tools for correct Doppler/range axes
            "frame_rate_hz":          frate,
            "center_frequency_hz":    _fc_hz,
            "bandwidth_hz":           round(_c / (2.0 * _rng_res), 0),
            "max_speed_m_s":          _max_spd,
            "speed_resolution_m_s":   _spd_res,
            "max_range_m":            _max_rng,
            "range_resolution_m":     _rng_res,
            "prf_hz":                 round(_prf_hz, 2),
            "num_chirps_nominal":     _num_chirps_nominal,
        }
        key = s3ctx.key(day_folder, session_stem, f"{session_stem}_radar_meta.json")
        s3ctx.put_json(key, radar_meta)
        print(f"Uploaded radar meta → s3://{s3ctx.bucket}/{key}")

    def s3_keys_for_segment(day_folder, session_stem):
        csv_key   = s3ctx.key(day_folder, session_stem, f"{session_stem}.csv")
        mp4_key   = s3ctx.key(day_folder, session_stem, f"{session_stem}.mp4")
        wav_key   = s3ctx.key(day_folder, session_stem, f"{session_stem}.wav")
        wavm_key  = s3ctx.key(day_folder, session_stem, f"{session_stem}_audio_meta.json")
        vts_key   = s3ctx.key(day_folder, session_stem, f"{session_stem}_video_ts.csv")
        return csv_key, mp4_key, wav_key, wavm_key, vts_key

    # First segment timing/paths
    user_id = (args.user or DEFAULT_USER).strip()
    segment_minutes = max(0, args.segment_minutes)
    segment_seconds = segment_minutes * 60
    session_start = time.time()
    next_rotate_at = session_start + segment_seconds if segment_seconds > 0 else float('inf')

    # First segment names/keys/locals
    day_folder, session_stem = mk_names_for_segment(user_id, session_start)
    csv_key, mp4_key, wav_key, wav_meta_key, vts_key = s3_keys_for_segment(day_folder, session_stem)
    video_local_path = os.path.join(workdir, f"{session_stem}.mp4")
    wav_local_path   = os.path.join(workdir, f"{session_stem}.wav")
    vts_local_path   = os.path.join(workdir, f"{session_stem}_video_ts.csv")

    # Upload meta, open writers
    upload_radar_meta(user_id, session_stem, day_folder, args.raw_float, args.frate)
    csv_writer = new_csv_writer(csv_key)

    # start video writer for this segment
    if cam_mode is not None and cam_obj is not None and frame_size is not None:
        video_writer = rotate_video_writer(cam_mode, cam_obj, video_writer, video_local_path, args.frate, frame_size)
    # start audio for this segment
    audio_rec = start_audio_recorder(wav_local_path, wav_key, wav_meta_key)

    # Video timestamp sidecar (one row per captured video frame: index + unix timestamp)
    vts_file, vts_csv = None, None
    if cam_mode is not None:
        try:
            vts_file = open(vts_local_path, 'w', newline='', encoding='utf-8')
            vts_csv  = csv.writer(vts_file)
            vts_csv.writerow(['video_frame_index', 'timestamp_unix'])
        except Exception as e:
            print(f"WARNING: Failed to open video timestamp sidecar: {e}")

    total_frames = 0
    cam_consec_errors = 0  # consecutive Picamera2 capture failures; disable after 3

    print(f"Begin collecting → session {session_stem} (CSV/MP4/WAV).")
    with dev0 as d0, dev1 as d1:
        rx0 = configure_device(d0, args.frate)
        rx1 = configure_device(d1, args.frate)

        try:
            while True:
                # guards: frames cap OR time cap
                now = time.time()
                if stop_requested.is_set():
                    print("[STOP] Graceful stop requested.")
                    break
                if args.nframes > 0 and total_frames >= args.nframes:
                    print("Reached nframes limit.")
                    break
                if args.max_run_seconds > 0 and (now - session_start) >= args.max_run_seconds:
                    print("Reached max-run-seconds limit.")
                    break

                # Get frames
                fc0 = d0.get_next_frame()
                fc1 = d1.get_next_frame()
                data0 = fc0[0]
                data1 = fc1[0]
                # Fix #1: timestamp captured AFTER both blocking get_next_frame() calls
                t_host = time.time()

                # Video frame
                if cam_mode == "picam2" and cam_obj is not None and video_writer is not None:
                    try:
                        frame = cam_obj.capture_array("main")
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.putText(frame, f"Frame: {total_frames+1}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        cv2.putText(frame, ts, (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        video_writer.write(frame)
                        cam_consec_errors = 0
                        # Fix #2: log frame index + unix timestamp to sidecar
                        if vts_csv is not None:
                            vts_csv.writerow([total_frames + 1, t_host])
                    except Exception as e:
                        cam_consec_errors += 1
                        print(f"WARNING: Picamera2 capture failed ({cam_consec_errors}/3): {e}")
                        if cam_consec_errors >= 3:
                            print("WARNING: 3 consecutive Picamera2 failures — disabling video.")
                            try: cam_obj.stop()
                            except Exception: pass
                            try: video_writer.release()
                            except Exception: pass
                            cam_mode, cam_obj, video_writer = None, None, None

                elif cam_mode == "opencv" and cam_obj is not None and video_writer is not None:
                    try:
                        ok, frame = cam_obj.read()
                        if not ok or frame is None:
                            raise RuntimeError("USB camera read failed")
                        cv2.putText(frame, f"Frame: {total_frames+1}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                        cv2.putText(frame, ts, (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        video_writer.write(frame)
                        # Fix #2: log frame index + unix timestamp to sidecar
                        if vts_csv is not None:
                            vts_csv.writerow([total_frames + 1, t_host])
                    except Exception as e:
                        print(f"WARNING: USB camera capture failed: {e}. Disabling video.")
                        try: cam_obj.release()
                        except Exception: pass
                        try: video_writer.release()
                        except Exception: pass
                        cam_mode, cam_obj, video_writer = None, None, None

                # Prepare radar chirps
                chirps0 = [data0[i, :, :] for i in range(min(rx0, data0.shape[0]))]
                chirps1 = [data1[i, :, :] for i in range(min(rx1, data1.shape[0]))]
                row = [
                    total_frames + 1,
                    t_host,
                    *dumps3(chirps0, dtype=chosen_dtype),
                    *dumps3(chirps1, dtype=chosen_dtype),
                ]
                csv_writer.write_row(row)
                total_frames += 1

                # Rotate at boundary?
                if now >= next_rotate_at:
                    # Close CSV segment
                    try:
                        csv_writer.close()
                        print(f"Finalized segment CSV: s3://{s3ctx.bucket}/{csv_key}")
                    except Exception as e:
                        print(f"WARNING: segment CSV close failed: {e}")

                    # Close & upload video segment
                    if video_writer is not None and os.path.exists(video_local_path):
                        try:
                            video_writer.release()
                        except Exception:
                            pass
                        try:
                            s3ctx.upload_file(video_local_path, mp4_key, content_type="video/mp4")
                            print(f"Uploaded video → s3://{s3ctx.bucket}/{mp4_key}")
                            if not keep_local:
                                os.remove(video_local_path)
                        except Exception as e:
                            print(f"WARNING: Failed to upload video: {e}")
                        video_writer = None  # will reopen for next segment

                    # Close & upload video timestamp sidecar
                    if vts_file is not None:
                        try:
                            vts_file.flush()
                            vts_file.close()
                        except Exception:
                            pass
                        vts_file, vts_csv = None, None
                        if os.path.exists(vts_local_path):
                            try:
                                s3ctx.upload_file(vts_local_path, vts_key, content_type="text/csv")
                                print(f"Uploaded video timestamps → s3://{s3ctx.bucket}/{vts_key}")
                                if not keep_local:
                                    os.remove(vts_local_path)
                            except Exception as e:
                                print(f"WARNING: Failed to upload video timestamps: {e}")

                    # Stop audio (uploads & cleans up)
                    if audio_rec is not None:
                        try:
                            audio_rec.stop()
                        except Exception as e:
                            print(f"WARNING: Failed to stop audio: {e}")
                        audio_rec = None

                    # Start next segment
                    seg_start_time = now  # segment boundary time
                    day_folder, session_stem = mk_names_for_segment(user_id, seg_start_time)
                    csv_key, mp4_key, wav_key, wav_meta_key, vts_key = s3_keys_for_segment(day_folder, session_stem)
                    video_local_path = os.path.join(workdir, f"{session_stem}.mp4")
                    wav_local_path   = os.path.join(workdir, f"{session_stem}.wav")
                    vts_local_path   = os.path.join(workdir, f"{session_stem}_video_ts.csv")

                    upload_radar_meta(user_id, session_stem, day_folder, args.raw_float, args.frate)
                    csv_writer = new_csv_writer(csv_key)

                    if cam_mode is not None and cam_obj is not None and frame_size is not None:
                        video_writer = rotate_video_writer(cam_mode, cam_obj, video_writer, video_local_path, args.frate, frame_size)

                    audio_rec = start_audio_recorder(wav_local_path, wav_key, wav_meta_key)

                    # Open new video timestamp sidecar for next segment
                    vts_file, vts_csv = None, None
                    if cam_mode is not None:
                        try:
                            vts_file = open(vts_local_path, 'w', newline='', encoding='utf-8')
                            vts_csv  = csv.writer(vts_file)
                            vts_csv.writerow(['video_frame_index', 'timestamp_unix'])
                        except Exception as e:
                            print(f"WARNING: Failed to open video timestamp sidecar: {e}")

                    # schedule next rotation
                    next_rotate_at = now + (segment_seconds if segment_seconds > 0 else float('inf'))

        except KeyboardInterrupt:
            stop_requested.set()
            print("Collection stopped by user (KeyboardInterrupt).")

    # During finalization+uploads, ignore further interrupts so we don't cut uploads mid-flight.
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except Exception:
        pass

    # Finalize last CSV
    try:
        csv_writer.close()
        print(f"Finalized last CSV: s3://{s3ctx.bucket}/{csv_key}")
    except Exception as e:
        print(f"WARNING: final CSV close failed: {e}")

    # Finalize last video
    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass
    if cam_mode == "picam2" and cam_obj is not None and os.path.exists(video_local_path):
        try: cam_obj.stop()
        except Exception: pass
    if cam_mode == "opencv" and cam_obj is not None:
        try: cam_obj.release()
        except Exception: pass
    if cam_mode is not None and os.path.exists(video_local_path):
        try:
            s3ctx.upload_file(video_local_path, mp4_key, content_type="video/mp4")
            print(f"Uploaded video → s3://{s3ctx.bucket}/{mp4_key}")
            if not keep_local:
                os.remove(video_local_path)
        except Exception as e:
            print(f"WARNING: Failed to upload video: {e}")

    # Finalize last video timestamp sidecar
    if vts_file is not None:
        try:
            vts_file.flush()
            vts_file.close()
        except Exception:
            pass
        vts_file, vts_csv = None, None
        if os.path.exists(vts_local_path):
            try:
                s3ctx.upload_file(vts_local_path, vts_key, content_type="text/csv")
                print(f"Uploaded video timestamps → s3://{s3ctx.bucket}/{vts_key}")
                if not keep_local:
                    os.remove(vts_local_path)
            except Exception as e:
                print(f"WARNING: Failed to upload video timestamps: {e}")

    # Finalize last audio
    if audio_rec is not None:
        try:
            audio_rec.stop()
        except Exception as e:
            print(f"WARNING: Failed to stop audio: {e}")

    # Cleanup
    if not keep_local:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass

    print("End collecting.")

if __name__ == "__main__":
    main()
