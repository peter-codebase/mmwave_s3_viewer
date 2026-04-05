#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPIO button + LED launcher for sense_motion_interact.py

Wiring (BCM):
- Button: GPIO17 -> button -> GND (internal pull-up; active-low)
- LED:    GPIO27 -> 330Ω -> LED anode; LED cathode -> GND

Behavior:
- Press button once: start sense_motion_interact.py (as a subprocess)
- Press button again: request graceful stop (SIGINT) and wait for exit
- LED ON while subprocess is running

Notes:
- This script never uses SIGKILL unless the process refuses to exit.
- SIGINT is important so the collector can finalize the last segment and upload to S3.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from gpiozero import Button, LED


# --- Configuration ---
GPIO_BUTTON = 17
GPIO_LED = 27

# Use your venv python and the patched sense script by default.
DEFAULT_VENV_PY = Path.home() / "Desktop/mmwave_collector_2radar/.venv/bin/python"
DEFAULT_SENSE_SCRIPT = Path.home() / "Desktop/mmwave_collector_2radar/sense_motion_interact.py"

# How long we wait for graceful shutdown before escalating
GRACEFUL_STOP_TIMEOUT_S = 300.0


class ToggleRunner:
    def __init__(self, python_bin: Path, script_path: Path):
        self.python_bin = python_bin
        self.script_path = script_path
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        with self.lock:
            return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return

            cmd = [str(self.python_bin), str(self.script_path)]
            # Use a separate process group so we can signal the whole group if needed.
            self.proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
            )

    def stop(self) -> None:
        with self.lock:
            proc = self.proc

        if proc is None:
            return
        if proc.poll() is not None:
            return

        # Graceful stop: SIGINT
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except Exception:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    return

        # Wait for clean exit
        try:
            proc.wait(timeout=GRACEFUL_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # Escalate: SIGTERM then SIGKILL
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=10.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        with self.lock:
            self.proc = None


def main():
    led = LED(GPIO_LED)
    btn = Button(GPIO_BUTTON, pull_up=True, bounce_time=0.05)

    runner = ToggleRunner(DEFAULT_VENV_PY, DEFAULT_SENSE_SCRIPT)

    def _toggle():
        if runner.is_running():
            runner.stop()
        else:
            runner.start()

    btn.when_pressed = _toggle

    print(f"[GPIO] Button on BCM{GPIO_BUTTON}, LED on BCM{GPIO_LED}")
    print(f"[GPIO] Python: {DEFAULT_VENV_PY}")
    print(f"[GPIO] Script: {DEFAULT_SENSE_SCRIPT}")
    print("[GPIO] Press button to start/stop. Ctrl+C here will stop the child if running.")

    try:
        while True:
            led.value = 1 if runner.is_running() else 0
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        led.off()
        if runner.is_running():
            runner.stop()


if __name__ == "__main__":
    main()
