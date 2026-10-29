"""Stage 1-1: detect what the PICO setup is streaming and pick a tracker profile.

What can be detected from the XRoboToolkit stream:
  * packet rate, headset / controller / hand-tracking activity
  * body tracking active (PICO only streams it in BodyTracking mode with >= 2 calibrated
    Motion Trackers) -> profile body_2trk, or body_3trk when a waist tracker is declared
  * object-mode Motion Trackers: count (<= 3) and serial numbers -> sparse_trackers

What cannot be read through XRoboToolkit and must be declared (CLI flag / config):
  * the number of trackers while in body-tracking mode (2 ankles vs. 2 ankles + waist)
  * tracker hardware generation (Beta "Swift" vs. 2024 Motion Tracker): the PICO SDK
    exposes it (`GetMotionTrackerDeviceType`: MT_1=Beta, MT_2=Official) only to apps on
    the headset, not through the PC service.
Optionally `adb` (USB) is queried for the headset model / OS build.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import yaml


@dataclass
class ProbeReport:
    connected: bool = False
    packet_rate_hz: float = 0.0
    headset_active: bool = False
    controllers: dict = field(default_factory=lambda: {"left": False, "right": False})
    hand_tracking: dict = field(default_factory=lambda: {"left": False, "right": False})
    body_tracking: bool = False
    body_rate_hz: float = 0.0
    body_joints_moving: int = 0
    object_trackers: int = 0
    tracker_serials: list = field(default_factory=list)
    tracker_rate_hz: float = 0.0
    declared_body_trackers: int | None = None
    tracker_hw: str = "unknown"
    headset_model: str = "unknown"
    headset_os: str = "unknown"
    profile: str = "none"
    notes: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def _adb_props() -> dict:
    adb = find_adb()
    if adb is None:
        return {}
    out = {}
    for key, prop in (("model", "ro.product.model"), ("os", "ro.build.display.id"),
                      ("brand", "ro.product.brand"), ("android_sdk", "ro.build.version.sdk")):
        try:
            r = subprocess.run([adb, "shell", "getprop", prop], capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                out[key] = r.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass
    return out


def find_adb() -> str | None:
    """adb on PATH, else the platform-tools folder this project's README points to."""
    from pathlib import Path

    adb = shutil.which("adb")
    if adb is None:
        cand = Path.home() / "Downloads" / "platform-tools-latest-windows" / "platform-tools" / "adb.exe"
        adb = str(cand) if cand.is_file() else None
    return adb


def usb_tunnel(port: int = 63901) -> bool:
    """`adb reverse`: the headset's 127.0.0.1:<port> reaches the PC service over USB.

    Enter 127.0.0.1 as the PC IP in the headset app; no Wi-Fi, firewall rule or headset sleep
    issues, lower latency. Needs USB debugging allowed on the headset.
    """
    adb = find_adb()
    if adb is None:
        return False
    try:
        # re-registering an existing reverse can drop the app's open connection: only add it when missing
        cur = subprocess.run([adb, "reverse", "--list"], capture_output=True, text=True, timeout=10)
        if cur.returncode == 0 and f"tcp:{port} tcp:{port}" in cur.stdout:
            return True
        r = subprocess.run([adb, "reverse", f"tcp:{port}", f"tcp:{port}"], capture_output=True, text=True, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return False
    return r.returncode == 0


def probe(duration: float = 3.0, source=None, declared_body_trackers: int | None = None,
          tracker_hw: str = "unknown") -> ProbeReport:
    """Sample the live stream for `duration` seconds and summarise it."""
    from .xrt_source import XRTSource

    rep = ProbeReport(declared_body_trackers=declared_body_trackers, tracker_hw=tracker_hw)
    props = _adb_props()
    rep.headset_model = props.get("model", "unknown (adb not available)" if not find_adb() else "unknown")
    if rep.headset_model == "A9210":
        rep.headset_model = "A9210 (PICO 4 Ultra Enterprise)"
    rep.headset_os = props.get("os", "unknown") + (f" / API {props['android_sdk']}" if "android_sdk" in props else "")

    own = source is None
    src = source or XRTSource(with_hands=False).open()
    frames = []
    try:
        t_end = time.time() + duration
        while time.time() < t_end:
            try:
                fr = src.poll()
            except StopIteration:  # finite (simulated / replayed) stream
                break
            if fr is None:
                time.sleep(0.002)
                continue
            frames.append(fr)
    finally:
        if own:
            src.close()

    if len(frames) < 2:
        rep.notes.append("No packets received. Check in order: PC service running, firewall inbound rule "
                         "(TCP 63901), PC IP entered on the headset (127.0.0.1 with --usb) and app Status WORKING "
                         "(service log: 'new tcp connect device'), 'Send' on, Motion Trackers calibrated if in Full "
                         "body mode, headset worn (PICO pauses the app when it is taken off).")
        rep.profile = "none"
        return rep
    rep.connected = True
    t = np.array([f.t_ns for f in frames]) * 1e-9
    rep.packet_rate_hz = float((len(t) - 1) / max(t[-1] - t[0], 1e-6))

    head = np.stack([f.headset for f in frames])
    rep.headset_active = bool(np.ptp(head[:, :3], axis=0).max() > 1e-4 or np.abs(head[:, 3:]).sum() > 0)
    for side, key in (("left", "left_ctrl"), ("right", "right_ctrl")):
        arr = np.stack([getattr(f, key) for f in frames])
        rep.controllers[side] = bool(np.abs(arr).sum(axis=1).mean() > 0 and np.ptp(arr, axis=0).max() > 1e-5)
    rep.hand_tracking["left"] = any(f.left_hand_active for f in frames)
    rep.hand_tracking["right"] = any(f.right_hand_active for f in frames)

    body_frames = [f for f in frames if f.body is not None]
    if len(body_frames) >= 2:
        rep.body_tracking = True
        bt = np.array([f.t_ns for f in body_frames]) * 1e-9
        rep.body_rate_hz = float((len(bt) - 1) / max(bt[-1] - bt[0], 1e-6))
        b = np.stack([f.body[:, :3] for f in body_frames])
        rep.body_joints_moving = int((np.ptp(b, axis=0).max(axis=-1) > 1e-4).sum())

    trk_frames = [f for f in frames if f.trackers is not None and len(f.trackers)]
    if trk_frames:
        rep.object_trackers = max(len(f.trackers) for f in trk_frames)
        serials = []
        for f in trk_frames:
            serials += [s for s in f.tracker_serials if s not in serials]
        rep.tracker_serials = serials
        tt = np.array([f.t_ns for f in trk_frames]) * 1e-9
        rep.tracker_rate_hz = float((len(tt) - 1) / max(tt[-1] - tt[0], 1e-6)) if len(tt) > 1 else 0.0

    rep.profile = select_profile(rep)
    if rep.body_tracking and declared_body_trackers is None:
        rep.notes.append("Body tracking is active (>= 2 trackers). The exact tracker count is not exposed "
                         "by XRoboToolkit; pass --body-trackers 3 if a waist tracker is worn.")
    if rep.tracker_hw == "unknown":
        rep.notes.append("Tracker generation (Beta Swift / 2024 Motion Tracker) cannot be read over the PC "
                         "service; declare it with --tracker-hw for the dataset metadata.")
    if not rep.body_tracking and rep.object_trackers == 0:
        rep.notes.append("Neither body tracking nor object trackers detected: 3-point (head + controllers) "
                         "profile; lower-body labels/retargeting will be disabled.")
    return rep


def select_profile(rep: ProbeReport) -> str:
    if not rep.connected:
        return "none"
    if rep.body_tracking:
        return "body_3trk" if (rep.declared_body_trackers or 2) >= 3 else "body_2trk"
    if rep.object_trackers > 0:
        return "sparse_trackers"
    if rep.headset_active:
        return "hmd_3pt"
    return "none"


def load_profiles(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
