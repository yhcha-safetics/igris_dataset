"""Live PICO data through XRoboToolkit (stage 1).

Requirements (see README "PICO 연결"):
  * PC: XRoboToolkit-PC-Service running (RoboticsServiceProcess.exe, Windows build)
  * Headset: XRoboToolkit Unity client APK (use the *-SDK29.apk build on PICO 4),
    PC IP entered, Motion Tracker set to "Full body" for body tracking, "Send" toggled on
  * Python: ``pip install xrobotoolkit-sdk`` (>=3.10)

Known quirks of the SDK that this module works around:
  * ``is_body_data_available()`` latches True forever -> freshness is detected from the
    per-joint IMU timestamps / pose changes instead.
  * ``get_body_timestamp_ns()`` / ``get_motion_timestamp_ns()`` are not filled by the
    client -> ``get_time_stamp_ns()`` (headset clock, ms resolution) is the frame time.
  * body velocity / acceleration fields are mislabelled and partly corrupted by the
    vendored PICO SDK -> derivatives are recomputed from poses downstream.
  * object-mode trackers are limited to 3 by the bindings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .frames import pose7_to_pipe

MAX_TRACKERS = 3


@dataclass
class XRFrame:
    t_ns: int
    headset: np.ndarray                 # (7,) native
    left_ctrl: np.ndarray               # (7,)
    right_ctrl: np.ndarray              # (7,)
    buttons: dict = field(default_factory=dict)
    body: np.ndarray | None = None      # (24,7) native, None if not fresh
    body_joint_ts: np.ndarray | None = None
    trackers: np.ndarray | None = None  # (N,7)
    tracker_serials: list[str] = field(default_factory=list)
    left_hand_active: int = 0
    right_hand_active: int = 0
    left_hand: np.ndarray | None = None  # (26,7)
    right_hand: np.ndarray | None = None


class XRTSource:
    """Polls xrobotoolkit_sdk and yields only *new* packets."""

    def __init__(self, with_hands: bool = False):
        import xrobotoolkit_sdk as xrt  # imported lazily so the rest works without it

        self.xrt = xrt
        self.with_hands = with_hands
        self._last_ts = None
        self._last_body_ts = None
        self._last_body = None
        self._opened = False

    def open(self):
        self.xrt.init()
        self._opened = True
        return self

    def close(self):
        if self._opened:
            self.xrt.close()
            self._opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    def poll(self) -> XRFrame | None:
        x = self.xrt
        ts = int(x.get_time_stamp_ns())
        if ts == 0 or ts == self._last_ts:
            return None
        self._last_ts = ts
        fr = XRFrame(
            t_ns=ts,
            headset=np.asarray(x.get_headset_pose(), dtype=np.float64),
            left_ctrl=np.asarray(x.get_left_controller_pose(), dtype=np.float64),
            right_ctrl=np.asarray(x.get_right_controller_pose(), dtype=np.float64),
            buttons={
                "left_trigger": x.get_left_trigger(), "right_trigger": x.get_right_trigger(),
                "left_grip": x.get_left_grip(), "right_grip": x.get_right_grip(),
                "A": x.get_A_button(), "B": x.get_B_button(), "X": x.get_X_button(), "Y": x.get_Y_button(),
                "left_menu": x.get_left_menu_button(), "right_menu": x.get_right_menu_button(),
            },
            left_hand_active=int(x.get_left_hand_is_active()),
            right_hand_active=int(x.get_right_hand_is_active()),
        )
        if x.is_body_data_available():
            body = np.asarray(x.get_body_joints_pose(), dtype=np.float64)
            jts = np.asarray(x.get_body_joints_timestamp(), dtype=np.int64)
            fresh = (self._last_body_ts is None or np.any(jts != self._last_body_ts)
                     or not np.array_equal(body, self._last_body))
            if fresh:
                fr.body, fr.body_joint_ts = body, jts
            self._last_body_ts, self._last_body = jts, body
        n = int(x.num_motion_data_available())
        if n > 0:
            fr.trackers = np.asarray(x.get_motion_tracker_pose(), dtype=np.float64).reshape(-1, 7)[:MAX_TRACKERS]
            fr.tracker_serials = list(x.get_motion_tracker_serial_numbers())[:MAX_TRACKERS]
        if self.with_hands:
            if fr.left_hand_active:
                fr.left_hand = np.asarray(x.get_left_hand_tracking_state(), dtype=np.float64)
            if fr.right_hand_active:
                fr.right_hand = np.asarray(x.get_right_hand_tracking_state(), dtype=np.float64)
        return fr

    def frames(self, poll_hz: float = 500.0):
        period = 1.0 / poll_hz
        while True:
            fr = self.poll()
            if fr is not None:
                yield fr
            else:
                time.sleep(period)


class FrameBuffer:
    """Accumulates XRFrames and converts them to the episode `raw/*` arrays."""

    def __init__(self, with_hands: bool = False):
        self.frames: list[XRFrame] = []
        self.with_hands = with_hands
        self.events: list[tuple[float, str]] = []

    def append(self, fr: XRFrame):
        self.frames.append(fr)

    def __len__(self):
        return len(self.frames)

    def to_arrays(self) -> dict[str, np.ndarray]:
        F = self.frames
        T = len(F)
        t0 = F[0].t_ns
        out: dict[str, np.ndarray] = {"raw/t": np.array([(f.t_ns - t0) * 1e-9 for f in F])}
        out["raw/t_ns"] = np.array([f.t_ns for f in F], dtype=np.int64)

        def pose_block(key, arrs):
            p, q, v = pose7_to_pipe(np.stack(arrs))
            out[f"{key}/pos"], out[f"{key}/quat"], out[f"{key}/valid"] = p, q, v

        pose_block("raw/headset", [f.headset for f in F])
        pose_block("raw/controller/left", [f.left_ctrl for f in F])
        pose_block("raw/controller/right", [f.right_ctrl for f in F])

        body = np.zeros((T, 24, 7))
        bvalid = np.zeros(T, dtype=bool)
        bts = np.zeros((T, 24), dtype=np.int64)
        for i, f in enumerate(F):
            if f.body is not None:
                body[i], bvalid[i], bts[i] = f.body, True, f.body_joint_ts
        p, q, v = pose7_to_pipe(body)
        out["raw/body/pos"], out["raw/body/quat"] = p, q
        out["raw/body/valid"] = bvalid & v.all(axis=-1)
        out["raw/body/joint_ts"] = bts

        trk = np.zeros((T, MAX_TRACKERS, 7))
        serials: list[str] = []
        for f in F:
            for s in f.tracker_serials:
                if s not in serials:
                    serials.append(s)
        for i, f in enumerate(F):
            if f.trackers is None:
                continue
            for k, s in enumerate(f.tracker_serials):
                if s in serials[:MAX_TRACKERS]:
                    trk[i, serials.index(s)] = f.trackers[k]
        p, q, v = pose7_to_pipe(trk)
        out["raw/tracker/pos"], out["raw/tracker/quat"], out["raw/tracker/valid"] = p, q, v
        out["_tracker_serials"] = np.array(serials[:MAX_TRACKERS] or [""], dtype="S32")

        for k in ("left_trigger", "right_trigger", "left_grip", "right_grip"):
            out[f"raw/buttons/{k}"] = np.array([f.buttons.get(k, 0.0) for f in F], dtype=np.float32)
        for k in ("A", "B", "X", "Y"):
            out[f"raw/buttons/{k}"] = np.array([bool(f.buttons.get(k, False)) for f in F])
        if self.with_hands:
            for side in ("left", "right"):
                h = np.zeros((T, 26, 7))
                act = np.zeros(T, dtype=np.int8)
                for i, f in enumerate(F):
                    arr = getattr(f, f"{side}_hand")
                    if arr is not None:
                        h[i], act[i] = arr, getattr(f, f"{side}_hand_active")
                p, q, _ = pose7_to_pipe(h)
                out[f"raw/hand/{side}/pos"], out[f"raw/hand/{side}/quat"] = p, q
                out[f"raw/hand/{side}/active"] = act
        return out
