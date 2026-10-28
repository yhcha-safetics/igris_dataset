"""Tracking sources for the simulation server (all with the XRTSource poll() interface).

    xrt                 live PICO data through the XRoboToolkit PC service
    sim:<profile>[:seed]  simulated stream (igris_pipeline.pico.simulate), e.g. sim:body_2trk
    replay:<episode.h5>   the raw/* streams of a recorded episode
    udp                 the PICO app over the server's UDP port (frames come from protocol.Hub)
"""
from __future__ import annotations

import json
import time

import numpy as np

from igris_pipeline.episode import Episode
from igris_pipeline.pico.frames import pipe_to_pose7
from igris_pipeline.pico.xrt_source import XRFrame


class EpisodeReplaySource:
    """Re-emit the raw/* streams of an episode as native-frame XRFrames."""

    def __init__(self, path, loop: bool = False):
        ep = Episode.load(path)
        self.loop = loop
        self.T = ep.num_frames
        self.t_ns = np.asarray(ep["raw/t_ns"], np.int64)

        def pose7(prefix, valid=None):
            p7 = pipe_to_pose7(ep[f"{prefix}/pos"], ep[f"{prefix}/quat"])
            v = np.asarray(ep[f"{prefix}/valid"] if valid is None else valid, bool)
            return np.where(v.reshape(v.shape + (1,) * (p7.ndim - v.ndim)), p7, 0.0)

        self.hmd = pose7("raw/headset")
        self.ctrl = {s: pose7(f"raw/controller/{s}") for s in ("left", "right")}
        self.body = pose7("raw/body", ep["raw/body/valid"]) if "raw/body/pos" in ep else None
        self.body_valid = np.asarray(ep["raw/body/valid"], bool) if self.body is not None else None
        self.body_ts = np.asarray(ep["raw/body/joint_ts"], np.int64) if "raw/body/joint_ts" in ep else None
        self.trk = pose7("raw/tracker") if "raw/tracker/pos" in ep else None
        self.trk_valid = np.asarray(ep["raw/tracker/valid"], bool) if self.trk is not None else None
        serials = ep.attrs.get("tracker_serials", "[]")
        self.serials = json.loads(serials) if isinstance(serials, str) else list(serials)
        self.buttons = {k.split("/")[-1]: np.asarray(ep[k]) for k in ep.data if k.startswith("raw/buttons/")}
        self.axes = {k.split("/")[-1]: np.asarray(ep[k]) for k in ep.data if k.startswith("raw/axes/")}
        self._i = 0
        self._offset = 0

    def open(self):
        return self

    def close(self):
        pass

    def poll(self) -> XRFrame:
        if self._i >= self.T:
            if not self.loop:
                raise StopIteration
            self._offset += int(self.t_ns[-1] - self.t_ns[0]) + 20_000_000
            self._i = 0
        i = self._i
        self._i += 1
        fr = XRFrame(t_ns=int(self.t_ns[i]) + self._offset, headset=self.hmd[i], left_ctrl=self.ctrl["left"][i],
                     right_ctrl=self.ctrl["right"][i])
        fr.buttons = {k: (float(v[i]) if v.dtype.kind == "f" else bool(v[i])) for k, v in self.buttons.items()}
        fr.axes = {k: v[i].astype(np.float64) for k, v in self.axes.items()}
        if self.body is not None and self.body_valid[i]:
            fr.body = self.body[i]
            fr.body_joint_ts = (self.body_ts[i] if self.body_ts is not None
                                else np.full(24, fr.t_ns // 1_000_000, dtype=np.int64))
        if self.trk is not None and self.trk_valid[i].any():
            rows = [k for k in range(self.trk.shape[1]) if self.trk_valid[i, k] and k < len(self.serials)]
            if rows:
                fr.trackers = self.trk[i, rows]
                fr.tracker_serials = [self.serials[k] for k in rows]
        return fr


class PacedSource:
    """Release a recorded / simulated source's frames at their original pace (wall clock x speed)."""

    def __init__(self, source, speed: float = 1.0):
        self.src = source
        self.speed = float(speed)
        self._pending = None
        self._t0 = None

    def open(self):
        return self

    def close(self):
        self.src.close()

    def poll(self):
        if self._pending is None:
            self._pending = self.src.poll()  # StopIteration ends the stream
        fr = self._pending
        now = time.time()
        if self._t0 is None:
            self._t0 = (fr.t_ns, now)
        if (fr.t_ns - self._t0[0]) * 1e-9 <= (now - self._t0[1]) * self.speed:
            self._pending = None
            return fr
        return None


def make_source(spec: str, seed: int = 0):
    """Source for a --source string; None for 'udp' (frames arrive through protocol.Hub)."""
    kind, _, arg = spec.partition(":")
    if kind == "udp":
        return None
    if kind == "xrt":
        from igris_pipeline.pico.xrt_source import XRTSource
        return XRTSource(with_hands=False).open()
    if kind == "sim":
        from igris_pipeline.pico.simulate import SimulatedXRTSource
        profile, _, s = (arg or "body_2trk").partition(":")
        return SimulatedXRTSource(profile, seed=int(s) if s else seed)
    if kind == "replay":
        return EpisodeReplaySource(arg)
    raise ValueError(f"unknown source {spec!r} (xrt | sim:<profile> | replay:<episode.h5> | udp)")
