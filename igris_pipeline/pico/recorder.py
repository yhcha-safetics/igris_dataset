"""Record XRoboToolkit packets into episode files (stage 1).

Controller shortcuts while recording (configurable in `record()`):
  A (right primary)  : start / stop an episode
  B (right secondary): drop an event marker (stored in attrs["events"])
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..episode import Episode
from .xrt_source import FrameBuffer


def buffer_to_episode(buf: FrameBuffer, meta: dict) -> Episode:
    arrs = buf.to_arrays()
    serials = [s.decode() if isinstance(s, bytes) else str(s) for s in arrs.pop("_tracker_serials")]
    t = arrs["raw/t"]
    fps = float((len(t) - 1) / max(t[-1] - t[0], 1e-6)) if len(t) > 1 else 0.0
    ep = Episode(fps=fps, data=arrs)
    ep.attrs.update({k: v for k, v in meta.items() if not isinstance(v, (dict, list))})
    ep.set_json_attr("tracker_serials", [s for s in serials if s])
    ep.set_json_attr("events", buf.events)
    for k, v in meta.items():
        if isinstance(v, (dict, list)):
            ep.set_json_attr(k, v)
    return ep


def next_episode_index(out_dir: Path) -> int:
    """One past the highest existing ep_NNNN.h5 (counting files would overwrite after a deleted take)."""
    idx = [int(p.stem[3:]) for p in Path(out_dir).glob("ep_*.h5") if p.stem[3:].isdigit()]
    return max(idx) + 1 if idx else 0


class RecordingSession:
    """Episode recording fed one XRFrame at a time (used by record() and the live viewer).

    With use_buttons the A button toggles recording and B drops a marker; otherwise every
    frame is recorded until close().
    """

    def __init__(self, out_dir: str | Path, meta: dict, use_buttons: bool = True, autostart: bool = False,
                 with_hands: bool = False, on_status=print):
        self.out_dir = Path(out_dir)
        self.meta = meta
        self.use_buttons = use_buttons
        self.with_hands = with_hands
        self.on_status = on_status
        self.saved: list[Path] = []
        self.buf = FrameBuffer(with_hands=with_hands)
        self.recording = autostart or not use_buttons
        self._prev = {"A": False, "B": False}
        self._ep_idx = next_episode_index(self.out_dir)
        if self.recording:
            on_status("[rec] recording ...")

    def _flush(self):
        if len(self.buf) > 10:
            ep = buffer_to_episode(self.buf, self.meta)
            path = self.out_dir / f"ep_{self._ep_idx:04d}.h5"
            ep.save(path)
            self.saved.append(path)
            self.on_status(f"[rec] saved {path} ({len(self.buf)} frames, {ep.fps:.1f} Hz)")
            self._ep_idx += 1
        self.buf = FrameBuffer(with_hands=self.with_hands)

    def feed(self, fr) -> None:
        if self.use_buttons:
            a, b = bool(fr.buttons.get("A")), bool(fr.buttons.get("B"))
            if a and not self._prev["A"]:
                if self.recording:
                    self.recording = False
                    self._flush()
                    self.on_status("[rec] stopped")
                else:
                    self.recording = True
                    self.on_status("[rec] recording ...")
            if b and not self._prev["B"] and self.recording and len(self.buf):
                self.buf.events.append(((fr.t_ns - self.buf.frames[0].t_ns) * 1e-9, "marker"))
            self._prev = {"A": a, "B": b}
        if self.recording:
            self.buf.append(fr)

    def close(self) -> list[Path]:
        if self.recording or len(self.buf):
            self._flush()
        self.recording = False
        return self.saved


def record(source, out_dir: str | Path, meta: dict, max_seconds: float | None = None,
           use_buttons: bool = True, autostart: bool = False, with_hands: bool = False,
           on_status=print) -> list[Path]:
    """Record one or more episodes from `source` (XRTSource or SimulatedXRTSource).

    With use_buttons the A button toggles recording; otherwise recording runs from
    the start until max_seconds / Ctrl+C / end of a simulated stream.
    """
    rec = RecordingSession(out_dir, meta, use_buttons, autostart, with_hands, on_status)
    t_start = time.time()
    try:
        while True:
            if max_seconds is not None and time.time() - t_start > max_seconds:
                break
            try:
                fr = source.poll()
            except StopIteration:
                break
            if fr is None:
                time.sleep(0.001)
                continue
            rec.feed(fr)
    except KeyboardInterrupt:
        on_status("[rec] interrupted")
    return rec.close()


def describe_episode(path) -> str:
    ep = Episode.load(path)
    T = ep.num_frames
    lines = [f"{path}: {T} frames, {ep.fps:.1f} Hz, {ep['raw/t'][-1]:.1f} s"]
    lines.append(f"  profile={ep.attrs.get('tracker_profile')}  body_valid={ep['raw/body/valid'].mean():.2%}  "
                 f"trackers={json.loads(ep.attrs.get('tracker_serials', '[]'))}")
    gaps = np.diff(ep["raw/t"])
    lines.append(f"  dt median={np.median(gaps)*1e3:.1f} ms, max={gaps.max()*1e3:.1f} ms")
    return "\n".join(lines)
