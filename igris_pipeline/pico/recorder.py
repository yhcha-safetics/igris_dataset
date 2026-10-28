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


def next_episode_index(out_dir: str | Path) -> int:
    """Next free number after the highest existing ep_NNNN.h5 (a file count would collide after deletions)."""
    out_dir = Path(out_dir)
    nums = [int(p.stem[3:]) for p in out_dir.glob("ep_*.h5") if p.stem[3:].isdigit()] if out_dir.exists() else []
    return max(nums) + 1 if nums else 0


def record(source, out_dir: str | Path, meta: dict, max_seconds: float | None = None,
           use_buttons: bool = True, autostart: bool = False, with_hands: bool = False,
           on_status=print) -> list[Path]:
    """Record one or more episodes from `source` (XRTSource or SimulatedXRTSource).

    With use_buttons the A button toggles recording; otherwise recording runs from
    the start until max_seconds / Ctrl+C / end of a simulated stream.
    """
    out_dir = Path(out_dir)
    saved: list[Path] = []
    buf = FrameBuffer(with_hands=with_hands)
    recording = autostart or not use_buttons
    prev = {"A": False, "B": False}
    t_start = time.time()
    ep_idx = next_episode_index(out_dir)

    def flush():
        nonlocal buf, ep_idx
        if len(buf) > 10:
            ep = buffer_to_episode(buf, meta)
            path = out_dir / f"ep_{ep_idx:04d}.h5"
            ep.save(path)
            saved.append(path)
            on_status(f"[rec] saved {path} ({len(buf)} frames, {ep.fps:.1f} Hz)")
            ep_idx += 1
        buf = FrameBuffer(with_hands=with_hands)

    if recording:
        on_status("[rec] recording ...")
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
            if use_buttons:
                a, b = bool(fr.buttons.get("A")), bool(fr.buttons.get("B"))
                if a and not prev["A"]:
                    if recording:
                        recording = False
                        flush()
                        on_status("[rec] stopped")
                    else:
                        recording = True
                        on_status("[rec] recording ...")
                if b and not prev["B"] and recording and len(buf):
                    buf.events.append(((fr.t_ns - buf.frames[0].t_ns) * 1e-9, "marker"))
                prev = {"A": a, "B": b}
            if recording:
                buf.append(fr)
    except KeyboardInterrupt:
        on_status("[rec] interrupted")
    if recording or len(buf):
        flush()
    return saved


def describe_episode(path) -> str:
    ep = Episode.load(path)
    T = ep.num_frames
    lines = [f"{path}: {T} frames, {ep.fps:.1f} Hz, {ep['raw/t'][-1]:.1f} s"]
    lines.append(f"  profile={ep.attrs.get('tracker_profile')}  body_valid={ep['raw/body/valid'].mean():.2%}  "
                 f"trackers={json.loads(ep.attrs.get('tracker_serials', '[]'))}")
    gaps = np.diff(ep["raw/t"])
    lines.append(f"  dt median={np.median(gaps)*1e3:.1f} ms, max={gaps.max()*1e3:.1f} ms")
    return "\n".join(lines)
