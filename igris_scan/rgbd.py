"""RGB-D captures for scanning a scene: the on-disk format, readers, writers and depth clean-up.

    <capture>/
      capture.json    {"format": "igris_rgbd", "version": 1, "source": "realsense" | "synthetic" | ...,
                       "device": "...", "fps": 30.0, "depth_scale": 1000.0,
                       "intrinsics": {"width", "height", "fx", "fy", "cx", "cy"}}
      frames.csv      index,t_s           one row per frame, t_s in seconds from the first frame
      color/%06d.png  8-bit RGB (.jpg is read too), registered to the depth image (same intrinsics)
      depth/%06d.png  16-bit; value / depth_scale = metres along the optical axis, 0 = no data
      poses.txt       optional: per frame a camera-to-world 4x4 matrix (16 numbers, row major) from an
                      external tracker, "nan" x 16 for frames without one; otherwise the camera is
                      tracked (igris_scan.reconstruct)

Camera axes are OpenCV's: x right, y down, z forward.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

FORMAT, VERSION = "igris_rgbd", 1


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        return cls(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]))

    def to_dict(self) -> dict:
        return asdict(self)


def clean_depth(depth: np.ndarray, min_depth: float = 0.15, max_depth: float = 4.0,
                edge: float = 0.04) -> np.ndarray:
    """Metres -> metres with unreliable pixels set to 0: out of range, and "flying pixels" on depth
    edges (a 4-neighbour differs by more than `edge` x depth), which smear foreground and background
    together in stereo and time-of-flight cameras."""
    d = np.asarray(depth, np.float32).copy()
    d[~np.isfinite(d) | (d < min_depth) | (d > max_depth)] = 0.0
    p = np.pad(d, 1, mode="edge")
    jump = np.zeros_like(d)
    for nb in (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]):
        jump = np.maximum(jump, np.where(nb > 0, np.abs(d - nb), 0.0))
    d[jump > edge * d] = 0.0
    return d


class Capture:
    """A capture folder (read only)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        meta = json.loads((self.folder / "capture.json").read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{self.folder}: not an {FORMAT} capture")
        self.meta = meta
        self.intr = Intrinsics.from_dict(meta["intrinsics"])
        self.depth_scale = float(meta.get("depth_scale", 1000.0))
        with open(self.folder / "frames.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.index = [int(r["index"]) for r in rows]
        self.t = np.array([float(r["t_s"]) for r in rows])
        self.poses = None
        pp = self.folder / "poses.txt"
        if pp.exists():
            self.poses = np.loadtxt(pp, ndmin=2).reshape(-1, 4, 4)
            if len(self.poses) != len(self.index):
                raise ValueError(f"{pp}: {len(self.poses)} poses for {len(self.index)} frames")

    def __len__(self) -> int:
        return len(self.index)

    def _file(self, kind: str, i: int) -> Path:
        stem = f"{self.index[i]:06d}"
        for ext in (".png", ".jpg", ".jpeg"):
            p = self.folder / kind / (stem + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"{self.folder / kind / stem}.*")

    def color(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self._file("color", i)), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def depth_raw(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self._file("depth", i)), cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16:
            raise ValueError(f"depth frame {i}: expected a 16-bit PNG")
        return img

    def depth(self, i: int, clean: bool = True, **kw) -> np.ndarray:
        d = self.depth_raw(i).astype(np.float32) / self.depth_scale
        return clean_depth(d, **kw) if clean else d

    def frames(self, start_s: float | None = None, end_s: float | None = None, step: int = 1) -> list[int]:
        """Frame numbers (0..len-1) with start_s <= t < end_s."""
        lo = -np.inf if start_s is None else start_s
        hi = np.inf if end_s is None else end_s
        return [i for i in range(0, len(self), max(1, int(step))) if lo <= self.t[i] < hi]


class CaptureWriter:
    """Write a capture frame by frame (with-statement; the metadata is written on close)."""

    def __init__(self, folder: str | Path, intr: Intrinsics, depth_scale: float = 1000.0, source: str = "",
                 device: str = "", fps: float | None = None, color_ext: str = ".png"):
        self.folder = Path(folder)
        (self.folder / "color").mkdir(parents=True, exist_ok=True)
        (self.folder / "depth").mkdir(parents=True, exist_ok=True)
        self.intr, self.depth_scale, self.color_ext = intr, float(depth_scale), color_ext
        self.meta = {"format": FORMAT, "version": VERSION, "source": source, "device": device,
                     "fps": fps, "depth_scale": self.depth_scale, "intrinsics": intr.to_dict()}
        self.t, self.poses = [], []

    def add(self, color_rgb: np.ndarray, depth: np.ndarray, t_s: float, pose: np.ndarray | None = None) -> None:
        """depth: uint16 in capture units, or float metres."""
        k = len(self.t)
        if depth.dtype != np.uint16:
            depth = np.clip(np.nan_to_num(depth) * self.depth_scale + 0.5, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(self.folder / "color" / f"{k:06d}{self.color_ext}"), cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.folder / "depth" / f"{k:06d}.png"), depth)
        self.t.append(float(t_s))
        self.poses.append(None if pose is None else np.asarray(pose, float).reshape(4, 4))

    def close(self) -> Path:
        t0 = self.t[0] if self.t else 0.0
        with open(self.folder / "frames.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "t_s"])
            w.writerows([k, f"{t - t0:.6f}"] for k, t in enumerate(self.t))
        if any(p is not None for p in self.poses):
            rows = [np.full(16, np.nan) if p is None else p.ravel() for p in self.poses]
            np.savetxt(self.folder / "poses.txt", np.array(rows), fmt="%.9g")
        if self.meta["fps"] is None and len(self.t) > 1:
            self.meta["fps"] = round((len(self.t) - 1) / max(self.t[-1] - t0, 1e-9), 3)
        (self.folder / "capture.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        return self.folder

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def import_realsense_bag(bag: str | Path, out: str | Path, every: int = 1, max_frames: int | None = None,
                         start_s: float = 0.0) -> Path:
    """Convert a RealSense recording (.bag, e.g. from realsense-viewer) to a capture: depth aligned
    to the colour camera, the colour intrinsics, device timestamps."""
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    rs.config.enable_device_from_file(cfg, str(bag), repeat_playback=False)
    cfg.enable_stream(rs.stream.depth)
    cfg.enable_stream(rs.stream.color)
    prof = pipe.start(cfg)
    prof.get_device().as_playback().set_real_time(False)
    align = rs.align(rs.stream.color)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()  # metres per unit
    cp = prof.get_stream(rs.stream.color).as_video_stream_profile()
    ci = cp.get_intrinsics()
    intr = Intrinsics(ci.width, ci.height, ci.fx, ci.fy, ci.ppx, ci.ppy)
    dev = prof.get_device().get_info(rs.camera_info.name)
    w = CaptureWriter(out, intr, depth_scale=1.0 / scale, source="realsense", device=dev)
    n, t0 = 0, None
    try:
        while max_frames is None or len(w.t) < max_frames:
            ok, fs = pipe.try_wait_for_frames(2000)
            if not ok:
                break
            fs = align.process(fs)
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if not d or not c:
                continue
            t = fs.get_timestamp() * 1e-3
            t0 = t if t0 is None else t0
            n += 1
            if t - t0 < start_s or (n - 1) % every:
                continue
            color = np.asanyarray(c.get_data())
            if c.get_profile().format() == rs.format.bgr8:
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            w.add(color, np.asanyarray(d.get_data()).copy(), t)
    finally:
        pipe.stop()
        w.close()
    return Path(out)
