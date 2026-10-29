"""Episode container and HDF5 schema shared by every pipeline stage.

Layout of one episode file (``data/<session>/raw/ep_XXXX.h5``). Positions in metres, world Z-up,
+X forward, quaternions wxyz (converted from PICO's Y-up xyzw by pico/frames.py)::

    attrs: schema_version, fps, created, subject, tracker_profile, recorded_at, [subject_height],
           device_profile(json), tracker_serials(json), events(json: [t_s, "marker"])
    raw/t, raw/t_ns           (T,)      seconds from the first packet / headset clock in ns
    raw/headset/{pos,quat,valid}                    (T,3) (T,4) (T,)
    raw/controller/{left,right}/{pos,quat,valid}    (T,3) (T,4) (T,)
    raw/body/{pos,quat}       (T,24,3) (T,24,4)  PICO body joints (skeleton.py order)
    raw/body/valid            (T,)      a fresh body-solver packet arrived with this frame
    raw/body/joint_ts         (T,24)    per-joint IMU timestamps
    raw/tracker/{pos,quat,valid}  (T,3,3) (T,3,4) (T,3)  object-mode trackers (tracker_serials order)
    raw/buttons/{A,B,X,Y,left_trigger,right_trigger,left_grip,right_grip}  (T,)
    raw/hand/{left,right}/{pos,quat,active}  (T,26,3) (T,26,4) (T,)   only with --hands
    proc/t                    (T',)     uniform rate (group attr proc.fps), stage 1-2
    proc/body/{pos,quat}      (T',24,3) (T',24,4) corrected, world-aligned skeleton
    proc/body/confidence      (T',24)   per-joint confidence 0..1
    proc/contact              (T',2)    left / right foot contact
    proc/valid                (T',)     False inside tracking dropouts longer than max_gap_s
    proc/{headset,controller/left,controller/right}/{pos,quat}  aligned sensor poses
      group attrs proc: fps, profile, calibration(json), report(json), observed_axes(json)
    labels/{loco,lower,upper} (T',)     int16 label index per frame, stage 2
      group attrs labels: {axis}_names(json), segments(json: start,end,t_start,t_end,loco,lower,upper,
                          name,confidence,needs_review), reviewed (set by --import-review)
    retarget/igris_c/qpos     (T',38)   root xyz + root wxyz + 31 joints (MJCF order), stage 3
      group attrs retarget/igris_c: fps, joint_names(json), qpos_layout, diagnostics(json)

Every stage only *adds* groups, so an episode can be re-processed from any stage.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCHEMA_VERSION = 1


@dataclass
class Episode:
    fps: float
    data: dict[str, np.ndarray] = field(default_factory=dict)  # "raw/body/pos" -> array
    attrs: dict[str, Any] = field(default_factory=dict)        # root attrs
    group_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- convenience accessors -------------------------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        self.data[key] = np.asarray(value)

    def __contains__(self, key: str) -> bool:
        return key in self.data

    @property
    def num_frames(self) -> int:
        return int(self.data["raw/t"].shape[0])

    def body(self, stage: str = "proc") -> tuple[np.ndarray, np.ndarray]:
        """Return (pos, quat) of the best available skeleton."""
        if stage == "proc" and "proc/body/pos" in self.data:
            return self.data["proc/body/pos"], self.data["proc/body/quat"]
        return self.data["raw/body/pos"], self.data["raw/body/quat"]

    def set_json_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = json.dumps(value, ensure_ascii=False)

    def get_json_attr(self, key: str, default: Any = None) -> Any:
        v = self.attrs.get(key)
        if v is None:
            return default
        return json.loads(v) if isinstance(v, (str, bytes)) else v

    # ---- IO ---------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["fps"] = float(self.fps)
            f.attrs.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
            for k, v in self.attrs.items():
                f.attrs[k] = v
            for k, arr in self.data.items():
                arr = np.asarray(arr)
                kw = {"compression": "gzip", "compression_opts": 4} if arr.size > 1024 else {}
                f.create_dataset(k, data=arr, **kw)
            for g, ga in self.group_attrs.items():
                grp = f.require_group(g)
                for k, v in ga.items():
                    grp.attrs[k] = v
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        data: dict[str, np.ndarray] = {}
        group_attrs: dict[str, dict[str, Any]] = {}
        with h5py.File(path, "r") as f:
            attrs = {k: _decode(v) for k, v in f.attrs.items()}

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data[name] = obj[()]
                elif isinstance(obj, h5py.Group) and len(obj.attrs):
                    group_attrs[name] = {k: _decode(v) for k, v in obj.attrs.items()}

            f.visititems(visit)
        ep = cls(fps=float(attrs.get("fps", 0.0)), data=data, attrs=attrs, group_attrs=group_attrs)
        ep.attrs["source_path"] = str(path)
        return ep


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
        return [x.decode("utf-8") if isinstance(x, bytes) else x for x in v.tolist()]
    return v


def list_episodes(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("*.h5"))
