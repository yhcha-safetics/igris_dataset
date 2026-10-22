"""Episode container and HDF5 schema shared by every pipeline stage.

Layout of one episode file (``data/<session>/<episode>.h5``)::

    attrs: schema_version, fps, source, subject, created, device_profile(json),
           tracker_profile, calibration(json)
    raw/t                     (T,)      seconds, device clock
    raw/body/pos              (T,24,3)  PICO body joints, world Z-up, meters
    raw/body/quat             (T,24,4)  wxyz
    raw/body/valid            (T,)      bool, body solver produced data
    raw/headset/pose          (T,7)     xyz + wxyz
    raw/controller/{left,right}/pose   (T,7)
    raw/controller/{left,right}/valid  (T,)
    raw/tracker/pose          (T,K,7)   motion trackers (object-tracking mode)
    raw/tracker/valid         (T,K)
    proc/body/pos|quat        corrected skeleton (stage 1-2)
    proc/body/confidence      (T,24)    per-joint confidence 0..1
    labels/frame              (T,)      int label id per frame (stage 2)
    labels/segments           json attr list of {start,end,label,score}
    retarget/<robot>/qpos     (T,nq)    root xyz + root wxyz + joint angles (stage 3)
    retarget/<robot>/...      joint_names attr, fps attr, ik residuals

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
