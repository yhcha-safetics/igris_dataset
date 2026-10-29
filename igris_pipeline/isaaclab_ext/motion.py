"""Reference-motion loading for the IGRIS-C tracking task (numpy / torch only, no Isaac Lab).

NPZ format (stage-3 retargeter output, MuJoCo naming):
    fps, joint_names (J_mj,), joint_pos (T,J_mj), joint_vel (T,J_mj),
    body_names (B_mj,), body_pos_w (T,B_mj,3), body_quat_w (T,B_mj,4) [wxyz],
    body_lin_vel_w (T,B_mj,3), body_ang_vel_w (T,B_mj,3)          -- world frame, Z-up

Joints / bodies are mapped BY NAME onto the requested (articulation) order. Joints or bodies that
exist only in the NPZ (wrists, neck, Left_Hand, ...) are ignored unless explicitly requested.
Several clips are concatenated into one flat buffer; ``offsets`` / ``lengths`` index the clips.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_relative(paths: Sequence[str]) -> list[str]:
    """Paths inside the project as project-relative strings, so run outputs (policy.json, ...) stay
    valid when the project folder is moved or copied to another PC; other paths stay absolute."""
    out = []
    for p in paths:
        try:
            rel = os.path.relpath(os.path.abspath(p), PROJECT_ROOT)
        except ValueError:  # different drive
            rel = ".."
        out.append(os.path.abspath(p) if rel.startswith("..") else rel)
    return out


def from_project(paths: Sequence[str]) -> list[str]:
    """Inverse of project_relative()."""
    return [p if os.path.isabs(p) else str(PROJECT_ROOT / p) for p in paths]

REQUIRED_KEYS = (
    "fps",
    "joint_names",
    "joint_pos",
    "joint_vel",
    "body_names",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def resolve_motion_files(spec: str | Sequence[str]) -> list[str]:
    """Accept a file, a directory (all ``*.npz`` recursively, sorted), a glob, a comma separated list,
    a ``.txt`` file with one path per line, or a python list of any of those."""
    if isinstance(spec, (list, tuple)):
        out: list[str] = []
        for s in spec:
            out += resolve_motion_files(s)
        return out
    spec = str(spec).strip()
    if "," in spec:
        return resolve_motion_files([s for s in spec.split(",") if s.strip()])
    if os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, "**", "*.npz"), recursive=True))
    elif spec.endswith(".txt") and os.path.isfile(spec):
        base = os.path.dirname(os.path.abspath(spec))
        with open(spec, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        files = [p if os.path.isabs(p) else os.path.join(base, p) for p in lines]
    elif any(ch in spec for ch in "*?["):
        files = sorted(glob.glob(spec, recursive=True))
    else:
        files = [spec]
    files = [os.path.abspath(f) for f in files]
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"Motion file(s) not found: {missing}")
    if not files:
        raise FileNotFoundError(f"No motion files found for '{spec}'")
    return files


def _quat_nlerp(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    q = (1.0 - a) * q0 + a * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample_clip(data: dict[str, np.ndarray], src_fps: float, dst_fps: float) -> dict[str, np.ndarray]:
    """Linear (nlerp for quaternions) resampling of all time-series arrays to ``dst_fps``."""
    T = data["joint_pos"].shape[0]
    duration = (T - 1) / src_fps
    n = int(np.floor(duration * dst_fps + 1e-6)) + 1
    t = np.arange(n) / dst_fps * src_fps
    i0 = np.clip(np.floor(t).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    a = (t - i0).astype(np.float32)
    out = dict(data)
    for k in ("joint_pos", "joint_vel", "body_pos_w", "body_lin_vel_w", "body_ang_vel_w"):
        x = data[k]
        aa = a.reshape((-1,) + (1,) * (x.ndim - 1))
        out[k] = (1.0 - aa) * x[i0] + aa * x[i1]
    out["body_quat_w"] = _quat_nlerp(data["body_quat_w"][i0], data["body_quat_w"][i1], a[:, None, None])
    return out


def load_clip(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        missing = [k for k in REQUIRED_KEYS if k not in f.files]
        if missing:
            raise KeyError(f"{path}: missing keys {missing}")
        data = {k: f[k] for k in f.files}
    data["joint_names"] = [str(s) for s in data["joint_names"].tolist()]
    data["body_names"] = [str(s) for s in data["body_names"].tolist()]
    data["fps"] = float(np.asarray(data["fps"]).reshape(-1)[0])
    return data


@dataclass
class MotionLibInfo:
    files: list[str]
    lengths: list[int]
    fps: float


class MotionLoader:
    """Loads one or many NPZ clips, maps them by name and stores them as flat torch tensors.

    Args:
        motion_files: anything accepted by :func:`resolve_motion_files`.
        joint_names: target joint order (e.g. ``robot.joint_names``) -- every name must exist in the NPZ.
        body_names: target body order -- every name must exist in the NPZ.
        fps: target frame rate (policy rate); clips with a different fps are resampled.
        device: torch device.
        z_offset: constant added to every body z (e.g. -0.009 for clips whose soles follow the MJCF foot box,
            which is 9 mm lower than the URDF/Isaac foot box). Default 0.
    """

    def __init__(
        self,
        motion_files: str | Sequence[str],
        joint_names: Sequence[str],
        body_names: Sequence[str],
        fps: float,
        device: str | torch.device = "cpu",
        z_offset: float = 0.0,
    ):
        import torch  # lazy: the numpy helpers of this module are used torch-free by the MuJoCo runner

        self.files = resolve_motion_files(motion_files)
        self.joint_names = list(joint_names)
        self.body_names = list(body_names)
        self.fps = float(fps)
        self.device = device

        jp, jv, bp, bq, bl, ba, lengths = [], [], [], [], [], [], []
        for path in self.files:
            clip = load_clip(path)
            if abs(clip["fps"] - self.fps) > 1e-6:
                clip = resample_clip(clip, clip["fps"], self.fps)
            j_idx = self._map(clip["joint_names"], self.joint_names, "joint", path)
            b_idx = self._map(clip["body_names"], self.body_names, "body", path)
            jp.append(clip["joint_pos"][:, j_idx])
            jv.append(clip["joint_vel"][:, j_idx])
            bp.append(clip["body_pos_w"][:, b_idx])
            quat = clip["body_quat_w"][:, b_idx]
            bq.append(quat / np.linalg.norm(quat, axis=-1, keepdims=True))
            bl.append(clip["body_lin_vel_w"][:, b_idx])
            ba.append(clip["body_ang_vel_w"][:, b_idx])
            if clip["joint_pos"].shape[0] < 2:
                raise ValueError(f"{path}: motion needs at least 2 frames")
            lengths.append(clip["joint_pos"].shape[0])

        def cat(xs):
            return torch.tensor(np.concatenate(xs, axis=0), dtype=torch.float32, device=device)

        self.joint_pos = cat(jp)
        self.joint_vel = cat(jv)
        self.body_pos_w = cat(bp)
        self.z_offset = float(z_offset)
        if self.z_offset != 0.0:
            self.body_pos_w[..., 2] += self.z_offset
        self.body_quat_w = cat(bq)
        self.body_lin_vel_w = cat(bl)
        self.body_ang_vel_w = cat(ba)
        self.lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self.offsets = torch.cumsum(self.lengths, dim=0) - self.lengths
        self.num_motions = len(self.files)
        self.total_frames = int(self.lengths.sum().item())
        self.motion_names = [os.path.splitext(os.path.basename(f))[0] for f in self.files]

    @staticmethod
    def _map(src: list[str], dst: list[str], kind: str, path: str) -> list[int]:
        lookup = {n: i for i, n in enumerate(src)}
        missing = [n for n in dst if n not in lookup]
        if missing:
            raise KeyError(f"{path}: {kind}(s) {missing} required by the articulation are not in the motion file")
        return [lookup[n] for n in dst]

    def body_index(self, name: str) -> int:
        return self.body_names.index(name)

    def global_index(self, motion_ids, time_steps):
        return self.offsets[motion_ids] + time_steps

    def info(self) -> MotionLibInfo:
        return MotionLibInfo(self.files, self.lengths.tolist(), self.fps)


def available_body_names(motion_files: str | Sequence[str]) -> set[str]:
    """Intersection of body names over all clips (used to decide which EE points exist in the NPZ)."""
    names: set[str] | None = None
    for path in resolve_motion_files(motion_files):
        with np.load(path, allow_pickle=False) as f:
            cur = {str(s) for s in f["body_names"].tolist()}
        names = cur if names is None else names & cur
    return names or set()


def load_segments(path: str, target_fps: float | None = None) -> list[dict]:
    """Label segments from the optional NPZ ``meta`` JSON string (key ``segments``: dicts with ``start`` /
    ``end`` frame indices, end exclusive). Indices are rescaled if the clip is resampled to ``target_fps``."""
    import json

    with np.load(path, allow_pickle=False) as f:
        if "meta" not in f.files:
            return []
        try:
            meta = json.loads(str(f["meta"]))
        except (ValueError, TypeError):
            return []
        fps = float(np.asarray(f["fps"]).reshape(-1)[0])
    segs = meta.get("segments", []) if isinstance(meta, dict) else []
    scale = 1.0 if target_fps is None else target_fps / fps
    out = []
    for s in segs:
        if "start" in s and "end" in s:
            s = dict(s)
            s["start"] = int(round(s["start"] * scale))
            s["end"] = int(round(s["end"] * scale))
            out.append(s)
    return out

