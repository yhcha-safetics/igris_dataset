"""Tracking metrics + small quaternion helpers in pure numpy.

Used by BOTH evaluators so the numbers are computed by literally the same code:
  * scripts/isaaclab/play_eval.py      (Isaac Lab; tensors are moved to numpy every step)
  * igris_pipeline/sim/mujoco_policy.py (MuJoCo sim2sim)

Quaternions are wxyz. Quaternion helpers mirror isaaclab.utils.math semantics.

Per policy step (reference frame k paired with the robot state *after* the step):
  mpkpe_global          mean_b ||p_ref_b - p_rob_b||                       (tracked bodies, world frame) [m]
  mpkpe_root_relative   mean_b ||p_ref_b^rel - p_rob_b||, reference re-anchored to the robot anchor
                        (robot anchor xy, reference anchor z, yaw difference) = mjlab R-MPKPE      [m]
  joint_pos_error       mean_j |q_ref - q|                                                       [rad]
  joint_vel_error       sqrt(mean_j (qd_ref - qd)^2)  (RMS, as mjlab)                            [rad/s]
  ee_pos_error          mean_e ||p_ref_e^rel - p_rob_e||  hands (virtual points) + feet, re-anchored [m]
  anchor_pos_error      ||p_ref_anchor - p_rob_anchor||                                           [m]
  anchor_height_error   |z_ref_anchor - z_rob_anchor|                                             [m]
  anchor_ori_error      angle(q_ref_anchor * q_rob_anchor^-1)                                    [rad]
Episode: per-step values are averaged over the steps an env was alive; ``anchor_drift_xy_final`` is the xy
anchor error at the last scored step; ``success`` = all T frames of the clip tracked without termination.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

# --------------------------------------------------------------------------------------------------
# quaternion helpers (wxyz), broadcasting over leading dims
# --------------------------------------------------------------------------------------------------


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz = q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(q), v)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    two_s = 2.0 / np.sum(q * q, axis=-1)
    m = np.stack(
        [
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ],
        axis=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def rot6(q: np.ndarray) -> np.ndarray:
    """First two columns of R(q), row-major flattened: [R00,R01,R10,R11,R20,R21]."""
    m = matrix_from_quat(q)
    return m[..., :, :2].reshape(q.shape[:-1] + (6,))


def yaw_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.zeros_like(q)
    out[..., 0] = np.cos(yaw / 2.0)
    out[..., 3] = np.sin(yaw / 2.0)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    d = quat_mul(q1, quat_conj(q2))
    d = d * np.where(d[..., :1] < 0.0, -1.0, 1.0)
    return 2.0 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), d[..., 0])


def subtract_frame_transforms(p01, q01, p02, q02):
    """Pose of frame 2 expressed in frame 1 (isaaclab.utils.math.subtract_frame_transforms)."""
    q10 = quat_conj(q01)
    return quat_apply(q10, p02 - p01), quat_mul(q10, q02)


def relative_to_robot_anchor(ref_anchor_pos, ref_anchor_quat, rob_anchor_pos, rob_anchor_quat, pos, quat=None):
    """numpy twin of mdp.commands.relative_to_robot_anchor. Shapes: anchors (N,3)/(N,4), pos (N,K,3)."""
    dq = yaw_quat(quat_mul(rob_anchor_quat, quat_conj(ref_anchor_quat)))[:, None, :]
    dq = np.broadcast_to(dq, pos.shape[:-1] + (4,))
    dp = rob_anchor_pos.copy()
    dp[:, 2] = ref_anchor_pos[:, 2]
    rel_pos = dp[:, None, :] + quat_apply(dq, pos - ref_anchor_pos[:, None, :])
    rel_quat = quat_mul(dq, quat) if quat is not None else None
    return rel_pos, rel_quat


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------
STEP_METRICS = (
    "mpkpe_global",
    "mpkpe_root_relative",
    "joint_pos_error",
    "joint_vel_error",
    "ee_pos_error",
    "anchor_pos_error",
    "anchor_height_error",
    "anchor_ori_error",
)


def step_metrics(ref: dict[str, np.ndarray], rob: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-env metrics for one step.

    ref / rob keys: body_pos (N,K,3), joint_pos (N,J), joint_vel (N,J), anchor_pos (N,3), anchor_quat (N,4),
    ee_pos (N,E,3).  ``ref`` must also hold the reference ``body_pos`` / ``ee_pos`` in the same world frame.
    """
    rel_body, _ = relative_to_robot_anchor(ref["anchor_pos"], ref["anchor_quat"], rob["anchor_pos"], rob["anchor_quat"], ref["body_pos"])
    rel_ee, _ = relative_to_robot_anchor(ref["anchor_pos"], ref["anchor_quat"], rob["anchor_pos"], rob["anchor_quat"], ref["ee_pos"])
    out = {
        "mpkpe_global": np.linalg.norm(ref["body_pos"] - rob["body_pos"], axis=-1).mean(-1),
        "mpkpe_root_relative": np.linalg.norm(rel_body - rob["body_pos"], axis=-1).mean(-1),
        "joint_pos_error": np.abs(ref["joint_pos"] - rob["joint_pos"]).mean(-1),
        "joint_vel_error": np.sqrt(np.mean((ref["joint_vel"] - rob["joint_vel"]) ** 2, axis=-1)),
        "ee_pos_error": np.linalg.norm(rel_ee - rob["ee_pos"], axis=-1).mean(-1),
        "anchor_pos_error": np.linalg.norm(ref["anchor_pos"] - rob["anchor_pos"], axis=-1),
        "anchor_height_error": np.abs(ref["anchor_pos"][:, 2] - rob["anchor_pos"][:, 2]),
        "anchor_ori_error": quat_error_magnitude(ref["anchor_quat"], rob["anchor_quat"]),
        "_anchor_xy_error": np.linalg.norm(ref["anchor_pos"][:, :2] - rob["anchor_pos"][:, :2], axis=-1),
    }
    return out


class EpisodeAccumulator:
    """Accumulates per-env step metrics until each env succeeds or fails (one episode per env)."""

    def __init__(
        self,
        num_envs: int,
        motion_ids: np.ndarray,
        motion_names: list[str],
        motion_lengths: list[int],
        segments: dict[int, list[dict]] | None = None,
    ):
        """``segments``: optional {motion_id: [{"name", "start", "end", ...}, ...]} (frame indices, end exclusive)
        -> adds per-segment metrics to the per-motion summary."""
        self.n = num_envs
        self.segments = segments or {}
        self.frame_log: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(num_envs)]
        self.motion_ids = np.asarray(motion_ids).copy()
        self.motion_names = list(motion_names)
        self.motion_lengths = list(motion_lengths)
        self.sums = {k: np.zeros(num_envs) for k in STEP_METRICS}
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.done = np.zeros(num_envs, dtype=bool)
        self.success = np.zeros(num_envs, dtype=bool)
        self.fail_reason = [""] * num_envs
        self.final_xy = np.zeros(num_envs)

    def add(self, metrics: dict[str, np.ndarray], active: np.ndarray, frames: np.ndarray | None = None):
        """``frames``: reference frame index per env (needed for per-segment metrics)."""
        active = active & ~self.done
        for k in STEP_METRICS:
            self.sums[k][active] += metrics[k][active]
        self.final_xy[active] = metrics["_anchor_xy_error"][active]
        self.steps[active] += 1
        if frames is not None and self.segments:
            for i in np.where(active)[0]:
                if int(self.motion_ids[i]) in self.segments:
                    self.frame_log[i].append((int(frames[i]), np.array([metrics[k][i] for k in STEP_METRICS])))

    def _segment_summary(self, idx: np.ndarray, segs: list[dict]) -> list[dict]:
        entries = [(f, v) for i in idx for f, v in self.frame_log[i]]
        fail_frames = [self.frame_log[i][-1][0] if self.frame_log[i] else 0 for i in idx if self.done[i] and not self.success[i]]
        out = []
        for seg in segs:
            s0, s1 = int(seg["start"]), int(seg["end"])
            vals = [v for f, v in entries if s0 <= f < s1]
            d = {k: seg[k] for k in ("name", "start", "end") if k in seg}
            for extra in ("loco", "lower", "upper", "t_start", "t_end"):
                if extra in seg:
                    d[extra] = seg[extra]
            d["tracked_ratio"] = float(len(vals) / max(1, (s1 - s0) * len(idx)))
            d["failures_in_segment"] = int(sum(s0 <= f < s1 for f in fail_frames))
            if vals:
                mean = np.mean(np.stack(vals), axis=0)
                d.update({k: float(mean[j]) for j, k in enumerate(STEP_METRICS)})
            out.append(d)
        return out

    def finish(self, env_mask: np.ndarray, success: bool, reasons: list[str] | None = None):
        idx = np.where(env_mask & ~self.done)[0]
        for i in idx:
            self.done[i] = True
            self.success[i] = success
            if not success:
                self.fail_reason[i] = reasons[i] if reasons else "terminated"

    def _summ(self, idx: np.ndarray) -> dict:
        if len(idx) == 0:
            return {}
        steps = np.maximum(self.steps[idx], 1)
        out = {"num_episodes": int(len(idx)), "success_rate": float(self.success[idx].mean())}
        for k in STEP_METRICS:
            out[k] = float(np.mean(self.sums[k][idx] / steps))
        out["anchor_drift_xy_final"] = float(np.mean(self.final_xy[idx]))
        lengths = np.array([self.motion_lengths[m] for m in self.motion_ids[idx]])
        out["completion_ratio"] = float(np.mean(self.steps[idx] / lengths))
        out["num_falls"] = int(np.sum(~self.success[idx] & self.done[idx]))
        reasons = defaultdict(int)
        for i in idx:
            if self.done[i] and not self.success[i]:
                reasons[self.fail_reason[i]] += 1
        out["failure_reasons"] = dict(reasons)
        return out

    def summary(self) -> dict:
        all_idx = np.arange(self.n)
        res = self._summ(all_idx)
        res["unfinished_episodes"] = int(np.sum(~self.done))
        per_motion = {}
        for m, name in enumerate(self.motion_names):
            idx = np.where(self.motion_ids == m)[0]
            if len(idx):
                d = self._summ(idx)
                d["num_frames"] = int(self.motion_lengths[m])
                if m in self.segments:
                    d["segments"] = self._segment_summary(idx, self.segments[m])
                per_motion[name] = d
        res["per_motion"] = per_motion
        return res


def write_json(path: str, data: dict):
    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default)
