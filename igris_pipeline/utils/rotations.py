"""Quaternion / rotation helpers.

Internal convention for the whole pipeline:
  * quaternions are stored as **wxyz** (MuJoCo / Isaac Lab / BeyondMimic convention)
  * positions in meters, world frame is right-handed **Z-up** (MuJoCo / Isaac)
  * arrays are vectorised over leading dims: (..., 4) / (..., 3)

scipy's Rotation uses xyzw, so every conversion goes through the helpers below.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, 1e-12, None)


def quat_to_rot(q_wxyz: np.ndarray) -> R:
    flat = np.asarray(q_wxyz, dtype=np.float64).reshape(-1, 4)
    return R.from_quat(wxyz_to_xyzw(flat))


def rot_to_quat(rot: R, shape=None) -> np.ndarray:
    q = xyzw_to_wxyz(rot.as_quat())
    return q.reshape(*shape, 4) if shape is not None else q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b, wxyz."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1.0
    return q


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (...,3) by quaternions q (...,4, wxyz)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle (rad) between two orientations."""
    d = np.abs(np.sum(quat_normalize(a) * quat_normalize(b), axis=-1))
    return 2.0 * np.arccos(np.clip(d, -1.0, 1.0))


def quat_continuity(q: np.ndarray, axis: int = 0) -> np.ndarray:
    """Flip signs so consecutive quaternions along `axis` lie in the same hemisphere.

    Needed before any filtering / interpolation of quaternion time series.
    """
    q = np.moveaxis(np.asarray(q, dtype=np.float64).copy(), axis, 0)
    for t in range(1, q.shape[0]):
        flip = np.sum(q[t] * q[t - 1], axis=-1) < 0.0
        q[t][flip] *= -1.0
    return np.moveaxis(q, 0, axis)


def yaw_of(q_wxyz: np.ndarray) -> np.ndarray:
    """Heading angle about world +Z (Z-up frame)."""
    fwd = quat_apply(q_wxyz, np.array([1.0, 0.0, 0.0]))
    return np.arctan2(fwd[..., 1], fwd[..., 0])


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    t = np.asarray(t, dtype=np.float64)[..., None] if np.ndim(t) else float(t)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t = np.sin(theta)
    small = sin_t < 1e-6
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin_t))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin_t))
    return quat_normalize(w0 * q0 + w1 * q1)


def angular_velocity(q_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from an orientation sequence (T,...,4) -> (T,...,3).

    Central difference inside, one-sided at the ends (same timing as the linear velocities of
    retarget/export.py, so both describe the same instant)."""
    q = quat_continuity(q_wxyz, axis=0)
    out = np.zeros(q.shape[:-1] + (3,))
    if q.shape[0] < 2:
        return out

    def rate(a, b, span):
        dq = quat_normalize(quat_mul(b, quat_conj(a)))
        return quat_to_rot(dq).as_rotvec().reshape(*dq.shape[:-1], 3) / (span * dt)

    if q.shape[0] > 2:
        out[1:-1] = rate(q[:-2], q[2:], 2.0)
    out[:1] = rate(q[:1], q[1:2], 1.0)
    out[-1:] = rate(q[-2:-1], q[-1:], 1.0)
    return out


def finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    v = np.gradient(np.asarray(x, dtype=np.float64), dt, axis=0)
    return v
