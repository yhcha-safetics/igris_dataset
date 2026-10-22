"""Vectorised geometry helpers: two-bone IK, swing rotations, filtering, resampling."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

from .rotations import quat_continuity, quat_normalize, rot_to_quat, slerp


def unit(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float64)
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)


def any_perpendicular(v):
    v = unit(v)
    a = np.where(np.abs(v[..., :1]) < 0.9, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    return unit(np.cross(v, a))


def swing_quat(u, v):
    """Shortest rotation (wxyz) taking direction u to direction v. (...,3) -> (...,4)"""
    u, v = unit(u), unit(v)
    c = np.cross(u, v)
    d = np.sum(u * v, axis=-1, keepdims=True)
    q = np.concatenate([1.0 + d, c], axis=-1)
    anti = (d[..., 0] < -0.999999)
    if np.any(anti):
        q[anti] = np.concatenate([np.zeros((anti.sum(), 1)), any_perpendicular(u[anti])], axis=-1)
    return quat_normalize(q)


def frame_from_two_vectors(d0, n0, d, n):
    """Rotation (wxyz) with R d0 = d exactly and R n0 as close as possible to n."""
    def basis(a, b):
        a = unit(a)
        b = b - np.sum(b * a, -1, keepdims=True) * a
        bad = np.linalg.norm(b, axis=-1) < 1e-6
        if np.any(bad):
            b[bad] = any_perpendicular(a[bad])
        b = unit(b)
        return np.stack([a, b, np.cross(a, b)], axis=-1)  # columns

    d0 = np.broadcast_to(d0, np.shape(d)).astype(np.float64)
    n0 = np.broadcast_to(n0, np.shape(d)).astype(np.float64)
    B0, B = basis(d0, n0.copy()), basis(np.asarray(d, float), np.asarray(n, float).copy())
    Rm = np.einsum("...ij,...kj->...ik", B, B0)
    shape = Rm.shape[:-2]
    return rot_to_quat(R.from_matrix(Rm.reshape(-1, 3, 3))).reshape(*shape, 4)


def two_bone_ik(root, target, l1, l2, pole):
    """Analytic 2-link IK. Returns (mid, end) positions. All (...,3); l1,l2 scalars or (...,)."""
    root, target, pole = (np.asarray(x, dtype=np.float64) for x in (root, target, pole))
    l1 = np.asarray(l1, dtype=np.float64)[..., None] if np.ndim(l1) else float(l1)
    l2 = np.asarray(l2, dtype=np.float64)[..., None] if np.ndim(l2) else float(l2)
    dv = target - root
    dist = np.linalg.norm(dv, axis=-1, keepdims=True)
    dist_c = np.clip(dist, np.abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)
    dirv = dv / np.clip(dist, 1e-9, None)
    cos_a = np.clip((l1 ** 2 + dist_c ** 2 - l2 ** 2) / (2 * l1 * dist_c), -1.0, 1.0)
    sin_a = np.sqrt(1.0 - cos_a ** 2)
    pv = pole - root
    pv = pv - np.sum(pv * dirv, -1, keepdims=True) * dirv
    bad = np.linalg.norm(pv, axis=-1) < 1e-6
    if np.any(bad):
        pv[bad] = any_perpendicular(dirv[bad])
    pv = unit(pv)
    mid = root + l1 * (cos_a * dirv + sin_a * pv)
    end = root + dist_c * dirv
    return mid, end


def lowpass(x, fps, cutoff, order=2, axis=0):
    """Zero-phase Butterworth low-pass along `axis`."""
    x = np.asarray(x, dtype=np.float64)
    if cutoff is None or cutoff <= 0 or cutoff >= 0.5 * fps:
        return x
    b, a = butter(order, cutoff / (0.5 * fps))
    if x.shape[axis] <= 3 * max(len(a), len(b)):
        return x
    return filtfilt(b, a, x, axis=axis, method="gust")


def lowpass_quat(q, fps, cutoff):
    q = quat_continuity(q, axis=0)
    return quat_normalize(lowpass(q, fps, cutoff))


def resample_series(t, x, valid, t_new, max_gap, is_quat=False):
    """Resample samples x(t) (T,...,D) at t_new using only `valid` samples (T,).

    Returns (x_new, valid_new, gap_filled) where gap_filled marks frames that were
    interpolated across a hole in the valid data (<= max_gap).
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out_shape = (len(t_new),) + x.shape[1:]
    if valid.sum() < 2:
        x_new = np.zeros(out_shape)
        if is_quat:
            x_new[..., 0] = 1.0
        return x_new, np.zeros(len(t_new), bool), np.zeros(len(t_new), bool)
    tv, xv = t[valid], x[valid]
    if is_quat:
        xv = quat_continuity(xv, axis=0)
    idx = np.clip(np.searchsorted(tv, t_new, side="right"), 1, len(tv) - 1)
    lo, hi = idx - 1, idx
    span = tv[hi] - tv[lo]
    alpha = np.clip((t_new - tv[lo]) / np.clip(span, 1e-9, None), 0.0, 1.0)
    inside = (t_new >= tv[0] - 1e-9) & (t_new <= tv[-1] + 1e-9)
    valid_new = inside & (span <= max_gap)
    # a nominal sample interval: anything much larger than it is a filled gap
    nominal = np.median(np.diff(tv)) if len(tv) > 2 else span.min()
    gap_filled = valid_new & (span > 2.5 * nominal)
    a = alpha.reshape((-1,) + (1,) * (x.ndim - 1))
    if is_quat:
        x_new = slerp(xv[lo], xv[hi], np.broadcast_to(a[..., 0], xv[lo].shape[:-1]))
    else:
        x_new = (1 - a) * xv[lo] + a * xv[hi]
    return x_new, valid_new, gap_filled


def fill_invalid(x, valid, is_quat=False):
    """Hold nearest valid sample for frames outside the valid set (keeps arrays finite)."""
    x = np.asarray(x, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    if valid.all() or not valid.any():
        return x
    idx = np.arange(len(valid))
    good = idx[valid]
    nearest = good[np.clip(np.searchsorted(good, idx), 0, len(good) - 1)]
    prev = good[np.clip(np.searchsorted(good, idx) - 1, 0, len(good) - 1)]
    pick = np.where(np.abs(prev - idx) < np.abs(nearest - idx), prev, nearest)
    x[~valid] = x[pick[~valid]]
    return x
