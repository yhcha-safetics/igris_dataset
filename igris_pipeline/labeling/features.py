"""Interpretable per-frame posture features computed from the 24-joint skeleton.

All features are expressed in the subject's *heading frame* (yaw-only, Z-up) and are
normalised by subject stature, so the same thresholds work for different people and
for both raw PICO data and synthetic data.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from ..skeleton import J


def _unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9, None)


def _angle(a, b):
    return np.arccos(np.clip(np.sum(_unit(a) * _unit(b), axis=-1), -1.0, 1.0))


def _smooth(x, fps, win_s=0.25):
    win = max(5, int(win_s * fps) | 1)
    if x.shape[0] <= win:
        return x
    return savgol_filter(x, win, 2, axis=0)


def _rolling_rms(x, n):
    return np.sqrt(np.clip(uniform_filter1d(x ** 2, size=max(1, n), axis=0, mode="nearest"), 0, None))


HEAD_TOP = 0.12  # m from the head joint to the top of the head (correction.py adds it to get stature)


def compute_features(pos: np.ndarray, fps: float, stature: float | None = None) -> dict[str, np.ndarray]:
    """pos: (T,24,3) world Z-up. Returns dict of (T,) feature arrays.

    Lengths are normalised by the standing head-joint height. Without `stature` it is the 95th
    percentile of the head height (what the rule thresholds were tuned on); a calibrated
    stature (top of head, stage 1-2) is converted to the same reference.
    """
    p = np.asarray(pos, dtype=np.float64)
    T = p.shape[0]
    up = np.array([0.0, 0.0, 1.0])

    foot_z = np.minimum(p[:, J["left_foot"], 2], p[:, J["right_foot"], 2])
    ankle_z = np.minimum(p[:, J["left_ankle"], 2], p[:, J["right_ankle"], 2])
    floor = float(np.percentile(np.minimum(foot_z, ankle_z), 5))
    head_h = p[:, J["head"], 2] - floor
    if stature is None:
        stature = float(np.percentile(head_h, 95))
    else:
        stature = float(stature) - HEAD_TOP
    stature = max(stature, 0.5)

    # heading frame from hips + shoulders
    lat = (p[:, J["left_hip"]] - p[:, J["right_hip"]]) + (p[:, J["left_shoulder"]] - p[:, J["right_shoulder"]])
    lat[:, 2] = 0.0
    left = _unit(_smooth(lat, fps))
    fwd = _unit(np.cross(left, up))
    heading = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))

    def to_body(v):  # world vector -> (fwd, left, up) components
        return np.stack([np.sum(v * fwd, -1), np.sum(v * left, -1), v[:, 2]], axis=-1)

    pelvis = p[:, J["pelvis"]]
    pelvis_h = (pelvis[:, 2] - floor) / stature
    pelvis_ref = float(np.percentile(pelvis_h, 90))

    trunk = p[:, J["neck"]] - pelvis
    trunk_b = to_body(trunk)
    trunk_pitch = np.arctan2(trunk_b[:, 0], np.clip(trunk_b[:, 2], 1e-6, None))  # + = leaning forward
    trunk_tilt = _angle(trunk, up[None])

    f: dict[str, np.ndarray] = {}
    f["stature"] = np.full(T, stature)
    f["pelvis_h"] = pelvis_h
    f["pelvis_rel"] = pelvis_h / max(pelvis_ref, 1e-3)
    f["head_rel"] = head_h / stature
    f["trunk_pitch"] = np.degrees(trunk_pitch)
    f["trunk_tilt"] = np.degrees(trunk_tilt)

    for side in ("left", "right"):
        hip, knee, ank = p[:, J[f"{side}_hip"]], p[:, J[f"{side}_knee"]], p[:, J[f"{side}_ankle"]]
        f[f"{side}_knee_flex"] = np.degrees(np.pi - _angle(hip - knee, ank - knee))
        f[f"{side}_hip_flex"] = np.degrees(_angle(knee - hip, -trunk))
        f[f"{side}_ankle_h"] = (ank[:, 2] - floor) / stature

        sh, el, wr, hd = (p[:, J[f"{side}_shoulder"]], p[:, J[f"{side}_elbow"]],
                          p[:, J[f"{side}_wrist"]], p[:, J[f"{side}_hand"]])
        arm_len = np.median(np.linalg.norm(el - sh, axis=-1) + np.linalg.norm(wr - el, axis=-1)) + 1e-6
        rel = to_body(wr - sh) / arm_len
        sgn = 1.0 if side == "left" else -1.0
        f[f"{side}_hand_fwd"] = rel[:, 0]
        # low-passed version: sustained reaching vs. arm swing during gait
        f[f"{side}_hand_fwd_lp"] = uniform_filter1d(rel[:, 0], size=max(1, int(0.8 * fps)), mode="nearest")
        f[f"{side}_hand_lat"] = sgn * rel[:, 1]          # + = away from body midline
        f[f"{side}_hand_up"] = rel[:, 2]                 # + = above shoulder
        f[f"{side}_hand_over_head"] = (wr[:, 2] - p[:, J["head"], 2]) / stature
        f[f"{side}_elbow_flex"] = np.degrees(np.pi - _angle(sh - el, wr - el))
        # oscillation energy of the forearm (waving): high-pass lateral hand motion
        fore = to_body(hd - el)[:, 1]
        hp = fore - uniform_filter1d(fore, size=max(3, int(0.6 * fps)), mode="nearest")
        f[f"{side}_hand_osc"] = _rolling_rms(hp, int(1.0 * fps)) / arm_len

    f["knee_flex_mean"] = 0.5 * (f["left_knee_flex"] + f["right_knee_flex"])
    f["knee_h_min"] = (np.minimum(p[:, J["left_knee"], 2], p[:, J["right_knee"], 2]) - floor) / stature
    f["ankle_h_diff"] = np.abs(f["left_ankle_h"] - f["right_ankle_h"])
    f["ankle_h_max"] = np.maximum(f["left_ankle_h"], f["right_ankle_h"]) - np.percentile(
        np.minimum(f["left_ankle_h"], f["right_ankle_h"]), 50)

    # locomotion
    xy = _smooth(pelvis[:, :2], fps, 0.5)
    vel = np.gradient(xy, 1.0 / fps, axis=0)
    speed = np.linalg.norm(vel, axis=-1)
    f["speed"] = uniform_filter1d(speed, size=max(1, int(0.5 * fps)), mode="nearest")
    f["speed_norm"] = f["speed"] / stature
    yaw_rate = np.gradient(_smooth(heading, fps, 0.5), 1.0 / fps)
    f["yaw_rate"] = np.degrees(uniform_filter1d(yaw_rate, size=max(1, int(0.5 * fps)), mode="nearest"))
    f["abs_yaw_rate"] = np.abs(f["yaw_rate"])
    ank_rel = to_body(p[:, J["left_ankle"]] - p[:, J["right_ankle"]])[:, 0]
    f["gait_energy"] = _rolling_rms(np.gradient(ank_rel, 1.0 / fps), int(1.0 * fps)) / stature
    f["heading"] = np.degrees(heading)
    return f


FEATURE_KEYS_FOR_ML = [
    "pelvis_rel", "head_rel", "trunk_pitch", "trunk_tilt", "left_knee_flex", "right_knee_flex",
    "left_hip_flex", "right_hip_flex", "left_ankle_h", "right_ankle_h", "left_hand_fwd",
    "right_hand_fwd", "left_hand_lat", "right_hand_lat", "left_hand_up", "right_hand_up",
    "left_hand_over_head", "right_hand_over_head", "left_elbow_flex", "right_elbow_flex",
    "left_hand_osc", "right_hand_osc", "speed_norm", "abs_yaw_rate", "gait_energy", "ankle_h_diff",
    "knee_h_min",
]


def feature_matrix(feats: dict[str, np.ndarray], fps: float, context_s: float = 0.5) -> np.ndarray:
    """Stack ML features with short temporal context (mean and std over a window)."""
    X = np.stack([feats[k] for k in FEATURE_KEYS_FOR_ML], axis=-1)
    n = max(1, int(context_s * fps))
    mu = uniform_filter1d(X, size=n, axis=0, mode="nearest")
    sd = np.sqrt(np.clip(uniform_filter1d(X ** 2, size=n, axis=0, mode="nearest") - mu ** 2, 0, None))
    return np.concatenate([X, mu, sd], axis=-1)
