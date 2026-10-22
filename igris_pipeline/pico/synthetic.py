"""Synthetic PICO-like motion generator.

Used to (a) test every stage without the headset, (b) give ground-truth posture labels
for measuring the auto-labeler, and (c) simulate the error characteristics of each
tracker profile so the correction stage can be validated.

Frame: world Z-up, subject faces +X at yaw 0, +Y is the subject's left.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..skeleton import J, NUM_JOINTS, PARENTS, PICO_JOINTS
from ..utils.rotations import rot_to_quat

# Rest (T-pose) offsets of each joint relative to its parent, parent frame.
REST_OFFSETS = np.zeros((NUM_JOINTS, 3))
_OFF = {
    "left_hip": (0.0, 0.09, -0.07), "right_hip": (0.0, -0.09, -0.07),
    "spine1": (0.0, 0.0, 0.11), "left_knee": (0.0, 0.0, -0.40), "right_knee": (0.0, 0.0, -0.40),
    "spine2": (0.0, 0.0, 0.13), "left_ankle": (0.0, 0.0, -0.41), "right_ankle": (0.0, 0.0, -0.41),
    "spine3": (0.0, 0.0, 0.06), "left_foot": (0.12, 0.0, -0.06), "right_foot": (0.12, 0.0, -0.06),
    "neck": (0.0, 0.0, 0.21), "left_collar": (0.0, 0.07, 0.14), "right_collar": (0.0, -0.07, 0.14),
    "head": (0.02, 0.0, 0.09), "left_shoulder": (0.0, 0.11, 0.0), "right_shoulder": (0.0, -0.11, 0.0),
    "left_elbow": (0.0, 0.27, 0.0), "right_elbow": (0.0, -0.27, 0.0),
    "left_wrist": (0.0, 0.25, 0.0), "right_wrist": (0.0, -0.25, 0.0),
    "left_hand": (0.0, 0.09, 0.0), "right_hand": (0.0, -0.09, 0.0),
}
for _n, _o in _OFF.items():
    REST_OFFSETS[J[_n]] = _o
ANKLE_HEIGHT = 0.08  # ankle joint height above floor at rest

LOWER_JOINTS = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
                "left_foot", "right_foot", "spine1", "spine2", "spine3"]
UPPER_JOINTS = ["neck", "head", "left_collar", "right_collar", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"]

LOCO_LABELS = ["idle", "walk", "run", "turn"]
LOWER_LABELS = ["upright", "squat", "bend_forward", "single_leg"]
UPPER_LABELS = ["neutral", "t_pose", "arms_up", "reach_forward", "wave_left", "wave_right"]


def _rx(d):
    return R.from_euler("x", d, degrees=True)


def _ry(d):
    return R.from_euler("y", d, degrees=True)


def _rz(d):
    return R.from_euler("z", d, degrees=True)


# ----------------------------------------------------------------------------------
# Lower-body primitives: return (local rotvecs for LOWER_JOINTS, forward speed, yaw rate,
# arm swing phase/amp, pelvis height offset). t is time inside the segment.
# ----------------------------------------------------------------------------------
@dataclass
class LowerOut:
    rot: dict
    speed: float = 0.0
    yaw_rate: float = 0.0
    arm_swing: float = 0.0  # deg, signed (left arm forward when positive)


def _gait(t, freq, hip_amp, knee_amp):
    ph = 2 * np.pi * freq * t
    lh = hip_amp * np.sin(ph)
    rh = -lh
    lk = knee_amp * max(0.0, np.sin(ph + np.pi / 2)) ** 2 + 5
    rk = knee_amp * max(0.0, np.sin(ph - np.pi / 2)) ** 2 + 5
    return {
        "left_hip": _ry(-lh), "right_hip": _ry(-rh),
        "left_knee": _ry(lk), "right_knee": _ry(rk),
        "left_ankle": _ry(-0.3 * lh), "right_ankle": _ry(-0.3 * rh),
        "spine1": _ry(4.0),
    }, lh


def lower_primitive(name: str, t: float, rng: np.random.Generator) -> LowerOut:
    sway = 1.5 * np.sin(2 * np.pi * 0.3 * t)
    if name == "idle":
        return LowerOut({"spine1": _ry(sway), "left_hip": _rz(sway), "right_hip": _rz(-sway)})
    if name == "walk":
        rot, lh = _gait(t, 0.9, 22.0, 55.0)
        return LowerOut(rot, speed=1.1, arm_swing=0.8 * lh)
    if name == "run":
        rot, lh = _gait(t, 1.4, 35.0, 90.0)
        rot["spine1"] = _ry(12.0)
        return LowerOut(rot, speed=2.6, arm_swing=1.2 * lh)
    if name == "turn":
        rot, lh = _gait(t, 0.8, 8.0, 25.0)
        return LowerOut(rot, speed=0.05, yaw_rate=np.deg2rad(70.0), arm_swing=0.3 * lh)
    if name == "squat":
        d = 0.5 - 0.5 * np.cos(2 * np.pi * min(t, 4.0) / 4.0)  # one squat in 4 s, then hold low
        d = 1.0 if t > 2.0 else d
        return LowerOut({
            "left_hip": _ry(-105 * d) * _rz(8 * d), "right_hip": _ry(-105 * d) * _rz(-8 * d),
            "left_knee": _ry(120 * d), "right_knee": _ry(120 * d),
            "left_ankle": _ry(-25 * d), "right_ankle": _ry(-25 * d),
            "spine1": _ry(35 * d), "spine2": _ry(5 * d),
        })
    if name == "bend_forward":
        d = min(1.0, t / 1.2)
        return LowerOut({
            "left_hip": _ry(-15 * d), "right_hip": _ry(-15 * d),
            "left_knee": _ry(10 * d), "right_knee": _ry(10 * d),
            "spine1": _ry(45 * d), "spine2": _ry(25 * d), "spine3": _ry(10 * d),
        })
    if name == "single_leg":
        d = min(1.0, t / 0.8)
        return LowerOut({
            "left_hip": _ry(-75 * d), "left_knee": _ry(95 * d), "left_ankle": _ry(-10 * d),
            "right_hip": _rz(-3 * d), "spine1": _ry(sway),
        })
    raise KeyError(name)


def upper_primitive(name: str, t: float, arm_swing: float) -> dict:
    # neutral = arms hanging (A-pose ~75 deg down) with optional gait swing
    neutral = {
        "left_shoulder": _ry(-arm_swing) * _rx(-75), "right_shoulder": _ry(arm_swing) * _rx(75),
        "left_elbow": _rz(-10 - max(0, arm_swing) * 0.5), "right_elbow": _rz(10 + max(0, -arm_swing) * 0.5),
    }
    if name == "neutral":
        return neutral
    d = min(1.0, t / 0.8)
    if name == "t_pose":
        return {"left_shoulder": _rx(-75 * (1 - d)), "right_shoulder": _rx(75 * (1 - d))}
    if name == "arms_up":
        return {"left_shoulder": _rx(-75 + 160 * d), "right_shoulder": _rx(75 - 160 * d),
                "left_elbow": _rz(-5), "right_elbow": _rz(5)}
    if name == "reach_forward":
        reach = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t)
        return {"left_shoulder": _rz(-85 * d) * _rx(-10 * d), "right_shoulder": _rz(85 * d) * _rx(10 * d),
                "left_elbow": _rz(-40 * (1 - reach) * d), "right_elbow": _rz(40 * (1 - reach) * d)}
    if name in ("wave_right", "wave_left"):
        osc = 25 * np.sin(2 * np.pi * 1.6 * t) * d
        out = dict(neutral)
        if name == "wave_right":
            out["right_shoulder"] = _rx(75 - 95 * d)
            out["right_elbow"] = _rx(-90 * d + osc)
        else:
            out["left_shoulder"] = _rx(-75 + 95 * d)
            out["left_elbow"] = _rx(90 * d - osc)
        return out
    raise KeyError(name)


def forward_kinematics(root_pos, root_rot: R, local: list[R]):
    """Single-frame FK. local: list of 24 Rotations (root's local is ignored)."""
    gpos = np.zeros((NUM_JOINTS, 3))
    grot = [None] * NUM_JOINTS
    gpos[0] = root_pos
    grot[0] = root_rot
    for i in range(1, NUM_JOINTS):
        p = PARENTS[i]
        gpos[i] = gpos[p] + grot[p].apply(REST_OFFSETS[i])
        grot[i] = grot[p] * local[i]
    return gpos, grot


@dataclass
class SyntheticClip:
    fps: float
    t: np.ndarray
    pos: np.ndarray          # (T,24,3) ground truth
    quat: np.ndarray         # (T,24,4) wxyz, rest pose = yaw-only orientation
    labels: dict             # {"loco": [...], "lower": [...], "upper": [...]} per-frame strings
    segments: list           # [(start, end, (loco, lower, upper))]


DEFAULT_PROGRAM = [
    ("idle", "upright", "t_pose", 3.0),
    ("idle", "upright", "neutral", 2.0),
    ("walk", "upright", "neutral", 4.0),
    ("idle", "upright", "wave_right", 3.0),
    ("idle", "squat", "neutral", 4.0),
    ("idle", "upright", "arms_up", 3.0),
    ("turn", "upright", "neutral", 3.0),
    ("idle", "bend_forward", "neutral", 3.0),
    ("walk", "upright", "wave_left", 4.0),
    ("idle", "upright", "reach_forward", 3.5),
    ("idle", "single_leg", "neutral", 3.0),
    ("run", "upright", "neutral", 3.0),
    ("idle", "upright", "neutral", 2.0),
]


def random_program(seed: int, n: int = 14, start_tpose: bool = True) -> list:
    """Random but plausible sequence of (loco, lower, upper, duration) segments."""
    rng = np.random.default_rng(seed)
    prog = [("idle", "upright", "t_pose", 3.0)] if start_tpose else []
    for _ in range(n):
        loco = rng.choice(LOCO_LABELS, p=[0.55, 0.25, 0.08, 0.12])
        lower = "upright" if loco != "idle" else rng.choice(LOWER_LABELS, p=[0.55, 0.17, 0.15, 0.13])
        if loco == "run" or lower in ("squat", "bend_forward"):
            upper = "neutral" if rng.random() < 0.7 else "reach_forward"
        else:
            upper = rng.choice(UPPER_LABELS, p=[0.35, 0.05, 0.15, 0.15, 0.15, 0.15])
        prog.append((str(loco), str(lower), str(upper), float(np.round(rng.uniform(2.0, 4.5), 1))))
    return prog


def generate(program=None, fps: float = 60.0, blend: float = 0.4, seed: int = 0,
             subject_scale: float = 1.0) -> SyntheticClip:
    """Generate a clip from a program of (loco, lower, upper, duration) tuples."""
    program = program or DEFAULT_PROGRAM
    rng = np.random.default_rng(seed)
    offsets_backup = REST_OFFSETS.copy()
    REST_OFFSETS[:] = offsets_backup * subject_scale
    try:
        dur = np.array([p[3] for p in program])
        starts = np.concatenate([[0.0], np.cumsum(dur)[:-1]])
        T = int(np.sum(dur) * fps)
        t_arr = np.arange(T) / fps
        pos = np.zeros((T, NUM_JOINTS, 3))
        quat = np.zeros((T, NUM_JOINTS, 4))
        lab = {"loco": [], "lower": [], "upper": []}
        xy = np.zeros(2)
        yaw = 0.0
        prev_rv = None
        seg_idx_of_frame = np.searchsorted(starts, t_arr, side="right") - 1
        for f in range(T):
            k = seg_idx_of_frame[f]
            tl = t_arr[f] - starts[k]
            rv, speed, yaw_rate = _frame_rotvecs(program[k], tl, rng)
            # cross-fade into the next segment over `blend` seconds at the end of this one
            if k + 1 < len(program) and dur[k] - tl < blend:
                a = 1.0 - (dur[k] - tl) / blend
                a = a * a * (3 - 2 * a)
                rv2, sp2, yr2 = _frame_rotvecs(program[k + 1], 0.0, rng)
                rv = (1 - a) * rv + a * rv2
                speed = (1 - a) * speed + a * sp2
                yaw_rate = (1 - a) * yaw_rate + a * yr2
            if prev_rv is not None:
                rv = 0.6 * rv + 0.4 * prev_rv  # light smoothing of the generator itself
            prev_rv = rv
            yaw += yaw_rate / fps
            xy += speed / fps * np.array([np.cos(yaw), np.sin(yaw)])
            root_rot = _rz(np.rad2deg(yaw))
            local = [R.from_rotvec(rv[i]) for i in range(NUM_JOINTS)]
            gp, gr = forward_kinematics(np.array([xy[0], xy[1], 0.0]), root_rot, local)
            gp[:, 2] += ANKLE_HEIGHT * subject_scale - min(gp[J["left_ankle"], 2], gp[J["right_ankle"], 2])
            pos[f] = gp
            quat[f] = np.stack([rot_to_quat(g) for g in gr])
            lo, lw, up, _ = program[k]
            if k + 1 < len(program) and dur[k] - tl < blend / 2:
                lo, lw, up, _ = program[k + 1]
            lab["loco"].append(lo)
            lab["lower"].append(lw)
            lab["upper"].append(up)
        segs = []
        for k, (lo, lw, up, d) in enumerate(program):
            s = int(round(starts[k] * fps))
            segs.append((s, min(T, int(round((starts[k] + d) * fps))), (lo, lw, up)))
        return SyntheticClip(fps=fps, t=t_arr, pos=pos, quat=quat, labels=lab, segments=segs)
    finally:
        REST_OFFSETS[:] = offsets_backup


def _frame_rotvecs(prog, tl, rng):
    lo, lw, up, _ = prog
    loco_out = lower_primitive(lo, tl, rng)
    lower_out = lower_primitive(lw, tl, rng) if lw != "upright" else LowerOut({})
    rot = dict(loco_out.rot)
    if lw != "upright":  # posture overrides gait on the joints it drives
        rot.update(lower_out.rot)
    upper = upper_primitive(up, tl, loco_out.arm_swing)
    if up == "neutral":  # hanging arms stay roughly vertical when the trunk pitches
        pitch = sum(rot[n].as_euler("yxz", degrees=True)[0] for n in ("spine1", "spine2", "spine3") if n in rot)
        for side in ("left_shoulder", "right_shoulder"):
            upper[side] = _ry(-0.9 * pitch) * upper[side]
    rot.update(upper)
    rv = np.zeros((NUM_JOINTS, 3))
    for n, r in rot.items():
        rv[J[n]] = r.as_rotvec()
    return rv, loco_out.speed, loco_out.yaw_rate


__all__ = ["generate", "SyntheticClip", "DEFAULT_PROGRAM", "LOCO_LABELS", "LOWER_LABELS",
           "UPPER_LABELS", "REST_OFFSETS", "PICO_JOINTS"]
