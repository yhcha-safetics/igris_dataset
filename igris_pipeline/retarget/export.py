"""Export retargeted IGRIS-C motions to downstream formats.

* tracking NPZ  (Isaac Lab tracking task in igris_pipeline/isaaclab_ext, BeyondMimic-style keys)
* GMR canonical PKL for IGRIS-C (Robros GMR fork, IGRIS_C_CANONICAL_FORM.md)
* CSV rows [root xyz, root quat xyzw, 31 joints] (BeyondMimic / mjlab csv_to_npz input)

NPZ and PKL can carry extra per-frame `series` (resampled with the motion, e.g. the hand command
hand_cmd (T, 12) of igris_pipeline/hands.py: the canonical form has no finger joints) and `static`
arrays (their names / layout); consumers that do not know them ignore them.
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..utils.rotations import angular_velocity, quat_continuity, quat_normalize, slerp, wxyz_to_xyzw, xyzw_to_wxyz
from .robot import IgrisModel

# Key bodies of the IGRIS-C canonical form, in order.
CANONICAL_KEYBODIES = [
    "Link_Waist_Yaw", "Link_Waist_Roll", "Link_Waist_Pitch",
    "Link_Hip_Pitch_Left", "Link_Hip_Roll_Left", "Link_Hip_Yaw_Left", "Link_Knee_Pitch_Left",
    "Link_Ankle_Pitch_Left", "Link_Ankle_Roll_Left",
    "Link_Hip_Pitch_Right", "Link_Hip_Roll_Right", "Link_Hip_Yaw_Right", "Link_Knee_Pitch_Right",
    "Link_Ankle_Pitch_Right", "Link_Ankle_Roll_Right",
    "Link_Shoulder_Pitch_Left", "Link_Shoulder_Roll_Left", "Link_Shoulder_Yaw_Left", "Link_Elbow_Pitch_Left",
    "Link_Wrist_Yaw_Left", "Link_Wrist_Roll_Left", "Link_Wrist_Pitch_Left", "Left_Hand",
    "Link_Shoulder_Pitch_Right", "Link_Shoulder_Roll_Right", "Link_Shoulder_Yaw_Right", "Link_Elbow_Pitch_Right",
    "Link_Wrist_Yaw_Right", "Link_Wrist_Roll_Right", "Link_Wrist_Pitch_Right", "Right_Hand",
    "Link_Neck_Yaw", "Link_Neck_Pitch",
]


def resample_qpos(qpos: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    if abs(fps_in - fps_out) < 1e-6:
        return qpos.copy()
    T = qpos.shape[0]
    t_in = np.arange(T) / fps_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    out = np.zeros((len(t_out), qpos.shape[1]))
    for k in (0, 1, 2, *range(7, qpos.shape[1])):
        out[:, k] = np.interp(t_out, t_in, qpos[:, k])
    q = quat_continuity(qpos[:, 3:7])
    idx = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, T - 2)
    a = (t_out - t_in[idx]) * fps_in
    out[:, 3:7] = slerp(q[idx], q[idx + 1], np.clip(a, 0, 1))
    return out


def resample_array(x: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Linear resampling of a (T, ...) series on the time grid of resample_qpos."""
    x = np.asarray(x, float)
    if abs(fps_in - fps_out) < 1e-6:
        return x.copy()
    t_in = np.arange(len(x)) / fps_in
    t_out = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fps_out)
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(t_out, t_in, flat[:, k]) for k in range(flat.shape[1])], axis=-1)
    return out.reshape((len(t_out),) + x.shape[1:])


def _diff(x, dt):
    v = np.zeros_like(x)
    if len(x) < 2:
        return v
    v[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


def body_states(robot: IgrisModel, qpos: np.ndarray, fps: float, bodies: list[str] | None = None):
    bodies = bodies or [b for b in robot.body_names if b != "world"]
    P, Q = robot.fk_sequence(qpos, bodies)
    Q = quat_continuity(Q, axis=0)
    dt = 1.0 / fps
    return bodies, P, Q, _diff(P, dt), angular_velocity(Q, dt)


def to_tracking_npz(path, robot: IgrisModel, qpos: np.ndarray, fps: float, fps_out: float = 50.0, meta: dict | None = None,
                    series: dict | None = None, static: dict | None = None):
    q = resample_qpos(qpos, fps, fps_out)
    bodies, P, Q, V, W = body_states(robot, q, fps_out)
    jp = q[:, 7:]
    extra = {k: resample_array(v, fps, fps_out)[:len(q)].astype(np.float32) for k, v in (series or {}).items()}
    extra.update(static or {})
    np.savez_compressed(
        path, fps=np.float64(fps_out), joint_names=np.array(robot.joint_names), joint_pos=jp.astype(np.float32),
        joint_vel=_diff(jp, 1.0 / fps_out).astype(np.float32), body_names=np.array(bodies),
        body_pos_w=P.astype(np.float32), body_quat_w=Q.astype(np.float32), body_lin_vel_w=V.astype(np.float32),
        body_ang_vel_w=W.astype(np.float32), root_pos=q[:, :3].astype(np.float32), root_quat=q[:, 3:7].astype(np.float32),
        meta=np.array(json.dumps(meta or {}, ensure_ascii=False)), **extra)
    return Path(path)


def to_canonical_pkl(path, robot: IgrisModel, qpos: np.ndarray, fps: float, meta: dict | None = None,
                     series: dict | None = None, static: dict | None = None):
    """IGRIS-C canonical GMR PKL (float32, wxyz), see Robros IGRIS_C_CANONICAL_FORM.md."""
    dt = 1.0 / fps
    root_pos = qpos[:, :3]
    root_rot = quat_normalize(quat_continuity(qpos[:, 3:7]))
    dof = qpos[:, 7:]
    rot = R.from_quat(wxyz_to_xyzw(root_rot))
    T = len(qpos)
    angvel = np.zeros((T, 3))
    if T > 1:
        angvel[1:-1] = (rot[:-2].inv() * rot[2:]).as_rotvec() / (2 * dt)
        angvel[0] = (rot[0].inv() * rot[1]).as_rotvec() / dt
        angvel[-1] = (rot[-2].inv() * rot[-1]).as_rotvec() / dt
    P, Q = robot.fk_sequence(qpos, CANONICAL_KEYBODIES)
    inv = rot.inv()
    pl = np.stack([inv[t].apply(P[t] - root_pos[t]) for t in range(T)])
    ql = np.stack([xyzw_to_wxyz((inv[t] * R.from_quat(wxyz_to_xyzw(Q[t]))).as_quat()) for t in range(T)])
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731
    d = {
        "fps": float(fps), "root_pos": f32(root_pos), "root_rot": f32(root_rot), "root_vel": f32(_diff(root_pos, dt)),
        "root_angvel": f32(angvel), "dof_pos": f32(dof), "dof_vel": f32(_diff(dof, dt)),
        "keybody_pos_world": f32(P), "keybody_rot_world": f32(Q), "keybody_pos_local": f32(pl),
        "keybody_rot_local": f32(ql), "keybody_pos": f32(P), "link_body_list": list(CANONICAL_KEYBODIES),
        "local_body_link_body_list": None, "local_body_pos": None, "retarget_meta": dict(meta or {}),
    }
    d.update({k: f32(v) for k, v in (series or {}).items()})  # extensions of the canonical form (e.g. hand_cmd)
    d.update({k: np.asarray(v).tolist() for k, v in (static or {}).items()})
    with open(path, "wb") as f:
        pickle.dump(d, f)
    return Path(path)


def to_csv(path, qpos: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for row in qpos:
            w.writerow(np.concatenate([row[:3], wxyz_to_xyzw(row[3:7]), row[7:]]).round(6).tolist())
    return Path(path)


def store_in_episode(ep, res, robot_key: str = "igris_c"):
    g = f"retarget/{robot_key}"
    ep[f"{g}/qpos"] = res.qpos.astype(np.float32)
    ep.group_attrs[g] = {"fps": float(res.fps), "joint_names": json.dumps(res.joint_names),
                         "qpos_layout": "root_xyz(3), root_quat_wxyz(4), joints(31) in joint_names order",
                         "diagnostics": json.dumps(res.diagnostics, ensure_ascii=False)}
