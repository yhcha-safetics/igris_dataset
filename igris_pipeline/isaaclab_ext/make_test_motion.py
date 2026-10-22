"""Generate a reproducible synthetic test motion for the IGRIS-C tracking task.

Run with the project venv (needs mujoco + numpy only):

    .venv\\Scripts\\python.exe -m igris_pipeline.isaaclab_ext.make_test_motion \
        --out data\\sample\\motions\\test_sine.npz

Motion: default standing pose + slow sinusoidal arm / waist / squat (knee) motion, amplitude ramped in
over the first second so that frame 0 is the static default pose. The floating base is solved so the
pelvis stays upright and both feet stay planted and flat with the URDF sole (z = -0.071 in the
ankle-roll frame; the Isaac articulation uses the URDF collision boxes) touching z = 0.

Output NPZ (the stage-3 retargeter format):
    fps (scalar), joint_names (31, MuJoCo order), joint_pos (T,31), joint_vel (T,31),
    body_names (B incl. base_link), body_pos_w (T,B,3), body_quat_w (T,B,4) wxyz,
    body_lin_vel_w (T,B,3) (velocity of the body-frame origin, world frame),
    body_ang_vel_w (T,B,3) (world frame).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))

from igris_pipeline.isaaclab_ext import constants as C  # noqa: E402


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def joint_trajectory(t: np.ndarray, names: list[str], wrist_amp: float = 0.0) -> np.ndarray:
    """(T, J) joint positions for the MuJoCo joint list ``names``."""
    env = _smoothstep(t / 1.0)  # ramp-in over 1 s
    w = 2.0 * np.pi
    q = {n: np.full_like(t, C.default_joint_pos(n)) for n in names}
    # waist
    q["Joint_Waist_Yaw"] += env * 0.30 * np.sin(w * 0.20 * t)
    q["Joint_Waist_Roll"] += env * 0.06 * np.sin(w * 0.13 * t)
    q["Joint_Waist_Pitch"] += env * 0.10 * np.sin(w * 0.15 * t)
    # squat: knee +a, hip/ankle pitch -a/2 -> foot stays parallel to the pelvis
    a = env * 0.30 * 0.5 * (1.0 - np.cos(w * 0.20 * t))
    for side in ("Left", "Right"):
        q[f"Joint_Knee_Pitch_{side}"] += a
        q[f"Joint_Hip_Pitch_{side}"] -= 0.5 * a
        q[f"Joint_Ankle_Pitch_{side}"] -= 0.5 * a
    # arms (anti-phase swing)
    s = np.sin(w * 0.25 * t)
    q["Joint_Shoulder_Pitch_Left"] += env * 0.50 * s
    q["Joint_Shoulder_Pitch_Right"] -= env * 0.50 * s
    roll = env * 0.20 * 0.5 * (1.0 - np.cos(w * 0.17 * t))
    q["Joint_Shoulder_Roll_Left"] += roll
    q["Joint_Shoulder_Roll_Right"] -= roll
    q["Joint_Shoulder_Yaw_Left"] += env * 0.20 * np.sin(w * 0.21 * t)
    q["Joint_Shoulder_Yaw_Right"] -= env * 0.20 * np.sin(w * 0.21 * t)
    elbow = env * 0.50 * 0.5 * (1.0 - np.cos(w * 0.30 * t))
    q["Joint_Elbow_Pitch_Left"] -= elbow
    q["Joint_Elbow_Pitch_Right"] -= elbow
    if wrist_amp > 0.0:  # exercises the "ignore joints absent in the articulation" path
        for n in names:
            if "Wrist" in n or "Neck" in n:
                q[n] += env * wrist_amp * np.sin(w * 0.4 * t)
    return np.stack([q[n] for n in names], axis=1)


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.reshape(9))
    return quat


def build_motion(mjcf: Path, duration: float, fps: float, wrist_amp: float, sole: str) -> dict[str, np.ndarray]:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    joint_names = [m.joint(j).name for j in hinge]
    qadr = np.array([m.jnt_qposadr[j] for j in hinge])
    body_ids = list(range(1, m.nbody))
    body_names = [m.body(b).name for b in body_ids]
    pelvis = m.body("Link_Waist_Pitch").id
    ankles = [m.body(n).id for n in C.FOOT_BODY_NAMES]
    sole_z = C.URDF_SOLE_Z if sole == "urdf" else _mjcf_sole_z(m)

    num = int(round(duration * fps))
    t = np.arange(num) / fps
    q = joint_trajectory(t, joint_names, wrist_amp)

    qpos = np.zeros((num, m.nq))
    c = None
    for i in range(num):
        d.qpos[:] = 0.0
        d.qpos[3] = 1.0
        d.qpos[qadr] = q[i]
        mujoco.mj_kinematics(m, d)
        R_rp = d.xmat[pelvis].reshape(3, 3).copy()  # pelvis orientation in root frame (root = identity)
        a_r = d.xpos[ankles].mean(axis=0).copy()  # mean ankle position in root frame
        R_wr = R_rp.T  # keep the pelvis upright with zero yaw
        if c is None:  # frame 0: root xy at origin, soles on z = 0
            c = R_wr @ a_r
            c[2] = -sole_z
            c[:2] = (R_wr @ a_r)[:2]
        p_wr = c - R_wr @ a_r
        qpos[i, 0:3] = p_wr
        qpos[i, 3:7] = _mat_to_quat(R_wr)
        qpos[i, qadr] = q[i]

    dt = 1.0 / fps
    qvel = np.zeros((num, m.nv))
    for i in range(num):
        i0, i1 = max(i - 1, 0), min(i + 1, num - 1)
        mujoco.mj_differentiatePos(m, qvel[i], (i1 - i0) * dt, qpos[i0], qpos[i1])

    B = len(body_ids)
    body_pos = np.zeros((num, B, 3))
    body_quat = np.zeros((num, B, 4))
    body_lin = np.zeros((num, B, 3))
    body_ang = np.zeros((num, B, 3))
    res = np.zeros(6)
    for i in range(num):
        d.qpos[:] = qpos[i]
        d.qvel[:] = qvel[i]
        mujoco.mj_forward(m, d)
        body_pos[i] = d.xpos[body_ids]
        body_quat[i] = d.xquat[body_ids]
        for k, b in enumerate(body_ids):
            # XBODY = body frame origin (BODY would be the COM). world-oriented 6D velocity [ang(3), lin(3)]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
            body_ang[i, k] = res[:3]
            body_lin[i, k] = res[3:]
    joint_vel = qvel[:, [m.jnt_dofadr[j] for j in hinge]]
    return {
        "fps": np.array(fps, dtype=np.float64),
        "joint_names": np.array(joint_names),
        "joint_pos": q.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "body_names": np.array(body_names),
        "body_pos_w": body_pos.astype(np.float32),
        "body_quat_w": body_quat.astype(np.float32),
        "body_lin_vel_w": body_lin.astype(np.float32),
        "body_ang_vel_w": body_ang.astype(np.float32),
    }


def _mjcf_sole_z(m) -> float:
    g = m.geom("foot_box_left").id
    return float(m.geom_pos[g][2] - m.geom_size[g][2])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mjcf", type=Path, default=C.IGRIS_C_MJCF_PATH)
    p.add_argument("--out", type=Path, default=C.SAMPLE_MOTION_DIR / "test_sine.npz")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--fps", type=float, default=C.MOTION_FPS)
    p.add_argument("--wrist_amp", type=float, default=0.0, help="optional wrist/neck motion (ignored by the policy)")
    p.add_argument("--sole", choices=["urdf", "mjcf"], default="urdf", help="foot sole height convention")
    args = p.parse_args()
    data = build_motion(args.mjcf, args.duration, args.fps, args.wrist_amp, args.sole)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **data)
    bi = list(data["body_names"]).index("base_link")
    print(
        f"wrote {args.out}: T={data['joint_pos'].shape[0]} fps={float(data['fps'])} "
        f"J={len(data['joint_names'])} B={len(data['body_names'])} "
        f"root z[0]={data['body_pos_w'][0, bi, 2]:.4f} root z range=({data['body_pos_w'][:, bi, 2].min():.4f}, "
        f"{data['body_pos_w'][:, bi, 2].max():.4f})"
    )


if __name__ == "__main__":
    main()
