"""Stage 4 (MuJoCo): physical feasibility of retargeted IGRIS-C clips.

Run before spending GPU hours on imitation learning, to catch clips no policy can track:
  1. kinematic checks   joint-limit saturation, joint speed vs. actuator limits,
                        self-collision depth, floor penetration, foot skating
  2. static balance     fraction of frames whose CoM ground projection lies inside the
                        support polygon of the feet on the floor (slow motions must satisfy it)
  3. PD replay          physics replay at 200 Hz with joint PD
       harness=True   a virtual 6-DoF spring holds the base on the reference; the harness
                      force (% body weight) says how much "external help" the clip needs.
                      Uses stiff gains (gain_scale) so the number reflects the clip, not the
                      deliberately soft Robros gains that an RL policy compensates for.
       harness=False  no help, Robros gains: time to fall / completed fraction (open-loop
                      lower bound; a tracking policy is needed for real tracking)
     metrics: MPKPE global / root-relative (cm), joint RMSE (deg), ee error (cm),
              torque saturation ratio
Inverse dynamics (mj_inverse) is intentionally not used: with soles exactly on the floor the
soft-contact model produces ~zero ground reaction for kinematic clips, which is misleading.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from ..retarget.robot import ROOT, IgrisModel
from ..utils.rotations import quat_conj, quat_mul, quat_normalize

EE_BODIES = ["Left_Hand", "Right_Hand", "Link_Ankle_Roll_Left", "Link_Ankle_Roll_Right", "Link_Neck_Pitch"]


@dataclass
class JointParams:
    kp: np.ndarray
    kd: np.ndarray
    effort: np.ndarray
    velocity: np.ndarray
    armature: np.ndarray
    friction: np.ndarray
    viscous: np.ndarray


def load_joint_params(robot: IgrisModel, path=ROOT / "configs" / "igris_c_actuators.yaml") -> JointParams:
    with open(path, encoding="utf-8") as f:
        table = yaml.safe_load(f)["joints"]
    cols = {k: np.zeros(robot.ndof) for k in ("kp", "kd", "effort", "velocity", "armature", "friction", "viscous")}
    for i, name in enumerate(robot.joint_names):
        for row in table:
            if re.fullmatch(row["match"], name):
                for k in cols:
                    cols[k][i] = row[k]
                break
        else:
            raise KeyError(f"no actuator entry for {name}")
    return JointParams(**cols)


def apply_joint_params(robot: IgrisModel, jp: JointParams):
    m = robot.model
    for i, name in enumerate(robot.joint_names):
        dof = m.jnt_dofadr[robot.jid(name)]
        m.dof_armature[dof] = jp.armature[i]
        m.dof_damping[dof] = jp.viscous[i]
        m.dof_frictionloss[dof] = jp.friction[i]


def qvel_from_qpos(model, qpos, fps):
    T = len(qpos)
    qvel = np.zeros((T, model.nv))
    dt = 1.0 / fps
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        mujoco.mj_differentiatePos(model, qvel[t], (b - a) * dt, qpos[a], qpos[b])
    return qvel


def kinematic_report(robot: IgrisModel, qpos: np.ndarray, fps: float, jp: JointParams) -> dict:
    m = robot.model
    rep = {}
    rng = robot.jnt_range
    dofq = qpos[:, 7:]
    near = (dofq < rng[:, 0] + np.radians(1)) | (dofq > rng[:, 1] - np.radians(1))
    rep["joint_limit_saturation"] = {robot.joint_names[i]: round(float(near[:, i].mean()), 3)
                                     for i in range(robot.ndof) if near[:, i].mean() > 0.01}
    qd = np.abs(qvel_from_qpos(m, qpos, fps)[:, 6:])
    over = qd > jp.velocity
    rep["joint_speed_over_limit"] = {robot.joint_names[i]: round(float(over[:, i].mean()), 3)
                                     for i in range(robot.ndof) if over[:, i].any()}
    rep["peak_joint_speed_ratio"] = round(float((qd / jp.velocity).max()), 3)
    # self-collision: copy model with robot collision geoms colliding with each other
    m2 = mujoco.MjModel.from_xml_path(str(robot.path))
    floor = robot.gid("floor") if robot.gid("floor") >= 0 else -1
    for g in range(m2.ngeom):
        if g != floor and m2.geom_conaffinity[g]:
            m2.geom_contype[g] = 1
    d2 = mujoco.MjData(m2)
    worst, frames_hit = 0.0, 0
    pairs: dict[str, float] = {}
    stride = max(1, int(fps // 25))
    for t in range(0, len(qpos), stride):
        d2.qpos[:] = qpos[t]
        mujoco.mj_forward(m2, d2)
        hit = False
        for c in d2.contact[:d2.ncon]:
            if floor in (c.geom1, c.geom2):
                continue
            if c.dist < -0.01:
                hit = True
                worst = min(worst, c.dist)
                b1 = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, m2.geom_bodyid[c.geom1])
                b2 = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, m2.geom_bodyid[c.geom2])
                key = "|".join(sorted([b1, b2]))
                pairs[key] = min(pairs.get(key, 0.0), float(c.dist))
        frames_hit += hit
    rep["self_collision_frame_ratio"] = round(frames_hit / max(1, len(range(0, len(qpos), stride))), 3)
    rep["self_collision_worst_depth_cm"] = round(-worst * 100, 2)
    rep["self_collision_pairs_cm"] = {k: round(-v * 100, 1) for k, v in sorted(pairs.items(), key=lambda kv: kv[1])[:8]}
    sole = np.stack([robot.sole_min_z(robot.forward(q)) for q in qpos[::stride]])
    rep["floor_penetration_max_cm"] = round(float(max(0.0, -sole.min()) * 100), 2)
    return rep


def static_balance_report(robot: IgrisModel, qpos: np.ndarray, contact_height: float = 0.015,
                          fps: float | None = None, max_com_speed: float = 0.15) -> dict:
    """CoM ground projection inside the convex hull of the sole boxes touching the floor.

    The criterion only holds without dynamics: during walking the CoM is legitimately outside
    the stance foot. With `fps`, the ratio is taken over quasi-static frames (horizontal CoM
    speed below max_com_speed); `com_in_support_ratio_all` keeps the all-frames value.
    """
    from scipy.spatial import ConvexHull, Delaunay

    m = robot.model
    T = len(qpos)
    com_xy = np.zeros((T, 2))
    grounded = np.zeros(T, bool)
    inside = np.zeros(T, bool)
    margin = np.full(T, np.nan)
    for i, q in enumerate(qpos):
        d = robot.forward(q)
        mujoco.mj_comPos(m, d)
        com = d.subtree_com[robot.bid("base_link")][:2].copy()
        com_xy[i] = com
        pts = []
        for g in robot.foot_geoms:
            half = m.geom_size[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
            w = d.geom_xpos[g] + corners @ d.geom_xmat[g].reshape(3, 3).T
            if w[:, 2].min() < contact_height:
                pts.append(w[w[:, 2] < w[:, 2].min() + contact_height, :2])
        if not pts:
            continue
        pts = np.concatenate(pts)
        grounded[i] = True
        try:
            hull = ConvexHull(pts)
            inside[i] = Delaunay(pts[hull.vertices]).find_simplex(com) >= 0
            # signed distance to the hull boundary (positive inside)
            eq = hull.equations
            margin[i] = float(-np.max(eq[:, :2] @ com + eq[:, 2]))
        except Exception:  # degenerate hull (single edge)
            pass
    static = grounded.copy()
    if fps and T > 2:
        speed = np.linalg.norm(np.gradient(com_xy, 1.0 / fps, axis=0), axis=-1)
        static &= speed < max_com_speed
    ratio = lambda mask: round(float(inside[mask].mean()), 3) if mask.any() else 1.0  # noqa: E731
    return {"com_in_support_ratio": ratio(static), "com_in_support_ratio_all": ratio(grounded),
            "static_ratio": round(float(static.mean()), 3), "airborne_ratio": round(float(1 - grounded.mean()), 3),
            "median_support_margin_cm": round(float(np.nanmedian(margin[static])) * 100, 1)
            if np.isfinite(margin[static]).any() else None}


def _quat_err(q_ref, q):
    e = quat_mul(q_ref, quat_conj(q))
    e = e * np.sign(e[..., :1] + 1e-12)
    return 2.0 * e[..., 1:]


def configure_position_servos(robot: IgrisModel, kp, kd, effort):
    """Turn the MJCF torque motors into PD servos integrated implicitly by MuJoCo.

    tau = kp*(ctrl - q) - kd*qdot, clipped to +-effort. With ctrl = q_ref + kd/kp * qdot_ref
    this is exactly kp*(q_ref - q) + kd*(qdot_ref - qdot), but the damping term is handled by
    the implicitfast integrator, so stiff gains on light links stay stable at 200 Hz.
    """
    m = robot.model
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    ids = []
    for i, n in enumerate(robot.joint_names):
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        ids.append(a)
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[a, :] = 0.0
        m.actuator_gainprm[a, 0] = kp[i]
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[a, :] = 0.0
        m.actuator_biasprm[a, 1] = -kp[i]
        m.actuator_biasprm[a, 2] = -kd[i]
        m.actuator_ctrllimited[a] = 0
        m.actuator_forcelimited[a] = 1
        m.actuator_forcerange[a] = [-effort[i], effort[i]]
    return np.array(ids)


def pd_replay(robot: IgrisModel, qpos_ref: np.ndarray, fps: float, jp: JointParams, harness: bool = True,
              harness_kp: float = 3000.0, harness_kd: float = 300.0, fall_height: float = 0.45,
              gain_scale: float = 1.0) -> dict:
    """Physics replay of a reference clip with joint PD (+ optional base harness)."""
    m = robot.model
    kp, kd = jp.kp * gain_scale, jp.kd * np.sqrt(gain_scale)
    apply_joint_params(robot, jp)
    act_id = configure_position_servos(robot, kp, kd, jp.effort)
    d = mujoco.MjData(m)
    dt = m.opt.timestep
    T = len(qpos_ref)
    t_ref = np.arange(T) / fps
    qvel_ref = qvel_from_qpos(m, qpos_ref, fps)
    base = robot.bid("base_link")
    mass = float(np.sum(m.body_mass))
    d.qpos[:] = qpos_ref[0]
    d.qvel[:] = qvel_ref[0]
    mujoco.mj_forward(m, d)
    ee = [robot.bid(b) for b in EE_BODIES]
    allb = list(range(1, m.nbody))
    steps = int(t_ref[-1] / dt)
    ref_d = mujoco.MjData(m)
    errs, rel_errs, jerr, ee_err, hforce, sat = [], [], [], [], [], []
    fell_at = None
    for k in range(steps):
        t = k * dt
        i = min(int(t * fps), T - 2)
        a = t * fps - i
        q_ref = (1 - a) * qpos_ref[i] + a * qpos_ref[i + 1]
        q_ref[3:7] = quat_normalize(q_ref[3:7])
        v_ref = (1 - a) * qvel_ref[i] + a * qvel_ref[i + 1]
        d.ctrl[act_id] = q_ref[7:] + kd / kp * v_ref[6:]
        d.xfrc_applied[:] = 0.0
        if harness:
            f = harness_kp * (q_ref[:3] - d.qpos[:3]) + harness_kd * (v_ref[:3] - d.qvel[:3])
            # qvel[3:6] of a free joint is the angular velocity in the *body* frame
            w_ref_world = _rotate(q_ref[3:7], v_ref[3:6])
            w_world = _rotate(d.qpos[3:7], d.qvel[3:6])
            tq = (harness_kp * 0.1) * _quat_err(q_ref[3:7], d.qpos[3:7]) + (harness_kd * 0.1) * (w_ref_world - w_world)
            d.xfrc_applied[base, :3] = f
            d.xfrc_applied[base, 3:] = tq
            hforce.append(np.linalg.norm(f))
        mujoco.mj_step(m, d)
        sat.append(np.abs(d.actuator_force[act_id]) >= jp.effort - 1e-6)
        if k % 4 == 0:  # evaluate at 50 Hz
            ref_d.qpos[:] = q_ref
            mujoco.mj_kinematics(m, ref_d)
            mujoco.mj_kinematics(m, d)
            p, pr = d.xpos[allb], ref_d.xpos[allb]
            errs.append(np.linalg.norm(p - pr, axis=-1).mean())
            rel_errs.append(np.linalg.norm((p - d.xpos[base]) - (pr - ref_d.xpos[base]), axis=-1).mean())
            ee_err.append(np.linalg.norm(d.xpos[ee] - ref_d.xpos[ee], axis=-1).mean())
            jerr.append(np.sqrt(np.mean((d.qpos[7:] - q_ref[7:]) ** 2)))
        if not harness and fell_at is None and d.qpos[2] < fall_height:
            fell_at = t
            break
    dur = t_ref[-1]
    sat = np.array(sat)
    rep = {
        "harness": harness,
        "gain_scale": gain_scale,
        "torque_saturation": {robot.joint_names[i]: round(float(sat[:, i].mean()), 3)
                              for i in range(robot.ndof) if sat[:, i].mean() > 0.01},
        "mpkpe_global_cm": round(float(np.mean(errs)) * 100, 2),
        "mpkpe_root_rel_cm": round(float(np.mean(rel_errs)) * 100, 2),
        "ee_err_cm": round(float(np.mean(ee_err)) * 100, 2),
        "joint_rmse_deg": round(float(np.degrees(np.mean(jerr))), 2),
    }
    if harness:
        hf = np.array(hforce) / (mass * 9.81) * 100
        rep["harness_force_pct_mg"] = {"mean": round(float(hf.mean()), 1), "p95": round(float(np.percentile(hf, 95)), 1)}
    else:
        rep["fell"] = fell_at is not None
        rep["completed_fraction"] = round(float((fell_at if fell_at is not None else dur) / dur), 3)
    return rep


def _rotate(q, v):
    from ..utils.rotations import quat_apply
    return quat_apply(q, v)


def evaluate_clip(qpos: np.ndarray, fps: float, robot: IgrisModel | None = None, harness_free: bool = True,
                  segments: list[dict] | None = None) -> dict:
    """Full feasibility report; with `segments` (stage-2 labels, frame indices at this fps)
    the balance and harness replay are also reported per labelled segment."""
    robot = robot or IgrisModel()
    jp = load_joint_params(robot)
    out = {"duration_s": round(len(qpos) / fps, 2)}
    out["kinematic"] = kinematic_report(robot, qpos, fps, jp)
    out["static_balance"] = static_balance_report(robot, qpos, fps=fps)
    out["pd_replay_harness"] = pd_replay(IgrisModel(robot.path), qpos, fps, jp, harness=True, gain_scale=4.0)
    if harness_free:
        out["pd_replay_free"] = pd_replay(IgrisModel(robot.path), qpos, fps, jp, harness=False)
    if segments:
        rows = []
        for sg in segments:
            a, b = int(sg["start"]), int(sg["end"])
            if b - a < int(0.5 * fps):
                continue
            hb = pd_replay(IgrisModel(robot.path), qpos[a:b], fps, jp, harness=True, gain_scale=4.0)
            sb = static_balance_report(robot, qpos[a:b], fps=fps)
            rows.append({"name": sg.get("name", ""), "t_start": round(a / fps, 2), "t_end": round(b / fps, 2),
                         "com_in_support": sb["com_in_support_ratio"], "harness_mean_pct_mg": hb["harness_force_pct_mg"]["mean"],
                         "harness_p95_pct_mg": hb["harness_force_pct_mg"]["p95"], "mpkpe_cm": hb["mpkpe_global_cm"],
                         "saturated_joints": sorted(k for k, v in hb["torque_saturation"].items() if v > 0.1)})
        out["per_segment"] = rows
    out["verdict"] = verdict(out)
    return out


def verdict(rep: dict) -> dict:
    """Heuristic go / review flags for a clip (thresholds are deliberately conservative)."""
    flags = []
    k = rep["kinematic"]
    if k["self_collision_worst_depth_cm"] > 3:
        flags.append("self-collision > 3 cm")
    if k["peak_joint_speed_ratio"] > 1.0:
        flags.append("joint speed above actuator limit")
    if k["floor_penetration_max_cm"] > 2:
        flags.append("floor penetration > 2 cm")
    sb = rep["static_balance"]
    if sb["com_in_support_ratio"] < 0.6:
        flags.append("CoM outside support polygon in >40% of quasi-static grounded frames")
    h = rep["pd_replay_harness"]
    if h["harness_force_pct_mg"]["p95"] > 50:
        flags.append("needs strong external support (harness p95 > 50% mg)")
    if any(v > 0.1 for v in h["torque_saturation"].values()):
        flags.append("torque saturation > 10% of steps in harness replay")
    return {"trackable_guess": len(flags) == 0, "flags": flags}


def load_qpos(path: str | Path):
    """Load qpos + fps from a tracking npz, canonical pkl or episode h5."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        return np.concatenate([z["root_pos"], z["root_quat"], z["joint_pos"]], axis=1).astype(np.float64), float(z["fps"])
    if path.suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        return np.concatenate([d["root_pos"], d["root_rot"], d["dof_pos"]], axis=1).astype(np.float64), float(d["fps"])
    from ..episode import Episode
    ep = Episode.load(path)
    return ep["retarget/igris_c/qpos"].astype(np.float64), float(ep.group_attrs["retarget/igris_c"]["fps"])
