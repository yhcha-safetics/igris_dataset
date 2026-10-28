"""Stage 3: retarget the corrected PICO skeleton onto IGRIS-C with differential IK (mink).

Per frame (warm-started from the previous solution), two IK stages as in GMR:
  stage 1: mostly orientation tasks + feet / pelvis / head positions
  stage 2: adds hip / knee positions and stronger trunk weights
Each task cost is multiplied by the stage 1-2 confidence of its source joint, and a weak
posture task pulls unobserved joints (e.g. legs in the 3-point profile) to the default
stance. Joint position and velocity limits are enforced inside the QP.

Afterwards: contact-aware ground alignment (soles on the floor whenever the human foot
is in contact), zero-phase smoothing and joint-range clipping.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mink
import mujoco
import numpy as np
from scipy.interpolate import interp1d

from ..skeleton import BONES, J, NUM_JOINTS, PICO_JOINTS
from ..utils.geometry import lowpass, lowpass_quat, swing_quat, unit
from ..utils.rotations import quat_apply, quat_mul, quat_normalize, slerp, yaw_of, yaw_quat
from .robot import ROOT, IgrisModel

# bone -> body-part group for kinematic scaling
def _bone_group(child_name: str) -> str:
    if child_name.endswith(("_knee", "_ankle", "_foot")):
        return "leg"
    if child_name.endswith("_hip"):
        return "hip_width"
    if child_name.endswith(("_elbow", "_wrist", "_hand")):
        return "arm"
    if child_name.endswith(("_collar", "_shoulder")):
        return "shoulder_width"
    return "trunk"  # spine1-3, neck, head


_BONE_GROUP = {c: _bone_group(PICO_JOINTS[c]) for _, c in BONES}


@dataclass
class RetargetResult:
    fps: float
    qpos: np.ndarray                       # (T, nq) root xyz, root wxyz, 31 joints (MJCF order)
    joint_names: list
    diagnostics: dict = field(default_factory=dict)
    scaled_targets: dict = field(default_factory=dict)


class Retargeter:
    def __init__(self, cfg: dict, robot: IgrisModel | None = None):
        self.cfg = cfg
        mjcf = Path(cfg["robot"]["mjcf"])
        self.robot = robot or IgrisModel(mjcf if mjcf.is_absolute() else ROOT / mjcf)
        self.model = self.robot.model
        sc = cfg["solver"]
        self.solver = sc.get("name", "daqp")
        self.damping = float(sc.get("damping", 0.5))
        self.max_iter = int(sc.get("max_iter", 10))
        self.tol = float(sc.get("tol", 1e-3))
        self.min_conf = float(sc.get("min_confidence", 0.05))
        self.dt = float(self.model.opt.timestep)
        geo = self._geometry_offsets(cfg.get("geometry_offsets") or {})
        self.stages = []
        for key in ("stage1", "stage2"):
            table = cfg.get(key) or {}
            entries = []
            for body, (human, wp, wr, po, ro) in table.items():
                if wp == 0 and wr == 0:
                    continue
                task = mink.FrameTask(body, "body", position_cost=float(wp), orientation_cost=float(wr), lm_damping=1.0)
                rot_off = quat_normalize(np.array(ro, float))
                if body in geo:
                    rot_off = quat_normalize(quat_mul(rot_off, geo[body]))
                entries.append({"body": body, "joint": J[human], "wp": float(wp), "wr": float(wr),
                                "pos_off": np.array(po, float), "rot_off": rot_off, "task": task})
            self.stages.append(entries)
        cost = np.full(self.model.nv, float(sc.get("posture_cost", 0.02)))
        cost[:6] = 0.0
        # per-joint overrides, e.g. shoulder pitch/yaw become collinear at shoulder roll ~90 deg
        # (T-pose) and drift along that null space without a stronger posture pull
        import re
        for pattern, c in (sc.get("posture_cost_overrides") or {}).items():
            for n in self.robot.joint_names:
                if re.fullmatch(pattern, n):
                    cost[self.model.jnt_dofadr[self.robot.jid(n)]] = float(c)
        self.posture = mink.PostureTask(self.model, cost=cost)
        self.posture.set_target(self.robot.default_qpos)
        bal = sc.get("balance") or {}
        self.balance = bal if bal.get("enabled", False) else None
        if self.balance:
            self.com_task = mink.ComTask(cost=np.array([float(bal.get("cost", 50.0))] * 2 + [0.0]))
        self.limits = [mink.ConfigurationLimit(self.model)]
        vmax = sc.get("velocity_limit")
        if vmax:
            self.limits.append(mink.VelocityLimit(self.model, {n: float(vmax) for n in self.robot.joint_names}))

    def _geometry_offsets(self, mapping: dict) -> dict:
        """Swing (wxyz) taking each body's zero-pose segment direction to -Z, in body frame."""
        r = self.robot
        d = r.forward(r.make_qpos(np.array([0, 0, 1.0]), np.array([1.0, 0, 0, 0]), np.zeros(r.ndof)))
        out = {}
        for body, child in mapping.items():
            seg = d.xpos[r.bid(child)] - d.xpos[r.bid(body)]
            local = d.xmat[r.bid(body)].reshape(3, 3).T @ seg
            out[body] = swing_quat(local, np.array([0.0, 0.0, -1.0]))
        return out

    # ------------------------------------------------------------------------------------
    def scale_human(self, pos: np.ndarray, quat: np.ndarray, stature: float | None) -> np.ndarray:
        return self.apply_scale(pos, self.scale_params(pos, stature))

    def scale_params(self, pos: np.ndarray, stature: float | None) -> dict:
        """Body-part scale factors from a (T,24,3) clip (also stored as self.scale_info).

        A streaming caller computes them once on a calibration window and then applies them per
        frame with apply_scale()."""
        sc = self.cfg.get("scaling", {})
        if sc.get("mode", "kinematic") == "gmr":
            h = stature or sc.get("human_height_assumption", 1.8)
            self.scale_info = {"mode": "gmr", "ratio": h / float(sc.get("human_height_assumption", 1.8))}
            return self.scale_info
        # kinematic scaling from segment-length ratios
        L = self.robot.segment_lengths()
        hum = np.median(np.stack([np.linalg.norm(pos[:, c] - pos[:, p], axis=-1) for p, c in BONES], -1), 0)
        bl = {(p, c): hum[k] for k, (p, c) in enumerate(BONES)}

        def hlen(*pairs):
            return sum(bl[(J[a], J[b])] for a, b in pairs)

        human_leg = 0.5 * (hlen(("left_hip", "left_knee"), ("left_knee", "left_ankle"))
                           + hlen(("right_hip", "right_knee"), ("right_knee", "right_ankle")))
        robot_leg = 0.5 * (L["thigh_Left"] + L["shank_Left"] + L["thigh_Right"] + L["shank_Right"])
        human_arm = 0.5 * (hlen(("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"))
                           + hlen(("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")))
        robot_arm = 0.5 * (L["upper_arm_Left"] + L["forearm_Left"] + L["upper_arm_Right"] + L["forearm_Right"])
        human_trunk = np.median(np.linalg.norm(pos[:, J["neck"]] - pos[:, J["pelvis"]], axis=-1))
        human_hipw = np.median(np.linalg.norm(pos[:, J["left_hip"]] - pos[:, J["right_hip"]], axis=-1))
        human_shw = np.median(np.linalg.norm(pos[:, J["left_shoulder"]] - pos[:, J["right_shoulder"]], axis=-1))
        r = {"leg": robot_leg / human_leg, "arm": robot_arm / human_arm, "trunk": L["trunk"] / human_trunk,
             "hip_width": L["hip_width"] / human_hipw, "shoulder_width": L["shoulder_width"] / human_shw}
        self.scale_info = {"mode": "kinematic", "ratios": {k: round(float(v), 4) for k, v in r.items()}}
        return {"mode": "kinematic", "ratios": r}

    def apply_scale(self, pos: np.ndarray, params: dict) -> np.ndarray:
        """Scale a (T,24,3) human skeleton with scale_params() output."""
        root = J[self.cfg["robot"].get("human_root", "pelvis")]
        if params["mode"] == "gmr":
            ratio = params["ratio"]
            table = self.cfg.get("scaling", {}).get("gmr_scale_table", {})
            s_root = table.get("pelvis", 1.0) * ratio
            out = pos.copy()
            root_pos = pos[:, root] * s_root
            for j in range(NUM_JOINTS):
                s = table.get(PICO_JOINTS[j])
                out[:, j] = root_pos + (pos[:, j] - pos[:, root]) * (s * ratio if s is not None else s_root)
            out[:, root] = root_pos
            return out
        r = params["ratios"]
        out = np.zeros_like(pos)
        out[:, root] = pos[:, root] * r["leg"]
        for p, c in BONES:  # BONES are in topological order
            out[:, c] = out[:, p] + r[_BONE_GROUP[c]] * (pos[:, c] - pos[:, p])
        return out

    def _pin_contact_feet(self, spos, quat, contact, fps):
        """Feet in contact: ankle at the robot's flat-sole height, foot orientation yaw-only.

        Blended with a smoothed contact weight so lift-off / touch-down stay continuous.
        """
        foot_entries = {e["joint"] for st in self.stages for e in st if e["body"].startswith("Link_Ankle_Roll")}
        h = self.robot.ankle_flat_height
        for k, side in enumerate(("left", "right")):
            j = J[f"{side}_ankle"]
            if j not in foot_entries:
                continue
            w = np.clip(lowpass(contact[:, k].astype(float), fps, 4.0), 0.0, 1.0)
            spos[:, j, 2] = (1 - w) * spos[:, j, 2] + w * h
            flat = yaw_quat(yaw_of(quat[:, j]))
            quat[:, j] = slerp(quat[:, j], flat, w)
        return spos, quat

    def _targets(self, entry, pos_t, quat_t):
        j = entry["joint"]
        q = quat_mul(quat_t[j], entry["rot_off"])
        p = pos_t[j] + quat_apply(q, entry["pos_off"])
        return p, q

    def _stage_error(self, cfg, tasks):
        return float(np.linalg.norm(np.concatenate([t.compute_error(cfg) for t in tasks])))

    def _support_target(self, data, contact_t, margin):
        """Closest point to the CoM ground projection inside the (shrunk) support polygon."""
        from scipy.spatial import ConvexHull

        mujoco.mj_comPos(self.model, data)
        com = data.subtree_com[self.robot.bid(self.cfg["robot"]["root_body"])].copy()
        pts = []
        for k, g in enumerate(self.robot.foot_geoms):
            if not contact_t[k]:
                continue
            half = self.model.geom_size[g].copy()
            half[:2] = np.maximum(half[:2] - margin, 0.005)
            corners = np.array([[sx, sy, -1.0] for sx in (-1, 1) for sy in (-1, 1)]) * half
            pts.append(data.geom_xpos[g][:2] + (corners @ data.geom_xmat[g].reshape(3, 3).T)[:, :2])
        if not pts:
            return None, com
        pts = np.concatenate(pts)
        hull = ConvexHull(pts)
        if np.all(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2] <= 0):
            return com.copy(), com  # already inside
        v = pts[hull.vertices]
        best, bd = None, np.inf
        for a, b in zip(v, np.roll(v, -1, axis=0)):
            ab = b - a
            tproj = np.clip(np.dot(com[:2] - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            c = a + tproj * ab
            dd = np.linalg.norm(c - com[:2])
            if dd < bd:
                best, bd = c, dd
        tgt = com.copy()
        tgt[:2] = best
        return tgt, com

    def solve_frame(self, cfg, pos_t, quat_t, conf_t, first=False, contact_t=None, staticness=0.0):
        for si, entries in enumerate(self.stages):
            if not entries:
                continue
            tasks = []
            for e in entries:
                p, q = self._targets(e, pos_t, quat_t)
                c = max(float(conf_t[e["joint"]]), self.min_conf)
                e["task"].set_position_cost(e["wp"] * c)
                e["task"].set_orientation_cost(e["wr"] * c)
                e["task"].set_target(mink.SE3.from_rotation_and_translation(mink.SO3(q), p))
                tasks.append(e["task"])
            all_tasks = tasks + [self.posture]
            n_iter = self.max_iter * (10 if first else 1)
            err = self._stage_error(cfg, tasks)
            for _ in range(n_iter + 1):
                vel = mink.solve_ik(cfg, all_tasks, self.dt, self.solver, damping=self.damping, limits=self.limits)
                cfg.integrate_inplace(vel, self.dt)
                new_err = self._stage_error(cfg, tasks)
                if err - new_err < self.tol:
                    break
                err = new_err
            last_tasks = all_tasks
        # balance stage: quasi-static frames keep the CoM over the support polygon. The CoM
        # offset is rate-limited over time, so support-polygon changes (lift-off, touch-down)
        # do not make the trunk snap.
        if self.balance:
            want = np.zeros(2)
            if contact_t is not None and staticness > 0.05 and np.any(contact_t):
                tgt, com = self._support_target(cfg.data, contact_t, float(self.balance.get("margin", 0.02)))
                if tgt is not None:
                    want = (tgt[:2] - com[:2]) * staticness
            step = float(self.balance.get("max_shift_rate", 0.3)) * self.dt_frame
            self._com_shift = self._com_shift + np.clip(want - self._com_shift, -step, step)
            if np.linalg.norm(self._com_shift) > 1e-3:
                mujoco.mj_comPos(self.model, cfg.data)
                tgt = cfg.data.subtree_com[self.robot.bid(self.cfg["robot"]["root_body"])].copy()
                tgt[:2] += self._com_shift
                w = float(self.balance.get("cost", 500.0))
                self.com_task.set_cost(np.array([w, w, 0.0]))
                self.com_task.set_target(tgt)
                for _ in range(int(self.balance.get("iters", 10))):
                    vel = mink.solve_ik(cfg, last_tasks + [self.com_task], self.dt, self.solver,
                                        damping=self.damping, limits=self.limits)
                    cfg.integrate_inplace(vel, self.dt)
        return cfg.q.copy()

    # ------------------------------------------------------------------------------------
    def retarget(self, pos, quat, fps, confidence=None, contact=None, stature=None, verbose=False) -> RetargetResult:
        T = pos.shape[0]
        conf = np.ones((T, NUM_JOINTS)) if confidence is None else np.asarray(confidence, float)
        spos = self.scale_human(np.asarray(pos, float), quat, stature)
        quat = np.asarray(quat, float).copy()
        if contact is not None and self.cfg.get("postprocess", {}).get("flat_feet_in_contact", True):
            spos, quat = self._pin_contact_feet(spos, quat, np.asarray(contact, bool), fps)
        cfg = mink.Configuration(self.model)
        q0 = self.robot.default_qpos.copy()
        q0[:2] = spos[0, J["pelvis"], :2]
        cfg.update(q0)
        qpos = np.zeros((T, self.model.nq))
        # quasi-static weight for the balance stage from the human pelvis speed
        v = np.linalg.norm(np.gradient(lowpass(spos[:, J["pelvis"], :2], fps, 2.0), 1.0 / fps, axis=0), axis=-1)
        vs = float((self.balance or {}).get("speed_scale", 0.25))
        staticness = np.exp(-(v / vs) ** 2)
        c_arr = None if contact is None else np.asarray(contact, bool)
        self._com_shift = np.zeros(2)
        self.dt_frame = 1.0 / fps
        t0 = time.time()
        for t in range(T):
            qpos[t] = self.solve_frame(cfg, spos[t], quat[t], conf[t], first=(t == 0),
                                       contact_t=None if c_arr is None else c_arr[t], staticness=staticness[t])
            if verbose and t % 500 == 0:
                print(f"  [retarget] frame {t}/{T}  {time.time() - t0:.1f}s")
        solve_time = time.time() - t0
        qpos = self.postprocess(qpos, fps, contact)
        res = RetargetResult(fps=fps, qpos=qpos, joint_names=list(self.robot.joint_names),
                             scaled_targets={"pos": spos, "quat": quat})
        res.diagnostics = self.diagnose(qpos, spos, quat, conf, contact, fps)
        res.diagnostics["solve_time_s"] = round(solve_time, 2)
        res.diagnostics["scaling"] = getattr(self, "scale_info", {})
        return res

    def postprocess(self, qpos, fps, contact):
        pp = self.cfg.get("postprocess", {})
        q = qpos.copy()
        hz = pp.get("smooth_hz")
        if hz:
            q[:, :3] = lowpass(q[:, :3], fps, hz)
            q[:, 3:7] = lowpass_quat(q[:, 3:7], fps, hz)
            q[:, 7:] = lowpass(q[:, 7:], fps, hz)
        margin = float(pp.get("joint_margin", 0.0))
        rng = self.robot.jnt_range
        q[:, 7:] = np.clip(q[:, 7:], rng[:, 0] + margin, rng[:, 1] - margin)
        ratio = pp.get("max_speed_ratio")
        if ratio:  # rate-limit every joint to a fraction of its actuator speed limit
            from ..sim.mujoco_eval import load_joint_params
            vmax = load_joint_params(self.robot).velocity * float(ratio) / fps
            for t in range(1, len(q)):
                q[t, 7:] = q[t - 1, 7:] + np.clip(q[t, 7:] - q[t - 1, 7:], -vmax, vmax)
        mode = pp.get("ground", "contact")
        if mode != "none":
            sole = np.stack([self.robot.sole_min_z(self.robot.forward(q[t])) for t in range(len(q))])
            if mode == "contact" and contact is not None and np.asarray(contact).any():
                c = np.asarray(contact, bool)
                dz = np.full(len(q), np.nan)
                anyc = c.any(axis=1)
                low = np.where(c, sole, np.inf).min(axis=1)
                dz[anyc] = -low[anyc]
                idx = np.nonzero(anyc)[0]
                if len(idx) >= 2:
                    dz = interp1d(idx, dz[idx], bounds_error=False, fill_value=(dz[idx[0]], dz[idx[-1]]))(np.arange(len(q)))
                else:
                    dz = np.nan_to_num(dz, nan=-np.min(sole[anyc])) if anyc.any() else -sole.min(1)
                dz = lowpass(dz, fps, 3.0)
            else:
                dz = np.maximum(0.0, -sole.min(axis=1))
                dz = lowpass(dz, fps, 3.0)
            dz = np.maximum(dz, -sole.min(axis=1))  # never below the floor
            q[:, 2] += dz
        return q

    # ------------------------------------------------------------------------------------
    def diagnose(self, qpos, spos, quat, conf, contact, fps) -> dict:
        robot = self.robot
        bodies = sorted({e["body"] for st in self.stages for e in st})
        P, Q = robot.fk_sequence(qpos, bodies)
        diag = {"frames": int(len(qpos))}
        # tracking residual vs. scaled targets (stage-2 table when present)
        last = {e["body"]: e for st in self.stages for e in st}
        pos_err = {}
        for b, e in last.items():
            if e["wp"] <= 0:
                continue
            tp = np.stack([self._targets(e, spos[t], quat[t])[0] for t in range(len(qpos))])
            k = bodies.index(b)
            pos_err[b] = float(np.median(np.linalg.norm(P[:, k] - tp, axis=-1)) * 100)
        diag["target_pos_err_cm_median"] = {k: round(v, 2) for k, v in pos_err.items()}
        # segment direction similarity human vs robot (scale-free retarget quality)
        pairs = {
            "thigh_L": (("left_hip", "left_knee"), ("Link_Hip_Pitch_Left", "Link_Knee_Pitch_Left")),
            "shank_L": (("left_knee", "left_ankle"), ("Link_Knee_Pitch_Left", "Link_Ankle_Roll_Left")),
            "thigh_R": (("right_hip", "right_knee"), ("Link_Hip_Pitch_Right", "Link_Knee_Pitch_Right")),
            "shank_R": (("right_knee", "right_ankle"), ("Link_Knee_Pitch_Right", "Link_Ankle_Roll_Right")),
            "upper_arm_L": (("left_shoulder", "left_elbow"), ("Link_Shoulder_Pitch_Left", "Link_Elbow_Pitch_Left")),
            "forearm_L": (("left_elbow", "left_wrist"), ("Link_Elbow_Pitch_Left", "Link_Wrist_Pitch_Left")),
            "upper_arm_R": (("right_shoulder", "right_elbow"), ("Link_Shoulder_Pitch_Right", "Link_Elbow_Pitch_Right")),
            "forearm_R": (("right_elbow", "right_wrist"), ("Link_Elbow_Pitch_Right", "Link_Wrist_Pitch_Right")),
            "trunk": (("pelvis", "neck"), ("Link_Waist_Pitch", "Link_Neck_Pitch")),
        }
        names = sorted({b for _, (_, rb) in pairs.items() for b in rb})
        RP, _ = robot.fk_sequence(qpos, names)
        ang = {}
        for k, ((ha, hb), (ra, rb)) in pairs.items():
            hd = unit(spos[:, J[hb]] - spos[:, J[ha]])
            rd = unit(RP[:, names.index(rb)] - RP[:, names.index(ra)])
            a = np.degrees(np.arccos(np.clip(np.sum(hd * rd, -1), -1, 1)))
            w = np.minimum(conf[:, J[ha]], conf[:, J[hb]])
            ang[k] = round(float(np.sum(a * w) / max(w.sum(), 1e-6)), 2)
        diag["segment_dir_err_deg"] = ang
        # joint limits
        rng = robot.jnt_range
        near = (qpos[:, 7:] < rng[:, 0] + np.radians(1)) | (qpos[:, 7:] > rng[:, 1] - np.radians(1))
        diag["joint_limit_ratio"] = {robot.joint_names[i]: round(float(near[:, i].mean()), 3)
                                     for i in range(len(robot.joint_names)) if near[:, i].mean() > 0.02}
        # feet: skating during contact, penetration
        sole_ids = robot.foot_geoms
        skate, pen = [], []
        for t in range(len(qpos)):
            d = robot.forward(qpos[t])
            pen.append(robot.sole_min_z(d))
            skate.append(d.geom_xpos[sole_ids][:, :2].copy())
        skate = np.array(skate)
        pen = np.array(pen)
        if contact is not None:
            c = np.asarray(contact, bool)
            v = np.linalg.norm(np.diff(skate, axis=0), axis=-1) * fps
            both = c[1:] & c[:-1]
            diag["foot_skate_mps_in_contact"] = round(float(v[both].mean()) if both.any() else 0.0, 4)
            diag["sole_height_in_contact_cm"] = round(float(np.abs(pen[c]).mean() * 100) if c.any() else 0.0, 2)
        diag["max_penetration_cm"] = round(float(max(0.0, -pen.min()) * 100), 2)
        jv = np.abs(np.diff(qpos[:, 7:], axis=0)) * fps
        diag["max_joint_vel"] = round(float(jv.max()), 2)
        diag["p99_joint_vel"] = round(float(np.percentile(jv, 99)), 2)
        return diag
