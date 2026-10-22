"""MuJoCo sim2sim runner for IGRIS-C tracking policies exported from Isaac Lab.

Loads ``policy.onnx`` (onnxruntime) or ``policy.pt`` (TorchScript) + ``policy.json`` (written by
igris_pipeline/isaaclab_ext/export.py), the 31-DoF MJCF and a reference motion NPZ, and reproduces the
Isaac Lab control loop:

  * 200 Hz physics (JSON ``sim_dt``), 50 Hz policy (``decimation`` = 4)
  * q_target = default_joint_pos + action_scale * action;   every physics step
    tau = clip(kp (q_target - q) - kd qd, +-effort_limit)  -> data.qfrc_applied   (explicit PD, like Isaac's
    DelayedPD/IdealPD actuators; MJCF motors are not used)
  * the 8 MJCF joints that are FIXED in the URDF (wrists, neck) are held at 0 by a stiff PD
  * joint armature / friction / viscous friction / position limits of the policy joints are overwritten
    from the JSON (Isaac values). ``joint_dynamics``:
      "isaac" (default)   dof_armature = armature, dof_damping = joint_viscous_friction,
                          dof_frictionloss = joint_dynamic_friction (= 0 for the Robros config)
      "isaac_static"      same but dof_frictionloss = joint_friction (PhysX *static* friction effort)
      "none"              armature only;  "mjcf"  keep the MJCF values
    PhysX (Isaac Sim 5) applies the static friction effort only as stiction at rest, MuJoCo frictionloss is a
    Coulomb force that also acts while moving; in the zero-action Isaac-vs-MuJoCo rollout "isaac" matched
    PhysX ~2-4x better than "isaac_static" (see README).
  * by default the MJCF foot boxes are replaced by the URDF ones (sole 9 mm higher) and the ankle-roll mesh
    colliders are disabled, so ground contact geometry matches the Isaac articulation (``foot_geom="urdf"``)
  * observations are built term-by-term from the JSON ``observation.terms`` order with the same formulas
    (igris_pipeline/isaaclab_ext/metrics.py quaternion helpers); MuJoCo freejoint qvel[3:6] is already the
    body-frame angular velocity (= Isaac ``root_ang_vel_b``), qvel[0:3] is the world linear velocity
  * metrics / terminations identical to scripts/isaaclab/play_eval.py (shared code in metrics.py)

Programmatic use::

    from igris_pipeline.sim.mujoco_policy import MujocoPolicyRunner
    runner = MujocoPolicyRunner("logs/isaaclab/igris_c_tracking/<run>/exported")
    result = runner.evaluate(["data/sample/motions/test_sine.npz"])
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from igris_pipeline.isaaclab_ext import constants as C  # noqa: E402  (pure python)
from igris_pipeline.isaaclab_ext.metrics import (  # noqa: E402  (pure numpy)
    EpisodeAccumulator,
    quat_apply,
    quat_apply_inverse,
    quat_conj,
    quat_mul,
    rot6,
    step_metrics,
    write_json,
)
from igris_pipeline.isaaclab_ext.motion import load_clip, load_segments, resample_clip, resolve_motion_files  # noqa: E402

GRAVITY_DIR = np.array([0.0, 0.0, -1.0])


# --------------------------------------------------------------------------------------------------
# policy backends
# --------------------------------------------------------------------------------------------------
class OnnxPolicy:
    def __init__(self, path: str, input_name: str = "obs"):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name if input_name not in [i.name for i in self.sess.get_inputs()] else input_name
        self.output_name = self.sess.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.sess.run([self.output_name], {self.input_name: obs.astype(np.float32)[None]})[0]
        return np.asarray(out[0], dtype=np.float64)


class TorchScriptPolicy:
    def __init__(self, path: str):
        import torch

        self.torch = torch
        self.module = torch.jit.load(path, map_location="cpu").eval()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            out = self.module(self.torch.as_tensor(obs, dtype=self.torch.float32)[None])
        return out[0].numpy().astype(np.float64)


class ZeroPolicy:
    def __init__(self, num_actions: int):
        self.n = num_actions

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.n)


# --------------------------------------------------------------------------------------------------
# reference motion (numpy)
# --------------------------------------------------------------------------------------------------
@dataclass
class Reference:
    name: str
    path: str
    joint_pos: np.ndarray  # (T, J) policy order
    joint_vel: np.ndarray
    body_pos: dict = field(default_factory=dict)  # name -> (T,3)
    body_quat: dict = field(default_factory=dict)  # name -> (T,4)
    body_lin_vel: dict = field(default_factory=dict)
    body_ang_vel: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.joint_pos.shape[0]


def load_reference(path: str, meta: dict) -> Reference:
    clip = load_clip(path)
    fps = float(meta["motion_fps"])
    if abs(clip["fps"] - fps) > 1e-6:
        clip = resample_clip(clip, clip["fps"], fps)
    jl = {n: i for i, n in enumerate(clip["joint_names"])}
    missing = [n for n in meta["joint_names"] if n not in jl]
    if missing:
        raise KeyError(f"{path}: joints {missing} missing in motion")
    jidx = [jl[n] for n in meta["joint_names"]]
    ref = Reference(
        name=os.path.splitext(os.path.basename(path))[0],
        path=path,
        joint_pos=clip["joint_pos"][:, jidx].astype(np.float64),
        joint_vel=clip["joint_vel"][:, jidx].astype(np.float64),
    )
    bl = {n: i for i, n in enumerate(clip["body_names"])}
    need = set(meta["body_names"]) | {meta["anchor_body_name"], meta["root_body_name"]}
    need |= {v["parent"] for v in meta["ee_points"].values()} | set(meta["ee_points"].keys())
    for n in need:
        if n not in bl:
            continue
        i = bl[n]
        q = clip["body_quat_w"][:, i].astype(np.float64)
        ref.body_pos[n] = clip["body_pos_w"][:, i].astype(np.float64)
        ref.body_pos[n][:, 2] += float(meta.get("motion_z_offset", 0.0))
        ref.body_quat[n] = q / np.linalg.norm(q, axis=-1, keepdims=True)
        ref.body_lin_vel[n] = clip["body_lin_vel_w"][:, i].astype(np.float64)
        ref.body_ang_vel[n] = clip["body_ang_vel_w"][:, i].astype(np.float64)
    for n in list(meta["body_names"]) + [meta["anchor_body_name"], meta["root_body_name"]]:
        if n not in ref.body_pos:
            raise KeyError(f"{path}: body {n} missing in motion")
    return ref


# --------------------------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------------------------
class MujocoPolicyRunner:
    """Sim2sim evaluation of an exported Isaac Lab tracking policy in MuJoCo."""

    def __init__(
        self,
        policy_dir: str | None = None,
        json_path: str | None = None,
        policy_path: str | None = None,
        backend: str = "onnx",
        mjcf_path: str | None = None,
        foot_geom: str = "urdf",
        joint_dynamics: str = "isaac",
        passive_kp: float = 100.0,
        passive_kd: float = 1.0,
        zero_action: bool = False,
        solver_iterations: int | None = None,
        noslip_iterations: int | None = None,
    ):
        import mujoco

        self.mj = mujoco
        if policy_dir is not None:
            json_path = json_path or os.path.join(policy_dir, "policy.json")
        with open(json_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        meta = self.meta
        base = os.path.dirname(os.path.abspath(json_path))
        if zero_action:
            self.policy = ZeroPolicy(meta["num_joints"])
        elif backend == "onnx":
            self.policy = OnnxPolicy(policy_path or os.path.join(base, meta["files"]["onnx"]), meta["onnx_io"]["input"])
        elif backend in ("jit", "torchscript"):
            self.policy = TorchScriptPolicy(policy_path or os.path.join(base, meta["files"]["torchscript"]))
        else:
            raise ValueError(backend)

        self.mjcf_path = str(mjcf_path or C.IGRIS_C_MJCF_PATH)
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        m = self.model
        m.opt.timestep = float(meta["sim_dt"])
        if solver_iterations is not None:
            m.opt.iterations = solver_iterations
        if noslip_iterations is not None:
            m.opt.noslip_iterations = noslip_iterations
        self.data = mujoco.MjData(m)
        self.decimation = int(meta["decimation"])

        # ---- joints ----
        self.joint_names = list(meta["joint_names"])
        J = len(self.joint_names)
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.joint_names])
        self.dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.joint_names])
        hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.passive_names = [m.joint(j).name for j in hinge if m.joint(j).name not in self.joint_names]
        self.p_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.passive_names], dtype=int)
        self.p_dadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.passive_names], dtype=int)
        self.p_effort = np.array(
            [np.max(np.abs(m.actuator_ctrlrange[a])) for n in self.passive_names for a in range(m.nu)
             if m.actuator_trnid[a, 0] == m.joint(n).id]
        ) if self.passive_names else np.zeros(0)
        if len(self.p_effort) != len(self.passive_names):
            self.p_effort = np.full(len(self.passive_names), 10.0)
        self.passive_kp, self.passive_kd = float(passive_kp), float(passive_kd)

        self.default_q = np.array(meta["default_joint_pos"], dtype=np.float64)
        self.action_scale = np.array(meta["action_scale"], dtype=np.float64)
        self.kp = np.array(meta["kp"], dtype=np.float64)
        self.kd = np.array(meta["kd"], dtype=np.float64)
        self.effort = np.array(meta["effort_limit"], dtype=np.float64)
        self.soft_limits = np.array(meta["soft_joint_pos_limits"], dtype=np.float64)
        self.action_clip = meta.get("action_clip")
        self.action_mode = meta.get("action_mode", "absolute")
        if self.action_mode not in ("absolute", "residual"):
            raise ValueError(f"unknown action_mode {self.action_mode}")

        # ---- dynamics overrides (Isaac values) ----
        self.joint_dynamics = joint_dynamics
        if joint_dynamics in ("isaac", "isaac_static"):
            m.dof_armature[self.dadr] = np.array(meta["armature"])
            key = "joint_friction" if joint_dynamics == "isaac_static" else "joint_dynamic_friction"
            m.dof_frictionloss[self.dadr] = np.array(meta.get(key, [0.0] * J))
            m.dof_damping[self.dadr] = np.array(meta["joint_viscous_friction"])
        elif joint_dynamics == "none":
            m.dof_armature[self.dadr] = np.array(meta["armature"])
            m.dof_frictionloss[self.dadr] = 0.0
            m.dof_damping[self.dadr] = 0.0
        elif joint_dynamics != "mjcf":
            raise ValueError(joint_dynamics)
        lim = np.array(meta["joint_pos_limits"], dtype=np.float64)
        for k, n in enumerate(self.joint_names):
            jid = m.joint(n).id
            m.jnt_range[jid] = lim[k]
            m.jnt_limited[jid] = 1
        # we apply torques through qfrc_applied; disable the MJCF motors entirely
        m.actuator_gainprm[:, 0] = 0.0
        m.actuator_biasprm[:] = 0.0
        self.foot_geom = foot_geom
        if foot_geom == "urdf":
            self._patch_feet()

        # ---- bodies ----
        self.body_names = list(meta["body_names"])
        self.bid = {n: m.body(n).id for n in set(self.body_names) | {meta["anchor_body_name"], meta["root_body_name"]}}
        self.anchor = meta["anchor_body_name"]
        self.root = meta["root_body_name"]
        self.ee_names = list(meta["ee_points"].keys())
        self.ee_parent_bid = np.array([m.body(meta["ee_points"][n]["parent"]).id for n in self.ee_names])
        self.ee_offset = np.array([meta["ee_points"][n]["offset"] for n in self.ee_names], dtype=np.float64)
        self.obs_terms = list(meta["observation"]["terms"])
        self.obs_dim = int(meta["observation"]["dim"])
        self.thresholds = {k: v["threshold"] for k, v in meta.get("terminations", {}).items()}
        self.last_action = np.zeros(J)
        self.k = 0
        self.ref: Reference | None = None
        self.viewer = None

    # ------------------------------------------------------------------------------------------
    def _patch_feet(self):
        m = self.model
        for side in ("left", "right"):
            gid = m.geom(f"foot_box_{side}").id
            m.geom_size[gid] = np.array(C.URDF_FOOT_BOX_HALF_SIZE)
            m.geom_pos[gid] = np.array(C.URDF_FOOT_BOX_POS)
            m.geom_quat[gid] = np.array([1.0, 0.0, 0.0, 0.0])
        for name in ("Link_Ankle_Roll_Left_collision", "Link_Ankle_Roll_Right_collision"):
            try:
                gid = m.geom(name).id
            except KeyError:
                continue
            m.geom_contype[gid] = 0
            m.geom_conaffinity[gid] = 0

    # ------------------------------------------------------------------------------------------
    # state access
    # ------------------------------------------------------------------------------------------
    def robot_state(self) -> dict:
        d = self.data
        q = d.qpos[self.qadr].copy()
        qd = d.qvel[self.dadr].copy()
        root_quat = d.qpos[3:7] / np.linalg.norm(d.qpos[3:7])
        anchor_pos = d.xpos[self.bid[self.anchor]].copy()
        anchor_quat = d.xquat[self.bid[self.anchor]].copy()
        body_pos = np.stack([d.xpos[self.bid[n]] for n in self.body_names])
        pq = d.xquat[self.ee_parent_bid]
        ee_pos = d.xpos[self.ee_parent_bid] + quat_apply(pq, self.ee_offset)
        return {
            "joint_pos": q,
            "joint_vel": qd,
            "root_quat": root_quat,
            "root_ang_vel_b": d.qvel[3:6].copy(),  # freejoint angular velocity is body-frame
            "anchor_pos": anchor_pos,
            "anchor_quat": anchor_quat,
            "body_pos": body_pos,
            "ee_pos": ee_pos,
        }

    def reference_frame(self, k: int) -> dict:
        r = self.ref
        k = min(k, r.length - 1)
        ee = []
        for n, pb, off in zip(self.ee_names, self.ee_parent_bid, self.ee_offset):
            if n in r.body_pos:
                ee.append(r.body_pos[n][k])
            else:
                parent = self.meta["ee_points"][n]["parent"]
                ee.append(r.body_pos[parent][k] + quat_apply(r.body_quat[parent][k], off))
        return {
            "joint_pos": r.joint_pos[k],
            "joint_vel": r.joint_vel[k],
            "anchor_pos": r.body_pos[self.anchor][k],
            "anchor_quat": r.body_quat[self.anchor][k],
            "body_pos": np.stack([r.body_pos[n][k] for n in self.body_names]),
            "ee_pos": np.stack(ee),
        }

    # ------------------------------------------------------------------------------------------
    # observation (same formulas as mdp/observations.py, order/dims from the JSON)
    # ------------------------------------------------------------------------------------------
    def observation_terms(self) -> dict[str, np.ndarray]:
        s = self.robot_state()
        ref = self.reference_frame(self.k)
        qa_inv = quat_conj(s["anchor_quat"])
        return {
            "motion_joint_pos": ref["joint_pos"],
            "motion_joint_vel": ref["joint_vel"],
            "motion_anchor_pos_b": quat_apply(qa_inv, ref["anchor_pos"] - s["anchor_pos"]),
            "motion_anchor_ori_b": rot6(quat_mul(qa_inv, ref["anchor_quat"])),
            "base_ang_vel": s["root_ang_vel_b"],
            "projected_gravity": quat_apply_inverse(s["root_quat"], GRAVITY_DIR),
            "joint_pos": s["joint_pos"] - self.default_q,
            "joint_vel": s["joint_vel"],
            "actions": self.last_action.copy(),
        }

    def observation(self) -> np.ndarray:
        terms = self.observation_terms()
        parts = []
        for t in self.obs_terms:
            if t["name"] not in terms:
                raise KeyError(f"observation term '{t['name']}' is not implemented in the MuJoCo runner")
            v = np.asarray(terms[t["name"]], dtype=np.float64).reshape(-1)
            if v.shape[0] != t["dim"]:
                raise ValueError(f"term {t['name']}: dim {v.shape[0]} != {t['dim']} (JSON)")
            if t.get("clip"):
                v = np.clip(v, t["clip"][0], t["clip"][1])
            if t.get("scale") is not None:
                v = v * np.asarray(t["scale"])
            parts.append(v)
        obs = np.concatenate(parts)
        assert obs.shape[0] == self.obs_dim
        return obs

    # ------------------------------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------------------------------
    def reset(self, ref: Reference, frame: int = 0):
        mj, m, d = self.mj, self.model, self.data
        self.ref = ref
        self.k = frame
        mj.mj_resetData(m, d)
        root = self.root
        rq = ref.body_quat[root][frame]
        d.qpos[0:3] = ref.body_pos[root][frame]
        d.qpos[3:7] = rq
        d.qvel[0:3] = ref.body_lin_vel[root][frame]  # world frame, body-frame-origin velocity
        d.qvel[3:6] = quat_apply_inverse(rq, ref.body_ang_vel[root][frame])  # body frame
        q0 = np.clip(ref.joint_pos[frame], self.soft_limits[:, 0], self.soft_limits[:, 1])  # as Isaac reset
        d.qpos[self.qadr] = q0
        d.qvel[self.dadr] = ref.joint_vel[frame]
        if len(self.p_qadr):
            d.qpos[self.p_qadr] = 0.0
            d.qvel[self.p_dadr] = 0.0
        self.last_action[:] = 0.0
        mj.mj_forward(m, d)

    def _apply_pd(self, q_target: np.ndarray):
        d = self.data
        d.qfrc_applied[:] = 0.0
        q = d.qpos[self.qadr]
        qd = d.qvel[self.dadr]
        tau = np.clip(self.kp * (q_target - q) - self.kd * qd, -self.effort, self.effort)
        d.qfrc_applied[self.dadr] = tau
        if len(self.p_dadr):
            qp = d.qpos[self.p_qadr]
            qdp = d.qvel[self.p_dadr]
            tau_p = np.clip(-self.passive_kp * qp - self.passive_kd * qdp, -self.p_effort, self.p_effort)
            d.qfrc_applied[self.p_dadr] = tau_p
        return tau

    def action_to_target(self, action: np.ndarray, k: int | None = None) -> np.ndarray:
        """absolute: q_default + scale*a;  residual: q_ref[k] + scale*a with k = the observation's frame."""
        if self.action_mode == "residual":
            k = self.k if k is None else k
            base = self.ref.joint_pos[min(k, self.ref.length - 1)]
        else:
            base = self.default_q
        q_target = base + self.action_scale * action
        if self.action_clip:  # JointAction.clip acts on the processed target
            lo, hi = np.array(self.action_clip.get(".*", (-np.inf, np.inf)), dtype=np.float64)
            q_target = np.clip(q_target, lo, hi)
        return q_target

    def step(self, action: np.ndarray):
        q_target = self.action_to_target(action)
        for _ in range(self.decimation):
            self._apply_pd(q_target)
            self.mj.mj_step(self.model, self.data)
        # mj_step integrates qpos AFTER computing kinematics -> xpos/xquat are one substep stale; refresh them
        # so body/anchor poses match qpos (Isaac reads PhysX poses after the step).
        self.mj.mj_kinematics(self.model, self.data)
        self.last_action = np.asarray(action, dtype=np.float64).copy()
        if self.viewer is not None:
            self.viewer.sync()

    def check_terminations(self, ref: dict, rob: dict) -> list[str]:
        out = []
        th = self.thresholds
        if "anchor_pos" in th and abs(ref["anchor_pos"][2] - rob["anchor_pos"][2]) > th["anchor_pos"]:
            out.append("anchor_pos")
        if "anchor_ori" in th:
            g_ref = quat_apply_inverse(ref["anchor_quat"], GRAVITY_DIR)
            g_rob = quat_apply_inverse(rob["anchor_quat"], GRAVITY_DIR)
            if abs(g_ref[2] - g_rob[2]) > th["anchor_ori"]:
                out.append("anchor_ori")
        if "ee_body_pos" in th and np.any(np.abs(ref["ee_pos"][:, 2] - rob["ee_pos"][:, 2]) > th["ee_body_pos"]):
            out.append("ee_body_pos")
        return out

    # ------------------------------------------------------------------------------------------
    def run_clip(self, ref: Reference, acc: EpisodeAccumulator, env_index: int, max_steps: int | None = None,
                 realtime: bool = False, trajectory: dict | None = None):
        n = acc.n
        self.reset(ref, 0)
        T = ref.length
        steps = T if max_steps is None else min(T, max_steps)
        mask = np.zeros(n, dtype=bool)
        mask[env_index] = True
        for _ in range(steps):
            t0 = time.time()
            obs = self.observation()
            action = self.policy(obs)
            ref_k = self.reference_frame(self.k)
            self.step(action)
            rob = self.robot_state()
            mets = step_metrics({k: v[None] for k, v in ref_k.items()}, {k: v[None] for k, v in rob.items()})
            mets_n = {k: np.repeat(v, n) for k, v in mets.items()}
            acc.add(mets_n, mask, frames=np.full(n, self.k))
            if trajectory is not None:
                trajectory.setdefault("obs", []).append(obs)
                trajectory.setdefault("action", []).append(action)
                trajectory.setdefault("qpos", []).append(self.data.qpos.copy())
            reasons = self.check_terminations(ref_k, rob)
            if reasons:
                r = [""] * n
                r[env_index] = "+".join(reasons)
                acc.finish(mask, success=False, reasons=r)
                return
            if self.k == T - 1:
                acc.finish(mask, success=True)
                return
            self.k += 1
            if realtime:
                time.sleep(max(0.0, float(self.meta["policy_dt"]) - (time.time() - t0)))

    def evaluate(self, motions, max_steps: int | None = None, render: bool = False, record_trajectory: bool = False) -> dict:
        files = resolve_motion_files(motions)
        refs = [load_reference(f, self.meta) for f in files]
        names = [r.name for r in refs]
        names = [nm if names.count(nm) == 1 else f"{nm}#{i}" for i, nm in enumerate(names)]
        fps = float(self.meta["motion_fps"])
        segments = {i: segs for i, f in enumerate(files) if (segs := load_segments(f, fps))}
        acc = EpisodeAccumulator(len(refs), np.arange(len(refs)), names, [r.length for r in refs], segments=segments)
        trajs = [dict() for _ in refs] if record_trajectory else None
        if render:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        t0 = time.time()
        for i, ref in enumerate(refs):
            self.run_clip(ref, acc, i, max_steps=max_steps, realtime=render,
                          trajectory=trajs[i] if trajs is not None else None)
        wall = time.time() - t0
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        res = {
            "simulator": "mujoco",
            "mujoco_version": self.mj.__version__,
            "mjcf": self.mjcf_path,
            "checkpoint": self.meta.get("checkpoint", ""),
            "motion_files": files,
            "options": {"action_mode": self.action_mode, "foot_geom": self.foot_geom, "joint_dynamics": self.joint_dynamics,
                        "passive_kp": self.passive_kp, "passive_kd": self.passive_kd},
            "wall_time_s": wall,
            "metric_definitions": "see igris_pipeline/isaaclab_ext/metrics.py",
            **acc.summary(),
        }
        if trajs is not None:
            res["_trajectories"] = trajs
        return res


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="MuJoCo sim2sim evaluation of an exported IGRIS-C tracking policy.")
    p.add_argument("--policy_dir", type=str, required=True, help="directory with policy.json (+ policy.onnx / policy.pt)")
    p.add_argument("--motion", type=str, default=None, help="NPZ / dir / list (default: the policy's training motions)")
    p.add_argument("--backend", choices=["onnx", "jit"], default="onnx")
    p.add_argument("--mjcf", type=str, default=None)
    p.add_argument("--foot_geom", choices=["urdf", "mjcf"], default="urdf")
    p.add_argument("--joint_dynamics", choices=["isaac", "isaac_static", "mjcf", "none"], default="isaac")
    p.add_argument("--passive_kp", type=float, default=100.0)
    p.add_argument("--passive_kd", type=float, default=1.0)
    p.add_argument("--zero_action", action="store_true", help="diagnostic: hold the default pose")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--render", action="store_true", help="open the passive MuJoCo viewer (real time)")
    p.add_argument("--output", type=str, default=None, help="metrics JSON (default <policy_dir>/../eval/mujoco_metrics.json)")
    p.add_argument("--save_trajectory", type=str, default=None, help="optional npz with obs/actions/qpos per clip")
    args = p.parse_args(argv)

    runner = MujocoPolicyRunner(
        policy_dir=args.policy_dir, backend=args.backend, mjcf_path=args.mjcf, foot_geom=args.foot_geom,
        joint_dynamics=args.joint_dynamics, passive_kp=args.passive_kp, passive_kd=args.passive_kd,
        zero_action=args.zero_action,
    )
    motions = args.motion or runner.meta["training_motion_files"]
    res = runner.evaluate(motions, max_steps=args.max_steps, render=args.render, record_trajectory=bool(args.save_trajectory))
    trajs = res.pop("_trajectories", None)
    if trajs is not None:
        flat = {}
        for i, tr in enumerate(trajs):
            for k, v in tr.items():
                flat[f"clip{i}/{k}"] = np.asarray(v)
        np.savez(args.save_trajectory, **flat)
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.policy_dir)), "eval", "mujoco_metrics.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_json(out, res)
    keys = ("success_rate", "mpkpe_global", "mpkpe_root_relative", "joint_pos_error", "joint_vel_error",
            "ee_pos_error", "anchor_pos_error", "anchor_drift_xy_final", "completion_ratio")
    print("[MUJOCO] " + ", ".join(f"{k}={res[k]:.4f}" for k in keys))
    print(f"[MUJOCO] num_falls={res['num_falls']} failure_reasons={res['failure_reasons']} wall={res['wall_time_s']:.1f}s")
    print(f"[MUJOCO] wrote {out}")
    return res


if __name__ == "__main__":
    main()
