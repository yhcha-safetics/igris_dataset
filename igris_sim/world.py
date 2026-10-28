"""MuJoCo world for teleoperation: IGRIS-C in a scene, following the streamed retarget targets.

Robot modes (configs/sim.yaml physics.mode):
  kinematic  the robot's qpos / qvel are set to the (interpolated) targets before every physics
             step, so the robot follows the operator exactly and pushes objects with the matching
             velocity; only the objects are simulated.
  harness    the robot is simulated: joint PD servos track the targets (igris_c_actuators.yaml
             gains x gain_scale) and a virtual 6-DoF spring holds the base on the target, as in
             igris_pipeline.sim.mujoco_eval.pd_replay. Contacts, joint limits and torque limits are
             physical; the harness force says how much external help the motion needs.

Grasping: pressing a controller grip attaches the nearest graspable object whose surface is within
grasp.radius of that palm; releasing the grip lets it go (with its current velocity, so it can be
thrown). In harness mode the attachment is a MuJoCo weld constraint between hand and object; in
kinematic mode the robot does not respond to forces, so a weld would be solved mostly on the
(overwritten) hand side and the object would lag: the object is carried kinematically instead,
its pose set from the hand pose before every physics step.
"""
from __future__ import annotations

import mujoco
import numpy as np

from igris_pipeline.retarget.robot import IgrisModel
from igris_pipeline.sim.mujoco_eval import (_quat_err, _rotate, apply_joint_params, configure_position_servos,
                                            load_joint_params)
from igris_pipeline.utils.rotations import quat_conj, quat_mul, quat_normalize, slerp, yaw_quat

from .scene import HANDS, Scene, build_model


class SimWorld:
    def __init__(self, scene: Scene, sim_cfg: dict, mode: str | None = None):
        self.scene = scene
        self.cfg = sim_cfg
        self.model, self.info = build_model(scene)
        m = self.model
        ph = sim_cfg.get("physics", {})
        m.opt.iterations = int(ph.get("solver_iterations", m.opt.iterations))
        m.opt.noslip_iterations = int(ph.get("noslip_iterations", m.opt.noslip_iterations))
        self.robot = IgrisModel(model=m)
        self.data = mujoco.MjData(m)
        r = self.robot
        self.nq_r, self.nv_r = r.nq, r.nv
        fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        if m.jnt_qposadr[fb] != 0 or max(m.jnt_qposadr[r.jid(n)] for n in r.joint_names) >= self.nq_r:
            raise RuntimeError("the robot must be the first body tree of the scene model")
        self.mode = mode or ph.get("mode", "kinematic")
        if self.mode not in ("kinematic", "harness"):
            raise ValueError(f"unknown physics mode {self.mode!r}")
        self.base = r.bid("base_link")
        self.mass = float(np.sum(m.body_mass[[b for b in range(m.nbody) if self._is_robot_body(b)]]))
        if self.mode == "harness":
            hc = ph.get("harness", {})
            jp = load_joint_params(r)
            gs = float(hc.get("gain_scale", 4.0))
            self.kp, self.kd = jp.kp * gs, jp.kd * np.sqrt(gs)
            apply_joint_params(r, jp)
            self.act = configure_position_servos(r, self.kp, self.kd, jp.effort)
            self.h_kp, self.h_kd = float(hc.get("kp", 3000.0)), float(hc.get("kd", 300.0))
        self.palm = {s: r.gid(f"{b}_collision") for s, b in HANDS.items()}
        self.hand_body = {s: r.bid(b) for s, b in HANDS.items()}
        g = sim_cfg.get("grasp", {})
        self.grasp_radius = float(g.get("radius", 0.10))
        self.grip_on, self.grip_off = float(g.get("grip_on", 0.6)), float(g.get("grip_off", 0.4))
        self.dynamic = [n for n in self.info.objects if self.info.dynamic[n]]
        self.dofadr = {n: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{n}_free")])
                       for n in self.dynamic}
        self.harness_force = 0.0
        self.reset()

    def _is_robot_body(self, b):
        return b == self.base or self.robot._descends(b, self.base)

    # -- state ----------------------------------------------------------------------------
    def spawn_qpos(self) -> np.ndarray:
        xy, yaw = self.scene.spawn
        q = self.robot.default_qpos.copy()
        q[:2] = xy
        q[3:7] = yaw_quat(np.array(yaw))
        return q

    def reset(self, q_robot: np.ndarray | None = None):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.eq_active[:] = 0
        self.grasped = {s: None for s in HANDS}
        self._carried = {}  # kinematic mode: side -> (object, pos in hand frame, quat in hand frame, last pose)
        q = self.spawn_qpos() if q_robot is None else np.asarray(q_robot, float)
        d.qpos[:self.nq_r] = q
        mujoco.mj_forward(m, d)
        self._q_prev = q.copy()

    def robot_qpos(self) -> np.ndarray:
        return self.data.qpos[:self.nq_r].copy()

    def object_poses(self, names=None):
        names = self.dynamic if names is None else names
        ids = [self.info.body[n] for n in names]
        return self.data.xpos[ids].copy(), self.data.xquat[ids].copy()

    def palm_pos(self, side) -> np.ndarray:
        return self.data.geom_xpos[self.palm[side]].copy()

    # -- stepping -------------------------------------------------------------------------
    def step(self, q_target: np.ndarray, dt: float):
        """Advance by dt (s), moving the robot from the previous target to q_target."""
        m, d = self.model, self.data
        h = float(m.opt.timestep)
        n = max(1, int(round(dt / h)))
        q0, q1 = self._q_prev, np.asarray(q_target, float)
        v_tick = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, v_tick, n * h, self._full(q0), self._full(q1))
        v_ref = v_tick[:self.nv_r]
        forces = []
        q_prev_sub = q0
        for k in range(1, n + 1):
            q_sub = self._interp(q0, q1, k / n)
            if self.mode == "kinematic":
                d.qpos[:self.nq_r] = q_prev_sub
                d.qvel[:self.nv_r] = v_ref
                self._carry(h)
                mujoco.mj_step(m, d)
                d.qpos[:self.nq_r] = q_sub
                d.qvel[:self.nv_r] = v_ref
            else:
                d.ctrl[self.act] = q_sub[7:] + self.kd / self.kp * v_ref[6:]
                d.xfrc_applied[:] = 0.0
                f = self.h_kp * (q_sub[:3] - d.qpos[:3]) + self.h_kd * (v_ref[:3] - d.qvel[:3])
                w_ref = _rotate(q_sub[3:7], v_ref[3:6])
                w = _rotate(d.qpos[3:7], d.qvel[3:6])
                tq = (self.h_kp * 0.1) * _quat_err(q_sub[3:7], d.qpos[3:7]) + (self.h_kd * 0.1) * (w_ref - w)
                d.xfrc_applied[self.base, :3] = f
                d.xfrc_applied[self.base, 3:] = tq
                mujoco.mj_step(m, d)
                forces.append(float(np.linalg.norm(f)))
            q_prev_sub = q_sub
        if self.mode == "kinematic":
            self._carry(h, update_velocity=False)  # leave carried objects at the final hand pose
            mujoco.mj_kinematics(m, d)
        else:
            self.harness_force = float(np.mean(forces)) / (self.mass * 9.81) * 100.0  # % body weight
        self._q_prev = q1.copy()

    def _full(self, q_robot):
        q = self.data.qpos.copy()
        q[:self.nq_r] = q_robot
        return q

    @staticmethod
    def _interp(q0, q1, a):
        q = (1 - a) * q0 + a * q1
        q[3:7] = quat_normalize(slerp(q0[3:7], q1[3:7], a))
        return q

    # -- grasping -------------------------------------------------------------------------
    def update_grasp(self, grips: dict) -> list[tuple[str, str, str]]:
        """Grip values per side (0..1) -> events [(side, "grasp"|"release", object)]."""
        events = []
        for side, value in grips.items():
            held = self.grasped.get(side)
            if held is not None and value < self.grip_off:
                self._set_weld(side, held, False)
                self.grasped[side] = None
                events.append((side, "release", held))
            elif held is None and value > self.grip_on:
                obj = self.nearest_graspable(side)
                if obj is not None:
                    self._set_weld(side, obj, True)
                    self.grasped[side] = obj
                    events.append((side, "grasp", obj))
        return events

    def nearest_graspable(self, side) -> str | None:
        m, d = self.model, self.data
        taken = {o for s, o in self.grasped.items() if o is not None and s != side}
        best, best_d = None, self.grasp_radius
        for name in self.info.graspable:
            if name in taken:
                continue
            for g in self.info.geoms[name]:
                dist = mujoco.mj_geomDistance(m, d, self.palm[side], g, self.grasp_radius, None)
                if dist < best_d:
                    best, best_d = name, dist
        return best

    def _carry(self, h, update_velocity=True):
        """Kinematic mode: put carried objects at their hand-relative pose (velocity from the motion)."""
        if not self._carried:
            return
        d = self.data
        mujoco.mj_kinematics(self.model, d)
        for side, (obj, rp, rq, last) in list(self._carried.items()):
            hb = self.hand_body[side]
            p = d.xpos[hb] + d.xmat[hb].reshape(3, 3) @ rp
            q = quat_normalize(quat_mul(d.xquat[hb], rq))
            a, v = self.info.qposadr[obj], self.dofadr[obj]
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = p, q
            if update_velocity:
                d.qvel[v:v + 3] = (p - last[0]) / h
                w = np.zeros(3)
                mujoco.mju_subQuat(w, q, last[1])  # rotation from last to q, in the object frame
                d.qvel[v + 3:v + 6] = w / h
                self._carried[side] = (obj, rp, rq, (p.copy(), q.copy()))

    def _set_weld(self, side, obj, on: bool):
        m, d = self.model, self.data
        if self.mode == "kinematic":
            if on:
                hb, b = self.hand_body[side], self.info.body[obj]
                rp = d.xmat[hb].reshape(3, 3).T @ (d.xpos[b] - d.xpos[hb])
                rq = quat_normalize(quat_mul(quat_conj(d.xquat[hb]), d.xquat[b]))
                self._carried[side] = (obj, rp, rq, (d.xpos[b].copy(), d.xquat[b].copy()))
            else:
                self._carried.pop(side, None)  # the object keeps its velocity (throwing works)
            return
        eq = self.info.weld[(side, obj)]
        if on:
            b1, b2 = self.hand_body[side], self.info.body[obj]
            p1, q1 = d.xpos[b1], d.xquat[b1]
            p2, q2 = d.xpos[b2], d.xquat[b2]
            # weld relpose = pose of body2 (object) in the frame of body1 (hand); anchor at body2's origin
            m.eq_data[eq, 0:3] = 0.0
            m.eq_data[eq, 3:6] = d.xmat[b1].reshape(3, 3).T @ (p2 - p1)
            m.eq_data[eq, 6:10] = quat_normalize(quat_mul(quat_conj(q1), q2))
            m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = int(on)
