"""MuJoCo world for teleoperation: IGRIS-C in a scene, following the streamed retarget targets.

Robot modes (configs/sim.yaml physics.mode):
  kinematic  the robot's qpos / qvel are set to the (interpolated) targets before every physics
             step, so the robot follows the operator exactly and pushes objects with the matching
             velocity; only the objects are simulated.
  harness    the robot is simulated: joint PD servos track the targets (igris_c_actuators.yaml
             gains x gain_scale) and a virtual 6-DoF spring holds the base on the target, as in
             igris_pipeline.sim.mujoco_eval.pd_replay. Contacts, joint limits and torque limits are
             physical; the harness force says how much external help the motion needs.

Hands (configs/sim.yaml hands.enabled): the controller grips also close articulated fingers
(igris_sim/hand_model.py) through the Robros hand command (igris_pipeline/hands.py).

Grasping (configs/sim.yaml grasp.mode):
  physics  the hands are physical (hand_model.PhysicalHands): a free rigid palm + fingers pulled onto
           the robot hand by a force-limited spring (it stops at surfaces instead of entering them),
           finger motors servo to the command and stop on contact. Objects are held only by contact
           forces and friction: they must be squeezed (between fingers and thumb or palm, or between
           both hands) or carried on the hand. grasped[side] / the grasp events report an object
           this hand presses with >= grasp.hold_force while it is held (opposing hand contacts, or
           no other support); it is released after the hand force stays below grasp.release_force
           for grasp.release_time_s. In harness mode the spring's reaction loads the robot's arm.
  weld     pressing a controller grip attaches the nearest graspable object whose surface is within
           grasp.radius of that palm, touching or not; releasing the grip lets it go (with its
           current velocity, so it can be thrown). In harness mode the attachment is a MuJoCo weld
           constraint between hand and object; in kinematic mode the robot does not respond to
           forces, so a weld would be solved mostly on the (overwritten) hand side and the object
           would lag: the object is carried kinematically instead, its pose set from the hand pose
           before every physics step. The fingers are visual overlays. Used when the hands are
           disabled.
"""
from __future__ import annotations

import mujoco
import numpy as np

from igris_pipeline.retarget.robot import IgrisModel
from igris_pipeline.sim.mujoco_eval import (_quat_err, _rotate, apply_joint_params, configure_position_servos,
                                            load_joint_params)
from igris_pipeline.utils.rotations import quat_conj, quat_mul, quat_normalize, slerp, yaw_quat

from igris_pipeline.hands import HAND_COMMAND_DIM, HAND_JOINT_NAMES, command_to_joint_pos

from .hand_model import HandOverlay, PhysicalHands, physics_params, pose_rate
from .scene import HANDS, Scene, build_model

SIDES = ("left", "right")


class SimWorld:
    def __init__(self, scene: Scene, sim_cfg: dict, mode: str | None = None):
        self.scene = scene
        self.cfg = sim_cfg
        hc = sim_cfg.get("hands", {})
        g = sim_cfg.get("grasp", {})
        self.hands_enabled = bool(hc.get("enabled", False))
        self.grasp_mode = str(g.get("mode", "physics"))
        if self.grasp_mode not in ("physics", "weld"):
            raise ValueError(f"unknown grasp mode {self.grasp_mode!r}")
        if not self.hands_enabled:
            self.grasp_mode = "weld"  # holding by contact needs the physical hands
        self.hand_params = physics_params(hc.get("physics")) if self.grasp_mode == "physics" else None
        self.model, self.info = build_model(scene, hands=self.hands_enabled, hand_physics=self.hand_params)
        m = self.model
        ph = sim_cfg.get("physics", {})
        m.opt.timestep = float(ph.get("timestep", m.opt.timestep))
        m.opt.impratio = float(ph.get("impratio", m.opt.impratio))
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
        self.overlay = None
        if self.hand_params is not None:
            self.overlay = PhysicalHands(m, self.hand_body, HANDS, self.hand_params)
        elif self.hands_enabled:
            self.overlay = HandOverlay(m, self.hand_body, HANDS)
        self.physical = self.hand_params is not None
        self._kin = mujoco.MjData(m)  # scratch: robot hand poses of the interpolated targets
        self.grasp_radius = float(g.get("radius", 0.10))
        self.grip_on, self.grip_off = float(g.get("grip_on", 0.6)), float(g.get("grip_off", 0.4))
        self.hold_force, self.release_force = float(g.get("hold_force", 1.0)), float(g.get("release_force", 0.3))
        self.release_time = float(g.get("release_time_s", 0.1))
        self.dynamic = [n for n in self.info.objects if self.info.dynamic[n]]
        self.body_object = {self.info.body[n]: n for n in self.info.graspable}
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
        d.eq_active[:] = m.eq_active0  # grasp welds off, finger couplings on
        self.hand_cmd = np.zeros(HAND_COMMAND_DIM)
        self.hand_target = np.zeros(len(HAND_JOINT_NAMES))
        self.hand_q = np.zeros(len(HAND_JOINT_NAMES))
        self.hand_force = np.zeros(2)  # physical hands: contact normal force on objects (N), left / right
        self.contacts = {}             # physical hands: PhysicalHands.contacts() of the last step
        self.grasped = {s: None for s in HANDS}
        self._carried = {}  # kinematic mode: side -> (object, pos in hand frame, quat in hand frame, last pose)
        self._weak = {s: 0.0 for s in HANDS}  # physics: time the held object has been pressed too weakly
        self._events = []
        q = self.spawn_qpos() if q_robot is None else np.asarray(q_robot, float)
        d.qpos[:self.nq_r] = q
        mujoco.mj_forward(m, d)
        if self.physical:
            self.overlay.place(d, {s: (d.xpos[b].copy(), d.xquat[b].copy()) for s, b in self.hand_body.items()})
            self.overlay.set_targets(d, self.hand_target)
            mujoco.mj_forward(m, d)
        self._update_hands()
        self._q_prev = q.copy()

    def robot_qpos(self) -> np.ndarray:
        return self.data.qpos[:self.nq_r].copy()

    # -- hands ----------------------------------------------------------------------------
    def set_hand_command(self, cmd) -> None:
        """Normalized Robros hand command (12, motor order); drives the fingers."""
        self.hand_cmd = np.clip(np.asarray(cmd, float), 0.0, 1.0)
        self.hand_target = command_to_joint_pos(self.hand_cmd)
        if self.physical:
            self.overlay.set_targets(self.data, self.hand_target)  # finger angles follow from the physics
        else:
            self.hand_q = self.hand_target

    def _update_hands(self):
        if self.physical:
            self.hand_q = self.overlay.joint_pos(self.data)
        elif self.overlay is not None:
            self.overlay.apply(self.data, self.hand_q)
            mujoco.mj_kinematics(self.model, self.data)

    def joint_names_ext(self) -> list[str]:
        """Names of qpos_ext()[7:7 + n]: the robot joints, then the finger joints when hands are enabled."""
        return list(self.robot.joint_names) + (list(HAND_JOINT_NAMES) if self.overlay is not None else [])

    def hand_poses(self) -> np.ndarray:
        """Physical hands: (2, 7) world pose [pos, quat wxyz] of the left / right hand root."""
        return self.overlay.poses(self.data)

    def qpos_ext(self) -> np.ndarray:
        """What the app draws (robot_export layout): robot qpos, the finger angles when hands are
        enabled, and with physical hands their root poses (left, right; 7 each)."""
        q = self.robot_qpos()
        if self.overlay is None:
            return q
        parts = [q, self.hand_q] + ([self.hand_poses().ravel()] if self.physical else [])
        return np.concatenate(parts)

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
                if self.physical:
                    self.overlay.drive(m, d, self._hand_targets(q_prev_sub, q_sub, h))
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
                if self.physical:  # the hands follow the simulated robot hands, which feel the reaction
                    q_next = d.qpos.copy()
                    mujoco.mj_integratePos(m, q_next, d.qvel, h)
                    self.overlay.drive(m, d, self._hand_targets(d.qpos[:self.nq_r], q_next[:self.nq_r], h))
                    self._hand_reactions()
                mujoco.mj_step(m, d)
                forces.append(float(np.linalg.norm(f)))
            q_prev_sub = q_sub
        if self.mode == "kinematic":
            self._carry(h, update_velocity=False)  # leave carried objects at the final hand pose
            mujoco.mj_kinematics(m, d)
        else:
            self.harness_force = float(np.mean(forces)) / (self.mass * 9.81) * 100.0  # % body weight
        self._update_hands()  # visual overlays on the final hand pose (no contacts: once a tick is enough)
        if self.physical:
            self._contact_grasp(n * h)
        self._q_prev = q1.copy()

    def _hand_poses_at(self, q_robot):
        kd = self._kin
        kd.qpos[:self.nq_r] = q_robot
        mujoco.mj_kinematics(self.model, kd)
        return {s: (kd.xpos[b].copy(), kd.xquat[b].copy()) for s, b in self.hand_body.items()}

    def _hand_targets(self, q0, q1, h):
        """Physical hands' targets for a step from robot pose q0 to q1: the robot hand at q1 and its velocity."""
        a, b = self._hand_poses_at(q0), self._hand_poses_at(q1)
        return {s: (*b[s], *pose_rate(*a[s], *b[s], h)) for s in a}

    def _hand_reactions(self):
        """Harness mode: the spring that pulls a physical hand pulls its robot hand the other way."""
        d = self.data
        for s, b in self.hand_body.items():
            w = self.overlay.reaction[s]
            f, tq = -w[:3], -w[3:]  # at the robot hand origin; xfrc_applied acts at its centre of mass
            d.xfrc_applied[b, :3] = f
            d.xfrc_applied[b, 3:] = tq - np.cross(d.xipos[b] - d.xpos[b], f)

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
        """Grip values per side (0..1) -> events [(side, "grasp"|"release", object)].

        Physics mode: the grips only close the fingers (set_hand_command); the events are the
        contact grasps detected by step() since the last call."""
        if self.physical:
            events, self._events = self._events, []
            return events
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

    def _contact_grasp(self, dt):
        """Physics mode: grasped / released objects from the hand contacts of the last step."""
        self.contacts = cs = self.overlay.contacts(self.model, self.data, self.body_object)
        force = {s: {} for s in SIDES}
        for obj, e in cs.items():
            for s, f, _ in e["hand"]:
                force[s][obj] = force[s].get(obj, 0.0) + f
        self.hand_force = np.array([sum(force[s].values()) for s in SIDES])
        for side in SIDES:
            held = self.grasped[side]
            if held is not None:
                if force[side].get(held, 0.0) >= self.release_force:
                    self._weak[side] = 0.0
                    continue
                self._weak[side] += dt
                if self._weak[side] >= self.release_time:
                    self.grasped[side], self._weak[side] = None, 0.0
                    self._events.append((side, "release", held))
                continue
            if force[side]:
                obj = max(force[side], key=force[side].get)
                if force[side][obj] >= self.hold_force and self.is_held(cs[obj]):
                    self.grasped[side] = obj
                    self._events.append((side, "grasp", obj))
        del self._events[:-20]  # not collected while calibrating

    def is_held(self, contact: dict) -> bool:
        """An object is held when the hands squeeze it (opposing contact normals, from one or both
        hands) or carry it (nothing else supports it)."""
        hand = [(f, n) for _, f, n in contact["hand"] if f > 0.0]
        total = sum(f for f, _ in hand)
        if total < self.hold_force:
            return False
        if contact["other"] < 1e-3:
            return True
        resultant = np.linalg.norm(sum(f * n for f, n in hand))
        return resultant < 0.8 * total

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
