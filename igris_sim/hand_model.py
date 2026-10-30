"""Simplified articulated fingers on the IGRIS-C hands, driven by the hand command (igris_pipeline.hands).

The MJCF hands are single rigid meshes with extended fingers, so a closing hand cannot be shown with
them. Each hand gets an overlay instead: a body carrying a palm and five fingers with the finger
joints of Robros' hand layout (l_0_joint_thumb_proximal ... r_10_joint_little_distal). The original
hand mesh is moved to a hidden geom group.

Two kinds of overlay (configs/sim.yaml grasp.mode):
  visual    a mocap body put on the robot hand every step; no contacts, gravity compensated,
            negligible mass; finger angles set from the command. Holding objects is the weld grasp
            of igris_sim/world.py. The hand mesh's collision geom stays (it is the grasp palm).
  physical  a free rigid hand (palm box + finger capsules that collide with the scene and its
            objects, not with the robot) pulled onto the robot hand by a force-limited 6-DoF spring
            (PhysicalHands.drive). The fingers are position servos on the six Robros motors (thumb
            flex, index, middle, ring, little, thumb CMC; torque limited), the distal joints are
            coupled to their middle joints by joint equalities as in the real linkage. An object is
            held only by contact forces and friction; the rigid hand mesh no longer collides.

Why an overlay and not child bodies of the hand: joints inside the robot tree would get qpos
addresses between the wrist and the next limb and break the robot's contiguous qpos[0:38] block
that the retargeter, the world and the recordings use. The overlay is a separate tree after the
robot (like the scene objects).

Dimensions follow the hand mesh in the hand body frame (z: wrist -> fingertips, x: little -> index /
thumb side, y: palm normal, palm towards +y on the left hand and -y on the right).
"""
from __future__ import annotations

import mujoco
import numpy as np

from igris_pipeline.hands import DISTAL_COUPLING, HAND_JOINT_NAMES, JOINT_CLOSED_POS_RAD, MOTOR_SLOT_TO_JOINT

HIDDEN_GROUP = 4  # not drawn by the viewer by default and not exported to the app (robot_export: group 1)
VISUAL_GROUP = 1
OVERLAY_SUFFIX = "_overlay"
HAND_CONTYPE, HAND_CONAFFINITY = 2, 1  # physical hands touch scene / object geoms (1/1), not the robot (0/1) or each other
_RADIUS = 0.0095
_KNUCKLE_Z = 0.097
# finger: (x of the finger axis, proximal length, distal length, joint index of the middle joint)
_FINGERS = {"index": (0.029, 0.044, 0.040, 3), "middle": (0.006, 0.046, 0.040, 5),
            "ring": (-0.017, 0.044, 0.038, 7), "little": (-0.040, 0.038, 0.034, 9)}
# thumb: at rest it points sideways / forward, 24 deg out of the palm plane; CMC (about z) turns it in
# front of the palm, the flex joints (about -y of the turned thumb) curl it towards the fingertips.
# Base and direction were fitted so the tip meets the index / middle fingertips at the full grip
# command (0.6) and is ~5 cm from them at 0.3: a pinch, as with the Robros hand.
_THUMB_BASE = np.array([0.0404, 0.0174, 0.060])
_THUMB_DIR = np.array([0.788, 0.408, 0.461]) / np.linalg.norm([0.788, 0.408, 0.461])
_THUMB_SEGMENTS = (0.030, 0.025, 0.022)  # CMC -> MCP, MCP -> IP, IP -> tip
_RANGE_MARGIN = 0.05  # rad beyond open / fully closed

DEFAULT_PHYSICS = {  # configs/sim.yaml hands.physics overrides these
    "density": 1000.0,                     # kg/m^3 of palm and fingers
    "friction": [1.2, 0.01, 0.0002],       # slide, torsion, rolling (MuJoCo takes the max of the pair)
    "condim": 4,                           # with torsional friction: pinched objects do not spin
    "solref": [0.01, 1.0],
    "kp": 3000.0, "kd": 70.0,              # N/m, N s/m of the spring onto the robot hand
    "kp_rot": 30.0, "kd_rot": 0.5,         # N m/rad, N m s/rad
    "max_force": 120.0, "max_torque": 12.0,
    "snap_distance": 0.25,                 # m: further from the robot hand (stuck, teleport) -> put back on it
    "finger_kp": 2.0,                      # N m/rad of the finger motors
    "finger_max_torque": 0.6,              # N m
    "finger_damping": 0.1, "finger_armature": 0.001,
}


def _mirror(v, side):
    # rounded like robot_export (6 decimals), so the app's FK reproduces MuJoCo's exactly
    v = np.round(np.array(v, float), 6)
    if side == "right":
        v[1] = -v[1]
    return v.tolist()


def physics_params(cfg: dict | None) -> dict:
    return {**DEFAULT_PHYSICS, **(cfg or {})}


def _joint_range(name: str) -> list[float]:
    closed = JOINT_CLOSED_POS_RAD.get(name)
    if closed is None:  # coupled distal joint
        middle, k = next((m, k) for m, d, k in DISTAL_COUPLING if d == name)
        closed = JOINT_CLOSED_POS_RAD[middle] * k
    lo, hi = sorted((0.0, closed))
    return [lo - _RANGE_MARGIN, hi + _RANGE_MARGIN]


def add_hand_overlays(spec: mujoco.MjSpec, hands: dict[str, str], physics: dict | None = None) -> None:
    """Add the finger overlays for {side: hand body name} to the spec and hide the hand meshes.

    physics: None for visual overlays, else the physical hand parameters (physics_params())."""
    for g in spec.geoms:
        if g.name in {f"{b}_visual" for b in hands.values()}:
            g.group = HIDDEN_GROUP
        elif g.name in {f"{b}_collision" for b in hands.values()}:
            g.group = 3  # the palm collider (grasp distance) is drawn in group 2 otherwise
            if physics is not None:  # the physical hand touches objects instead of the rigid mesh
                g.contype, g.conaffinity = 0, 0
    if physics is None:
        common = {"contype": 0, "conaffinity": 0, "group": VISUAL_GROUP, "density": 50.0,
                  "rgba": [0.2, 0.22, 0.25, 1.0]}
    else:
        common = {"contype": HAND_CONTYPE, "conaffinity": HAND_CONAFFINITY, "group": VISUAL_GROUP,
                  "density": float(physics["density"]), "friction": list(map(float, physics["friction"])),
                  "condim": int(physics["condim"]), "solref": list(map(float, physics["solref"])),
                  "rgba": [0.2, 0.22, 0.25, 1.0]}

    def joint(body, name, axis):
        kw = {}
        if physics is not None:
            kw = {"range": _joint_range(name), "limited": mujoco.mjtLimited.mjLIMITED_TRUE,
                  "damping": float(physics["finger_damping"]), "armature": float(physics["finger_armature"])}
        body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis, **kw)

    for side, hand in hands.items():
        p = side[0]  # joint prefix l_ / r_
        flex = [-1.0, 0.0, 0.0] if side == "left" else [1.0, 0.0, 0.0]  # positive angle curls towards the palm
        if physics is None:
            root = spec.worldbody.add_body(name=hand + OVERLAY_SUFFIX, mocap=True)
        else:
            root = spec.worldbody.add_body(name=hand + OVERLAY_SUFFIX, gravcomp=1.0)
            root.add_freejoint(name=hand + OVERLAY_SUFFIX + "_free")
        root.add_geom(name=f"{hand}_palm", type=mujoco.mjtGeom.mjGEOM_BOX, pos=_mirror([0.004, 0.005, 0.047], side),
                      size=[0.051, 0.015, 0.047], **common)
        for name, (x, l1, l2, j) in _FINGERS.items():
            b1 = root.add_body(name=f"{hand}_{name}1", pos=_mirror([x, 0.0015, _KNUCKLE_Z], side), gravcomp=1.0)
            joint(b1, f"{p}_{j}_joint_{name}_middle", flex)
            b1.add_geom(name=f"{hand}_{name}1_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[_RADIUS, 0, 0],
                        fromto=[0, 0, 0, 0, 0, l1], **common)
            b2 = b1.add_body(name=f"{hand}_{name}2", pos=[0.0, 0.0, l1], gravcomp=1.0)
            joint(b2, f"{p}_{j + 1}_joint_{name}_distal", flex)
            b2.add_geom(name=f"{hand}_{name}2_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[_RADIUS, 0, 0],
                        fromto=[0, 0, 0, 0, 0, l2 - _RADIUS], **common)
        # thumb: CMC swings it in front of the palm (Robros: positive on the left, negative on the right),
        # middle / distal flex positively on both hands (the mirror image of -y is -y)
        cmc_axis = [0.0, 0.0, 1.0]
        flex_axis = [0.0, -1.0, 0.0]
        d = np.array(_mirror(_THUMB_DIR, side))
        parent, pos = root, _mirror(_THUMB_BASE, side)
        for k, (seg, jname, axis) in enumerate(zip(_THUMB_SEGMENTS,
                                                   ("0_joint_thumb_proximal", "1_joint_thumb_middle", "2_joint_thumb_distal"),
                                                   (cmc_axis, flex_axis, flex_axis))):
            b = parent.add_body(name=f"{hand}_thumb{k}", pos=list(pos), gravcomp=1.0)
            joint(b, f"{p}_{jname}", axis)
            end = d * (seg - (_RADIUS if k == 2 else 0.0))
            b.add_geom(name=f"{hand}_thumb{k}_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[_RADIUS, 0, 0],
                       fromto=[0, 0, 0, *end.tolist()], **common)
            parent, pos = b, np.round(d * seg, 6).tolist()
        if physics is not None:
            for jname in MOTOR_SLOT_TO_JOINT:
                if jname[0] != p:
                    continue
                act = spec.add_actuator(name=f"{jname}_motor", target=jname, trntype=mujoco.mjtTrn.mjTRN_JOINT)
                act.set_to_position(kp=float(physics["finger_kp"]))
                act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE  # setpoints come from the command map
                act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
                act.forcerange = [-float(physics["finger_max_torque"]), float(physics["finger_max_torque"])]
            for middle, distal, k in DISTAL_COUPLING:
                if middle[0] == p:  # distal = k * middle (the finger linkage)
                    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=distal, name2=middle,
                                           objtype=mujoco.mjtObj.mjOBJ_JOINT)
                    eq.data = [0.0, float(k)] + [0.0] * 9
                    eq.name = f"{distal}_coupling"


class HandOverlay:
    """Indices of the visual overlays in a compiled model and the per-step update."""
    physical = False

    def __init__(self, model: mujoco.MjModel, hand_bodies: dict[str, int], hands: dict[str, str]):
        self.hand_body = hand_bodies
        self.root = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, h + OVERLAY_SUFFIX) for s, h in hands.items()}
        self.mocap = {s: int(model.body_mocapid[b]) for s, b in self.root.items()}
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in HAND_JOINT_NAMES]
        if min(jids) < 0:
            raise RuntimeError("hand overlay joints missing from the model")
        self.qposadr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dofadr = np.array([model.jnt_dofadr[j] for j in jids])

    def apply(self, data: mujoco.MjData, joint_pos: np.ndarray) -> None:
        """Put each overlay on its hand (current kinematics) and set the finger angles."""
        for s, b in self.hand_body.items():
            data.mocap_pos[self.mocap[s]] = data.xpos[b]
            data.mocap_quat[self.mocap[s]] = data.xquat[b]
        data.qpos[self.qposadr] = joint_pos
        data.qvel[self.dofadr] = 0.0

    def is_overlay_body(self, model: mujoco.MjModel, b: int) -> bool:
        roots = set(self.root.values())
        while b > 0:
            if b in roots:
                return True
            b = int(model.body_parentid[b])
        return False


def _rotvec(q: np.ndarray) -> np.ndarray:
    """Unit quaternion wxyz -> rotation vector (shortest way)."""
    if q[0] < 0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    return 2.0 * q[1:] if s < 1e-12 else (2.0 * np.arctan2(s, q[0]) / s) * q[1:]


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def _conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


class PhysicalHands(HandOverlay):
    """Physical hands (add_hand_overlays with physics): tracking spring, finger servos, contact readout."""
    physical = True

    def __init__(self, model: mujoco.MjModel, hand_bodies: dict[str, int], hands: dict[str, str], params: dict):
        super().__init__(model, hand_bodies, hands)
        self.p = params
        self.free_q, self.free_v = {}, {}
        for s, h in hands.items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, h + OVERLAY_SUFFIX + "_free")
            self.free_q[s], self.free_v[s] = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
            # the damping of the spring acts on the absolute velocity and is integrated implicitly
            # (stable for any gain); drive() adds the target velocity back so it damps the relative one
            model.dof_damping[self.free_v[s]:self.free_v[s] + 3] = float(params["kd"])
            model.dof_damping[self.free_v[s] + 3:self.free_v[s] + 6] = float(params["kd_rot"])
        idx = {n: k for k, n in enumerate(HAND_JOINT_NAMES)}
        self.act = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
                             for n in MOTOR_SLOT_TO_JOINT])
        self.act_joint = np.array([idx[n] for n in MOTOR_SLOT_TO_JOINT])  # actuator -> index in HAND_JOINT_NAMES
        if min(self.act) < 0:
            raise RuntimeError("finger motors missing from the model")
        self.side_of_body = {}
        for b in range(model.nbody):
            for s, r in self.root.items():
                if self.is_overlay_body(model, b) and self._root_of(model, b) == r:
                    self.side_of_body[b] = s
        self.reaction = {s: np.zeros(6) for s in hands}  # last spring wrench (world force, torque about the hand origin)

    @staticmethod
    def _root_of(model, b):
        while int(model.body_parentid[b]) > 0:
            b = int(model.body_parentid[b])
        return b

    def apply(self, data, joint_pos):  # the fingers are servoed, not set (see set_targets)
        raise TypeError("physical hands are driven by set_targets() / drive()")

    def place(self, data: mujoco.MjData, poses: dict) -> None:
        """Put the hands on {side: (pos, quat)} at rest with open fingers (reset)."""
        for s, (p, q) in poses.items():
            a, v = self.free_q[s], self.free_v[s]
            data.qpos[a:a + 3], data.qpos[a + 3:a + 7] = p, q
            data.qvel[v:v + 6] = 0.0
        data.qpos[self.qposadr] = 0.0
        data.qvel[self.dofadr] = 0.0

    def set_targets(self, data: mujoco.MjData, joint_target: np.ndarray) -> None:
        """Finger servo setpoints from the target angles (22, HAND_JOINT_NAMES order)."""
        data.ctrl[self.act] = np.asarray(joint_target)[self.act_joint]

    def joint_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qposadr].copy()

    def poses(self, data: mujoco.MjData) -> np.ndarray:
        """(2, 7) root pose [pos, quat wxyz] of the left and right hand."""
        return np.stack([data.qpos[self.free_q[s]:self.free_q[s] + 7] for s in ("left", "right")])

    def drive(self, model: mujoco.MjModel, data: mujoco.MjData, targets: dict) -> None:
        """Set the spring wrench on each hand for the next step.

        targets: {side: (pos, quat, lin_vel, ang_vel)} pose of the robot hand body at the end of
        the step and its velocity (world frame)."""
        P = self.p
        kp, kd, kr, kdr = float(P["kp"]), float(P["kd"]), float(P["kp_rot"]), float(P["kd_rot"])
        for s, (pt, qt, vt, wt) in targets.items():
            a, v, b = self.free_q[s], self.free_v[s], self.root[s]
            p, q = data.qpos[a:a + 3], data.qpos[a + 3:a + 7]
            ep = pt - p
            if np.linalg.norm(ep) > float(P["snap_distance"]):  # lost the hand (stuck, teleported): put back
                data.qpos[a:a + 3], data.qpos[a + 3:a + 7] = pt, qt
                data.qvel[v:v + 3] = vt
                data.qvel[v + 3:v + 6] = _rotate_inv(qt, wt)
                data.xfrc_applied[b] = 0.0
                self.reaction[s] = np.zeros(6)
                continue
            vel = data.qvel[v:v + 3]
            w = _rotate(q, data.qvel[v + 3:v + 6])
            f = kp * ep + kd * (vt - vel)
            fn = np.linalg.norm(f)
            if fn > P["max_force"]:
                f *= P["max_force"] / fn
            er = _rotvec(_quat_mul(qt, _conj(q)))
            tq = kr * er + kdr * (wt - w)
            tn = np.linalg.norm(tq)
            if tn > P["max_torque"]:
                tq *= P["max_torque"] / tn
            self.reaction[s] = np.concatenate([f, tq])
            f_app, t_app = f + kd * vel, tq + kdr * w  # the dof damping (-kd v) is applied by MuJoCo
            # xfrc_applied acts at the centre of mass; the spring acts at the hand origin
            arm = data.xipos[b] - p
            data.xfrc_applied[b, :3] = f_app
            data.xfrc_applied[b, 3:] = t_app - np.cross(arm, f_app)

    def contacts(self, model: mujoco.MjModel, data: mujoco.MjData, body_object: dict) -> dict:
        """Contacts of the hands with objects: {object: {"hand": [(side, force N, direction on the
        object)], "other": total normal force N from everything else}} for {body id: object}."""
        out = {}
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.efc_address < 0:
                continue
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            s1, s2 = self.side_of_body.get(b1), self.side_of_body.get(b2)
            o1, o2 = body_object.get(b1), body_object.get(b2)
            if o1 is None and o2 is None:
                continue
            mujoco.mj_contactForce(model, data, i, f6)
            n = c.frame[:3]  # from geom1 to geom2: the force on geom2 is along +n
            for obj, side, sign in ((o2, s1, 1.0), (o1, s2, -1.0)):
                if obj is None:
                    continue
                e = out.setdefault(obj, {"hand": [], "other": 0.0})
                if side is not None:
                    e["hand"].append((side, float(f6[0]), sign * n.copy()))
                else:
                    e["other"] += float(f6[0])
        return out


def pose_rate(p0, q0, p1, q1, h):
    """World linear and angular velocity that moves pose (p0, q0) to (p1, q1) in h seconds."""
    return (p1 - p0) / h, _rotvec(_quat_mul(q1, _conj(q0))) / h


def _rotate(q, v):
    """Rotate v by unit quaternion q (wxyz)."""
    u = q[1:]
    t = 2.0 * np.cross(u, v)
    return v + q[0] * t + np.cross(u, t)


def _rotate_inv(q, v):
    return _rotate(_conj(q), v)
