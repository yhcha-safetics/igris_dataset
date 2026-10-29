"""Simplified articulated fingers on the IGRIS-C hands, driven by the hand command (igris_pipeline.hands).

The MJCF hands are single rigid meshes with extended fingers, so a closing hand cannot be shown with
them. Each hand gets an overlay instead: a mocap body that is put on the robot hand every step,
carrying a palm and five fingers with the finger joints of Robros' hand layout
(l_0_joint_thumb_proximal ... r_10_joint_little_distal). The original hand mesh is moved to a hidden
geom group; its collision geom stays (it is the palm used for grasping).

Why an overlay and not child bodies of the hand: joints inside the robot tree would get qpos
addresses between the wrist and the next limb and break the robot's contiguous qpos[0:38] block
that the retargeter, the world and the recordings use. The overlay is a separate tree after the
robot (like the scene objects), visual only (no contacts, gravity compensated, negligible mass).

Dimensions follow the hand mesh in the hand body frame (z: wrist -> fingertips, x: little -> index /
thumb side, y: palm normal, palm towards +y on the left hand and -y on the right).
"""
from __future__ import annotations

import mujoco
import numpy as np

from igris_pipeline.hands import HAND_JOINT_NAMES

HIDDEN_GROUP = 4  # not drawn by the viewer by default and not exported to the app (robot_export: group 1)
VISUAL_GROUP = 1
OVERLAY_SUFFIX = "_overlay"
_RADIUS = 0.0095
_KNUCKLE_Z = 0.097
# finger: (x of the finger axis, proximal length, distal length, joint index of the middle joint)
_FINGERS = {"index": (0.029, 0.044, 0.040, 3), "middle": (0.006, 0.046, 0.040, 5),
            "ring": (-0.017, 0.044, 0.038, 7), "little": (-0.040, 0.038, 0.034, 9)}
_THUMB_BASE = np.array([0.050, 0.012, 0.052])
_THUMB_DIR = np.array([0.072, 0.007, 0.020]) / np.linalg.norm([0.072, 0.007, 0.020])
_THUMB_SEGMENTS = (0.030, 0.025, 0.022)  # CMC -> MCP, MCP -> IP, IP -> tip


def _mirror(v, side):
    # rounded like robot_export (6 decimals), so the app's FK reproduces MuJoCo's exactly
    v = np.round(np.array(v, float), 6)
    if side == "right":
        v[1] = -v[1]
    return v.tolist()


def add_hand_overlays(spec: mujoco.MjSpec, hands: dict[str, str]) -> None:
    """Add the finger overlays for {side: hand body name} to the spec and hide the hand meshes."""
    for g in spec.geoms:
        if g.name in {f"{b}_visual" for b in hands.values()}:
            g.group = HIDDEN_GROUP
        elif g.name in {f"{b}_collision" for b in hands.values()}:
            g.group = 3  # the palm collider (grasp distance; contype 0) is drawn in group 2 otherwise
    for side, hand in hands.items():
        p = side[0]  # joint prefix l_ / r_
        flex = [-1.0, 0.0, 0.0] if side == "left" else [1.0, 0.0, 0.0]  # positive angle curls towards the palm
        root = spec.worldbody.add_body(name=hand + OVERLAY_SUFFIX, mocap=True)
        common = {"contype": 0, "conaffinity": 0, "group": VISUAL_GROUP, "density": 50.0,
                  "rgba": [0.2, 0.22, 0.25, 1.0]}
        root.add_geom(name=f"{hand}_palm", type=mujoco.mjtGeom.mjGEOM_BOX, pos=_mirror([0.004, 0.005, 0.047], side),
                      size=[0.051, 0.015, 0.047], **common)
        for name, (x, l1, l2, j) in _FINGERS.items():
            b1 = root.add_body(name=f"{hand}_{name}1", pos=_mirror([x, 0.0015, _KNUCKLE_Z], side), gravcomp=1.0)
            b1.add_joint(name=f"{p}_{j}_joint_{name}_middle", type=mujoco.mjtJoint.mjJNT_HINGE, axis=flex)
            b1.add_geom(name=f"{hand}_{name}1_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[_RADIUS, 0, 0],
                        fromto=[0, 0, 0, 0, 0, l1], **common)
            b2 = b1.add_body(name=f"{hand}_{name}2", pos=[0.0, 0.0, l1], gravcomp=1.0)
            b2.add_joint(name=f"{p}_{j + 1}_joint_{name}_distal", type=mujoco.mjtJoint.mjJNT_HINGE, axis=flex)
            b2.add_geom(name=f"{hand}_{name}2_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[_RADIUS, 0, 0],
                        fromto=[0, 0, 0, 0, 0, l2 - _RADIUS], **common)
        # thumb: CMC swings it in front of the palm (Robros: positive on the left, negative on the right),
        # middle / distal flex positively on both hands
        cmc_axis = [0.0, 0.0, 1.0]
        flex_axis = [0.0, 0.0, 1.0] if side == "left" else [0.0, 0.0, -1.0]
        d = np.array(_mirror(_THUMB_DIR, side))
        parent, pos = root, _mirror(_THUMB_BASE, side)
        for k, (seg, jname, axis) in enumerate(zip(_THUMB_SEGMENTS,
                                                   ("0_joint_thumb_proximal", "1_joint_thumb_middle", "2_joint_thumb_distal"),
                                                   (cmc_axis, flex_axis, flex_axis))):
            b = parent.add_body(name=f"{hand}_thumb{k}", pos=list(pos), gravcomp=1.0)
            b.add_joint(name=f"{p}_{jname}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis)
            end = d * (seg - (_RADIUS if k == 2 else 0.0))
            b.add_geom(name=f"{hand}_thumb{k}_g", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[_RADIUS, 0, 0],
                       fromto=[0, 0, 0, *end.tolist()], **common)
            parent, pos = b, np.round(d * seg, 6).tolist()


class HandOverlay:
    """Indices of the overlays in a compiled model and the per-step update."""

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
