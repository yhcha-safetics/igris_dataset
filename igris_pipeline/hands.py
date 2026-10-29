"""IGRIS-C hand (H5_C) commands from the controller grips, in Robros' normalized hand format.

The layout and the mappings mirror robrosinc/GMR (MIT), general_motion_retargeting/hand_retarget/
igris_c_layout.py and igris_c_hand_command.py, so recorded hand commands can drive Robros' hand
consumers (real hand: one normalized target per Dynamixel motor) without conversion:

    hand_cmd     (..., 12) normalized motor targets in [0, 1] (0 = extended, 1 = flexed) in MOTOR order:
                 right hand motors 11..16, then left 21..26; per hand
                 [thumb_flex, index, middle, ring, little, thumb_cmc]
    grip  -> cmd every motor of a hand = clamp(controller grip, 0, 1) * GRIP_FULL_COMMAND
                 (0.6: the full-grip command Robros' operator chose in their sim)
    cmd -> joint angles  linear between open (0 rad) and closed (fist; thumb CMC at its limit),
                 distal joints follow their middle joint (DISTAL_COUPLING). Robros marks this a
                 placeholder until the hardware bend ratio is measured; it is kept identical here.

Pure numpy: used by the simulation server (igris_sim) and by the offline exports (pipeline.py).
"""
from __future__ import annotations

import math

import numpy as np

GRIP_FULL_COMMAND = 0.6
HAND_COMMAND_DIM = 12
MOTOR_IDS = (11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26)
HAND_SLOT_LABELS = ("thumb_flex", "index", "middle", "ring", "little", "thumb_cmc")
MOTOR_SLOT_LABELS = tuple(f"r_{s}" for s in HAND_SLOT_LABELS) + tuple(f"l_{s}" for s in HAND_SLOT_LABELS)
SIDE_MOTOR_SLOTS = {"right": slice(0, 6), "left": slice(6, 12)}
# motor slot -> actuated finger joint (the thumb CMC is the last slot of a hand, the first joint)
MOTOR_SLOT_TO_JOINT = (
    "r_1_joint_thumb_middle", "r_3_joint_index_middle", "r_5_joint_middle_middle",
    "r_7_joint_ring_middle", "r_9_joint_little_middle", "r_0_joint_thumb_proximal",
    "l_1_joint_thumb_middle", "l_3_joint_index_middle", "l_5_joint_middle_middle",
    "l_7_joint_ring_middle", "l_9_joint_little_middle", "l_0_joint_thumb_proximal",
)
# (actuated middle joint, coupled distal joint, multiplier): distal = middle * multiplier
DISTAL_COUPLING = tuple(
    (f"{s}_{a}", f"{s}_{b}", k)
    for s in ("l", "r")
    for a, b, k in (("1_joint_thumb_middle", "2_joint_thumb_distal", 1.285714286),
                    ("3_joint_index_middle", "4_joint_index_distal", 1.0),
                    ("5_joint_middle_middle", "6_joint_middle_distal", 1.0),
                    ("7_joint_ring_middle", "8_joint_ring_distal", 1.0),
                    ("9_joint_little_middle", "10_joint_little_distal", 1.0)))
# every finger joint of both hands (11 per hand), the order of hand_joint_pos
_FINGERS = ("0_joint_thumb_proximal", "1_joint_thumb_middle", "2_joint_thumb_distal", "3_joint_index_middle",
            "4_joint_index_distal", "5_joint_middle_middle", "6_joint_middle_distal", "7_joint_ring_middle",
            "8_joint_ring_distal", "9_joint_little_middle", "10_joint_little_distal")
HAND_JOINT_NAMES = tuple(f"l_{f}" for f in _FINGERS) + tuple(f"r_{f}" for f in _FINGERS)
# closed (command 1) angle of each actuated joint; open (command 0) is 0 rad for all
JOINT_CLOSED_POS_RAD = {
    **{f"{s}_{f}": math.pi / 2 for s in ("l", "r")
       for f in ("3_joint_index_middle", "5_joint_middle_middle", "7_joint_ring_middle", "9_joint_little_middle")},
    "l_1_joint_thumb_middle": math.radians(70.0), "r_1_joint_thumb_middle": math.radians(70.0),
    "l_0_joint_thumb_proximal": 1.727875959, "r_0_joint_thumb_proximal": -1.727875959,
}


def grip_to_command(left_grip, right_grip, full_command: float = GRIP_FULL_COMMAND) -> np.ndarray:
    """Controller grips (scalars or (T,) arrays) -> hand_cmd (..., 12) in motor order."""
    left = np.clip(np.asarray(left_grip, float), 0.0, 1.0) * float(full_command)
    right = np.clip(np.asarray(right_grip, float), 0.0, 1.0) * float(full_command)
    cmd = np.zeros(np.broadcast(left, right).shape + (HAND_COMMAND_DIM,))
    cmd[..., SIDE_MOTOR_SLOTS["right"]] = right[..., None]
    cmd[..., SIDE_MOTOR_SLOTS["left"]] = left[..., None]
    return cmd


def command_to_joint_pos(cmd) -> np.ndarray:
    """hand_cmd (..., 12) -> finger joint angles (..., 22) in HAND_JOINT_NAMES order (radians)."""
    cmd = np.clip(np.asarray(cmd, float), 0.0, 1.0)
    out = np.zeros(cmd.shape[:-1] + (len(HAND_JOINT_NAMES),))
    idx = {n: k for k, n in enumerate(HAND_JOINT_NAMES)}
    for slot, name in enumerate(MOTOR_SLOT_TO_JOINT):
        out[..., idx[name]] = cmd[..., slot] * JOINT_CLOSED_POS_RAD[name]
    for middle, distal, k in DISTAL_COUPLING:
        out[..., idx[distal]] = out[..., idx[middle]] * k
    return out
