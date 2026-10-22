"""PICO body-tracking skeleton definition (24 joints, SMPL ordering).

PICO's BodyTrackerRole enum (PXR_Input / XRoboToolkit `get_body_joints_pose`) uses the
same 24-joint layout and ordering as SMPL, which also lets the data be consumed by
SMPL-based tooling (GMR, PHC, ProtoMotions).
"""
from __future__ import annotations

import numpy as np

PICO_JOINTS = [
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
    "left_hand",       # 22
    "right_hand",      # 23
]
NUM_JOINTS = len(PICO_JOINTS)
J = {name: i for i, name in enumerate(PICO_JOINTS)}

PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)

LEFT_RIGHT_PAIRS = [
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_foot", "right_foot"),
    ("left_collar", "right_collar"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hand", "right_hand"),
]
MIRROR_INDEX = np.arange(NUM_JOINTS)
for _l, _r in LEFT_RIGHT_PAIRS:
    MIRROR_INDEX[J[_l]], MIRROR_INDEX[J[_r]] = J[_r], J[_l]

# Bones whose length should be constant for a given subject (used by calibration).
BONES = [(int(PARENTS[i]), i) for i in range(1, NUM_JOINTS)]

LOWER_BODY = [J[n] for n in ("pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
                             "left_ankle", "right_ankle", "left_foot", "right_foot")]
UPPER_BODY = [i for i in range(NUM_JOINTS) if i not in LOWER_BODY]

# Joints that are directly observed by a physical sensor, depending on the device setup.
# Everything else is estimated by PICO's body-tracking solver (or by our own IK fallback).
SENSOR_ANCHORS = {
    "headset": [J["head"]],
    "controllers": [J["left_hand"], J["right_hand"]],
    "hand_tracking": [J["left_wrist"], J["right_wrist"]],
    "ankle_trackers": [J["left_ankle"], J["right_ankle"]],
    "waist_tracker": [J["pelvis"]],
}

# Joint groups used for confidence / filtering parameters (configs/tracker_profiles.yaml).
_GROUP_MEMBERS = {
    "head": ["head"],
    "hands": ["left_wrist", "right_wrist", "left_hand", "right_hand"],
    "arms": ["left_collar", "right_collar", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
    "spine": ["spine1", "spine2", "spine3", "neck"],
    "pelvis": ["pelvis"],
    "legs": ["left_hip", "right_hip", "left_knee", "right_knee"],
    "feet": ["left_ankle", "right_ankle", "left_foot", "right_foot"],
}
JOINT_GROUP = [""] * NUM_JOINTS
for _g, _members in _GROUP_MEMBERS.items():
    for _n in _members:
        JOINT_GROUP[J[_n]] = _g
GROUPS = list(_GROUP_MEMBERS)

# Typical adult segment lengths (m) used when no calibration is available.
DEFAULT_BONE_LENGTHS = {
    ("pelvis", "left_hip"): 0.10, ("pelvis", "right_hip"): 0.10,
    ("pelvis", "spine1"): 0.11, ("left_hip", "left_knee"): 0.40,
    ("right_hip", "right_knee"): 0.40, ("spine1", "spine2"): 0.13,
    ("left_knee", "left_ankle"): 0.41, ("right_knee", "right_ankle"): 0.41,
    ("spine2", "spine3"): 0.06, ("left_ankle", "left_foot"): 0.13,
    ("right_ankle", "right_foot"): 0.13, ("spine3", "neck"): 0.21,
    ("spine3", "left_collar"): 0.15, ("spine3", "right_collar"): 0.15,
    ("neck", "head"): 0.09, ("left_collar", "left_shoulder"): 0.11,
    ("right_collar", "right_shoulder"): 0.11, ("left_shoulder", "left_elbow"): 0.27,
    ("right_shoulder", "right_elbow"): 0.27, ("left_elbow", "left_wrist"): 0.25,
    ("right_elbow", "right_wrist"): 0.25, ("left_wrist", "left_hand"): 0.09,
    ("right_wrist", "right_hand"): 0.09,
}


def bone_lengths(pos: np.ndarray) -> np.ndarray:
    """(T,24,3) -> (T,23) bone lengths in BONES order."""
    p = np.asarray(pos)
    return np.stack([np.linalg.norm(p[..., c, :] - p[..., pa, :], axis=-1) for pa, c in BONES], axis=-1)


def default_bone_length_array() -> np.ndarray:
    out = np.zeros(len(BONES))
    for k, (pa, c) in enumerate(BONES):
        out[k] = DEFAULT_BONE_LENGTHS[(PICO_JOINTS[pa], PICO_JOINTS[c])]
    return out
