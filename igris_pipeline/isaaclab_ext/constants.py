"""Pure-python constants for the IGRIS-C Isaac Lab tracking task.

This module must stay importable WITHOUT isaaclab / torch / mujoco so that the
MuJoCo sim2sim side (``.venv``) and the Isaac Lab side (conda ``isaaclab``) can
share the exact same numbers.  The *contract* between both simulators is the
exported policy JSON (see ``export.py``); these constants are only the source
that the Isaac Lab config is built from.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
"""C:\\Users\\c\\igris_dataset (resolved from this file)."""

ROBROS_IGRIS_C_DIR = (
    PROJECT_ROOT
    / "third_party"
    / "robros_lab_public"
    / "source"
    / "robros_lab_public"
    / "robros_lab_public"
    / "assets"
    / "robots"
    / "robros"
    / "igris_c"
    / "from_urdf"
)
IGRIS_C_URDF_PATH = Path(os.environ.get("IGRIS_C_URDF", ROBROS_IGRIS_C_DIR / "urdf" / "igris_c_v2.urdf"))
"""URDF with wrists + neck FIXED -> 23 actuated joints."""

IGRIS_C_ROBROS_USD_PATH = ROBROS_IGRIS_C_DIR / "usd" / "extras" / "locomotion" / "igris_c.usd"
"""Robros' pre-converted USD (same URDF, merge_fixed_joints, convex_hull). Used only as fallback."""

IGRIS_C_MJCF_PATH = PROJECT_ROOT / "third_party" / "robros_GMR" / "assets" / "robros_igris_c_v2" / "igris_c_v2.xml"
"""31-DOF MuJoCo model (wrists + neck actuated)."""

USD_CACHE_DIR = Path(os.environ.get("IGRIS_C_USD_CACHE", PROJECT_ROOT / "logs" / "isaaclab" / "usd_cache" / "igris_c_v2"))
LOG_ROOT = PROJECT_ROOT / "logs" / "isaaclab"
SAMPLE_MOTION_DIR = PROJECT_ROOT / "data" / "sample" / "motions"

# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
SIM_DT = 0.005  # 200 Hz physics
DECIMATION = 4  # 50 Hz policy
POLICY_DT = SIM_DT * DECIMATION
MOTION_FPS = 50.0

# --------------------------------------------------------------------------------------
# Joints
# --------------------------------------------------------------------------------------
ACTUATED_JOINT_NAMES = [
    "Joint_Waist_Yaw",
    "Joint_Waist_Roll",
    "Joint_Waist_Pitch",
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Yaw_Left",
    "Joint_Knee_Pitch_Left",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Roll_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Right",
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Elbow_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Shoulder_Roll_Right",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Elbow_Pitch_Right",
]
"""The 23 actuated joints of the URDF (canonical/MuJoCo-like order, NOT the PhysX order).
The policy order is the Isaac articulation order and is written to the export JSON."""

PASSIVE_MJCF_JOINT_NAMES = [
    "Joint_Wrist_Yaw_Left",
    "Joint_Wrist_Roll_Left",
    "Joint_Wrist_Pitch_Left",
    "Joint_Wrist_Yaw_Right",
    "Joint_Wrist_Roll_Right",
    "Joint_Wrist_Pitch_Right",
    "Joint_Neck_Yaw",
    "Joint_Neck_Pitch",
]
"""Joints that exist in the 31-DOF MJCF but are FIXED in the URDF (held at 0 in MuJoCo)."""

DEFAULT_JOINT_POS = {
    "Joint_Hip_Pitch_Left": -0.2,
    "Joint_Hip_Pitch_Right": -0.2,
    "Joint_Knee_Pitch_Left": 0.3,
    "Joint_Knee_Pitch_Right": 0.3,
    "Joint_Ankle_Pitch_Left": -0.1,
    "Joint_Ankle_Pitch_Right": -0.1,
    "Joint_Shoulder_Pitch_Left": 0.13,
    "Joint_Shoulder_Pitch_Right": 0.13,
    "Joint_Shoulder_Roll_Left": 0.13,
    "Joint_Shoulder_Roll_Right": -0.13,
    "Joint_Elbow_Pitch_Left": -0.3,
    "Joint_Elbow_Pitch_Right": -0.3,
}
"""Default standing pose (all other joints 0). Same as robros.py IGRIS_C_LOCOMOTION_CFG.init_state."""

DEFAULT_ROOT_HEIGHT = 0.98  # robros.py init_state.pos z (overwritten by the motion at reset anyway)

# --------------------------------------------------------------------------------------
# Actuators -- copied verbatim from
#   third_party/robros_lab_public/source/robros_lab_public/robros_lab_public/assets/robots/robros.py
#   (IGRISC_LOCOMOTION_ACTUATOR_CFG, DelayedPDActuatorCfg groups X12-150, X8-120, X6-60, PLA)
# Values are per joint-name *pattern* (python ``re.fullmatch`` on the joint name).
# friction        -> Isaac Sim >= 5.0 joint *static* friction EFFORT [Nm]  (PhysX)
# viscous_friction-> joint viscous friction coefficient [Nm s/rad]           (PhysX)
# --------------------------------------------------------------------------------------
ACTUATOR_GROUPS = {
    "X12-150": {
        "joint_names_expr": [".*Hip_Pitch.*", ".*Knee.*"],
        "velocity_limit": 10.4720,
        "effort_limit": 150.0,
        "stiffness": 118.435253,
        "damping": 15.079645,
        "armature": {".*Hip_Pitch.*": 0.3, ".*Knee.*": 0.18},
        "friction": {".*Hip_Pitch.*": 0.5, ".*Knee.*": 0.5},
        "viscous_friction": {".*Hip_Pitch.*": 2.4, ".*Knee.*": 2.2},
    },
    "X8-120": {
        "joint_names_expr": [".*Hip_Roll.*"],
        "velocity_limit": 13.2994,
        "effort_limit": 120.0,
        "stiffness": 122.136354,
        "damping": 10.367256,
        "armature": 0.06,
        "friction": 0.1,
        "viscous_friction": 3.0,
    },
    "X6-60": {
        "joint_names_expr": [".*Hip_Yaw.*", ".*Waist_Yaw.*", ".*_Shoulder_.*", ".*_Elbow_.*"],
        "velocity_limit": 16.0221,
        "effort_limit": 60.0,
        "stiffness": 121.514570,
        "damping": 6.077864,
        "armature": 0.02,
        "friction": 1.01,
        "viscous_friction": 1.8,
    },
    "PLA": {
        "joint_names_expr": [".*Ankle.*", ".*Waist_Roll.*", ".*Waist_Pitch.*"],
        "velocity_limit": {
            ".*Waist_Roll.*": 8.6608,
            ".*Waist_Pitch.*": 15.6655,
            ".*Ankle_Pitch.*": 18.1383,
            ".*Ankle_Roll.*": 19.8934,
        },
        "effort_limit": {
            ".*Waist_Roll.*": 217.0253,
            ".*Waist_Pitch.*": 120.0,
            ".*Ankle_Pitch.*": 163.1163,
            ".*Ankle_Roll.*": 148.7251,
        },
        "stiffness": {
            ".*Waist_Roll.*": 660.634418,
            ".*Waist_Pitch.*": 230.579719,
            ".*Ankle_Pitch.*": 93.373568,
            ".*Ankle_Roll.*": 77.634366,
        },
        "damping": {
            ".*Waist_Roll.*": 34.67,
            ".*Waist_Pitch.*": 12.10,
            ".*Ankle_Pitch.*": 14.863025,
            ".*Ankle_Roll.*": 12.355893,
        },
        "armature": {
            ".*Waist_Roll.*": 0.114,
            ".*Waist_Pitch.*": 0.047,
            ".*Ankle_Pitch.*": 0.14,
            ".*Ankle_Roll.*": 0.12,
        },
        "friction": {
            ".*Waist_Roll.*": 3.6,
            ".*Waist_Pitch.*": 1.2,
            ".*Ankle_Pitch.*": 0.5,
            ".*Ankle_Roll.*": 0.5,
        },
        "viscous_friction": {
            ".*Waist_Roll.*": 7.5,
            ".*Waist_Pitch.*": 2.5,
            ".*Ankle_Pitch.*": 0.1,
            ".*Ankle_Roll.*": 0.1,
        },
    },
}
ACTUATOR_DELAY_STEPS = (0, 2)  # robros.py min_delay/max_delay (physics steps) -- training only

# --------------------------------------------------------------------------------------
# Bodies
# --------------------------------------------------------------------------------------
ROOT_BODY_NAME = "base_link"  # articulation root (= torso, IMU body)
ANCHOR_BODY_NAME = "base_link"
"""Anchor for global tracking. base_link (torso) is chosen over Link_Waist_Pitch (pelvis) because
(1) it is the articulation root -> the anchor pose/velocity come straight from the floating-base state
(PhysX root / MuJoCo freejoint qpos[0:7], qvel[0:6]) without forward kinematics through the waist,
(2) the robot IMU is mounted on base_link, so on hardware anchor orientation/ang-vel come from the IMU,
(3) BeyondMimic also anchors on the torso for G1. The pelvis is still tracked as a body."""

TRACKED_BODY_NAMES = [
    "Link_Waist_Pitch",  # pelvis
    "Link_Hip_Roll_Left",
    "Link_Knee_Pitch_Left",
    "Link_Ankle_Roll_Left",
    "Link_Hip_Roll_Right",
    "Link_Knee_Pitch_Right",
    "Link_Ankle_Roll_Right",
    "base_link",  # torso
    "Link_Shoulder_Roll_Left",
    "Link_Elbow_Pitch_Left",
    "Link_Shoulder_Roll_Right",
    "Link_Elbow_Pitch_Right",
]
"""Bodies used for body pos/ori/vel rewards and MPKPE (mirrors BeyondMimic's G1 key-body list)."""

# After URDF->USD conversion with merge_fixed_joints=True the wrist/hand links are merged into
# Link_Elbow_Pitch_* (frame kept at the elbow joint). The hand frame is therefore a *virtual point*
# rigidly attached to the elbow link. Offset = sum of the fixed URDF joint origins
#   elbow -> wrist_yaw (-0.01, 0, -0.05) -> wrist_roll (0, -/+0.005, -0.2395) -> wrist_pitch (0,0,0)
#   -> hand (0, +/-0.0109, -0.024)   (all with identity rotation except the hand's rpy=(pi,0,0))
EE_POINTS = {
    "Left_Hand": ("Link_Elbow_Pitch_Left", (-0.01, 0.0059, -0.3135)),
    "Right_Hand": ("Link_Elbow_Pitch_Right", (-0.01, -0.0059, -0.3135)),
    "Link_Ankle_Roll_Left": ("Link_Ankle_Roll_Left", (0.0, 0.0, 0.0)),
    "Link_Ankle_Roll_Right": ("Link_Ankle_Roll_Right", (0.0, 0.0, 0.0)),
}
"""End-effector points: name -> (articulation body, offset in that body's frame). If the motion NPZ
contains a body with the same name (e.g. MuJoCo 'Left_Hand') its pose is used as the reference,
otherwise the reference is computed from the reference parent body + offset."""

FOOT_BODY_NAMES = ["Link_Ankle_Roll_Left", "Link_Ankle_Roll_Right"]
HAND_PARENT_BODY_NAMES = ["Link_Elbow_Pitch_Left", "Link_Elbow_Pitch_Right"]

# URDF foot collision box (Link_Ankle_Roll_*): origin (0.048, 0, -0.0685), size (0.2, 0.07, 0.005)
URDF_FOOT_BOX_POS = (0.048, 0.0, -0.0685)
URDF_FOOT_BOX_HALF_SIZE = (0.1, 0.035, 0.0025)
URDF_SOLE_Z = URDF_FOOT_BOX_POS[2] - URDF_FOOT_BOX_HALF_SIZE[2]  # -0.071 in ankle-roll frame


def default_joint_pos(name: str) -> float:
    return float(DEFAULT_JOINT_POS.get(name, 0.0))


def resolve_actuator_params(joint_name: str) -> dict[str, float]:
    """Return {group, kp, kd, effort_limit, velocity_limit, armature, friction, viscous_friction}
    for one joint by regex-matching the ACTUATOR_GROUPS table (same semantics as Isaac Lab)."""
    import re

    for group, cfg in ACTUATOR_GROUPS.items():
        if not any(re.fullmatch(expr, joint_name) for expr in cfg["joint_names_expr"]):
            continue
        out = {"group": group}
        for key, out_key in [
            ("stiffness", "kp"),
            ("damping", "kd"),
            ("effort_limit", "effort_limit"),
            ("velocity_limit", "velocity_limit"),
            ("armature", "armature"),
            ("friction", "friction"),
            ("viscous_friction", "viscous_friction"),
        ]:
            val = cfg[key]
            if isinstance(val, dict):
                matches = [v for k, v in val.items() if re.fullmatch(k, joint_name)]
                if len(matches) != 1:
                    raise ValueError(f"{joint_name}: {len(matches)} matches for {key} in group {group}")
                val = matches[0]
            out[out_key] = float(val)
        return out
    raise KeyError(f"No actuator group matches joint {joint_name}")


def action_scale(joint_name: str) -> float:
    """BeyondMimic convention: scale = 0.25 * effort_limit / stiffness (one action unit ~ 1/4 max torque)."""
    p = resolve_actuator_params(joint_name)
    return 0.25 * p["effort_limit"] / p["kp"]


ACTION_SCALE_PRESETS = {
    # BeyondMimic: 0.25 * effort / kp  -> arms 0.123, waist roll 0.082, ankles 0.44-0.48, hips 0.25-0.32 rad/unit
    "beyondmimic": {name: action_scale(name) for name in ACTUATED_JOINT_NAMES},
    # Robros igris_c_gmt_public deployment convention (config/action.yaml): 0.25 rad/unit for every joint
    "gmt": {name: 0.25 for name in ACTUATED_JOINT_NAMES},
}
DEFAULT_ACTION_SCALE_PRESET = "beyondmimic"
ACTION_SCALE = dict(ACTION_SCALE_PRESETS[DEFAULT_ACTION_SCALE_PRESET])
