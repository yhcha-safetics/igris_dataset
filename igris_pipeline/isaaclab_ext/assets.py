"""IGRIS-C articulation config for Isaac Lab 2.3 / Isaac Sim 5.1.

Spawn: ``UrdfFileCfg`` (URDF -> USD conversion at spawn, cached under logs/isaaclab/usd_cache and only
re-run when the URDF / converter settings change). Converter settings mirror the config.yaml that
Robros used for their shipped USD: merge_fixed_joints=True, convex_hull colliders, joint_drive=force with
zero gains, fix_base=False. Set the environment variable ``IGRIS_C_SPAWN=usd`` to spawn Robros' pre-
converted USD (third_party/.../usd/extras/locomotion/igris_c.usd, generated from the same URDF) instead,
e.g. if the URDF importer misbehaves.

Actuators: the numbers are copied from
``third_party/robros_lab_public/.../assets/robots/robros.py`` (IGRISC_LOCOMOTION_ACTUATOR_CFG) via
``constants.ACTUATOR_GROUPS``. We keep Robros' delayed explicit PD model (``DelayedPDActuatorCfg`` semantics,
implemented sync-free in ``actuators.FastDelayedPDActuatorCfg`` -- 2x faster env.step on this machine)
rather than ``ImplicitActuatorCfg`` because
  * the torque law tau = clip(kp (q* - q) - kd qd, +-effort) is evaluated by Isaac Lab at every 200 Hz
    physics step -- exactly what the MuJoCo sim2sim runner (and the robot's joint controller) does, so the
    two simulators see the same actuator model (an implicit PhysX drive integrates the PD term implicitly
    and clips differently, which is a classic sim2sim gap);
  * these gains/armatures/frictions were sim-to-real validated by Robros with this actuator type;
  * the 0-2 physics-step random command delay is useful robustness randomisation (disabled for PLAY).
At these gains (kp/armature -> omega*dt <= 0.4 at dt=5 ms) the explicit PD is stable.

Self-collisions are disabled (Robros enables them): MuJoCo's MJCF has no self-collision either, and the
URDF's coarse boxes (torso box vs. arm cylinders) can produce spurious contacts on arm motions.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from . import constants as C
from .actuators import FastDelayedPDActuatorCfg


def _actuators(min_delay: int, max_delay: int) -> dict[str, FastDelayedPDActuatorCfg]:
    out = {}
    for name, g in C.ACTUATOR_GROUPS.items():
        out[name] = FastDelayedPDActuatorCfg(
            joint_names_expr=list(g["joint_names_expr"]),
            velocity_limit=g["velocity_limit"],
            velocity_limit_sim=g["velocity_limit"],
            effort_limit=g["effort_limit"],
            effort_limit_sim=g["effort_limit"],
            stiffness=g["stiffness"],
            damping=g["damping"],
            armature=g["armature"],
            friction=g["friction"],
            viscous_friction=g["viscous_friction"],
            min_delay=min_delay,
            max_delay=max_delay,
        )
    return out


_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
)


def _spawn_cfg():
    mode = os.environ.get("IGRIS_C_SPAWN", "urdf").lower()
    if mode == "usd":
        return sim_utils.UsdFileCfg(
            usd_path=str(C.IGRIS_C_ROBROS_USD_PATH),
            activate_contact_sensors=True,
            rigid_props=_RIGID_PROPS,
            articulation_props=_ARTICULATION_PROPS,
        )
    return sim_utils.UrdfFileCfg(
        asset_path=str(C.IGRIS_C_URDF_PATH),
        usd_dir=str(C.USD_CACHE_DIR),
        usd_file_name="igris_c_v2.usd",
        force_usd_conversion=False,
        make_instanceable=True,
        fix_base=False,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=False,
        root_link_name=None,
        link_density=0.0,
        collider_type="convex_hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    )


IGRIS_C_CFG = ArticulationCfg(
    spawn=_spawn_cfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, C.DEFAULT_ROOT_HEIGHT),
        # exact names (Isaac Lab raises if one joint matches several regex keys)
        joint_pos={name: C.default_joint_pos(name) for name in C.ACTUATED_JOINT_NAMES},
        joint_vel={".*": 0.0},
    ),
    actuators=_actuators(*C.ACTUATOR_DELAY_STEPS),
    soft_joint_pos_limit_factor=C.SOFT_JOINT_POS_LIMIT_FACTOR,
)
"""IGRIS-C (23 DoF) with Robros DelayedPD actuators (0-2 step delay)."""

IGRIS_C_NO_DELAY_CFG = IGRIS_C_CFG.replace(actuators=_actuators(0, 0))
"""Same robot without actuator delay (PLAY / evaluation / sim2sim parity)."""

ACTION_SCALE = dict(C.ACTION_SCALE)
"""Per-joint action scale (0.25 * effort_limit / stiffness, BeyondMimic convention)."""
