"""Manager-based env config for IGRIS-C whole-body motion tracking (BeyondMimic-style).

Policy observation (order = concatenation order, dims for the 23-DoF robot, total 130):
    motion_joint_pos     23  reference joint positions (absolute, rad)
    motion_joint_vel     23  reference joint velocities (rad/s)
    motion_anchor_pos_b   3  reference anchor (base_link) position in the robot anchor frame
    motion_anchor_ori_b   6  reference anchor orientation in the robot anchor frame (rot6, 2 columns)
    base_ang_vel          3  robot base_link angular velocity, base frame
    projected_gravity     3  gravity direction (0,0,-1) in the base_link frame
    joint_pos            23  q - q_default
    joint_vel            23  qd (default joint velocity is 0)
    actions              23  last raw policy action
Critic additionally gets base_lin_vel and the tracked bodies' poses in the anchor frame.

Action (``action_mode``, see :func:`configure_action`):
    absolute (default): q_target = q_default + action_scale * a   (scale preset "beyondmimic" = 0.25*effort/kp)
    residual:           q_target = q_ref(t)  + action_scale * a   (scale preset "gmt" = 0.25 rad, uniform)
Explicit delayed PD at 200 Hz, policy at 50 Hz.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import constants as C
from . import mdp
from .assets import ACTION_SCALE, IGRIS_C_CFG, IGRIS_C_NO_DELAY_CFG

DEFAULT_MOTION = str(C.SAMPLE_MOTION_DIR / "test_sine.npz")

# push / reset velocity perturbation (moderate; BeyondMimic uses +-0.5 m/s, +-0.52/0.78 rad/s)
VELOCITY_RANGE = {
    "x": (-0.4, 0.4),
    "y": (-0.4, 0.4),
    "z": (-0.2, 0.2),
    "roll": (-0.4, 0.4),
    "pitch": (-0.4, 0.4),
    "yaw": (-0.6, 0.6),
}

# contacts allowed only on the feet and the hands (merged into the elbow links)
UNDESIRED_CONTACT_BODIES = (
    r"^(?!Link_Ankle_Roll_Left$)(?!Link_Ankle_Roll_Right$)(?!Link_Elbow_Pitch_Left$)(?!Link_Elbow_Pitch_Right$).+$"
)


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    robot: ArticulationCfg = IGRIS_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=DEFAULT_MOTION,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        anchor_body_name=C.ANCHOR_BODY_NAME,
        root_body_name=C.ROOT_BODY_NAME,
        body_names=list(C.TRACKED_BODY_NAMES),
        ee_points=dict(C.EE_POINTS),
        sampling_mode="adaptive",
        motion_end_behavior="resample",
        randomize_reset_state=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=dict(ACTION_SCALE), use_default_offset=True
    )


ACTION_MODES = ("absolute", "residual")
DEFAULT_SCALE_PRESET = {"absolute": "beyondmimic", "residual": "gmt"}


def configure_action(env_cfg, mode: str = "absolute", scale: str | dict | None = None):
    """Switch the action term between absolute (default-pose offset) and residual (reference offset).
    ``scale``: a preset name from constants.ACTION_SCALE_PRESETS, an explicit {joint: scale} dict, or None
    (mode default: absolute -> "beyondmimic", residual -> "gmt" = 0.25 rad)."""
    if mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {mode}")
    if scale is None:
        scale = DEFAULT_SCALE_PRESET[mode]
    scale = dict(C.ACTION_SCALE_PRESETS[scale]) if isinstance(scale, str) else dict(scale)
    if mode == "absolute":
        env_cfg.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=scale, use_default_offset=True
        )
    else:
        env_cfg.actions.joint_pos = mdp.ResidualJointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=scale, use_default_offset=True, command_name="motion"
        )
    return env_cfg


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # order matters: this is the ONNX input layout (see export JSON)
        motion_joint_pos = ObsTerm(func=mdp.motion_joint_pos, params={"command_name": "motion"})
        motion_joint_vel = ObsTerm(func=mdp.motion_joint_vel, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.25, n_max=0.25)
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        motion_joint_pos = ObsTerm(func=mdp.motion_joint_pos, params={"command_name": "motion"})
        motion_joint_vel = ObsTerm(func=mdp.motion_joint_vel, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # -- startup (moderate domain randomisation)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 64,
        },
    )
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )
    # -- interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    # weights / stds = BeyondMimic G1 defaults
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.3}
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.3}
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.4}
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0}
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])}
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[UNDESIRED_CONTACT_BODIES]),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(func=mdp.bad_anchor_pos_z_only, params={"command_name": "motion", "threshold": 0.25})
    anchor_ori = DoneTerm(func=mdp.bad_anchor_ori, params={"command_name": "motion", "threshold": 0.8})
    ee_body_pos = DoneTerm(func=mdp.bad_ee_pos_z_only, params={"command_name": "motion", "threshold": 0.25})


@configclass
class IgrisCTrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Training config: ``Igris-C-Tracking-Flat-v0``."""

    scene: TrackingSceneCfg = TrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = C.DECIMATION
        self.episode_length_s = 10.0
        self.sim.dt = C.SIM_DT
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class IgrisCTrackingEnvCfg_PLAY(IgrisCTrackingEnvCfg):
    """Evaluation config: ``Igris-C-Tracking-Flat-Play-v0`` -- no noise, no randomisation, no actuator
    delay, every env starts at frame 0 of clip (env_id % num_clips) and holds the last frame."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.robot = IGRIS_C_NO_DELAY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.base_mass = None
        self.events.push_robot = None
        self.commands.motion.sampling_mode = "start"
        self.commands.motion.motion_end_behavior = "hold"
        self.commands.motion.randomize_reset_state = False
        self.episode_length_s = 1.0e5  # the eval loop stops at the end of each clip
