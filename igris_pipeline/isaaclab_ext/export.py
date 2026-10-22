"""Export a trained IGRIS-C tracking actor to ONNX + TorchScript + JSON metadata.

The JSON is the sim2sim / deployment contract: together with the reference motion NPZ it contains
everything needed to rebuild the policy observation exactly and to apply the action with the same PD law
(see igris_pipeline/sim/mujoco_policy.py).

Both model files contain the empirical observation normaliser -> feed the RAW observation vector.
ONNX:        input "obs" float32 (1, obs_dim)  ->  output "actions" float32 (1, num_joints)
TorchScript: forward(obs: Tensor[N, obs_dim]) -> Tensor[N, num_joints]
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any

import torch

from . import constants as C

OBS_TERM_DEFINITIONS = {
    "motion_joint_pos": "reference joint_pos[t] (absolute rad) in policy joint order",
    "motion_joint_vel": "reference joint_vel[t] (rad/s) in policy joint order",
    "motion_anchor_pos_b": "R(q_robot_anchor)^T (p_ref_anchor - p_robot_anchor); anchor = anchor_body_name",
    "motion_anchor_ori_b": "rot6(q_robot_anchor^-1 * q_ref_anchor), rot6(R) = [R00,R01,R10,R11,R20,R21] "
    "(first two columns, row-major flatten)",
    "base_ang_vel": "root (base_link) angular velocity in the root frame = R(q_root)^T w_world "
    "(MuJoCo freejoint qvel[3:6] is already in this frame)",
    "projected_gravity": "R(q_root)^T (0, 0, -1)",
    "joint_pos": "q - default_joint_pos (policy joint order)",
    "joint_vel": "qd - 0 (policy joint order)",
    "actions": "previous raw policy output (before scaling), zeros at episode start",
}


def _to_list(x) -> list:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _per_joint(value, num_joints: int) -> list[float]:
    if isinstance(value, torch.Tensor):
        v = value.detach().cpu()
        if v.ndim == 2:
            v = v[0]
        if v.numel() == 1:
            return [float(v.item())] * num_joints
        return [float(a) for a in v.tolist()]
    return [float(value)] * num_joints


def build_policy_metadata(env, task_id: str = "", checkpoint: str = "", extra: dict | None = None) -> dict[str, Any]:
    """Collect everything a sim2sim runner needs from a live ManagerBasedRLEnv (unwrapped)."""
    env = env.unwrapped if hasattr(env, "unwrapped") else env
    robot = env.scene["robot"]
    J = robot.num_joints
    joint_names = list(robot.joint_names)

    if hasattr(robot.data, "default_joint_pos_nominal"):
        default_joint_pos = robot.data.default_joint_pos_nominal.cpu().tolist()
    else:
        default_joint_pos = robot.data.default_joint_pos[0].cpu().tolist()

    # ---- actuators (per joint, policy order) ----
    kp = [0.0] * J
    kd = [0.0] * J
    effort = [0.0] * J
    vel_limit = [0.0] * J
    delay = {}
    actuator_types = {}
    for name, act in robot.actuators.items():
        ids = act.joint_indices
        ids = list(range(J)) if isinstance(ids, slice) else _to_list(ids)
        for k, j in enumerate(ids):
            kp[j] = float(act.stiffness[0, k])
            kd[j] = float(act.damping[0, k])
            effort[j] = float(act.effort_limit[0, k])
            vel_limit[j] = float(act.velocity_limit[0, k])
        actuator_types[name] = type(act).__name__
        if hasattr(act.cfg, "min_delay"):
            delay[name] = [int(act.cfg.min_delay), int(act.cfg.max_delay)]
    data = robot.data
    armature = data.joint_armature[0].cpu().tolist()
    friction = data.joint_friction_coeff[0].cpu().tolist()
    dyn_friction = data.joint_dynamic_friction_coeff[0].cpu().tolist() if hasattr(data, "joint_dynamic_friction_coeff") else [0.0] * J
    visc_friction = data.joint_viscous_friction_coeff[0].cpu().tolist() if hasattr(data, "joint_viscous_friction_coeff") else [0.0] * J

    # ---- action ----
    act_term = env.action_manager.get_term("joint_pos")
    action_mode = "residual" if type(act_term).__name__ == "ResidualJointPositionAction" else "absolute"
    action_scale = _per_joint(act_term._scale, J)
    action_clip = None
    if getattr(act_term.cfg, "clip", None):
        action_clip = {k: list(v) for k, v in act_term.cfg.clip.items()}

    # ---- observations ----
    om = env.observation_manager
    names = list(om.active_terms["policy"])
    dims = [list(d) for d in om.group_obs_term_dim["policy"]]
    cfgs = om._group_obs_term_cfgs["policy"]
    terms = []
    start = 0
    for name, dim, cfg in zip(names, dims, cfgs):
        size = int(torch.tensor(dim).prod().item())
        terms.append(
            {
                "name": name,
                "func": getattr(cfg.func, "__name__", str(cfg.func)),
                "dim": size,
                "slice": [start, start + size],
                "scale": cfg.scale if cfg.scale is None or isinstance(cfg.scale, (int, float)) else _to_list(cfg.scale),
                "clip": list(cfg.clip) if cfg.clip is not None else None,
                "history_length": int(getattr(cfg, "history_length", 0) or 0),
                "training_noise": (
                    {"type": type(cfg.noise).__name__, "n_min": getattr(cfg.noise, "n_min", None), "n_max": getattr(cfg.noise, "n_max", None)}
                    if cfg.noise is not None
                    else None
                ),
                "definition": OBS_TERM_DEFINITIONS.get(name, ""),
            }
        )
        start += size
    obs_dim = start

    # ---- command / terminations ----
    cmd = env.command_manager.get_term("motion")
    terms_cfg = env.termination_manager.cfg
    term_thresholds = {}
    for tname in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        t = getattr(terms_cfg, tname, None)
        if t is not None:
            term_thresholds[tname] = {"func": t.func.__name__, "threshold": float(t.params["threshold"])}

    meta = {
        "format": "igris_c_tracking_policy",
        "format_version": 1,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "task": task_id,
        "checkpoint": checkpoint,
        "export_env_cfg": type(env.cfg).__name__,
        "files": {"onnx": "policy.onnx", "torchscript": "policy.pt"},
        "onnx_io": {"input": "obs", "input_shape": [1, obs_dim], "output": "actions", "output_shape": [1, J]},
        "normalizer": "embedded in policy.onnx / policy.pt (feed raw observations)",
        "conventions": {
            "quaternion": "wxyz",
            "world": "Z-up, gravity (0,0,-9.81)",
            "rot6": "R[:, :2] flattened row-major = [R00,R01,R10,R11,R20,R21]",
            "body_velocity": "linear velocity of the body-frame origin (not COM), world frame",
            "reference_alignment": "observation at policy step k uses reference frame k; after the step the "
            "robot is scored / terminated against frame k; then k += 1",
        },
        # timing
        "sim_dt": float(env.physics_dt),
        "decimation": int(env.cfg.decimation),
        "policy_dt": float(env.step_dt),
        "motion_fps": float(1.0 / env.step_dt),
        # joints / actuation (policy order)
        "num_joints": J,
        "joint_names": joint_names,
        "default_joint_pos": default_joint_pos,
        "action_scale": action_scale,
        "action_mode": action_mode,
        "action_offset": "default_joint_pos" if action_mode == "absolute" else "reference joint_pos[k] (motion_joint_pos obs)",
        "action_clip": action_clip,
        "action_to_target": (
            "q_target = default_joint_pos + action_scale * action"
            if action_mode == "absolute"
            else "q_target = motion_joint_pos[k] + action_scale * action  (k = reference frame of the observation)"
        ),
        "pd_law": "tau = clip(kp * (q_target - q) - kd * qd, -effort_limit, effort_limit) every sim_dt",
        "kp": kp,
        "kd": kd,
        "effort_limit": effort,
        "velocity_limit": vel_limit,
        "armature": armature,
        "joint_friction": friction,
        "joint_dynamic_friction": dyn_friction,
        "joint_viscous_friction": visc_friction,
        "joint_friction_semantics": "Isaac Sim >=5 PhysX: joint_friction = static friction effort [Nm], "
        "joint_dynamic_friction = dynamic friction effort [Nm], joint_viscous_friction = viscous coeff [Nm s/rad]",
        "actuator_types": actuator_types,
        "actuator_delay_steps": delay,
        "actuator_delay_note": "delay of the env used for export (training cfg: 0-2 physics steps, PLAY cfg: 0); "
        "the sim2sim runner applies no delay",
        "joint_pos_limits": data.joint_pos_limits[0].cpu().tolist(),
        "soft_joint_pos_limits": data.soft_joint_pos_limits[0].cpu().tolist(),
        "passive_mjcf_joints": list(C.PASSIVE_MJCF_JOINT_NAMES),
        # bodies
        "root_body_name": robot.body_names[0],
        "anchor_body_name": cmd.cfg.anchor_body_name,
        "body_names": list(cmd.body_names),
        "ee_points": {k: {"parent": v[0], "offset": list(v[1])} for k, v in cmd.cfg.ee_points.items()},
        "robot_body_names": list(robot.body_names),
        # observation
        "observation": {"group": "policy", "dim": obs_dim, "concatenate": True, "terms": terms},
        "observation_order": [t["name"] for t in terms],
        # terminations used for success / failure
        "terminations": term_thresholds,
        "default_root_height": C.DEFAULT_ROOT_HEIGHT,
        "training_motion_files": list(cmd.motion.files),
        "motion_z_offset": float(cmd.cfg.motion_z_offset),
    }
    if extra:
        meta.update(extra)
    return meta


def export_policy(
    policy_nn, normalizer, env, out_dir: str, task_id: str = "", checkpoint: str = "", extra: dict | None = None
) -> dict[str, str]:
    """Write policy.onnx, policy.pt and policy.json into ``out_dir``."""
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

    os.makedirs(out_dir, exist_ok=True)
    export_policy_as_onnx(policy_nn, path=out_dir, normalizer=normalizer, filename="policy.onnx")
    export_policy_as_jit(policy_nn, normalizer, path=out_dir, filename="policy.pt")
    meta = build_policy_metadata(env, task_id=task_id, checkpoint=checkpoint, extra=extra)
    json_path = os.path.join(out_dir, "policy.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # the exporters move the copied modules to CPU; the live policy is untouched
    return {
        "onnx": os.path.join(out_dir, "policy.onnx"),
        "torchscript": os.path.join(out_dir, "policy.pt"),
        "json": json_path,
    }


def get_policy_and_normalizer(runner):
    """rsl-rl >= 2.3: runner.alg.policy; normaliser lives in policy.actor_obs_normalizer."""
    try:
        policy_nn = runner.alg.policy
    except AttributeError:  # pragma: no cover
        policy_nn = runner.alg.actor_critic
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    if normalizer is None:
        normalizer = getattr(runner, "obs_normalizer", None)
    return policy_nn, normalizer
