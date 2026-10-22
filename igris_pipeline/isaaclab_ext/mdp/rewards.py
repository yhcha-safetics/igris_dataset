# Reward terms for IGRIS-C motion tracking (DeepMimic / BeyondMimic exponential kernels).
# Adapted from BeyondMimic whole_body_tracking (tasks/tracking/mdp/rewards.py, MIT License):
#   r = exp(-mean_k ||e_k||^2 / std^2)
# Body position/orientation targets are the reference re-anchored to the robot anchor (xy + yaw), so
# global drift is only penalised by the two anchor terms.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def _body_ids(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, n in enumerate(command.body_names) if body_names is None or n in body_names]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    c = _cmd(env, command_name)
    err = torch.sum(torch.square(c.anchor_pos_w - c.robot_anchor_pos_w), dim=-1)
    return torch.exp(-err / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    c = _cmd(env, command_name)
    err = quat_error_magnitude(c.anchor_quat_w, c.robot_anchor_quat_w) ** 2
    return torch.exp(-err / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    c = _cmd(env, command_name)
    ids = _body_ids(c, body_names)
    err = torch.sum(torch.square(c.body_pos_relative_w[:, ids] - c.robot_body_pos_w[:, ids]), dim=-1)
    return torch.exp(-err.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    c = _cmd(env, command_name)
    ids = _body_ids(c, body_names)
    err = quat_error_magnitude(c.body_quat_relative_w[:, ids], c.robot_body_quat_w[:, ids]) ** 2
    return torch.exp(-err.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    c = _cmd(env, command_name)
    ids = _body_ids(c, body_names)
    err = torch.sum(torch.square(c.body_lin_vel_w[:, ids] - c.robot_body_lin_vel_w[:, ids]), dim=-1)
    return torch.exp(-err.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    c = _cmd(env, command_name)
    ids = _body_ids(c, body_names)
    err = torch.sum(torch.square(c.body_ang_vel_w[:, ids] - c.robot_body_ang_vel_w[:, ids]), dim=-1)
    return torch.exp(-err.mean(-1) / std**2)


def motion_ee_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Hands (virtual points on the elbow links) + feet, re-anchored reference."""
    c = _cmd(env, command_name)
    err = torch.sum(torch.square(c.ee_pos_relative_w - c.robot_ee_pos_w), dim=-1)
    return torch.exp(-err.mean(-1) / std**2)
