# Termination terms for IGRIS-C motion tracking.
# Adapted from BeyondMimic whole_body_tracking (tasks/tracking/mdp/terminations.py, MIT License).
# The MuJoCo sim2sim runner re-implements the same three checks with the thresholds from the export JSON.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply_inverse

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """|z_ref_anchor - z_robot_anchor| > threshold."""
    c = _cmd(env, command_name)
    return torch.abs(c.anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]) > threshold


def bad_anchor_ori(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """|g_ref_b.z - g_robot_b.z| > threshold with g_b = R(q_anchor)^T (0, 0, -1) (projected gravity)."""
    c = _cmd(env, command_name)
    g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    g_ref = quat_apply_inverse(c.anchor_quat_w, g)
    g_rob = quat_apply_inverse(c.robot_anchor_quat_w, g)
    return torch.abs(g_ref[:, 2] - g_rob[:, 2]) > threshold


def bad_ee_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Any end-effector point (hands, feet) with |z_ref - z_robot| > threshold (z is unaffected by the
    xy/yaw re-anchoring, so this equals the BeyondMimic ``bad_motion_body_pos_z_only``)."""
    c = _cmd(env, command_name)
    err = torch.abs(c.ee_pos_w[..., 2] - c.robot_ee_pos_w[..., 2])
    return torch.any(err > threshold, dim=-1)
