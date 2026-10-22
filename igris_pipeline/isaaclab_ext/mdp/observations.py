# Observation terms for IGRIS-C motion tracking.
# Term definitions follow BeyondMimic whole_body_tracking (tasks/tracking/mdp/observations.py, MIT).
#
# Every term is documented with the exact formula so that the MuJoCo sim2sim runner
# (igris_pipeline/sim/mujoco_policy.py) reproduces it bit-for-bit (same formulas, numpy):
#   R(q)      rotation matrix of quaternion q (wxyz)
#   rot6(R)   = R[:, :2].reshape(-1) = [R00, R01, R10, R11, R20, R21]   (first two COLUMNS, row-major)

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _cmd(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    return env.command_manager.get_term(command_name)


def rot6(quat: torch.Tensor) -> torch.Tensor:
    mat = matrix_from_quat(quat)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference joint positions (absolute, rad, policy joint order). dim = 23."""
    return _cmd(env, command_name).joint_pos


def motion_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference joint velocities (rad/s, policy joint order). dim = 23."""
    return _cmd(env, command_name).joint_vel


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference anchor position in the robot anchor frame: R(q_rob)^T (p_ref - p_rob). dim = 3."""
    c = _cmd(env, command_name)
    pos, _ = subtract_frame_transforms(c.robot_anchor_pos_w, c.robot_anchor_quat_w, c.anchor_pos_w, c.anchor_quat_w)
    return pos


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference anchor orientation in the robot anchor frame: rot6(q_rob^-1 * q_ref). dim = 6."""
    c = _cmd(env, command_name)
    _, ori = subtract_frame_transforms(c.robot_anchor_pos_w, c.robot_anchor_quat_w, c.anchor_pos_w, c.anchor_quat_w)
    return rot6(ori)


# ---- privileged (critic only) -------------------------------------------------------------------


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked robot body positions in the robot anchor frame. dim = 3 * num_bodies."""
    c = _cmd(env, command_name)
    K = len(c.body_names)
    pos, _ = subtract_frame_transforms(
        c.robot_anchor_pos_w[:, None].expand(-1, K, -1),
        c.robot_anchor_quat_w[:, None].expand(-1, K, -1),
        c.robot_body_pos_w,
        c.robot_body_quat_w,
    )
    return pos.reshape(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked robot body orientations (rot6) in the robot anchor frame. dim = 6 * num_bodies."""
    c = _cmd(env, command_name)
    K = len(c.body_names)
    _, ori = subtract_frame_transforms(
        c.robot_anchor_pos_w[:, None].expand(-1, K, -1),
        c.robot_anchor_quat_w[:, None].expand(-1, K, -1),
        c.robot_body_pos_w,
        c.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori.reshape(-1, 4))
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_body_pos_error_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """(re-anchored reference body pos - robot body pos) expressed in the robot anchor frame. critic only."""
    c = _cmd(env, command_name)
    K = len(c.body_names)
    err_w = c.body_pos_relative_w - c.robot_body_pos_w
    from isaaclab.utils.math import quat_apply_inverse

    q = c.robot_anchor_quat_w[:, None].expand(-1, K, -1)
    return quat_apply_inverse(q, err_w).reshape(env.num_envs, -1)
