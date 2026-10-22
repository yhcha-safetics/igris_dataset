# Event terms for IGRIS-C motion tracking.
# ``randomize_joint_default_pos`` adapted from BeyondMimic whole_body_tracking
# (tasks/tracking/mdp/events.py, MIT License): models joint-encoder calibration offsets by perturbing the
# default joint position (= action offset and joint_pos_rel reference). The nominal default is stored in
# ``asset.data.default_joint_pos_nominal`` for export.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    action_name: str = "joint_pos",
):
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset.data, "default_joint_pos_nominal"):
        asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)
    if pos_distribution_params is None:
        return
    pos = asset.data.default_joint_pos.to(asset.device).clone()
    pos = _randomize_prop_by_op(
        pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
    )[env_ids][:, joint_ids]
    if env_ids != slice(None) and joint_ids != slice(None):
        env_ids = env_ids[:, None]
    asset.data.default_joint_pos[env_ids, joint_ids] = pos
    # the JointPositionAction offset is not updated automatically
    env.action_manager.get_term(action_name)._offset[env_ids, joint_ids] = pos
