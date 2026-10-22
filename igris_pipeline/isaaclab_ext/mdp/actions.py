"""Residual joint-position action around the motion reference.

    absolute (isaaclab JointPositionAction): q_target = q_default + scale * a
    residual (this term):                    q_target = q_ref(t) + scale * a

q_ref(t) is the motion command's current reference joint position, i.e. exactly the ``motion_joint_pos``
observation the policy saw for this step: ``process_actions`` runs at the start of ``env.step`` while the
command is still on reference frame k (the frame used to build the observation), and the target is held for
the ``decimation`` physics steps. The MuJoCo runner applies the identical rule (policy.json ``action_mode``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResidualJointPositionAction(JointPositionAction):
    cfg: ResidualJointPositionActionCfg

    def __init__(self, cfg: ResidualJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command = None

    def process_actions(self, actions: torch.Tensor):
        if self._command is None:  # the command manager is built before the action manager
            self._command = self._env.command_manager.get_term(self.cfg.command_name)
        self._raw_actions[:] = actions
        q_ref = self._command.joint_pos[:, self._joint_ids]  # command joint order == articulation order
        self._processed_actions = q_ref + self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class ResidualJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = ResidualJointPositionAction
    command_name: str = "motion"
