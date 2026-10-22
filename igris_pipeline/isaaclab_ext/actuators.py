"""Sync-free delayed PD actuator.

Functionally identical to ``isaaclab.actuators.DelayedPDActuator`` (explicit PD of IdealPDActuator, command
lagged by a per-env random number of physics steps in [min_delay, max_delay], re-sampled at reset, history
filled with the first command after a reset), but implemented with a plain ring buffer + gather.

Why: Isaac Lab's ``DelayBuffer``/``CircularBuffer`` call ``tensor.item()`` several times per physics step
and actuator group (12 GPU->CPU syncs per substep here). On this Windows/WDDM machine with a shared GPU every
sync costs ~0.4 ms+, which made env.step ~200 ms (profiled: ~50% of the step in delay_buffer.compute).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class FastDelayedPDActuator(IdealPDActuator):
    cfg: FastDelayedPDActuatorCfg

    def __init__(self, cfg: FastDelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._H = int(cfg.max_delay) + 1
        shape = (self._H, self._num_envs, self.num_joints)
        self._hist = [torch.zeros(shape, device=self._device) for _ in range(3)]  # pos, vel, effort
        self._ptr = 0
        self._lags = torch.randint(cfg.min_delay, cfg.max_delay + 1, (self._num_envs,), device=self._device)
        self._fresh = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._env_ar = torch.arange(self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int] | None):
        super().reset(env_ids)
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            env_ids = self._env_ar
        n = len(env_ids)
        self._lags[env_ids] = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (n,), device=self._device)
        self._fresh[env_ids] = True

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        if self._H > 1:
            self._ptr = (self._ptr + 1) % self._H
            fresh = self._fresh[None, :, None]
            cur = (control_action.joint_positions, control_action.joint_velocities, control_action.joint_efforts)
            out = []
            idx = (self._ptr - self._lags) % self._H
            for hist, c in zip(self._hist, cur):
                hist[self._ptr] = c
                hist.copy_(torch.where(fresh, c[None].expand_as(hist), hist))
                out.append(hist[idx, self._env_ar])
            self._fresh[:] = False
            control_action.joint_positions, control_action.joint_velocities, control_action.joint_efforts = out
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class FastDelayedPDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = FastDelayedPDActuator
    min_delay: int = 0
    max_delay: int = 0
