# Motion-tracking command for IGRIS-C.
#
# Structure (reference-state initialisation, yaw/xy re-anchored body targets, failure-adaptive time
# sampling) follows BeyondMimic "whole_body_tracking" (tasks/tracking/mdp/commands.py,
# Copyright (c) 2024 The Isaac Lab Project Developers / HybridRobotics, MIT License). Re-implemented here
# with: name-based NPZ mapping, multi-clip libraries (uniform over clips + adaptive bins per clip),
# root != anchor support, virtual end-effector points (hands merged into the elbow links) and an
# evaluation mode (start at frame 0, hold the last frame).

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .. import constants as C
from ..motion import MotionLoader, available_body_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def relative_to_robot_anchor(
    ref_anchor_pos: torch.Tensor,
    ref_anchor_quat: torch.Tensor,
    robot_anchor_pos: torch.Tensor,
    robot_anchor_quat: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Re-anchor reference body poses (N,K,3)/(N,K,4) onto the robot anchor: keep the reference anchor
    height, take the robot anchor xy, and rotate by the yaw difference robot_anchor * ref_anchor^-1.
    (BeyondMimic ``body_pos_relative_w``; the MuJoCo runner implements the identical numpy version.)"""
    K = pos.shape[1]
    delta_quat = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
    delta_quat = delta_quat[:, None, :].expand(-1, K, -1)
    delta_pos = robot_anchor_pos.clone()
    delta_pos[:, 2] = ref_anchor_pos[:, 2]
    rel_pos = delta_pos[:, None, :] + quat_apply(delta_quat, pos - ref_anchor_pos[:, None, :])
    rel_quat = quat_mul(delta_quat, quat) if quat is not None else None
    return rel_pos, rel_quat


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # ---- bodies --------------------------------------------------------------------------
        self.body_names = list(cfg.body_names)
        self.robot_body_ids = torch.tensor(
            self.robot.find_bodies(self.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.robot_anchor_id = self.robot.body_names.index(cfg.anchor_body_name)
        self.robot_root_name = cfg.root_body_name
        avail = available_body_names(cfg.motion_file)
        self.ee_names = list(cfg.ee_points.keys())
        ee_parents = [cfg.ee_points[n][0] for n in self.ee_names]
        self.ee_offsets = torch.tensor(
            [cfg.ee_points[n][1] for n in self.ee_names], dtype=torch.float32, device=self.device
        ).reshape(-1, 3)
        self.robot_ee_parent_ids = torch.tensor(
            self.robot.find_bodies(ee_parents, preserve_order=True)[0] if ee_parents else [],
            dtype=torch.long,
            device=self.device,
        )
        # ee reference: the NPZ body of the same name if present (e.g. MuJoCo 'Left_Hand'), else parent+offset
        self.ee_from_motion = [n in avail for n in self.ee_names]
        load = []
        for n in self.body_names + [cfg.anchor_body_name, cfg.root_body_name] + ee_parents:
            if n not in load:
                load.append(n)
        for n, from_motion in zip(self.ee_names, self.ee_from_motion):
            if from_motion and n not in load:
                load.append(n)

        # ---- motion library ------------------------------------------------------------------
        self.motion = MotionLoader(
            cfg.motion_file, self.robot.joint_names, load, fps=1.0 / env.step_dt, device=self.device,
            z_offset=cfg.motion_z_offset,
        )
        self.m_body_ids = torch.tensor([load.index(n) for n in self.body_names], dtype=torch.long, device=self.device)
        self.m_anchor_id = load.index(cfg.anchor_body_name)
        self.m_root_id = load.index(cfg.root_body_name)
        self.m_ee_parent_ids = torch.tensor([load.index(n) for n in ee_parents], dtype=torch.long, device=self.device)
        self.m_ee_src_ids = torch.tensor(
            [load.index(n) if fm else load.index(p) for n, p, fm in zip(self.ee_names, ee_parents, self.ee_from_motion)],
            dtype=torch.long,
            device=self.device,
        )
        self.ee_use_offset = torch.tensor([not fm for fm in self.ee_from_motion], dtype=torch.bool, device=self.device)

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reached_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ---- adaptive sampling bins (per clip, ~adaptive_bin_s seconds each) ------------------
        self.bin_frames = max(1, int(round(cfg.adaptive_bin_s / env.step_dt)))
        lengths = self.motion.lengths
        nb = torch.div(lengths + self.bin_frames - 1, self.bin_frames, rounding_mode="floor")
        self.motion_num_bins = nb
        self.motion_bin_offset = torch.cumsum(nb, 0) - nb
        self.num_bins = int(nb.sum().item())
        self.bin_motion = torch.repeat_interleave(torch.arange(self.motion.num_motions, device=self.device), nb)
        self.bin_start = (torch.arange(self.num_bins, device=self.device) - self.motion_bin_offset[self.bin_motion]) * self.bin_frames
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.num_bins, dtype=torch.float, device=self.device)
        kernel = torch.tensor([cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)], device=self.device)
        self.kernel = kernel / kernel.sum()

        for k in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_ee_pos",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
        ):
            self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            f"MotionCommand: {self.motion.num_motions} clip(s), {self.motion.total_frames} frames, "
            f"anchor={self.cfg.anchor_body_name}, bodies={len(self.body_names)}, ee={self.ee_names}, "
            f"sampling={self.cfg.sampling_mode}, end={self.cfg.motion_end_behavior}"
        )

    # ------------------------------------------------------------------------------------------
    # reference quantities (current frame)
    # ------------------------------------------------------------------------------------------
    @property
    def gidx(self) -> torch.Tensor:
        return self.motion.offsets[self.motion_ids] + self.time_steps

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.gidx]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.gidx]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.gidx][:, self.m_body_ids] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.gidx][:, self.m_body_ids]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.gidx][:, self.m_body_ids]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.gidx][:, self.m_body_ids]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.gidx, self.m_anchor_id] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.gidx, self.m_anchor_id]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.gidx, self.m_anchor_id]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.gidx, self.m_anchor_id]

    @property
    def ee_pos_w(self) -> torch.Tensor:
        g = self.gidx
        src_pos = self.motion.body_pos_w[g][:, self.m_ee_src_ids]
        src_quat = self.motion.body_quat_w[g][:, self.m_ee_src_ids]
        offs = torch.where(self.ee_use_offset[:, None], self.ee_offsets, torch.zeros_like(self.ee_offsets))
        offs = offs[None].expand(src_pos.shape[0], -1, -1)
        return src_pos + quat_apply(src_quat, offs) + self._env.scene.env_origins[:, None, :]

    # ------------------------------------------------------------------------------------------
    # robot quantities
    # ------------------------------------------------------------------------------------------
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_body_ids]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        # link-frame origin velocity (matches the NPZ definition), not the COM velocity
        return self.robot.data.body_link_lin_vel_w[:, self.robot_body_ids]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_body_ids]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_link_pos_w[:, self.robot_anchor_id]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_link_quat_w[:, self.robot_anchor_id]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_id]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_id]

    @property
    def robot_ee_pos_w(self) -> torch.Tensor:
        pos = self.robot.data.body_link_pos_w[:, self.robot_ee_parent_ids]
        quat = self.robot.data.body_link_quat_w[:, self.robot_ee_parent_ids]
        return pos + quat_apply(quat, self.ee_offsets[None].expand(pos.shape[0], -1, -1))

    # ------------------------------------------------------------------------------------------
    # re-anchored reference (computed lazily from the *current* robot anchor)
    # ------------------------------------------------------------------------------------------
    @property
    def body_pose_relative_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return relative_to_robot_anchor(
            self.anchor_pos_w, self.anchor_quat_w, self.robot_anchor_pos_w, self.robot_anchor_quat_w,
            self.body_pos_w, self.body_quat_w,
        )  # fmt: skip

    @property
    def body_pos_relative_w(self) -> torch.Tensor:
        return self.body_pose_relative_w[0]

    @property
    def body_quat_relative_w(self) -> torch.Tensor:
        return self.body_pose_relative_w[1]

    @property
    def ee_pos_relative_w(self) -> torch.Tensor:
        return relative_to_robot_anchor(
            self.anchor_pos_w, self.anchor_quat_w, self.robot_anchor_pos_w, self.robot_anchor_quat_w, self.ee_pos_w
        )[0]

    # ------------------------------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------------------------------
    def _update_metrics(self):
        m = self.metrics
        m["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        m["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        m["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        m["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        rel_pos, rel_quat = self.body_pose_relative_w
        m["error_body_pos"] = torch.norm(rel_pos - self.robot_body_pos_w, dim=-1).mean(-1)
        m["error_body_rot"] = quat_error_magnitude(rel_quat, self.robot_body_quat_w).mean(-1)
        m["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(-1)
        m["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(-1)
        m["error_ee_pos"] = torch.norm(self.ee_pos_relative_w - self.robot_ee_pos_w, dim=-1).mean(-1)
        m["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        m["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _record_failures(self, env_ids: torch.Tensor):
        tm = getattr(self._env, "termination_manager", None)
        if tm is None:
            return
        failed = tm.terminated[env_ids].float()  # weights instead of masking -> no GPU sync
        bins = self.motion_bin_offset[self.motion_ids[env_ids]] + torch.div(
            self.time_steps[env_ids], self.bin_frames, rounding_mode="floor"
        )
        bins = torch.clamp(bins, max=self.num_bins - 1)
        self._current_bin_failed += torch.bincount(bins, weights=failed, minlength=self.num_bins)

    def _sample(self, env_ids: torch.Tensor):
        n = len(env_ids)
        M = self.motion.num_motions
        lengths = self.motion.lengths
        mode = self.cfg.sampling_mode
        if mode == "start":
            self.motion_ids[env_ids] = env_ids % M
            self.time_steps[env_ids] = 0
            return
        if mode == "uniform":
            mids = torch.randint(0, M, (n,), device=self.device)
            self.motion_ids[env_ids] = mids
            t = (torch.rand(n, device=self.device) * lengths[mids]).long()
            self.time_steps[env_ids] = torch.minimum(t, lengths[mids] - 1)
            return
        if mode != "adaptive":
            raise ValueError(f"unknown sampling_mode {mode}")
        # failure-weighted bins (BeyondMimic) + per-clip uniform floor, clips sampled uniformly
        p = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / self.motion_num_bins[self.bin_motion].float()
        if self.cfg.adaptive_kernel_size > 1:
            p = torch.nn.functional.pad(p[None, None], (0, self.cfg.adaptive_kernel_size - 1), mode="replicate")
            p = torch.nn.functional.conv1d(p, self.kernel.view(1, 1, -1)).view(-1)
        per_motion = torch.zeros(M, device=self.device).scatter_add_(0, self.bin_motion, p)
        p = p / per_motion[self.bin_motion] / M
        bins = torch.multinomial(p, n, replacement=True)
        mids = self.bin_motion[bins]
        t = self.bin_start[bins] + (torch.rand(n, device=self.device) * self.bin_frames).long()
        self.motion_ids[env_ids] = mids
        self.time_steps[env_ids] = torch.minimum(t, lengths[mids] - 1)
        H = -(p * (p + 1e-12).log()).sum() / max(math.log(self.num_bins), 1e-6)
        self.metrics["sampling_entropy"][:] = H
        self.metrics["sampling_top1_prob"][:] = p.max()

    def _write_reference_state(self, env_ids: torch.Tensor, randomize: bool):
        g = self.motion.offsets[self.motion_ids[env_ids]] + self.time_steps[env_ids]
        root_pos = self.motion.body_pos_w[g, self.m_root_id] + self._env.scene.env_origins[env_ids]
        root_quat = self.motion.body_quat_w[g, self.m_root_id].clone()
        root_lin = self.motion.body_lin_vel_w[g, self.m_root_id].clone()
        root_ang = self.motion.body_ang_vel_w[g, self.m_root_id].clone()
        joint_pos = self.motion.joint_pos[g].clone()
        joint_vel = self.motion.joint_vel[g].clone()
        n = len(env_ids)
        if randomize:
            keys = ["x", "y", "z", "roll", "pitch", "yaw"]
            r = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
            s = sample_uniform(r[:, 0], r[:, 1], (n, 6), device=self.device)
            root_pos = root_pos + s[:, :3]
            root_quat = quat_mul(quat_from_euler_xyz(s[:, 3], s[:, 4], s[:, 5]), root_quat)
            r = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in keys], device=self.device)
            s = sample_uniform(r[:, 0], r[:, 1], (n, 6), device=self.device)
            root_lin = root_lin + s[:, :3]
            root_ang = root_ang + s[:, 3:]
            joint_pos = joint_pos + sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, self.device)
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, limits[..., 0], limits[..., 1])
        # root is written as LINK pose / LINK velocity: the NPZ stores body-frame-origin velocities
        self.robot.write_root_link_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat([root_lin, root_ang], dim=-1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        self._record_failures(env_ids)
        self._sample(env_ids)
        self.reached_end[env_ids] = False
        self._write_reference_state(env_ids, randomize=self.cfg.randomize_reset_state)

    def _update_command(self):
        self.time_steps += 1
        ended = self.time_steps >= self.motion.lengths[self.motion_ids]
        if self.cfg.motion_end_behavior == "hold":
            self.reached_end |= ended
            self.time_steps = torch.minimum(self.time_steps, self.motion.lengths[self.motion_ids] - 1)
        else:
            env_ids = torch.where(ended)[0]
            if len(env_ids) > 0:
                self._resample_command(env_ids)
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for :class:`MotionCommand`."""

    class_type: type = MotionCommand

    asset_name: str = "robot"
    motion_file: str | list[str] = MISSING
    """NPZ file, directory (recursive *.npz), glob, comma separated list or .txt list."""
    motion_z_offset: float = 0.0
    """Constant z shift of all reference bodies [m] (exported to the JSON; the MuJoCo runner applies it too)."""
    anchor_body_name: str = C.ANCHOR_BODY_NAME
    root_body_name: str = C.ROOT_BODY_NAME
    body_names: list[str] = list(C.TRACKED_BODY_NAMES)
    ee_points: dict = dict(C.EE_POINTS)

    sampling_mode: str = "adaptive"
    """'adaptive' (failure-weighted bins, training), 'uniform', or 'start' (frame 0, env i -> clip i % M)."""
    motion_end_behavior: str = "resample"
    """'resample' (training) or 'hold' (evaluation: clamp at the last frame and flag ``reached_end``)."""
    randomize_reset_state: bool = True

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)

    adaptive_bin_s: float = 1.0
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
