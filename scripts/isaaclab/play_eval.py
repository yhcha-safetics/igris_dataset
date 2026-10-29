"""Deterministic evaluation of a trained IGRIS-C tracking policy in Isaac Lab + policy export.

Every env starts at frame 0 of clip (env_id % num_clips) (no noise, no randomisation, no actuator delay),
runs the deterministic actor (mean action) and is scored until it either tracks all T frames of its clip
(success) or hits a termination (failure). Metrics are computed by igris_pipeline/isaaclab_ext/metrics.py
(the MuJoCo sim2sim runner uses the same code).

Example:
    C:\\ProgramData\\Anaconda3\\envs\\isaaclab\\python.exe scripts\\isaaclab\\play_eval.py --headless ^
        --checkpoint logs\\isaaclab\\igris_c_tracking\\<run>\\model_9.pt --motion data\\sample\\motions ^
        --dump_obs logs\\isaaclab\\igris_c_tracking\\<run>\\exported\\isaac_obs_dump.npz

Outputs (next to the checkpoint):
    exported/policy.onnx, exported/policy.pt, exported/policy.json   (unless --no_export)
    eval/metrics_<ckpt>.json
    optional --dump_obs npz: first observation (per term), policy action, robot state, zero-action rollout
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# Kit's shutdown ends the process without flushing Python's buffers -> line-buffer redirected output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate / export an IGRIS-C tracking policy.")
parser.add_argument("--task", type=str, default="Igris-C-Tracking-Flat-Play-v0")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="model_*.pt, or a run directory (latest model), default: latest run of the experiment")
parser.add_argument("--experiment_name", type=str, default="igris_c_tracking")
parser.add_argument("--log_root", type=str, default=str(PROJECT_ROOT / "logs" / "isaaclab"))
parser.add_argument("--motion", type=str, default=None, help="default: the run's params/motions.txt")
parser.add_argument("--num_envs", type=int, default=None, help="default: number of clips")
parser.add_argument("--max_steps", type=int, default=None, help="safety cap on policy steps (default: longest clip)")
parser.add_argument("--output", type=str, default=None, help="metrics JSON path")
parser.add_argument("--no_export", action="store_true")
parser.add_argument("--motion_z_offset", type=float, default=None, help="default: value used in training (params)")
parser.add_argument("--dump_obs", type=str, default=None, help="write the t=0 observation dump (npz) for parity checks")
parser.add_argument("--dump_steps", type=int, default=25, help="zero-action steps recorded in the dump")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import igris_pipeline.isaaclab_ext as ext  # noqa: E402
from igris_pipeline.isaaclab_ext.export import export_policy, get_policy_and_normalizer  # noqa: E402
from igris_pipeline.isaaclab_ext.metrics import EpisodeAccumulator, step_metrics, write_json  # noqa: E402
from igris_pipeline.isaaclab_ext.motion import load_segments, project_relative, resolve_motion_files  # noqa: E402
from igris_pipeline.isaaclab_ext.tracking_env_cfg import configure_action  # noqa: E402


def _latest(pattern_dir: str) -> str:
    models = glob.glob(os.path.join(pattern_dir, "model_*.pt"))
    if not models:
        raise FileNotFoundError(f"no model_*.pt in {pattern_dir}")
    return max(models, key=lambda p: int(re.findall(r"model_(\d+)\.pt", p)[0]))


def resolve_checkpoint() -> str:
    ck = args_cli.checkpoint
    if ck and os.path.isfile(ck):
        return os.path.abspath(ck)
    if ck and os.path.isdir(ck):
        return _latest(ck)
    root = os.path.join(args_cli.log_root, args_cli.experiment_name)
    runs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d) and glob.glob(os.path.join(d, "model_*.pt")))
    if not runs:
        raise FileNotFoundError(f"no runs with checkpoints under {root}")
    return _latest(runs[-1])


def _training_overrides(run_dir: str) -> dict:
    """Settings written by train.py that change the policy interface (action scale, reference z offset)."""
    path = os.path.join(run_dir, "params", "overrides.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def np_(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64)


def snapshot_reference(cmd) -> dict:
    return {
        "body_pos": np_(cmd.body_pos_w),
        "joint_pos": np_(cmd.joint_pos),
        "joint_vel": np_(cmd.joint_vel),
        "anchor_pos": np_(cmd.anchor_pos_w),
        "anchor_quat": np_(cmd.anchor_quat_w),
        "ee_pos": np_(cmd.ee_pos_w),
    }


def snapshot_robot(cmd) -> dict:
    return {
        "body_pos": np_(cmd.robot_body_pos_w),
        "joint_pos": np_(cmd.robot_joint_pos),
        "joint_vel": np_(cmd.robot_joint_vel),
        "anchor_pos": np_(cmd.robot_anchor_pos_w),
        "anchor_quat": np_(cmd.robot_anchor_quat_w),
        "ee_pos": np_(cmd.robot_ee_pos_w),
    }


def dump_first_obs(env, policy, path: str, steps: int):
    """Observation/state dump of env 0 right after reset (+ a zero-action rollout) for check_obs_parity.py."""
    uenv = env.unwrapped
    cmd = uenv.command_manager.get_term("motion")
    robot = uenv.scene["robot"]
    om = uenv.observation_manager
    obs = env.get_observations()
    origin = uenv.scene.env_origins[0]
    out = {
        "obs": np_(obs["policy"][0]),
        "action": np_(policy(obs)[0]),
        "obs_term_names": np.array(om.active_terms["policy"]),
        "joint_names": np.array(robot.joint_names),
        "body_names": np.array(robot.body_names),
        "motion_file": np.array(cmd.motion.files[int(cmd.motion_ids[0])]),
        "time_step": np.array(int(cmd.time_steps[0])),
        "root_pos": np_(robot.data.root_link_pos_w[0] - origin),
        "root_quat": np_(robot.data.root_link_quat_w[0]),
        "root_lin_vel": np_(robot.data.root_link_lin_vel_w[0]),
        "root_ang_vel": np_(robot.data.root_link_ang_vel_w[0]),
        "joint_pos": np_(robot.data.joint_pos[0]),
        "joint_vel": np_(robot.data.joint_vel[0]),
        "body_pos": np_(robot.data.body_link_pos_w[0] - origin),
        "body_quat": np_(robot.data.body_link_quat_w[0]),
    }
    for name, cfg in zip(om.active_terms["policy"], om._group_obs_term_cfgs["policy"]):
        out[f"term/{name}"] = np_(cfg.func(uenv, **cfg.params)[0])
    # action -> joint target mapping at t=0 for a fixed random action (term buffers only; env.step re-processes)
    term = uenv.action_manager.get_term("joint_pos")
    probe = torch.randn(env.num_envs, env.num_actions, generator=torch.Generator().manual_seed(0)).to(uenv.device)
    term.process_actions(probe)
    out["probe_action"] = np_(probe[0])
    out["probe_target"] = np_(term.processed_actions[0])
    out["action_term"] = np.array(type(term).__name__)
    # zero-action rollout (robot holds the default pose; diverges from MuJoCo only through physics)
    seq_obs, seq_q, seq_root = [out["obs"]], [out["joint_pos"]], [np.concatenate([out["root_pos"], out["root_quat"]])]
    seq_target = []  # joint position target applied during zero-action step j (reference frame j)
    zero = torch.zeros(env.num_envs, env.num_actions, device=uenv.device)
    for _ in range(steps):
        obs, _, dones, _ = env.step(zero)
        seq_target.append(np_(robot.data.joint_pos_target[0]))
        if bool(dones[0]):  # env 0 terminated and was reset -> later samples are a new episode
            break
        seq_obs.append(np_(obs["policy"][0]))
        seq_q.append(np_(robot.data.joint_pos[0]))
        seq_root.append(np.concatenate([np_(robot.data.root_link_pos_w[0] - origin), np_(robot.data.root_link_quat_w[0])]))
    out["zero_action_obs"] = np.stack(seq_obs)
    out["zero_action_joint_pos"] = np.stack(seq_q)
    out["zero_action_root_pose"] = np.stack(seq_root)
    out["zero_action_target"] = np.stack(seq_target) if seq_target else np.zeros((0, env.num_actions))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **out)
    print(f"[INFO] wrote observation dump {path} (obs dim {out['obs'].shape[0]}, "
          f"{len(seq_obs) - 1} zero-action steps before termination/limit)")


def main():
    ckpt = resolve_checkpoint()
    run_dir = os.path.dirname(ckpt)
    train_list = os.path.join(run_dir, "params", "motions.txt")
    training_files = resolve_motion_files(train_list) if os.path.isfile(train_list) else None
    if args_cli.motion:
        motion_files = resolve_motion_files(args_cli.motion)
    elif training_files:
        motion_files = training_files
    else:
        motion_files = resolve_motion_files(str(PROJECT_ROOT / "data" / "sample" / "motions" / "test_sine.npz"))

    env_cfg = ext.load_entry_point(args_cli.task, "env_cfg_entry_point")
    agent_cfg = ext.load_entry_point(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.commands.motion.motion_file = motion_files
    overrides = _training_overrides(run_dir)
    env_cfg.commands.motion.motion_z_offset = (
        args_cli.motion_z_offset if args_cli.motion_z_offset is not None else overrides.get("motion_z_offset", 0.0)
    )
    # action mode / scale must match training (they are part of the policy interface)
    configure_action(env_cfg, overrides.get("action_mode", "absolute"), overrides.get("action_scale"))
    env_cfg.scene.num_envs = args_cli.num_envs or len(motion_files)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.log_dir = run_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped
    cmd = uenv.command_manager.get_term("motion")
    print(f"[INFO] checkpoint={ckpt}\n[INFO] {cmd}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=uenv.device)

    export_paths = None
    if not args_cli.no_export:
        policy_nn, normalizer = get_policy_and_normalizer(runner)
        # policy.json keeps the run's training clips even when evaluating other clips (--motion)
        extra = {"training_motion_files": project_relative(training_files)} if training_files else None
        export_paths = export_policy(policy_nn, normalizer, uenv, os.path.join(run_dir, "exported"),
                                     task_id=args_cli.task.replace("-Play", ""), checkpoint=ckpt, extra=extra)
        print(f"[INFO] exported: {export_paths}")

    # NOTE: everything that steps the env runs under inference_mode (buffers created inside it cannot be
    # modified in-place outside of it afterwards).
    with torch.inference_mode():
        if args_cli.dump_obs:
            dump_first_obs(env, policy, args_cli.dump_obs, args_cli.dump_steps)
            env.reset()
        summary, step = evaluate(env, cmd, policy)

    result = {
        "simulator": "isaaclab",
        "task": args_cli.task,
        "checkpoint": ckpt,
        "motion_files": motion_files,
        "num_envs": uenv.num_envs,
        "policy_steps": step,
        "metric_definitions": "see igris_pipeline/isaaclab_ext/metrics.py",
        "export": export_paths,
        **summary,
    }
    out = args_cli.output or os.path.join(run_dir, "eval", f"metrics_{Path(ckpt).stem}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_json(out, result)
    print("[RESULT] " + ", ".join(f"{k}={summary[k]:.4f}" for k in ("success_rate", "mpkpe_global", "mpkpe_root_relative",
                                                                    "joint_pos_error", "joint_vel_error", "ee_pos_error",
                                                                    "anchor_pos_error", "anchor_drift_xy_final",
                                                                    "completion_ratio")))
    print(f"[RESULT] failure_reasons={summary['failure_reasons']} unfinished={summary['unfinished_episodes']}")
    print(f"[INFO] wrote {out}")
    env.close()


def evaluate(env, cmd, policy):
    """Run every env once through its clip; returns (summary dict, number of policy steps)."""
    uenv = env.unwrapped
    obs = env.get_observations()
    N = uenv.num_envs
    names = list(cmd.motion.motion_names)
    names = [n if names.count(n) == 1 else f"{n}#{i}" for i, n in enumerate(names)]
    lengths = cmd.motion.lengths.cpu().numpy()
    segments = {m: segs for m, f in enumerate(cmd.motion.files) if (segs := load_segments(f, cmd.motion.fps))}
    acc = EpisodeAccumulator(N, cmd.motion_ids.cpu().numpy(), names, lengths.tolist(), segments=segments)
    tm = uenv.termination_manager
    term_names = [t for t in tm.active_terms if t != "time_out"]
    max_steps = args_cli.max_steps or int(lengths.max()) + 5
    step = 0
    while not acc.done.all() and step < max_steps:
        k = cmd.time_steps.cpu().numpy().copy()
        mids = cmd.motion_ids.cpu().numpy().copy()
        ref = snapshot_reference(cmd)
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        dones = dones.bool().cpu().numpy()
        # envs that were reset inside env.step carry the new episode's state -> not scored
        active = ~dones
        acc.add(step_metrics(ref, snapshot_robot(cmd)), active, frames=k)
        reasons = [""] * N
        if dones.any():
            per_term = {t: tm.get_term(t).cpu().numpy() for t in term_names}
            timeouts = tm.time_outs.cpu().numpy()
            for i in np.where(dones)[0]:
                hit = [t for t in term_names if per_term[t][i]]
                reasons[i] = "+".join(hit) if hit else ("time_out" if timeouts[i] else "unknown")
            acc.finish(dones, success=False, reasons=reasons)
        reached = active & (k == lengths[mids] - 1)
        acc.finish(reached, success=True)
        step += 1
    return acc.summary(), step


if __name__ == "__main__":
    main()
    simulation_app.close()
