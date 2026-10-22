"""Train the IGRIS-C motion-tracking policy with RSL-RL PPO in Isaac Lab 2.3.

Example (from C:\\Users\\c\\igris_dataset):
    C:\\ProgramData\\Anaconda3\\envs\\isaaclab\\python.exe scripts\\isaaclab\\train.py --headless ^
        --num_envs 4096 --max_iterations 30000 --motion data\\sample\\motions

Logs / checkpoints: logs\\isaaclab\\<experiment_name>\\<YYYY-mm-dd_HH-MM-SS>[_<run_name>]\\model_<it>.pt
At the end the final policy is exported to <run_dir>\\exported\\policy.{onnx,pt,json}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# Kit's shutdown ends the process without flushing Python's buffers -> line-buffer redirected output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Train IGRIS-C motion tracking (RSL-RL PPO).")
parser.add_argument("--task", type=str, default="Igris-C-Tracking-Flat-v0")
parser.add_argument("--motion", type=str, default=str(PROJECT_ROOT / "data" / "sample" / "motions" / "test_sine.npz"),
                    help="NPZ file, directory (recursive), glob, comma list or .txt list")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--log_root", type=str, default=str(PROJECT_ROOT / "logs" / "isaaclab"))
parser.add_argument("--resume", type=str, default=None, help="checkpoint (.pt) to resume from")
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--logger", type=str, default=None, choices=["tensorboard", "wandb", "neptune"])
parser.add_argument("--no_export", action="store_true", help="skip ONNX/JIT/JSON export at the end")
parser.add_argument("--motion_z_offset", type=float, default=0.0,
                    help="constant z shift of the reference (e.g. -0.009 for MJCF-sole clips)")
parser.add_argument("--action_mode", type=str, default="absolute", choices=["absolute", "residual"],
                    help="absolute: q* = q_default + s*a (default) | residual: q* = q_ref(t) + s*a")
parser.add_argument("--action_scale", type=str, default=None, choices=["beyondmimic", "gmt"],
                    help="scale preset (constants.ACTION_SCALE_PRESETS); default: beyondmimic (absolute), gmt=0.25 (residual)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after the app is up."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import igris_pipeline.isaaclab_ext as ext  # noqa: E402  (registers the gym ids)
from igris_pipeline.isaaclab_ext.tracking_env_cfg import configure_action  # noqa: E402
from igris_pipeline.isaaclab_ext.export import export_policy, get_policy_and_normalizer  # noqa: E402
from igris_pipeline.isaaclab_ext.motion import resolve_motion_files  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg = ext.load_entry_point(args_cli.task, "env_cfg_entry_point")
    agent_cfg = ext.load_entry_point(args_cli.task, "rsl_rl_cfg_entry_point")

    motion_files = resolve_motion_files(args_cli.motion)
    env_cfg.commands.motion.motion_file = motion_files
    env_cfg.commands.motion.motion_z_offset = args_cli.motion_z_offset
    configure_action(env_cfg, args_cli.action_mode, args_cli.action_scale)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    agent_cfg.run_name = args_cli.run_name
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    log_root = os.path.join(os.path.abspath(args_cli.log_root), agent_cfg.experiment_name)
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{agent_cfg.run_name}" if agent_cfg.run_name else "")
    log_dir = os.path.join(log_root, run_dir)
    os.makedirs(log_dir, exist_ok=True)
    env_cfg.log_dir = log_dir
    print(f"[INFO] task={args_cli.task} num_envs={env_cfg.scene.num_envs} motions={len(motion_files)} log_dir={log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO] {env.unwrapped.command_manager.get_term('motion')}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.resume:
        print(f"[INFO] resuming from {args_cli.resume}")
        runner.load(args_cli.resume)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "motions.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(motion_files) + "\n")
    # settings play_eval.py must reproduce (read back from params/overrides.json)
    with open(os.path.join(log_dir, "params", "overrides.json"), "w", encoding="utf-8") as f:
        json.dump({"action_mode": args_cli.action_mode,
                   "action_scale": dict(env_cfg.actions.joint_pos.scale),
                   "motion_z_offset": env_cfg.commands.motion.motion_z_offset}, f, indent=2)

    t0 = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] training time: {time.time() - t0:.1f} s")

    if not args_cli.no_export:
        ckpt = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
        policy_nn, normalizer = get_policy_and_normalizer(runner)
        paths = export_policy(policy_nn, normalizer, env.unwrapped, os.path.join(log_dir, "exported"),
                              task_id=args_cli.task, checkpoint=ckpt)
        print(f"[INFO] exported: {paths}")
    print(f"[INFO] RUN_DIR={log_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
