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
DEFAULT_MOTION = str(PROJECT_ROOT / "data" / "sample" / "motions" / "test_sine.npz")
parser.add_argument("--motion", type=str, default=None,
                    help="NPZ file, directory (recursive), glob, comma list or .txt list "
                         "(default: the resumed run's clips, else data/sample/motions/test_sine.npz)")
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
parser.add_argument("--motion_z_offset", type=float, default=None,
                    help="constant z shift of the reference (e.g. -0.009 for MJCF-sole clips; default 0 / resumed run)")
parser.add_argument("--action_mode", type=str, default=None, choices=["absolute", "residual"],
                    help="absolute: q* = q_default + s*a (default) | residual: q* = q_ref(t) + s*a; "
                         "with --resume the run's mode is kept unless given")
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


def _rel_or_abs(path: str, start: str) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:  # different drive
        return path


def resumed_run_settings(ckpt: str) -> dict:
    """Policy-interface settings of the run a checkpoint belongs to (its params/ folder): resuming with
    a different action mode / scale / reference would silently continue training a different task."""
    params = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "params")
    out = {}
    if os.path.isfile(os.path.join(params, "overrides.json")):
        with open(os.path.join(params, "overrides.json"), encoding="utf-8") as f:
            out.update(json.load(f))
    if os.path.isfile(os.path.join(params, "motions.txt")):
        out["motions"] = os.path.join(params, "motions.txt")
    return out


def main():
    env_cfg = ext.load_entry_point(args_cli.task, "env_cfg_entry_point")
    agent_cfg = ext.load_entry_point(args_cli.task, "rsl_rl_cfg_entry_point")

    prev = resumed_run_settings(args_cli.resume) if args_cli.resume else {}
    if prev:
        print(f"[INFO] resume: run settings {json.dumps({k: v for k, v in prev.items() if k != 'action_scale'})}")
    action_mode = args_cli.action_mode or prev.get("action_mode", "absolute")
    if prev.get("action_mode") and action_mode != prev["action_mode"]:
        print(f"[WARN] resuming a {prev['action_mode']} policy with --action_mode {action_mode}")
    action_scale = args_cli.action_scale or (prev.get("action_scale") if action_mode == prev.get("action_mode") else None)
    motion_files = resolve_motion_files(args_cli.motion or prev.get("motions") or DEFAULT_MOTION)
    env_cfg.commands.motion.motion_file = motion_files
    env_cfg.commands.motion.motion_z_offset = (
        args_cli.motion_z_offset if args_cli.motion_z_offset is not None else prev.get("motion_z_offset", 0.0))
    configure_action(env_cfg, action_mode, action_scale)
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
    params_dir = os.path.join(log_dir, "params")
    with open(os.path.join(params_dir, "motions.txt"), "w", encoding="utf-8") as f:
        # relative to params/ (resolve_motion_files resolves list entries against the list's folder),
        # so a run copied together with data/ to another PC still finds its clips
        f.write("\n".join(_rel_or_abs(p, params_dir) for p in motion_files) + "\n")
    # settings play_eval.py must reproduce (read back from params/overrides.json)
    with open(os.path.join(params_dir, "overrides.json"), "w", encoding="utf-8") as f:
        json.dump({"action_mode": action_mode,
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
