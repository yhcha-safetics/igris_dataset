"""IGRIS-C motion-tracking extension for Isaac Lab 2.3 (stage 4 of the PICO-VR -> IGRIS-C pipeline).

Importing this package registers the gym ids (string entry points only -> no isaaclab import here):

* ``Igris-C-Tracking-Flat-v0``       training config (noise, domain randomisation, adaptive sampling)
* ``Igris-C-Tracking-Flat-Play-v0``  evaluation config (no noise/randomisation, start at frame 0, hold end)

The package is standalone: it never imports the rest of ``igris_pipeline`` (mujoco / mink).
``constants`` and ``motion`` are importable without Isaac Lab (numpy / torch only).
"""

try:  # gymnasium is present in the isaaclab env; the MuJoCo .venv may not have it.
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None

TASK_ID = "Igris-C-Tracking-Flat-v0"
PLAY_TASK_ID = "Igris-C-Tracking-Flat-Play-v0"

if gym is not None:
    _registered = set(gym.registry.keys())
    if TASK_ID not in _registered:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:IgrisCTrackingEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:IgrisCTrackingPPORunnerCfg",
            },
        )
    if PLAY_TASK_ID not in _registered:
        gym.register(
            id=PLAY_TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:IgrisCTrackingEnvCfg_PLAY",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:IgrisCTrackingPPORunnerCfg",
            },
        )


def load_entry_point(task_id: str, key: str):
    """Instantiate the cfg class registered under ``key`` for ``task_id`` (no hydra)."""
    import importlib

    spec = gym.spec(task_id)
    entry = spec.kwargs[key]
    if isinstance(entry, str):
        mod_name, attr = entry.split(":")
        entry = getattr(importlib.import_module(mod_name), attr)
    return entry() if callable(entry) else entry
