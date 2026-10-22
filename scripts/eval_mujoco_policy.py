"""CLI: MuJoCo sim2sim evaluation of an Isaac Lab-exported IGRIS-C tracking policy.

    .venv\\Scripts\\python.exe scripts\\eval_mujoco_policy.py ^
        --policy_dir logs\\isaaclab\\igris_c_tracking\\<run>\\exported ^
        --motion data\\sample\\motions\\test_sine.npz

See igris_pipeline/sim/mujoco_policy.py for options (--backend onnx|jit, --foot_geom urdf|mjcf,
--joint_dynamics isaac|mjcf|none, --render, --zero_action, --save_trajectory ...).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from igris_pipeline.sim.mujoco_policy import main  # noqa: E402

if __name__ == "__main__":
    main()
