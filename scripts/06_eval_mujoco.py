"""4. MuJoCo 물리 타당성 평가 (리타겟 모션을 로봇이 추종할 수 있는지).

    python scripts/06_eval_mujoco.py data/s01                       # 세션의 모든 에피소드 (라벨 구간별)
    python scripts/06_eval_mujoco.py data/s01/motions/ep_0000.npz   # 모션 파일 하나
"""
import argparse
import json
from pathlib import Path

import _common
from igris_pipeline.pipeline import stage_eval_mujoco
from igris_pipeline.sim.mujoco_eval import evaluate_clip, load_qpos

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--free", action="store_true", help="also run the harness-free open-loop replay")
a = ap.parse_args()
for arg in a.paths:
    p = Path(arg)
    if p.suffix in (".npz", ".pkl"):
        q, fps = load_qpos(p)
        reps = [(p, evaluate_clip(q, fps, harness_free=a.free))]
    else:
        reps = [(e, stage_eval_mujoco(e, harness_free=a.free)) for e in _common.episode_paths([arg])]
    for path, r in reps:
        print(f"\n## {path}")
        for k in ("kinematic", "static_balance", "pd_replay_harness", "pd_replay_free", "verdict"):
            if k in r:
                print(f"  {k}: {json.dumps(r[k], ensure_ascii=False)}")
        for s in r.get("per_segment", []):
            print(f"    {s['t_start']:6.1f}-{s['t_end']:6.1f}s {s['name']:<26s} CoM-in {s['com_in_support']:.2f} "
                  f"harness {s['harness_mean_pct_mg']:5.1f}%mg  mpkpe {s['mpkpe_cm']:5.1f}cm  "
                  f"sat {s['saturated_joints']}")
