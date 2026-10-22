"""전체 파이프라인: 보정 -> 라벨링 -> 리타겟팅 -> MuJoCo 평가 (+ manifest.json).

    python scripts/run_pipeline.py data/s01
    python scripts/run_pipeline.py data/demo_body2 --simulate body_2trk --subject-height 1.72
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from igris_pipeline.pipeline import run_session

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--simulate", default=None, help="first record a simulated episode with this profile")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--subject-height", type=float, default=None)
ap.add_argument("--profile", default=None)
ap.add_argument("--learned", default=None)
ap.add_argument("--stages", default="correct,label,retarget,eval")
a = ap.parse_args()
sess = Path(a.session)
if a.simulate:
    from igris_pipeline.pico.recorder import record
    from igris_pipeline.pico.simulate import SimulatedXRTSource

    meta = {"tracker_profile": a.simulate, "subject": f"synthetic_{a.seed}"}
    if a.subject_height:
        meta["subject_height"] = a.subject_height
    record(SimulatedXRTSource(a.simulate, seed=a.seed), sess / "raw", meta, use_buttons=False)
m = run_session(sess, tuple(a.stages.split(",")), a.profile, a.subject_height, a.learned)
print(json.dumps(m, ensure_ascii=False, indent=1))
