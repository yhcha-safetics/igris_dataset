"""3. IGRIS-C 리타겟팅 + 내보내기 (tracking npz / GMR canonical pkl / csv).

    python scripts/05_retarget.py data/s01
"""
import argparse
import json

import _common
from igris_pipeline.pipeline import stage_retarget

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--fps", type=float, default=50.0, help="export rate of the tracking npz")
a = ap.parse_args()
for p in _common.episode_paths(a.paths):
    d = stage_retarget(p, export_fps=a.fps)
    print(f"{p}: {d['frames']} frames in {d['solve_time_s']} s")
    print("   segment direction error (deg):", json.dumps(d["segment_dir_err_deg"]))
    print(f"   foot skate {d.get('foot_skate_mps_in_contact')} m/s, joint-limit hits {d['joint_limit_ratio']}")
