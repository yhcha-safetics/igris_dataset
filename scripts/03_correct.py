"""1-2. 트래커 구성별 캘리브레이션 / 정확도 보정 (proc/* 생성).

    python scripts/03_correct.py data/s01                 # 세션 전체
    python scripts/03_correct.py data/s01/raw/ep_0000.h5 --profile body_3trk --subject-height 1.72
"""
import argparse
import json

import _common
from igris_pipeline.pipeline import stage_correct

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--profile", default=None, help="override the recorded tracker profile")
ap.add_argument("--subject-height", type=float, default=None)
a = ap.parse_args()
for p in _common.episode_paths(a.paths):
    r = stage_correct(p, a.profile, a.subject_height)
    cal = r["calibration"]
    print(f"{p}: profile={cal['profile']} stature={cal['stature']:.2f} floor={cal['floor_z']:.3f}")
    for st in r["report"]["steps"]:
        print("   ", json.dumps(st, ensure_ascii=False)[:200])
