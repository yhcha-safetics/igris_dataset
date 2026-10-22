"""Retarget a corrected simulated episode and report MuJoCo feasibility per synthetic segment.

    python tests/eval_balance.py <episode.h5> [--no-balance]
"""
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igris_pipeline.episode import Episode  # noqa: E402
from igris_pipeline.pico.correction import apply_correction  # noqa: E402
from igris_pipeline.pico.synthetic import DEFAULT_PROGRAM  # noqa: E402
from igris_pipeline.retarget.retargeter import Retargeter  # noqa: E402
from igris_pipeline.sim.mujoco_eval import evaluate_clip, static_balance_report  # noqa: E402

if __name__ == "__main__":
    ep = Episode.load(sys.argv[1])
    if "proc/body/pos" not in ep:
        apply_correction(ep, yaml.safe_load(open(ROOT / "configs/tracker_profiles.yaml", encoding="utf-8")))
    cal = json.loads(ep.group_attrs["proc"]["calibration"])
    cfg = yaml.safe_load(open(ROOT / "configs/retarget_igris_c.yaml", encoding="utf-8"))
    if "--no-balance" in sys.argv:
        cfg = copy.deepcopy(cfg)
        cfg["solver"]["balance"]["enabled"] = False
    r = Retargeter(cfg)
    fps = 50.0
    res = r.retarget(ep["proc/body/pos"], ep["proc/body/quat"], fps, ep["proc/body/confidence"], ep["proc/contact"],
                     stature=cal["stature"])
    q = res.qpos
    print("retarget diagnostics:", json.dumps({k: res.diagnostics[k] for k in
                                                ("segment_dir_err_deg", "foot_skate_mps_in_contact", "solve_time_s")}))
    t = 0.0
    for lo, lw, up, d in DEFAULT_PROGRAM:
        a, b = int((t + 0.5) * fps), int((t + d - 0.3) * fps)
        t += d
        rep = static_balance_report(r.robot, q[a:b])
        print("%-32s com_in %.2f margin %s" % ("/".join([lo, lw, up]), rep["com_in_support_ratio"],
                                               rep["median_support_margin_cm"]))
    rep = evaluate_clip(q, fps, harness_free=False)
    for k in ("static_balance", "pd_replay_harness", "verdict"):
        print(k, json.dumps(rep[k]))
