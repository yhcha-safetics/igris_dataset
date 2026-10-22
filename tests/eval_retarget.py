"""Compare retargeting variants on a corrected (stage 1-2) simulated episode.

    python tests/eval_retarget.py <corrected_episode.h5 | raw episode.h5>

Variants: Robros' table (robot ankle <- human toe joint) vs. ankle<-ankle mapping, and
kinematic vs. GMR scaling. Metrics are scale-free (segment direction error, foot skate,
penetration, joint-limit saturation), see Retargeter.diagnose().
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
from igris_pipeline.retarget.retargeter import Retargeter  # noqa: E402


def variants(base):
    """base = configs/retarget_igris_c.yaml (ankle mapping + geometry offsets + kinematic scaling)."""
    out = {"default (ankle map + geometry offsets + kinematic)": copy.deepcopy(base)}
    v = copy.deepcopy(base)
    v["geometry_offsets"] = {}
    out["no geometry offsets"] = v
    v = copy.deepcopy(base)
    v["geometry_offsets"] = {}
    for st in ("stage1", "stage2"):
        for side in ("Left", "Right"):
            v[st][f"Link_Ankle_Roll_{side}"][0] = f"{side.lower()}_foot"
    out["Robros original table (toe map, no geometry offsets)"] = v
    g = copy.deepcopy(base)
    g["scaling"]["mode"] = "gmr"
    out["default but GMR uniform scaling"] = g
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1])
    ep = Episode.load(path)
    if "proc/body/pos" not in ep:
        tp = yaml.safe_load(open(ROOT / "configs/tracker_profiles.yaml", encoding="utf-8"))
        apply_correction(ep, tp)
    cal = json.loads(ep.group_attrs["proc"]["calibration"])
    base = yaml.safe_load(open(ROOT / "configs/retarget_igris_c.yaml", encoding="utf-8"))
    fps = float(ep.group_attrs["proc"]["fps"])
    for name, cfg in variants(base).items():
        r = Retargeter(cfg)
        res = r.retarget(ep["proc/body/pos"], ep["proc/body/quat"], fps, ep["proc/body/confidence"],
                         ep["proc/contact"], stature=cal["stature"])
        d = res.diagnostics
        seg = d["segment_dir_err_deg"]
        print(f"\n## {name}  ({d['solve_time_s']} s, {d['frames']} frames)")
        print("  segment dir err (deg):", seg, " mean %.2f" % (sum(seg.values()) / len(seg)))
        print("  foot skate in contact %.3f m/s | sole height in contact %.2f cm | max penetration %.2f cm"
              % (d.get("foot_skate_mps_in_contact", -1), d.get("sole_height_in_contact_cm", -1), d["max_penetration_cm"]))
        print("  joint limit ratio:", d["joint_limit_ratio"], "| p99 joint vel", d["p99_joint_vel"])
