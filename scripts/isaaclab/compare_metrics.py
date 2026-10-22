"""Side-by-side markdown tables from metrics JSONs written by play_eval.py / eval_mujoco_policy.py.

    .venv\\Scripts\\python.exe scripts\\isaaclab\\compare_metrics.py ^
        --run absolute=logs\\isaaclab\\igris_c_tracking\\<runA> --run residual=logs\\isaaclab\\igris_c_tracking\\<runB> ^
        --clip sim_body2trk_full [--out table.md]

For every run it reads <run>\\eval\\metrics_model_<last>.json (Isaac) and <run>\\eval\\mujoco_metrics.json (MuJoCo).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

KEYS = [
    ("success_rate", "success", "{:.2f}"),
    ("completion_ratio", "completion", "{:.3f}"),
    ("mpkpe_global", "MPKPE [m]", "{:.4f}"),
    ("mpkpe_root_relative", "R-MPKPE [m]", "{:.4f}"),
    ("joint_pos_error", "joint pos [rad]", "{:.4f}"),
    ("joint_vel_error", "joint vel [rad/s]", "{:.3f}"),
    ("ee_pos_error", "EE [m]", "{:.4f}"),
    ("anchor_pos_error", "anchor [m]", "{:.4f}"),
]


def _isaac_json(run: str) -> str:
    files = glob.glob(os.path.join(run, "eval", "metrics_model_*.json"))
    if not files:
        raise FileNotFoundError(f"no Isaac metrics in {run}/eval")
    return max(files, key=lambda p: int(re.findall(r"metrics_model_(\d+)\.json", p)[0]))


def _fmt(v, f):
    return "–" if v is None else f.format(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True, help="label=run_dir")
    p.add_argument("--clip", default=None, help="clip name for the per-clip / per-segment tables")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    runs = [r.split("=", 1) for r in args.run]
    data = {}
    for label, run in runs:
        data[label] = {
            "isaac": json.load(open(_isaac_json(run), encoding="utf-8")),
            "mujoco": json.load(open(os.path.join(run, "eval", "mujoco_metrics.json"), encoding="utf-8")),
        }
    lines = []
    cols = [f"{lab} {sim}" for lab, _ in runs for sim in ("Isaac", "MuJoCo")]
    for scope in ["all clips"] + ([args.clip] if args.clip else []):
        lines += [f"\n**{scope}**\n", "| metric | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for key, name, f in KEYS:
            row = []
            for lab, _ in runs:
                for sim in ("isaac", "mujoco"):
                    d = data[lab][sim] if scope == "all clips" else data[lab][sim]["per_motion"].get(scope, {})
                    row.append(_fmt(d.get(key), f))
            lines.append(f"| {name} | " + " | ".join(row) + " |")
    if args.clip:
        segs0 = data[runs[0][0]]["isaac"]["per_motion"][args.clip].get("segments", [])
        seg_cols = [f"{lab} {sim}" for lab, _ in runs for sim in ("Isaac", "MuJoCo")]
        lines += [f"\n**{args.clip}: per segment** (tracked fraction of the segment's frames / MPKPE m; F = failure here)\n",
                  "| # | segment | frames | " + " | ".join(seg_cols) + " |", "|---" * (len(seg_cols) + 3) + "|"]
        for i, s in enumerate(segs0):
            row = []
            for lab, _ in runs:
                for sim in ("isaac", "mujoco"):
                    ss = data[lab][sim]["per_motion"][args.clip]["segments"][i]
                    cell = f"{ss['tracked_ratio']:.2f}"
                    if ss.get("mpkpe_global") is not None:
                        cell += f" / {ss['mpkpe_global']:.3f}"
                    if ss.get("failures_in_segment"):
                        cell += " F"
                    row.append(cell)
            lines.append(f"| {i} | {s['name']} | {s['start']}-{s['end']} | " + " | ".join(row) + " |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
