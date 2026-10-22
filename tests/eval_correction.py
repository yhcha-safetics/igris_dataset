"""Accuracy of stage 1-2 per tracker profile on simulated PICO streams.

    python tests/eval_correction.py [out_dir]

Simulated stream -> recorder -> correction, compared against the synthetic ground truth.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igris_pipeline.episode import Episode  # noqa: E402
from igris_pipeline.pico.correction import apply_correction  # noqa: E402
from igris_pipeline.pico.recorder import record  # noqa: E402
from igris_pipeline.pico.simulate import SimulatedXRTSource  # noqa: E402
from igris_pipeline.pico.synthetic import generate  # noqa: E402
from igris_pipeline.skeleton import GROUPS, JOINT_GROUP  # noqa: E402
from igris_pipeline.utils.rotations import quat_angle_between, quat_mul, yaw_quat  # noqa: E402

PROFILES = ["body_2trk", "body_3trk", "sparse_trackers", "hmd_3pt"]


def align_yaw_translation(src, dst):
    """Best rigid (yaw about Z + translation) fit of src onto dst. (N,3) arrays -> transformed src (any shape)."""
    a, b = src.reshape(-1, 3), dst.reshape(-1, 3)
    ma, mb = a.mean(0), b.mean(0)
    A, B = a[:, :2] - ma[:2], b[:, :2] - mb[:2]
    H = A.T @ B
    yaw = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s_ = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    return (src - ma) @ Rz.T + mb, yaw


def run(out_root: Path, verbose=True) -> dict:
    cfg = yaml.safe_load(open(ROOT / "configs" / "tracker_profiles.yaml", encoding="utf-8"))
    clip = generate(fps=72.0, seed=1)
    summary = {}
    for prof in PROFILES:
        src = SimulatedXRTSource(prof, clip=clip, seed=2)
        paths = record(src, out_root / prof, {"tracker_profile": prof, "subject_height": 1.72},
                       use_buttons=False, on_status=lambda *a: None)
        ep = Episode.load(paths[0])
        t0 = time.time()
        res = apply_correction(ep, cfg, prof, subject_height=1.72)
        dt = time.time() - t0
        a = res["arrays"]
        tn = a["proc/t"]
        idx = np.clip(np.round(tn * clip.fps).astype(int), 0, len(clip.t) - 1)
        gt, gtq = clip.pos[idx], clip.quat[idx]
        w0, w1 = res["calibration"]["still_window"]
        off = gt[w0:w1, 0].mean(0)
        off[2] = 0
        gt = gt - off
        # calibration re-orients the world (face +X at start); undo that arbitrary rigid
        # transform before scoring, as is standard for MPJPE-style metrics
        est, yaw_fit = align_yaw_translation(a["proc/body/pos"], gt)
        err = np.linalg.norm(est - gt, axis=-1)
        rel_err = np.linalg.norm((est - est[:, :1]) - (gt - gt[:, :1]), axis=-1)
        q_fit = yaw_quat(np.array(yaw_fit))
        oerr = np.degrees(quat_angle_between(quat_mul(np.broadcast_to(q_fit, a["proc/body/quat"].shape),
                                                      a["proc/body/quat"]), gtq))
        raw_err = None
        if prof.startswith("body"):
            traw = ep["raw/t"] - ep["raw/t"][0]
            ridx = np.clip(np.searchsorted(traw, tn), 0, len(traw) - 1)
            rp, _ = align_yaw_translation(ep["raw/body/pos"][ridx], gt)
            raw_err = np.linalg.norm(rp - gt, axis=-1)
        prof_sum = {}
        if verbose:
            cal = res["calibration"]
            print(f"\n=== {prof}: {len(tn)} frames, corrected in {dt:.2f}s, floor {cal['floor_z']:.3f}, "
                  f"yaw {cal['heading_yaw_deg']:.1f}, stature {cal['stature']:.2f}")
        for g in GROUPS:
            js = [j for j in range(24) if JOINT_GROUP[j] == g]
            d = {"pos_cm": float(np.median(err[:, js]) * 100), "pos_p95_cm": float(np.percentile(err[:, js], 95) * 100),
                 "rel_cm": float(np.median(rel_err[:, js]) * 100),
                 "ori_deg": float(np.median(oerr[:, js])),
                 "conf": float(a["proc/body/confidence"][:, js].mean())}
            if raw_err is not None:
                d["raw_pos_cm"] = float(np.median(raw_err[:, js]) * 100)
                d["raw_pos_p95_cm"] = float(np.percentile(raw_err[:, js], 95) * 100)
            prof_sum[g] = d
            if verbose:
                line = (f"  {g:7s} pos {d['pos_cm']:5.1f} cm (p95 {d['pos_p95_cm']:5.1f}) root-rel {d['rel_cm']:5.1f}"
                        f"  ori {d['ori_deg']:5.1f} deg"
                        f"  conf {d['conf']:.2f}")
                if raw_err is not None:
                    line += f" | raw {d['raw_pos_cm']:5.1f} cm (p95 {d['raw_pos_p95_cm']:5.1f})"
                print(line)
        if verbose:
            print("  steps:", json.dumps(res["report"]["steps"], ensure_ascii=False)[:1200])
        summary[prof] = prof_sum
    return summary


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="igris_corr_"))
    run(out)
