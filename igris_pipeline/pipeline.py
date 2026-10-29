"""Stage drivers operating on episode files, and a session-level runner.

Session layout::

    data/<session>/raw/ep_0000.h5        recorded episodes; later stages add groups in place
    data/<session>/motions/ep_0000.{npz,pkl,csv}   IGRIS-C motions (tracking npz / GMR pkl / csv)
    data/<session>/reports/ep_0000_*.json|png     correction, labels, retarget, MuJoCo reports
    data/<session>/manifest.json         one row per episode (profile, duration, segments, verdict)
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from .episode import Episode, _decode, list_episodes

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs"


def _yaml(name):
    with open(CFG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _reports_dir(ep_path: Path) -> Path:
    d = ep_path.parent.parent / "reports" if ep_path.parent.name == "raw" else ep_path.parent / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _motions_dir(ep_path: Path) -> Path:
    d = ep_path.parent.parent / "motions" if ep_path.parent.name == "raw" else ep_path.parent / "motions"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -- stage 1-2 ------------------------------------------------------------------------
def stage_correct(ep_path, profile=None, subject_height=None, profiles_cfg=None) -> dict:
    from .pico.correction import apply_correction

    ep_path = Path(ep_path)
    ep = Episode.load(ep_path)
    res = apply_correction(ep, profiles_cfg or _yaml("tracker_profiles.yaml"), profile, subject_height)
    ep.save(ep_path)
    rep = {"calibration": res["calibration"], "report": res["report"]}
    (_reports_dir(ep_path) / f"{ep_path.stem}_correction.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


# -- stage 2 ----------------------------------------------------------------------------
def stage_label(ep_path, rules=None, learned_path=None, plot=True, overwrite_reviewed=False) -> list[dict]:
    """Auto-label an episode. Human-reviewed labels are kept (and returned) unless overwrite_reviewed."""
    from .labeling.auto_label import LearnedLabeler, label_motion, write_labels

    ep_path = Path(ep_path)
    ep = Episode.load(ep_path)
    if ep.group_attrs.get("labels", {}).get("reviewed") and not overwrite_reviewed:
        return json.loads(ep.group_attrs["labels"].get("segments", "[]"))
    rules = rules or _yaml("labels.yaml")
    learned = LearnedLabeler.load(learned_path) if learned_path else None
    proc = ep.group_attrs.get("proc", {})
    fps = float(proc.get("fps", ep.fps))
    axes = tuple(json.loads(proc.get("observed_axes", '["loco","lower","upper"]')))
    cal = json.loads(proc["calibration"]) if "calibration" in proc else {}
    pos, _ = ep.body("proc")
    out = label_motion(pos, fps, rules, learned=learned, stature=cal.get("stature"), valid_axes=axes)
    write_labels(ep, out)
    ep.save(ep_path)
    rd = _reports_dir(ep_path)
    (rd / f"{ep_path.stem}_labels.json").write_text(json.dumps(out["segments"], ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    if plot:
        from .labeling.review import plot_timeline
        plot_timeline(out, fps, rd / f"{ep_path.stem}_labels.png", title=ep_path.stem)
    return out["segments"]


# -- stage 3 ----------------------------------------------------------------------------
def stage_retarget(ep_path, retarget_cfg=None, export_fps=50.0, min_part_s=2.0) -> dict:
    from .retarget import export as X
    from .retarget.retargeter import Retargeter

    ep_path = Path(ep_path)
    ep = Episode.load(ep_path)
    proc = ep.group_attrs["proc"]
    fps = float(proc["fps"])
    cal = json.loads(proc["calibration"])
    r = Retargeter(retarget_cfg or _yaml("retarget_igris_c.yaml"))
    res = r.retarget(ep["proc/body/pos"], ep["proc/body/quat"], fps, ep["proc/body/confidence"],
                     ep["proc/contact"], stature=cal.get("stature"))
    X.store_in_episode(ep, res)
    ep.save(ep_path)
    md = _motions_dir(ep_path)
    for old in list(md.glob(f"{ep_path.stem}.*")) + list(md.glob(f"{ep_path.stem}_p[0-9]*.*")):
        old.unlink()  # a re-run may split differently; stale parts would be picked up by folder/glob loaders
    segs = json.loads(ep.group_attrs.get("labels", {}).get("segments", "[]"))
    valid = ep["proc/valid"].astype(bool) if "proc/valid" in ep else np.ones(len(res.qpos), bool)
    parts = motion_parts(valid, fps, min_part_s)
    exported = []
    for k, (a, b) in enumerate(parts):
        name = ep_path.stem if len(parts) == 1 and (a, b) == (0, len(valid)) else f"{ep_path.stem}_p{k}"
        q = res.qpos[a:b]
        meta = {"episode": str(ep_path), "profile": proc.get("profile"), "source_fps": fps,
                "source_frames": [int(a), int(b)], "retarget": res.diagnostics.get("scaling")}
        # segment indices at the source fps, re-based to the part (load_segments rescales via source_fps)
        X.to_tracking_npz(md / f"{name}.npz", r.robot, q, fps, export_fps,
                          meta={**meta, "segments": _segments_in(segs, a, b, fps)})
        X.to_canonical_pkl(md / f"{name}.pkl", r.robot, q, fps, meta={**meta, "segments": _segments_in(segs, a, b, fps)})
        X.to_csv(md / f"{name}.csv", X.resample_qpos(q, fps, 30.0))
        exported.append({"name": name, "t_start": round(a / fps, 2), "t_end": round(b / fps, 2)})
    res.diagnostics["motion_parts"] = exported
    res.diagnostics["invalid_s"] = round(float((~valid).sum() / fps), 2)
    (_reports_dir(ep_path) / f"{ep_path.stem}_retarget.json").write_text(
        json.dumps(res.diagnostics, ensure_ascii=False, indent=1), encoding="utf-8")
    return res.diagnostics


def motion_parts(valid: np.ndarray, fps: float, min_part_s: float = 2.0) -> list[tuple[int, int]]:
    """Frame ranges of the motion that are backed by tracking data.

    Correction bridges gaps up to max_gap_s; longer dropouts stay invalid there and are only
    interpolated, so they are cut out of the exported clips instead of becoming training data.
    """
    v = np.concatenate([[False], np.asarray(valid, bool), [False]])
    edges = np.flatnonzero(np.diff(v.astype(np.int8)))
    runs = list(zip(edges[0::2], edges[1::2]))
    return [(int(a), int(b)) for a, b in runs if b - a >= max(2, int(min_part_s * fps))]


def _segments_in(segs: list[dict], a: int, b: int, fps: float) -> list[dict]:
    """Label segments clipped to source frames [a, b) and re-based to a. Indices stay at the source
    fps (meta.source_fps), which is how isaaclab_ext.motion.load_segments reads them."""
    out = []
    for s in segs:
        s0, s1 = max(int(s["start"]), a), min(int(s["end"]), b)
        if s1 > s0:
            out.append(dict(s, start=s0 - a, end=s1 - a, t_start=round((s0 - a) / fps, 3),
                            t_end=round((s1 - a) / fps, 3)))
    return out


# -- stage 4 (MuJoCo feasibility) --------------------------------------------------------
def stage_eval_mujoco(ep_path, per_segment=True, harness_free=False) -> dict:
    from .sim.mujoco_eval import evaluate_clip

    ep_path = Path(ep_path)
    ep = Episode.load(ep_path)
    q = ep["retarget/igris_c/qpos"].astype(np.float64)
    fps = float(ep.group_attrs["retarget/igris_c"]["fps"])
    segs = json.loads(ep.group_attrs.get("labels", {}).get("segments", "[]")) if per_segment else None
    rep = evaluate_clip(q, fps, harness_free=harness_free, segments=segs)
    if "proc/valid" in ep and not ep["proc/valid"].all():
        invalid_s = float((~ep["proc/valid"].astype(bool)).sum() / fps)
        rep["verdict"]["flags"].append(f"tracking dropouts: {invalid_s:.1f} s interpolated (cut from motions/)")
    (_reports_dir(ep_path) / f"{ep_path.stem}_mujoco.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def run_session(session_dir, stages=("correct", "label", "retarget", "eval"), profile=None, subject_height=None,
                learned_path=None, log=print) -> dict:
    session_dir = Path(session_dir)
    raw = session_dir / "raw" if (session_dir / "raw").exists() else session_dir
    rows = []
    for p in list_episodes(raw):
        row = {"episode": p.name}
        log(f"== {p}")
        if "correct" in stages:
            c = stage_correct(p, profile, subject_height)
            row.update({"profile": c["calibration"]["profile"], "stature": round(c["calibration"]["stature"], 3)})
            log(f"   corrected ({row['profile']})")
        if "label" in stages:
            segs = stage_label(p, learned_path=learned_path)
            row["segments"] = [s["name"] for s in segs]
            row["segments_needing_review"] = sum(s["needs_review"] for s in segs)
            log(f"   labelled: {len(segs)} segments, {row['segments_needing_review']} to review")
        if "retarget" in stages:
            d = stage_retarget(p)
            row["segment_dir_err_deg_mean"] = round(float(np.mean(list(d["segment_dir_err_deg"].values()))), 2)
            row["foot_skate_mps"] = d.get("foot_skate_mps_in_contact")
            row["invalid_s"] = d["invalid_s"]
            row["motion_parts"] = [m["name"] for m in d["motion_parts"]]
            log(f"   retargeted ({d['solve_time_s']} s), exported {row['motion_parts']}")
        if "eval" in stages:
            r = stage_eval_mujoco(p)
            row["com_in_support"] = r["static_balance"]["com_in_support_ratio"]
            row["harness_p95_pct_mg"] = r["pd_replay_harness"]["harness_force_pct_mg"]["p95"]
            row["trackable_guess"] = r["verdict"]["trackable_guess"]
            row["flags"] = r["verdict"]["flags"]
            log(f"   MuJoCo: {r['verdict']}")
        with h5py.File(p, "r") as f:  # only raw/t and two attrs are needed, not a full Episode.load
            t = f["raw/t"]
            row["duration_s"] = round(float(t[-1] - t[0]), 2)
            prof = f["proc"].attrs.get("profile") if "proc" in f else None
            row.setdefault("profile", _decode(prof if prof is not None else f.attrs.get("tracker_profile")))
        rows.append(row)
    manifest = {"session": session_dir.name, "episodes": rows}
    (session_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest
