"""Human-in-the-loop tools for stage 2.

* plot_timeline      PNG with the three label axes, confidence and key features
* export_review_csv  segments -> CSV a person can correct in a spreadsheet
* import_review_csv  corrected CSV -> frame labels in the episode, marked as reviewed
* train_learned      fit the LearnedLabeler on reviewed episodes (+ optional synthetic data)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .auto_label import AXES, AxisResult, LearnedLabeler, axis_label_names, composite_segments, read_labels
from .features import compute_features, feature_matrix


def plot_timeline(out: dict, fps: float, path, title: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = out["features"]
    T = next(iter(feats.values())).shape[0]
    t = np.arange(T) / fps
    fig, axs = plt.subplots(5, 1, figsize=(14, 8), sharex=True,
                            gridspec_kw={"height_ratios": [1, 1, 1, 1.4, 1.4]})
    cmap = plt.get_cmap("tab10")
    for k, axis in enumerate(AXES):
        r = out["axes"][axis]
        ax = axs[k]
        for i, name in enumerate(r.names):
            m = r.frame == i
            if m.any():
                ax.fill_between(t, 0, 1, where=m, color=cmap(i % 10), alpha=0.85, step="mid",
                                label=name, linewidth=0)
        conf = r.scores[np.arange(T), r.frame]
        ax.plot(t, conf, color="k", linewidth=0.8)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(axis, rotation=0, ha="right", va="center")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=7, frameon=False)
    for s in out["segments"]:
        if s["needs_review"]:
            for ax in axs[:3]:
                ax.axvspan(s["t_start"], s["t_end"], facecolor="none", hatch="//", edgecolor="red", linewidth=0)
    axs[3].plot(t, feats["pelvis_rel"], label="pelvis height (rel)")
    axs[3].plot(t, feats["trunk_pitch"] / 90, label="trunk pitch /90deg")
    axs[3].plot(t, feats["knee_flex_mean"] / 120, label="knee flex /120deg")
    axs[3].plot(t, feats["speed"] / 2, label="speed /2 m/s")
    axs[3].legend(loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=7, frameon=False)
    axs[4].plot(t, feats["left_hand_up"], label="L hand up")
    axs[4].plot(t, feats["right_hand_up"], label="R hand up")
    axs[4].plot(t, feats["left_hand_fwd_lp"], label="L hand fwd")
    axs[4].plot(t, feats["right_hand_fwd_lp"], label="R hand fwd")
    axs[4].plot(t, 5 * feats["right_hand_osc"], label="R osc x5", alpha=0.6)
    axs[4].legend(loc="upper left", bbox_to_anchor=(1.0, 1.05), fontsize=7, frameon=False)
    axs[4].set_xlabel("time [s]")
    fig.suptitle(title + "   (black line: label probability, red hatch: needs review)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return Path(path)


def export_review_csv(ep, path):
    segs = json.loads(ep.group_attrs["labels"]["segments"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "loco", "lower", "upper", "confidence", "needs_review", "comment"])
        for s in segs:
            w.writerow([f"{s['t_start']:.2f}", f"{s['t_end']:.2f}", s["loco"], s["lower"], s["upper"],
                        f"{s['confidence']:.2f}", int(s["needs_review"]), ""])
    return Path(path)


def import_review_csv(ep, path, rules: dict):
    """Apply a corrected CSV (time based) to the episode's frame labels."""
    fps = float(ep.group_attrs.get("proc", {}).get("fps", ep.fps))
    T = ep["labels/loco"].shape[0]
    ga = ep.group_attrs.get("labels", {})
    # the stored frame labels index the episode's own name lists (which may already hold labels a
    # reviewer added earlier), so those lists are authoritative; the rules only seed missing ones
    names = {a: (json.loads(ga[f"{a}_names"]) if f"{a}_names" in ga else axis_label_names(rules["axes"][a]))
             for a in AXES}
    frame = {a: ep[f"labels/{a}"].copy() for a in AXES}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            s, e = int(round(float(row["t_start"]) * fps)), int(round(float(row["t_end"]) * fps))
            for a in AXES:
                lab = row[a].strip()
                if lab not in names[a]:  # new label introduced by the reviewer
                    names[a].append(lab)
                frame[a][max(0, s):min(T, e)] = names[a].index(lab)
    for a in AXES:
        ep[f"labels/{a}"] = frame[a].astype(np.int16)
        ep.group_attrs["labels"][f"{a}_names"] = json.dumps(names[a])
    # rebuild the composite segments from the reviewed frame labels (downstream stages and
    # export_review_csv read `segments`); reviewed labels have confidence 1
    results = {a: AxisResult(names[a], frame[a].astype(int), np.eye(len(names[a]))[frame[a]]) for a in AXES}
    ep.group_attrs["labels"]["segments"] = json.dumps(composite_segments(results, fps, 0.0), ensure_ascii=False)
    ep.group_attrs["labels"]["reviewed"] = 1
    return ep


def training_rows(pos, fps, labels: dict[str, list[str]], stature=None):
    feats = compute_features(pos, fps, stature=stature)
    return feature_matrix(feats, fps), labels


def train_learned(episodes: list, synthetic_clips: list | None = None, out_path=None) -> LearnedLabeler:
    """episodes: Episode objects with reviewed labels; synthetic_clips: SyntheticClip objects."""
    Xs, Ys = [], {a: [] for a in AXES}
    for ep in episodes:
        fps = float(ep.group_attrs.get("proc", {}).get("fps", ep.fps))
        pos, _ = ep.body("proc")
        X, lab = training_rows(pos, fps, read_labels(ep))
        Xs.append(X)
        for a in AXES:
            Ys[a] += list(lab[a])
    for c in synthetic_clips or []:
        X, lab = training_rows(c.pos, c.fps, c.labels)
        Xs.append(X)
        for a in AXES:
            Ys[a] += list(lab[a])
    model = LearnedLabeler().fit(np.concatenate(Xs), Ys)
    if out_path:
        model.save(out_path)
    return model
