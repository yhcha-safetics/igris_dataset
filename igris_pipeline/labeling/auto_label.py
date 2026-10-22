"""Automatic posture / action labelling (stage 2).

Pipeline per episode:
  1. interpretable features (features.py)
  2. rule scores per axis (configs/labels.yaml), fuzzy AND of soft thresholds
  3. optional learned classifier (trained on reviewed / synthetic labels) blended in
  4. Viterbi smoothing with a switch penalty + minimum-duration merge
  5. composite segments ("walk+wave_right") with confidence and a review flag
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .features import compute_features, feature_matrix

AXES = ("loco", "lower", "upper")


def load_rules(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def axis_label_names(axis_cfg: dict) -> list[str]:
    return [axis_cfg["default"]] + list(axis_cfg["labels"].keys())


def rule_scores(feats: dict[str, np.ndarray], axis_cfg: dict) -> np.ndarray:
    """-> (T, K) scores in [0,1], column 0 is the default label."""
    T = next(iter(feats.values())).shape[0]
    names = axis_label_names(axis_cfg)
    S = np.zeros((T, len(names)))
    S[:, 0] = axis_cfg.get("default_score", 0.5)
    for k, name in enumerate(names[1:], start=1):
        s = np.ones(T)
        for feat, op, thr, soft in axis_cfg["labels"][name]:
            x = feats[feat]
            z = (x - thr) / soft if op == ">" else (thr - x) / soft
            s = np.minimum(s, _sigmoid(z))
        S[:, k] = s
    return S


def rule_posterior(S: np.ndarray, floor: float = 0.02) -> np.ndarray:
    """Turn raw rule scores into a per-frame distribution.

    The default label's evidence is "no other rule fired": 1 - max(other scores).
    """
    P = S.copy()
    if S.shape[1] > 1:
        P[:, 0] = 1.0 - S[:, 1:].max(axis=1)
    P = np.clip(P, floor, None)
    return P / P.sum(1, keepdims=True)


def viterbi(scores: np.ndarray, switch_penalty: float) -> np.ndarray:
    """Most likely label path given per-frame scores and a constant switch cost."""
    T, K = scores.shape
    logp = np.log(np.clip(scores / scores.sum(1, keepdims=True), 1e-9, None))
    dp = logp[0].copy()
    back = np.zeros((T, K), dtype=np.int32)
    trans = -switch_penalty * (1.0 - np.eye(K))
    for t in range(1, T):
        cand = dp[:, None] + trans  # from i to j
        back[t] = np.argmax(cand, axis=0)
        dp = cand[back[t], np.arange(K)] + logp[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """[(start, end_exclusive, label)]"""
    out = []
    s = 0
    for t in range(1, len(labels) + 1):
        if t == len(labels) or labels[t] != labels[s]:
            out.append((s, t, int(labels[s])))
            s = t
    return out


def merge_short(labels: np.ndarray, scores: np.ndarray, min_len: int) -> np.ndarray:
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        r = runs(labels)
        if len(r) <= 1:
            break
        for i, (s, e, lab) in enumerate(r):
            if e - s >= min_len:
                continue
            cands = []
            if i > 0:
                cands.append(r[i - 1][2])
            if i + 1 < len(r):
                cands.append(r[i + 1][2])
            best = max(cands, key=lambda c: scores[s:e, c].mean())
            labels[s:e] = best
            changed = True
            break
    return labels


@dataclass
class AxisResult:
    names: list[str]
    frame: np.ndarray        # (T,) int
    scores: np.ndarray       # (T,K)


class LearnedLabeler:
    """Per-axis gradient-boosting classifier on windowed features.

    Trained from episodes whose labels were reviewed by a human (or from synthetic
    ground truth); its probabilities are blended with the rule scores.
    """

    def __init__(self):
        self.models: dict[str, object] = {}
        self.classes: dict[str, list[str]] = {}

    def fit(self, X: np.ndarray, Y: dict[str, list[str]]):
        from sklearn.ensemble import HistGradientBoostingClassifier

        for axis, y in Y.items():
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31)
            clf.fit(X, np.asarray(y))
            self.models[axis] = clf
            self.classes[axis] = list(clf.classes_)
        return self

    def proba(self, axis: str, X: np.ndarray, names: list[str]) -> np.ndarray | None:
        if axis not in self.models:
            return None
        p = self.models[axis].predict_proba(X)
        out = np.full((X.shape[0], len(names)), 1e-3)
        for j, c in enumerate(self.classes[axis]):
            if c in names:
                out[:, names.index(c)] = p[:, j]
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"models": self.models, "classes": self.classes}, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj.models, obj.classes = d["models"], d["classes"]
        return obj


def label_motion(pos: np.ndarray, fps: float, rules: dict, learned: LearnedLabeler | None = None,
                 learned_weight: float = 0.5, stature: float | None = None,
                 valid_axes: tuple[str, ...] = AXES) -> dict:
    feats = compute_features(pos, fps, stature=stature)
    X = feature_matrix(feats, fps) if learned is not None else None
    results: dict[str, AxisResult] = {}
    for axis in AXES:
        cfg = rules["axes"][axis]
        names = axis_label_names(cfg)
        S = rule_posterior(rule_scores(feats, cfg))
        if axis not in valid_axes:  # e.g. lower body unobserved with a 3-point setup
            S = np.zeros_like(S)
            S[:, 0] = 1.0
        elif learned is not None:
            P = learned.proba(axis, X, names)
            if P is not None:
                S = (1 - learned_weight) * S + learned_weight * P / P.sum(1, keepdims=True)
        path = viterbi(S, cfg.get("switch_penalty", 6.0))
        path = merge_short(path, S, int(cfg.get("min_duration", 0.4) * fps))
        results[axis] = AxisResult(names, path, S)
    snap_boundaries(results, int(rules.get("min_composite_duration", 0.3) * fps))
    segments = composite_segments(results, fps, rules.get("review_threshold", 0.65))
    return {"axes": results, "segments": segments, "features": feats}


def _composite_runs(results: dict[str, AxisResult]) -> list[tuple[int, int]]:
    T = results[AXES[0]].frame.shape[0]
    key = np.stack([results[a].frame for a in AXES], axis=-1)
    change = np.any(np.diff(key, axis=0) != 0, axis=-1)
    bounds = [0] + list(np.nonzero(change)[0] + 1) + [T]
    return list(zip(bounds[:-1], bounds[1:]))


def snap_boundaries(results: dict[str, AxisResult], min_len: int) -> None:
    """Remove sliver segments created when boundaries of different axes are a few frames apart."""
    for _ in range(1000):
        segs = _composite_runs(results)
        short = [i for i, (s, e) in enumerate(segs) if e - s < min_len]
        if not short or len(segs) == 1:
            return
        i = short[0]
        s, e = segs[i]
        nb = segs[i - 1] if i > 0 else segs[i + 1]
        if 0 < i < len(segs) - 1 and (segs[i + 1][1] - segs[i + 1][0]) > (nb[1] - nb[0]):
            nb = segs[i + 1]
        for a in AXES:
            results[a].frame[s:e] = results[a].frame[nb[0]]


def composite_segments(results: dict[str, AxisResult], fps: float, review_thr: float) -> list[dict]:
    segs = []
    for s, e in _composite_runs(results):
        parts = {}
        conf = []
        for a in AXES:
            r = results[a]
            lab = int(r.frame[s])
            parts[a] = r.names[lab]
            conf.append(float(r.scores[s:e, lab].mean()))
        tags = [parts["loco"]] if parts["loco"] != results["loco"].names[0] else []
        tags += [parts["lower"]] if parts["lower"] != results["lower"].names[0] else []
        tags += [parts["upper"]] if parts["upper"] != results["upper"].names[0] else []
        score = float(np.min(conf))
        segs.append({
            "start": int(s), "end": int(e), "t_start": s / fps, "t_end": e / fps,
            "loco": parts["loco"], "lower": parts["lower"], "upper": parts["upper"],
            "name": "+".join(tags) if tags else "idle",
            "confidence": score, "needs_review": bool(score < review_thr),
        })
    return segs


def write_labels(ep, out: dict) -> None:
    """Store labelling results into an Episode (labels/* group)."""
    for axis, r in out["axes"].items():
        ep[f"labels/{axis}"] = r.frame.astype(np.int16)
        ep.group_attrs.setdefault("labels", {})[f"{axis}_names"] = json.dumps(r.names)
    ep.group_attrs.setdefault("labels", {})["segments"] = json.dumps(out["segments"], ensure_ascii=False)


def read_labels(ep) -> dict[str, list[str]]:
    ga = ep.group_attrs.get("labels", {})
    out = {}
    for axis in AXES:
        names = json.loads(ga[f"{axis}_names"])
        out[axis] = [names[i] for i in ep[f"labels/{axis}"]]
    return out


def evaluate(pred: dict[str, AxisResult], gt: dict[str, list[str]], fps: float | None = None,
             tolerance_s: float = 0.0) -> dict:
    """Frame accuracy and macro-F1 per axis against ground-truth string labels.

    `tolerance_s` ignores frames within that distance of a ground-truth boundary, since
    a posture takes a moment to form after the instructed segment starts.
    """
    from sklearn.metrics import accuracy_score, f1_score

    rep = {}
    for axis in AXES:
        r = pred[axis]
        p = np.array([r.names[i] for i in r.frame])
        g = np.array(gt[axis])
        keep = np.ones(len(g), dtype=bool)
        if tolerance_s > 0 and fps:
            n = int(tolerance_s * fps)
            for b in np.nonzero(g[1:] != g[:-1])[0] + 1:
                keep[max(0, b - n): b + n] = False
        rep[axis] = {"acc": float(accuracy_score(g[keep], p[keep])),
                     "macro_f1": float(f1_score(g[keep], p[keep], average="macro", zero_division=0))}
    return rep
