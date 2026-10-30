"""Which objects were interacted with: change detection over the interaction phase of a capture.

  moved     each object's surface points are projected into every interaction frame and compared
            with the measured depth: about the same depth = still there, clearly farther = the camera
            sees through where the object was (it is gone), nearer = something in front (a hand, a
            person: no evidence). An object whose see-through fraction stays above gone_fraction for
            min_gone_s has been moved (taken away, maybe put back): it is an interacted object.
  final     the last final_s seconds are fused into a separate TSDF; geometry farther than
            new_distance from the scan is new. Object-sized new clusters resting on a surface are
            matched to the moved objects by size and colour: where they ended up.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .align import Alignment
from .reconstruct import ScanParams, cloud_arrays, fuse_with_poses, project
from .rgbd import Capture
from .segment import SegmentParams, Segmentation, _components, _footprint, _rgba, nearest


@dataclass
class ChangeParams:
    tolerance: float = 0.03        # m (+ 2 % of the depth): measured depth agrees with the object
    gone_fraction: float = 0.6     # of the informative points seeing through: the object is gone
    min_informative: int = 15      # points per frame to judge an object
    min_gone_s: float = 0.3        # s
    max_points: int = 500          # sampled per object
    frame_step: int = 1
    final_s: float = 1.0           # s at the end that show the final state
    new_distance: float = 0.025    # m from the scan surface: new geometry
    match_cost: float = 0.6        # size (relative) + colour distance for "this is where it went"


def _sample(idx, n, rng):
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)


def detect_moved(capture: Capture, poses: dict, seg: Segmentation, al: Alignment,
                 p: ChangeParams | None = None) -> dict:
    """{object: {"t": [...], "gone": [...], "moved": bool, "t_moved": s | None}} over the frames of
    poses {frame: (camera-to-reconstruction pose, tracked)}."""
    p = p or ChangeParams()
    rng = np.random.default_rng(0)
    Tinv = np.linalg.inv(al.T)
    R = al.T[:3, :3] / al.scale
    objs = seg.objects
    pts = {o.name: _sample(o.index, p.max_points, rng) for o in objs}
    out = {o.name: {"t": [], "gone": []} for o in objs}
    frames = sorted(poses)[::max(1, p.frame_step)]
    for i in frames:
        T, ok = poses[i]
        if not ok:
            continue
        depth = capture.depth(i)
        cam = T[:3, 3]
        for o in objs:
            idx = pts[o.name]
            X = seg.points[idx] @ Tinv[:3, :3].T + Tinv[:3, 3]  # scene -> reconstruction frame
            N = seg.normals[idx] @ R                             # (rotation only)
            front = np.einsum("ij,ij->i", N, cam - X) > 0
            u, v, z, inside = project(X[front], T, capture.intr)
            d = depth[v, u]
            valid = inside & (d > 0)
            tol = p.tolerance + 0.02 * z
            same = valid & (np.abs(d - z) < tol)
            through = valid & (d > z + tol)
            n_inf = int(same.sum() + through.sum())
            out[o.name]["t"].append(float(capture.t[i]))
            out[o.name]["gone"].append(float(through.sum() / n_inf) if n_inf >= p.min_informative else np.nan)
    for name, e in out.items():
        t, g = np.array(e["t"]), np.array(e["gone"])
        e["moved"], e["t_moved"] = False, None
        start = None
        for k in range(len(t)):  # runs of "gone" (frames without evidence neither break nor extend a run)
            if np.isnan(g[k]):
                continue
            if g[k] > p.gone_fraction:
                start = t[k] if start is None else start
                if t[k] - start >= p.min_gone_s:
                    e["moved"], e["t_moved"] = True, float(start)
                    break
            else:
                start = None
    return out


def final_state(capture: Capture, poses: dict, seg: Segmentation, al: Alignment, scan_params: ScanParams,
                seg_params: SegmentParams, p: ChangeParams | None = None) -> list[dict]:
    """New object-sized geometry at the end of the capture: [{"center", "size", "yaw", "rgba", "support"}]."""
    p = p or ChangeParams()
    t_end = max(capture.t[i] for i in poses)
    last = {i: T for i, (T, ok) in poses.items() if ok and capture.t[i] >= t_end - p.final_s}
    if not last:
        return []
    vbg = fuse_with_poses(capture, last, scan_params)
    pts, col, _ = cloud_arrays(vbg, scan_params, weight_threshold=min(3.0, len(last) / 3))
    if len(pts) == 0:
        return []
    P = al.apply(pts)
    new = nearest(P, seg.points) > p.new_distance
    P, col = P[new], col[new]
    surfaces = [s for s in seg.surfaces if s.kind in ("floor", "support")]
    out = []
    for comp in _components(P, seg_params.cluster_eps, 10):
        if len(comp) < seg_params.min_object_points // 2:
            continue
        q = P[comp]
        c2, s2, yaw, _ = _footprint(q[:, :2])
        z0, z1 = float(q[:, 2].min()), float(q[:, 2].max())
        rest = [s for s in surfaces if abs(z0 - s.height) < seg_params.rest_tolerance + seg_params.surface_layer
                and (s.kind == "floor" or s.contains(c2, margin=0.02))]
        if not rest or max(s2) > seg_params.max_object_size:
            continue
        if z1 - min(rest, key=lambda s: abs(z0 - s.height)).height < 2 * seg_params.surface_layer:
            continue  # flat: surface that the object used to hide
        sup = min(rest, key=lambda s: abs(z0 - s.height))
        out.append({"center": [float(c2[0]), float(c2[1]), 0.5 * (sup.height + z1)],
                    "size": [float(s2[0]), float(s2[1]), z1 - sup.height], "yaw": float(yaw),
                    "rgba": _rgba(col, comp), "support": sup.name, "points": int(len(comp))})
    return out


def match_final(seg: Segmentation, moved: list[str], candidates: list[dict], p: ChangeParams | None = None) -> dict:
    """{moved object: final candidate} by size and colour."""
    p = p or ChangeParams()
    objs = {o.name: o for o in seg.objects}
    if not moved or not candidates:
        return {}
    cost = np.zeros((len(moved), len(candidates)))
    for a, name in enumerate(moved):
        o = objs[name]
        so = np.sort(o.size)
        for b, c in enumerate(candidates):
            sc = np.sort(c["size"])
            cost[a, b] = (np.abs(so - sc).sum() / max(so.sum(), 1e-6)
                          + np.linalg.norm(np.array(o.rgba[:3]) - np.array(c["rgba"][:3])))
    rows, cols = linear_sum_assignment(cost)
    return {moved[a]: {**candidates[b], "cost": round(float(cost[a, b]), 3)}
            for a, b in zip(rows, cols) if cost[a, b] < p.match_cost}
