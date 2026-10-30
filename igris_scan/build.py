"""Capture folder -> simulation scene: reconstruct, correct, segment, detect interactions, export.

    from igris_scan.build import build_scene, load_config
    report = build_scene("data/scans/lab01", "lab01", load_config(split_s=20.0))

The capture is split at split_s into the scan phase (static scene, fused into the model) and the
interaction phase (people moving objects; registered to the model, never fused). Without split_s the
whole capture is the scan and no object is marked interacted. The scene is written to
scenes/<name>/ and compiled once with the simulator's scene builder to check it.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from .align import AlignParams, align
from .changes import ChangeParams, detect_moved, final_state, match_final
from .export import ExportParams, export_scene
from .reconstruct import ScanParams, fuse, localize
from .rgbd import Capture
from .segment import SegmentParams, segment

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "scan.yaml"
SCENES_DIR = ROOT / "scenes"


@dataclass
class BuildConfig:
    scan: ScanParams = field(default_factory=ScanParams)
    align: AlignParams = field(default_factory=AlignParams)
    segment: SegmentParams = field(default_factory=SegmentParams)
    changes: ChangeParams = field(default_factory=ChangeParams)
    export: ExportParams = field(default_factory=ExportParams)
    split_s: float | None = None          # end of the scan phase; None: the whole capture is the scan
    camera: str = "track"                 # scan phase: track | static | poses (poses.txt)
    interaction_camera: str = "track"     # track | static (tripod: keeps the last scan pose)
    scan_step: int = 1                    # use every n-th frame of the scan phase


def _fill(cls, d: dict | None):
    names = {f.name for f in fields(cls)}
    unknown = set(d or {}) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown settings {sorted(unknown)}")
    obj = cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in (d or {}).items()})
    return obj


def load_config(path: str | Path | None = CFG, **overrides) -> BuildConfig:
    """configs/scan.yaml (sections scan / align / segment / changes / export + top-level keys), then
    overrides: top-level keys, or "section.key" written as section__key."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else {}
    raw = raw or {}
    for k, v in overrides.items():
        if v is None:
            continue
        if "__" in k:
            sec, key = k.split("__", 1)
            raw.setdefault(sec, {})[key] = v
        else:
            raw[k] = v
    cfg = BuildConfig(scan=_fill(ScanParams, raw.get("scan")), align=_fill(AlignParams, raw.get("align")),
                      segment=_fill(SegmentParams, raw.get("segment")), changes=_fill(ChangeParams, raw.get("changes")),
                      export=_fill(ExportParams, raw.get("export")),
                      **{k: raw[k] for k in ("split_s", "camera", "interaction_camera", "scan_step") if k in raw})
    cfg.segment.voxel = cfg.scan.voxel
    return cfg


def build_scene(capture_dir: str | Path, name: str, cfg: BuildConfig | None = None,
                out_root: str | Path = SCENES_DIR, log=print) -> dict:
    cfg = cfg or load_config()
    t0 = time.time()
    capture = Capture(capture_dir)
    timing = {}
    scan = capture.frames(end_s=cfg.split_s, step=cfg.scan_step)
    if len(scan) < 2:
        raise ValueError("the scan phase has fewer than 2 frames")
    log(f"[1/5] reconstruct: {len(scan)} frames ({cfg.camera} camera)")
    model = fuse(capture, scan, cfg.scan, cfg.camera, log=log)
    pts, col, nrm = model.point_cloud()
    timing["reconstruct_s"] = round(time.time() - t0, 2)
    log(f"[2/5] correct: {len(pts)} surface points")
    al = align(pts, [model.poses[i] for i in model.frames], cfg.align)
    for n in al.notes:
        log(f"      note: {n}")
    R = al.T[:3, :3] / al.scale
    seg_p = cfg.segment
    seg_p.voxel = cfg.scan.voxel * al.scale
    seg = segment(al.apply(pts), col, nrm @ R.T, seg_p)
    timing["segment_s"] = round(time.time() - t0 - sum(timing.values()), 2)
    log(f"[3/5] segment: {sum(s.kind == 'support' for s in seg.surfaces)} supports, "
        f"{sum(s.kind == 'wall' for s in seg.surfaces)} walls, {len(seg.objects)} objects")
    moved, finals, inter_info = {}, {}, None
    inter = capture.frames(start_s=cfg.split_s) if cfg.split_s is not None else []
    if inter:
        loc = localize(model, capture, inter, static=cfg.interaction_camera == "static")
        moved = detect_moved(capture, loc, seg, al, cfg.changes)
        names = [k for k, e in moved.items() if e["moved"]]
        finals = match_final(seg, names, final_state(capture, loc, seg, al, cfg.scan, seg_p, cfg.changes), cfg.changes)
        inter_info = {"frames": len(inter), "tracked": int(sum(ok for _, ok in loc.values())),
                      "camera": cfg.interaction_camera, "moved": names}
        log(f"[4/5] interaction: {len(inter)} frames, moved: {', '.join(names) or 'none'}")
    else:
        log("[4/5] interaction: none (no split_s): objects stay fixed unless export.dynamic is 'all'")
    timing["interaction_s"] = round(time.time() - t0 - sum(timing.values()), 2)
    mesh = model.mesh()
    mesh.vertices = mesh.vertices.__class__(al.apply(np.asarray(mesh.vertices)))
    if mesh.has_vertex_normals():
        mesh.vertex_normals = mesh.vertex_normals.__class__(np.asarray(mesh.vertex_normals) @ R.T)
    report = {"capture": str(Path(capture_dir).resolve()), "capture_meta": capture.meta, "split_s": cfg.split_s,
              "scan": {"frames": len(scan), "fused": len(model.frames), "keyframes": len(model.keyframes),
                       "camera": cfg.camera},
              "interaction": inter_info, "alignment": al.to_dict(),
              "settings": {k: asdict(getattr(cfg, k)) for k in ("scan", "align", "segment", "changes")},
              "evidence": {k: {"moved": e["moved"], "t_moved": e["t_moved"],
                               "gone_max": float(np.nanmax(e["gone"])) if np.any(np.isfinite(e["gone"])) else None}
                           for k, e in moved.items()}}
    out = Path(out_root) / name
    spec = export_scene(out, name, seg, mesh, moved, finals, seg_p.voxel, cfg.export,
                        f"scanned from {Path(capture_dir).name} (igris_scan)", report)
    timing["export_s"] = round(time.time() - t0 - sum(timing.values()), 2)
    log(f"[5/5] export: {out} ({len(spec['objects'])} scene objects)")
    from igris_sim.scene import build_model, load_scene
    build_model(load_scene(out))  # the simulator can load it
    report["timing"] = timing
    report["scene"] = str(out)
    return report
