"""Depth-camera capture -> simulation scene (igris_scan, docs/scan_pipeline.md).

    python scripts/scan_build.py data/scans/lab01 --name lab01                      # scan only
    python scripts/scan_build.py data/scans/lab01 --name lab01 --split 25            # + interaction after 25 s
    python scripts/scan_build.py data/scans/lab01 --name lab01 --split 25 --interaction-camera static
    python scripts/scan_build.py data/scans/lab01 --name lab01 --support-height 0.74 --set scan.voxel=0.005
    python scripts/scan_build.py --synthetic data/scans/synthetic --name scan_demo   # rendered demo capture

The scene is written to scenes/<name>/ (scene.json, meshes, scan_report.json, preview.png) and can be
used right away: python scripts/sim_server.py --scene <name> --viewer
"""
import argparse
import json
import time
from pathlib import Path

import _common  # noqa: F401
import yaml

from igris_scan.build import CFG, SCENES_DIR, build_scene, load_config

ap = argparse.ArgumentParser()
ap.add_argument("capture", nargs="?", help="capture folder (igris_rgbd: capture.json, color/, depth/)")
ap.add_argument("--name", required=True, help="scene name: scenes/<name>/")
ap.add_argument("--split", type=float, default=None, help="s: end of the scan phase, interaction after it")
ap.add_argument("--camera", choices=("track", "static", "poses"), default=None, help="scan phase camera")
ap.add_argument("--interaction-camera", choices=("track", "static"), default=None)
ap.add_argument("--step", type=int, default=None, help="use every n-th frame of the scan phase")
ap.add_argument("--support-height", type=float, default=None,
                help="m: true height of the main table (scale correction; needed if the floor is not in view)")
ap.add_argument("--dynamic", choices=("interacted", "all", "none"), default=None, help="objects that become free bodies")
ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                help="any configs/scan.yaml setting, e.g. scan.voxel=0.005 (repeatable)")
ap.add_argument("--config", default=str(CFG))
ap.add_argument("--out-root", default=str(SCENES_DIR))
ap.add_argument("--synthetic", default=None, metavar="DIR",
                help="render the synthetic demo capture to DIR first and build from it (split from its gt.json)")
a = ap.parse_args()

over = {"split_s": a.split, "camera": a.camera, "interaction_camera": a.interaction_camera, "scan_step": a.step,
        "align__support_height": a.support_height, "export__dynamic": a.dynamic}
for s in a.set:
    key, _, val = s.partition("=")
    sec, _, k = key.partition(".")
    over[f"{sec}__{k}" if k else sec] = yaml.safe_load(val)
capture = a.capture
if a.synthetic:
    from igris_scan.synthetic import render_capture
    t0 = time.time()
    capture = render_capture(a.synthetic, 640, 480)
    over["split_s"] = over["split_s"] or json.loads((capture / "gt.json").read_text())["split_s"]
    over["scan_step"] = over["scan_step"] or 2
    print(f"synthetic capture: {capture} ({time.time() - t0:.1f} s)")
if not capture:
    ap.error("give a capture folder or --synthetic DIR")
report = build_scene(capture, a.name, load_config(a.config, **over), a.out_root)
print(f"done in {sum(report['timing'].values()):.1f} s: {report['scene']}")
for o in json.loads((Path(report["scene"]) / "scan_report.json").read_text())["objects"]:
    print(f"  {o['name']:10s} {'MOVED ' if o['interacted'] else '      '}{'dynamic' if o['dynamic'] else 'fixed  '} "
          f"centre {o['center']} size {o['size']} on {o['support']}"
          + (f" -> {o['final']['center']}" if 'final' in o else ""))
print(f"check preview.png, then: python scripts/sim_server.py --scene {a.name} --viewer")
