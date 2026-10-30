"""Tests for the depth-camera scan pipeline (igris_scan) on a rendered capture with ground truth."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igris_scan.align import AlignParams, align  # noqa: E402
from igris_scan.build import build_scene, load_config  # noqa: E402
from igris_scan.rgbd import Capture, CaptureWriter, Intrinsics, clean_depth  # noqa: E402
from igris_scan.synthetic import look_at, render_capture  # noqa: E402


def test_clean_depth_drops_flying_pixels_and_range():
    d = np.full((5, 6), 1.0, np.float32)
    d[:, 3:] = 2.0
    d[2, 3] = 1.5           # a pixel between foreground and background
    d[0, 0] = 9.0           # out of range
    c = clean_depth(d, max_depth=4.0, edge=0.04)
    assert c[0, 0] == 0 and c[2, 3] == 0
    assert c[4, 0] == 1.0 and c[4, 5] == 2.0  # flat regions are kept


def test_capture_roundtrip(tmp_path):
    intr = Intrinsics(8, 6, 5.0, 5.0, 3.5, 2.5)
    rng = np.random.default_rng(0)
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    with CaptureWriter(tmp_path / "cap", intr, 1000.0, source="test") as w:
        for k in range(3):
            w.add(rng.integers(0, 255, (6, 8, 3), dtype=np.uint8), np.full((6, 8), 1.234 + k, np.float32),
                  10.0 + k / 30, pose if k != 1 else None)
    c = Capture(tmp_path / "cap")
    assert len(c) == 3 and c.intr == intr and np.allclose(c.t, [0, 1 / 30, 2 / 30], atol=1e-6)
    assert np.allclose(c.depth(2, clean=False), 3.234, atol=5e-4)
    assert np.allclose(c.poses[0], pose) and np.isnan(c.poses[1]).all()
    assert c.frames(start_s=0.02) == [1, 2]


def test_alignment_without_floor_uses_the_given_table_height():
    """Only a tilted table top in view: it is put at support_height, level."""
    rng = np.random.default_rng(1)
    top = np.c_[rng.uniform(-0.4, 0.4, 3000), rng.uniform(-0.3, 0.3, 3000), np.zeros(3000)]
    tilt = np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
    T = np.eye(4)
    T[:3, :3] = tilt
    cam = T @ look_at([0, -0.85, 0.6], [0, 0, 0])  # 0.6 m above the table, looking 35 deg down at it
    al = align(top @ tilt.T, [cam], AlignParams(support_height=0.74))
    P = al.apply(top @ tilt.T)
    assert not al.floor_visible and np.abs(P[:, 2] - 0.74).max() < 0.005


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scan")
    cap = render_capture(tmp / "capture", 320, 240)
    gt = json.loads((cap / "gt.json").read_text())
    cfg = load_config(split_s=gt["split_s"], scan__voxel=0.01)
    rep = build_scene(cap, "scan_test", cfg, out_root=tmp / "scenes", log=lambda *a: None)
    return cap, gt, rep, json.loads((tmp / "scenes" / "scan_test" / "scan_report.json").read_text())


def _fit2d(a, b):
    """Rigid 2D transform taking points a onto b (Kabsch) -> residuals."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return np.linalg.norm((a - ca) @ R.T + cb - b, axis=1), R


def test_scan_is_tracked_and_levelled(built):
    cap, gt, rep, report = built
    assert rep["scan"]["fused"] == rep["scan"]["frames"]  # no frame lost
    al = report["alignment"]
    assert al["floor_visible"] and al["floor_rms_m"] < 0.01 and al["heading_source"] == "manhattan"
    cam_z = np.array(gt["camera_poses"][0])[2, 3]
    assert abs(al["camera_height_m"] - cam_z) < 0.03
    tables = [s for s in report["surfaces"] if s["kind"] == "support"]
    assert min(abs(s["height"] - gt["table_top"]) for s in tables) < 0.015


def test_objects_are_found_and_the_moved_one_is_identified(built):
    cap, gt, rep, report = built
    objs = report["objects"]
    assert len(objs) == len(gt["objects"])
    names = list(gt["objects"])
    g = np.array([gt["objects"][n]["start"] for n in names])
    found = np.array([o["center"] for o in objs])
    # pair by nearest after a rigid fit of the layouts (the scene frame starts at the camera)
    best = None
    from itertools import permutations
    for perm in permutations(range(len(objs))):
        res, R = _fit2d(found[list(perm), :2], g[:, :2])
        if best is None or res.max() < best[0].max():
            best = (res, R, perm)
    res, R, perm = best
    assert res.max() < 0.05
    for n, k in zip(names, perm):
        o, t = objs[k], gt["objects"][n]
        h_true = t["size"][2] if t["shape"] == "box" else t["size"][1]
        assert abs(o["size"][2] - h_true) < 0.02, n
        assert o["interacted"] == t["moved"] and o["dynamic"] == t["moved"], n
        if t["moved"]:  # where it went: displacement as in the ground truth (rotated into the scene frame)
            d_true = R.T @ (np.array(t["end"][:2]) - np.array(t["start"][:2]))
            d = np.array(o["final"]["center"][:2]) - np.array(o["center"][:2])
            assert np.linalg.norm(d - d_true) < 0.06


def test_scene_loads_and_objects_rest(built):
    from igris_sim.scene import load_scene
    from igris_sim.session import load_sim_config
    from igris_sim.world import SimWorld
    cap, gt, rep, report = built
    scene = load_scene(rep["scene"])
    kinds = {o["name"].split("_")[0] for o in scene.spec["objects"]}
    assert {"scan", "support", "wall", "object"} <= kinds
    assert all(o["rgba"][3] == 0 for o in scene.spec["objects"] if o["name"].startswith("wall"))  # colliders only
    w = SimWorld(scene, load_sim_config())
    moved = [o["name"] for o in report["objects"] if o["interacted"]]
    assert w.dynamic == moved
    p0, _ = w.object_poses()
    q = w.robot_qpos()
    for _ in range(50):
        w.step(q, 0.02)
    assert np.abs(w.object_poses()[0] - p0).max() < 0.01  # rests on its support
