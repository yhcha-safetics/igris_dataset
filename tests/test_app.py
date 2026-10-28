"""Tests for the PICO app support: robot description, asset server, and the app's C# core (pico_app)."""
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igris_pipeline.utils.rotations import rot_to_quat  # noqa: E402
from igris_sim import protocol  # noqa: E402
from igris_sim.robot_export import forward_kinematics, robot_description  # noqa: E402
from igris_sim.scene import _parts, list_scenes, load_scene, rot_quat  # noqa: E402
from igris_sim.session import SimSession, load_sim_config  # noqa: E402
from igris_sim.world import SimWorld  # noqa: E402

APP = ROOT / "pico_app"


@pytest.fixture(scope="module")
def world():
    return SimWorld(load_scene("tabletop"), load_sim_config())


def _random_qpos(w, rng):
    q = w.robot_qpos()
    q[:3] += rng.uniform(-0.5, 0.5, 3)
    q[3:7] = rng.normal(size=4)
    q[3:7] /= np.linalg.norm(q[3:7])
    q[7:] = rng.uniform(-1.2, 1.2, len(q) - 7)
    return q


def _mj_poses(w, q):
    m, d = w.model, w.data
    d.qpos[:w.nq_r] = q
    mujoco.mj_kinematics(m, d)
    return d


def _geom_quat(d, g):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, d.geom_xmat[g])
    return q


def test_robot_description_matches_mujoco(world):
    desc, files = robot_description(world)
    m = world.model
    assert desc["nq"] == world.nq_r == 38 and len(desc["joint_names"]) == 31
    assert all(b["parent"] < k for k, b in enumerate(desc["bodies"]))
    rng = np.random.default_rng(1)
    for _ in range(3):
        q = _random_qpos(world, rng)
        d = _mj_poses(world, q)
        xp, xq = forward_kinematics(desc, q)
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b["name"]) for b in desc["bodies"]]
        assert np.abs(xp - d.xpos[ids]).max() < 1e-9
        assert np.abs(np.abs(np.sum(xq * d.xquat[ids], 1)) - 1).max() < 1e-9
    # every visual mesh is exported in its compiled (geom) frame
    for g in desc["geoms"]:
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, Path(g["mesh"]).stem)
        text = files[g["mesh"]].decode()
        assert text.count("\nv ") == m.mesh_vertnum[k] and text.count("\nf ") == m.mesh_facenum[k]
    assert json.loads(files["robot.json"]) == desc


def test_asset_server_serves_generated_files(world):
    _, files = robot_description(world)
    srv = protocol.AssetServer(0, host="127.0.0.1")
    try:
        srv.add_files(files, protocol.ROBOT_ASSETS)
        base = f"http://127.0.0.1:{srv.port}/"
        with urllib.request.urlopen(base + "_robot/robot.json", timeout=5) as r:
            assert r.read() == files["robot.json"]
        mesh = next(k for k in files if k.endswith(".obj"))
        with urllib.request.urlopen(base + "_robot/" + mesh, timeout=5) as r:
            assert r.read() == files[mesh]
        with urllib.request.urlopen(base + "tabletop/scene.json", timeout=5) as r:  # scene folder still served
            assert json.loads(r.read())["name"] == "tabletop"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "_robot/missing.obj", timeout=5)
    finally:
        srv.close()


def test_scene_message_points_to_robot():
    s = SimSession(load_sim_config(), "tabletop")
    msg = s.scene_message("http://x/")
    assert msg["robot"] == protocol.ROBOT_ASSETS + "robot.json" and msg["folder"] == "tabletop"
    assert json.loads(protocol.encode(msg)) == msg


# -- the app's C# core ------------------------------------------------------------------------
def _find_csc():
    if os.environ.get("CSC"):
        return os.environ["CSC"]
    pats = [r"C:\Program Files*\Microsoft Visual Studio\*\*\MSBuild\Current\Bin\Roslyn\csc.exe",
            r"C:\Program Files*\Microsoft Visual Studio\*\*\MSBuild\*\Bin\Roslyn\csc.exe"]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[-1]
    return shutil.which("csc")


def _unity_scale(shape, size):
    if shape == "box":
        return [size[1], size[2], size[0]]
    if shape == "sphere":
        return [2 * size[0]] * 3
    if shape == "cylinder":
        return [2 * size[0], size[1] / 2, 2 * size[0]]
    if shape == "capsule":
        return [2 * size[0], (size[1] + 2 * size[0]) / 2, 2 * size[0]]
    return None


def _fixtures(tmp_path, world):
    desc, files = robot_description(world)
    robot_dir = tmp_path / "robot"
    for rel, data in files.items():
        (robot_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (robot_dir / rel).write_bytes(data)
    m = world.model
    rng = np.random.default_rng(7)
    fk = []
    for _ in range(3):
        q = _random_qpos(world, rng)
        d = _mj_poses(world, q)
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b["name"]) for b in desc["bodies"]]
        gids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g["name"]) for g in desc["geoms"]]
        fk.append({"qpos": q.tolist(), "body_pos": d.xpos[ids].tolist(), "body_quat": d.xquat[ids].tolist(),
                   "geom_pos": d.geom_xpos[gids].tolist(), "geom_quat": [_geom_quat(d, g).tolist() for g in gids]})
    mesh_counts = {}
    for g in desc["geoms"]:
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, Path(g["mesh"]).stem)
        mesh_counts[g["mesh"]] = [int(m.mesh_vertnum[k]), int(m.mesh_facenum[k])]
    conv_mj = []
    for _ in range(5):
        p, q = rng.normal(size=3), rng.normal(size=4)
        conv_mj.append({"p": p.tolist(), "q": q.tolist(), "up": protocol.unity_from_mujoco_pos(p).tolist(),
                        "uq": protocol.unity_from_mujoco_quat(q).tolist()})
    conv_native = []
    for _ in range(5):
        p, q = rng.normal(size=3), rng.normal(size=4)
        conv_native.append({"p": p.tolist(), "q": q.tolist(), "n": protocol.native_from_unity(p, q).tolist()})
    euler = []
    for _ in range(8):
        deg = rng.uniform(-180, 180, 3)
        euler.append({"deg": deg.tolist(), "quat": rot_to_quat(R.from_euler("xyz", deg, degrees=True)).tolist()})
    scenes = []
    for name in list_scenes():
        spec = load_scene(name).spec
        objs = []
        for o in spec["objects"]:
            parts = []
            for p in _parts(o):
                rgba = p.get("rgba", [0.7, 0.7, 0.7, 1.0])
                parts.append({"shape": p["shape"], "pos": p.get("pos", [0, 0, 0]), "quat": rot_quat(p.get("rot")).tolist(),
                              "rgba": rgba, "unity_scale": _unity_scale(p["shape"], p.get("size", []))})
            dyn = bool(o.get("dynamic", False))
            objs.append({"name": o["name"], "pos": o.get("pos", [0, 0, 0]), "quat": rot_quat(o.get("rot")).tolist(),
                         "dynamic": dyn, "graspable": dyn and bool(o.get("graspable", dyn)), "parts": parts})
        scenes.append({"spec": spec, "objects": objs})
    s = SimSession(load_sim_config(), "tabletop")
    s.tick(0.0)
    unity = {k: np.concatenate([rng.normal(size=3), (lambda v: v / np.linalg.norm(v))(rng.normal(size=4))]).tolist()
             for k in ("head", "left", "right", "tracker")}
    fx = {"robot_dir": "robot", "fk": fk, "mesh_counts": mesh_counts, "unity_from_mj": conv_mj,
          "native_from_unity": conv_native, "euler": euler, "scenes": scenes,
          "state": json.loads(protocol.encode(s.state_message())),
          "scene_message": json.loads(protocol.encode({**s.scene_message("http://192.168.0.10:9871/"), "asset_port": 9871})),
          "tracking_unity": unity}
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(fx), encoding="utf-8")
    return path, unity


def test_app_core_csharp(tmp_path, world):
    csc = _find_csc()
    if not csc:
        pytest.skip("no C# compiler (csc) found; set CSC to its path")
    sources = sorted(str(p) for p in (APP / "Assets" / "IgrisPico" / "Core").glob("*.cs"))
    sources.append(str(APP / "Assets" / "IgrisPico" / "Runtime" / "SimLink.cs"))  # Unity-free as well
    exe = tmp_path / "CoreTests.exe"
    # C# 7.3: the core must also compile in Unity without newer language features
    r = subprocess.run([csc, "-nologo", "-langversion:7.3", "-warnaserror+", f"-out:{exe}",
                        str(APP / "CoreTests" / "CoreTests.cs"), *sources], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    fx_path, unity = _fixtures(tmp_path, world)
    out = tmp_path / "out"
    out.mkdir()
    r = subprocess.run([str(exe), str(fx_path), str(out)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL PASSED" in r.stdout

    # the tracking message written by the app core is what the server expects
    fr = protocol.frame_from_tracking(protocol.decode((out / "tracking.json").read_bytes()))
    for key, got in (("head", fr.headset), ("left", fr.left_ctrl), ("right", fr.right_ctrl)):
        u = unity[key]
        assert np.allclose(got, protocol.native_from_unity(u[:3], u[3:]), atol=1e-12)
    u = unity["tracker"]
    assert fr.tracker_serials == ["PC2310MLL4210362G"]
    assert np.allclose(fr.trackers[0], protocol.native_from_unity(u[:3], u[3:]), atol=1e-12)
    assert fr.t_ns == 1761100000123456789
    assert fr.body.shape == (24, 7) and fr.body_joint_ts[3] == 123459
    assert np.allclose(fr.body[5, :3], [0.05, 0.85, -0.2])
    assert fr.buttons["A"] and fr.buttons["Y"] and not fr.buttons["B"]
    assert fr.buttons["right_grip"] == pytest.approx(0.8) and fr.buttons["left_trigger"] == pytest.approx(0.25)
    assert np.allclose(fr.axes["left"], [-0.5, 0.0]) and np.allclose(fr.axes["right"], [0.0, 0.75])
    cmd = protocol.decode((out / "cmd.json").read_bytes())
    assert cmd == {"v": 1, "type": "cmd", "cmd": "record", "on": False, "scene": "room"}
