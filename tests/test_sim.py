"""Tests for the real-time simulation server (igris_sim)."""
import json
import socket
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igris_pipeline.episode import Episode  # noqa: E402
from igris_pipeline.pico.frames import pose7_to_pipe  # noqa: E402
from igris_pipeline.pico.recorder import record  # noqa: E402
from igris_pipeline.pico.simulate import SimulatedXRTSource  # noqa: E402
from igris_pipeline.pico.synthetic import generate  # noqa: E402
from igris_pipeline.pico.xrt_source import FrameBuffer  # noqa: E402
from igris_pipeline.retarget.robot import IgrisModel  # noqa: E402
from igris_pipeline.utils.rotations import quat_apply, quat_mul, rot_to_quat  # noqa: E402
from igris_sim import protocol  # noqa: E402
from igris_sim.hand_model import pose_rate  # noqa: E402
from igris_sim.scene import Scene, build_model, list_scenes, load_scene, validate  # noqa: E402
from igris_sim.session import SimSession, load_sim_config  # noqa: E402
from igris_sim.sources import EpisodeReplaySource  # noqa: E402
from igris_sim.streaming import Biquad, StreamingCorrector, StreamingRetargeter  # noqa: E402
from igris_sim.world import SimWorld  # noqa: E402

PROGRAM = [("idle", "upright", "t_pose", 3.0), ("idle", "upright", "neutral", 1.0),
           ("walk", "upright", "neutral", 2.0), ("idle", "upright", "reach_forward", 1.5)]


def sim_frames(profile, program=PROGRAM, seed=3):
    clip = generate(program, fps=72, seed=seed)
    src = SimulatedXRTSource(profile, clip=clip, seed=seed)
    out = []
    while True:
        try:
            out.append(src.poll())
        except StopIteration:
            return out, clip


def frame_times(frames):
    t = np.array([f.t_ns for f in frames]) * 1e-9
    return t - t[0]


@pytest.fixture(scope="module")
def cfg():
    return load_sim_config()


@pytest.fixture(scope="module")
def weld_cfg(cfg):
    return {**cfg, "grasp": {**cfg["grasp"], "mode": "weld"}}


# -- scenes -----------------------------------------------------------------------------
def test_builtin_scenes_compile():
    names = list_scenes()
    assert {"empty", "tabletop", "room"} <= set(names)
    for n in names:
        model, info = build_model(load_scene(n))
        robot = IgrisModel(model=model)
        assert robot.ndof == 31
        assert model.jnt_qposadr[robot.jid(robot.joint_names[-1])] < 7 + robot.ndof  # robot comes first
        assert len(info.weld) == 2 * len(info.graspable)
        for name in info.graspable:
            assert model.body_mass[info.body[name]] > 0


def test_mesh_object_scene(tmp_path):
    v = [(x, y, z) for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05)]
    faces = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    obj = "\n".join([f"v {x} {y} {z}" for x, y, z in v] + [f"f {' '.join(map(str, f))}" for f in faces])
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "box.obj").write_text(obj)
    (tmp_path / "meshes" / "box_col.obj").write_text(obj)
    spec = {"format": "igris_scene", "version": 1, "name": "mesh_test", "objects": [
        {"name": "scan", "shape": "mesh", "file": "meshes/box.obj", "pos": [1, 0, 0.05]},
        {"name": "thing", "shape": "mesh", "file": "meshes/box.obj", "collision": ["meshes/box_col.obj"],
         "pos": [1, 0.5, 0.3], "dynamic": True}]}
    (tmp_path / "scene.json").write_text(json.dumps(spec))
    model, info = build_model(load_scene(tmp_path))
    assert info.dynamic["thing"] and not info.dynamic["scan"]
    assert model.body_mass[info.body["thing"]] > 0
    assert len(info.geoms["thing"]) == 1 and len(info.geoms["scan"]) == 1


def test_scene_validation():
    base = {"format": "igris_scene", "version": 1, "name": "x"}
    with pytest.raises(ValueError):
        validate({**base, "objects": [{"name": "a", "shape": "box", "size": [1, 1, 1]}, {"name": "a", "shape": "box"}]})
    with pytest.raises(ValueError):
        validate({**base, "objects": [{"name": "a", "shape": "torus"}]})
    with pytest.raises(ValueError):
        validate({**base, "version": 2})


# -- filters ----------------------------------------------------------------------------
def test_biquad_steady_state_and_attenuation():
    f = Biquad(50.0, 5.0)
    x0 = np.array([1.5, -2.0])
    assert np.allclose(f(x0), x0) and np.allclose(f(x0), x0)
    f = Biquad(50.0, 2.0)
    t = np.arange(500) / 50.0
    y = np.array([f(np.array([np.sin(2 * np.pi * 15 * tt)]))[0] for tt in t])
    assert np.abs(y[100:]).max() < 0.05  # 15 Hz well above a 2 Hz cutoff
    assert np.allclose(Biquad(50.0, None)(x0), x0)


# -- streaming correction + retargeting ------------------------------------------------------
@pytest.mark.parametrize("profile,max_err_cm", [("body_2trk", 8.0), ("body_3trk", 7.0),
                                                ("sparse_trackers", 6.0), ("hmd_3pt", 12.0)])
def test_streaming_profiles(cfg, profile, max_err_cm):
    import yaml
    from igris_pipeline.retarget.ik_config import load as load_ik
    cfg_all = yaml.safe_load(open(ROOT / "configs" / "tracker_profiles.yaml", encoding="utf-8"))
    frames, clip = sim_frames(profile)
    t = frame_times(frames)
    sc = StreamingCorrector(cfg_all, profile, subject_height=1.72)
    res = sc.calibrate([f for f, tt in zip(frames, t) if tt < 3.0])
    rt = StreamingRetargeter(load_ik(ROOT / "configs" / "retarget_igris_c.yaml"), sc.fps)
    rt.calibrate(res["arrays"]["proc/body/pos"], res["calibration"]["stature"])
    ticks = np.arange(3.0, t[-1], 1 / 50)
    idx = np.searchsorted(t, ticks, side="right") - 1
    gt_t = clip.t - clip.t[0]
    errs, qs = [], []
    for i in idx:
        pose = sc.process(frames[i])
        qs.append(rt.step(pose, 0.0, np.zeros(3)))
        g = min(int(np.searchsorted(gt_t, t[i])), len(gt_t) - 1)
        errs.append(np.linalg.norm((clip.pos[g] - clip.pos[g, 0]) - (pose.pos - pose.pos[0]), axis=-1).mean())
    qs = np.array(qs)
    assert np.isfinite(qs).all()
    assert np.mean(errs) * 100 < max_err_cm
    assert 0.5 < qs[:, 2].min() and qs[:, 2].max() < 1.1          # pelvis height stays plausible
    assert qs[-1, 0] - qs[0, 0] > 0.5                              # follows the walk forward
    rng = rt.robot.jnt_range
    assert np.all(qs[:, 7:] >= rng[:, 0] - 1e-6) and np.all(qs[:, 7:] <= rng[:, 1] + 1e-6)


# -- physics / grasping ---------------------------------------------------------------------
def _place_near_palm(w, obj, gap=0.03):
    """Put a dynamic object below the left palm with ~gap between the surfaces."""
    d, adr = w.data, w.info.qposadr[obj]
    palm = w.palm_pos("left")
    for dz in np.arange(0.02, 0.3, 0.005):
        d.qpos[adr:adr + 3] = palm - np.array([0.0, 0.0, dz])
        d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(w.model, d)
        dist = min(mujoco.mj_geomDistance(w.model, d, w.palm["left"], g, 1.0, None) for g in w.info.geoms[obj])
        if dist >= gap:
            return
    raise AssertionError("could not place the object")


@pytest.mark.parametrize("mode", ["kinematic", "harness"])
def test_grasp_follows_hand_and_releases(weld_cfg, mode):
    w = SimWorld(load_scene("tabletop"), weld_cfg, mode)
    _place_near_palm(w, "cube_red")
    assert w.update_grasp({"left": 1.0, "right": 0.0}) == [("left", "grasp", "cube_red")]
    body, hand = w.info.body["cube_red"], w.hand_body["left"]

    def in_hand():  # object position in the hand frame (what the weld keeps constant)
        d = w.data
        return d.xmat[hand].reshape(3, 3).T @ (d.xpos[body] - d.xpos[hand])

    rel0, palm0 = in_hand(), w.palm_pos("left")
    q = w.robot_qpos()
    j = w.robot.joint_names.index("Joint_Shoulder_Pitch_Left")
    for k in range(40):  # raise the arm forward
        qt = q.copy()
        qt[7 + j] = q[7 + j] - 0.8 * min(1.0, (k + 1) / 20)
        w.step(qt, 0.02)
    assert np.linalg.norm(w.palm_pos("left") - palm0) > 0.2   # the hand really moved
    assert np.linalg.norm(in_hand() - rel0) < 0.02
    assert w.update_grasp({"left": 0.0}) == [("left", "release", "cube_red")]
    p_release = w.data.xpos[body].copy()
    for k in range(40):  # lower the arm again: a released object must not follow
        qt2 = qt.copy()
        qt2[7 + j] = qt[7 + j] + 0.8 * min(1.0, (k + 1) / 20)
        w.step(qt2, 0.02)
    assert np.linalg.norm(w.data.xpos[body] - w.palm_pos("left")) > 0.15
    assert np.linalg.norm(w.data.xpos[body][:2] - p_release[:2]) < 0.1  # stays where it was put down
    v = w.data.qvel[w.dofadr["cube_red"]:w.dofadr["cube_red"] + 3]
    assert np.linalg.norm(v) < 0.05  # and comes to rest (on the table or the floor)


def test_grasp_needs_proximity(weld_cfg):
    w = SimWorld(load_scene("tabletop"), weld_cfg)
    assert w.update_grasp({"left": 1.0, "right": 1.0}) == []  # nothing within reach at the spawn pose


# -- physical hands (grasp.mode physics) ------------------------------------------------------------
def _hand_quat(x, y, z):
    """Hand orientation from its axes in the world (z: fingers, y: palm normal of the left hand)."""
    return rot_to_quat(R.from_matrix(np.stack([x, y, z], axis=1)))


def _bench_scene(*objects):
    """A table (top at z = 0.75) with the given objects."""
    table = {"name": "table", "pos": [0.8, 0.0, 0.0], "rgba": [0.55, 0.4, 0.28, 1.0],
             "geoms": [{"shape": "box", "size": [0.7, 1.2, 0.04], "pos": [0.0, 0.0, 0.73]},
                       {"shape": "box", "size": [0.05, 0.05, 0.71], "pos": [0.0, 0.0, 0.355]}]}
    spec = {"format": "igris_scene", "version": 1, "name": "bench", "objects": [table, *objects]}
    return Scene(spec, ROOT / "scenes" / "tabletop")


def _script_hands(w, poses_at):
    """Drive the physical hands along poses_at(t) -> {side: (pos, quat)} instead of the robot hands."""
    clock = [0.0]

    def targets(q0, q1, h):
        a = poses_at(clock[0])
        clock[0] += h
        b = poses_at(clock[0])
        return {s: (*b[s], *pose_rate(*a[s], *b[s], h)) for s in b}

    w._hand_targets = targets
    w.overlay.place(w.data, poses_at(0.0))
    mujoco.mj_forward(w.model, w.data)


def _run_world(w, seconds, grips=lambda t: (0.0, 0.0), log=None):
    from igris_pipeline.hands import grip_to_command
    q = w.robot_qpos()
    events = []
    for k in range(int(round(seconds * 50))):
        g = grips(k / 50)
        w.set_hand_command(grip_to_command(*g))
        events += w.update_grasp({"left": g[0], "right": g[1]})
        w.step(q, 0.02)
        if log is not None:
            log.append(w.data.xpos[list(w.body_object)].copy())
    return events + w.update_grasp({})


@pytest.mark.parametrize("mode", ["kinematic", "harness"])
def test_physical_hands_follow_the_robot_hands(cfg, mode):
    from igris_pipeline.hands import command_to_joint_pos, grip_to_command
    w = SimWorld(load_scene("empty"), cfg, mode)
    assert w.physical and w.grasp_mode == "physics" and len(w.qpos_ext()) == 38 + 22 + 14
    q = w.robot_qpos()
    j = w.robot.joint_names.index("Joint_Shoulder_Pitch_Left")
    err = []
    for k in range(60):  # raise the arm forward and close the left hand
        qt = q.copy()
        qt[7 + j] = q[7 + j] - 0.8 * min(1.0, (k + 1) / 30)
        w.set_hand_command(grip_to_command(min(1.0, k / 20), 0.0))
        w.step(qt, 0.02)
        hp = w.hand_poses()
        err += [np.linalg.norm(hp[i, :3] - w.data.xpos[w.hand_body[s]]) for i, s in enumerate(("left", "right"))]
    assert max(err) < 0.01 and err[-1] < 1e-3  # free space: on the robot hand (mm while moving)
    assert np.abs(w.hand_q - command_to_joint_pos(grip_to_command(1.0, 0.0))).max() < 0.02  # fingers reach the command
    assert w.grasped == {"left": None, "right": None} and np.all(w.hand_force == 0)


def test_physical_hand_stops_at_surfaces(cfg):
    w = SimWorld(_bench_scene(), cfg)
    down = _hand_quat(np.array([0, -1.0, 0]), np.array([0, 0, -1.0]), np.array([1.0, 0, 0]))  # palm down
    other = (w.data.xpos[w.hand_body["right"]].copy(), w.data.xquat[w.hand_body["right"]].copy())
    # the target pushes the palm (0.02 below the hand origin) 5 cm into the table top
    _script_hands(w, lambda t: {"left": (np.array([0.8, 0.0, 0.75 + 0.02 + 0.10 - min(t, 1.0) * 0.15]), down),
                                "right": other})
    _run_world(w, 1.5)
    palm_z = w.hand_poses()[0, 2] - 0.02
    assert -0.005 < palm_z - 0.75 < 0.002  # on the surface, not in it
    f = np.linalg.norm(w.overlay.reaction["left"][:3])
    assert abs(f - cfg["hands"]["physics"]["max_force"]) < 1.0  # pressing with the force limit


def test_grip_without_contact_grasps_nothing(cfg):
    """The weld grasp picked up anything within 10 cm; physical fingers have to touch it."""
    w = SimWorld(_bench_scene({"name": "cube", "shape": "box", "size": [0.05, 0.05, 0.05], "pos": [0.6, 0.0, 0.776],
                               "dynamic": True, "mass": 0.1}), cfg)
    down = _hand_quat(np.array([0, -1.0, 0]), np.array([0, 0, -1.0]), np.array([1.0, 0, 0]))
    other = (w.data.xpos[w.hand_body["right"]].copy(), w.data.xquat[w.hand_body["right"]].copy())
    hover = np.array([0.6 - 0.11, 0.0, 0.776 + 0.06 + 0.08])  # the pinch point 8 cm above the cube
    _script_hands(w, lambda t: {"left": (hover + [0, 0, max(0.0, t - 1.5) * 0.15], down), "right": other})
    log = []
    events = _run_world(w, 2.5, grips=lambda t: (min(1.0, t), 0.0), log=log)
    assert events == [] and w.grasped["left"] is None
    assert np.abs(log[-1] - log[0]).max() < 1e-3  # the cube stayed on the table


def _pinch(w, obj, lift=0.15, open_at=None):
    c0 = w.data.xpos[w.info.body[obj]].copy()
    down = _hand_quat(np.array([0, -1.0, 0]), np.array([0, 0, -1.0]), np.array([1.0, 0, 0]))
    other = (w.data.xpos[w.hand_body["right"]].copy(), w.data.xquat[w.hand_body["right"]].copy())
    grasp = c0 + np.array([-0.11, 0.02, 0.06])  # hand origin with the pinch point on the object
    _script_hands(w, lambda t: {"left": (grasp + [0, 0, 0.1 * max(0.0, 1 - t / 0.8) + lift * np.clip(t - 2.2, 0, 1)],
                                         down), "right": other})
    grips = (lambda t: (np.clip(t - 1.0, 0, 1) * (t < open_at if open_at else 1), 0.0))
    return c0, grips


def test_pinch_lifts_a_cube_and_lets_it_go(cfg):
    w = SimWorld(_bench_scene({"name": "cube", "shape": "box", "size": [0.05, 0.05, 0.05], "pos": [0.6, 0.0, 0.776],
                               "dynamic": True, "mass": 0.1}), cfg)
    c0, grips = _pinch(w, "cube", open_at=4.0)
    log = []
    events = _run_world(w, 4.0, grips, log)
    assert log[-1][0][2] - c0[2] > 0.12  # lifted with the hand (0.15)
    assert ("left", "grasp", "cube") in events and w.grasped["left"] == "cube" and w.hand_force[0] > 2.0
    events = _run_world(w, 1.0, lambda t: (0.0, 0.0), log)  # open the hand: it falls back on the table
    assert ("left", "release", "cube") in events and w.grasped["left"] is None
    assert abs(log[-1][0][2] - c0[2]) < 0.01


def test_two_hands_lift_a_box(cfg):
    box = {"name": "box", "shape": "box", "size": [0.22, 0.30, 0.18], "pos": [0.7, 0.0, 0.84], "dynamic": True,
           "mass": 1.5}
    w = SimWorld(_bench_scene(box), cfg)
    c0 = w.data.xpos[w.info.body["box"]].copy()
    q = _hand_quat(np.array([0, 0, 1.0]), np.array([0, -1.0, 0]), np.array([1.0, 0, 0]))  # palms facing, thumbs up

    def poses(t):
        off = 0.15 + 0.02 + 0.05 - np.clip(t / 0.6, 0, 1) * 0.07  # palms from 5 cm outside to 2 cm into the sides
        base = c0 + [-0.09, 0, 0.2 * np.clip(t - 1.0, 0, 1)]
        return {"left": (base + [0, off, 0], q), "right": (base - [0, off, 0], q)}

    _script_hands(w, poses)
    log = []
    events = _run_world(w, 2.5, log=log)
    assert log[-1][0][2] - c0[2] > 0.18  # lifted 0.2 by squeezing it between the palms
    assert {("left", "grasp", "box"), ("right", "grasp", "box")} <= set(events)
    assert all(abs(f - 60.0) < 15 for f in w.hand_force)  # each palm presses ~ kp * 2 cm


# -- protocol ------------------------------------------------------------------------------
def test_tracking_message_roundtrip():
    frames, _ = sim_frames("sparse_trackers", PROGRAM[:1])
    fr = frames[10]
    fr.buttons = {"A": True, "left_grip": 0.7}
    fr.axes = {"left": np.array([0.1, -0.5])}
    back = protocol.frame_from_tracking(protocol.decode(protocol.encode(protocol.tracking_from_frame(fr))))
    assert back.t_ns == fr.t_ns and back.buttons == {"A": True, "left_grip": 0.7}
    assert np.allclose(back.headset, fr.headset, atol=1e-6)
    assert np.allclose(back.trackers, fr.trackers, atol=1e-6) and back.tracker_serials == fr.tracker_serials
    assert np.allclose(back.axes["left"], [0.1, -0.5])
    with pytest.raises(ValueError):
        protocol.frame_from_tracking({"t_ns": 1, "head": [0, 0, 0]})
    with pytest.raises(ValueError):
        protocol.decode(b'{"v": 9, "type": "state"}')


def test_unity_conversions_are_consistent():
    rng = np.random.default_rng(0)
    from scipy.spatial.transform import Rotation as R
    for _ in range(20):
        q = rot_to_quat(R.random(random_state=rng))
        p = rng.normal(size=3)
        v = rng.normal(size=3)
        # rotating then converting == converting then rotating (Unity applies q*v with its own axes)
        qu = protocol.unity_from_mujoco_quat(q)
        lhs = protocol.unity_from_mujoco_pos(quat_apply(q, v))
        rhs = quat_apply(np.array([qu[3], qu[0], qu[1], qu[2]]), protocol.unity_from_mujoco_pos(v))
        assert np.allclose(lhs, rhs)
        # Unity -> native -> pipeline (server side) -> Unity is the identity
        pu = protocol.unity_from_mujoco_pos(p)
        pp, qp, _ = pose7_to_pipe(protocol.native_from_unity(pu, qu))
        assert np.allclose(pp, p) and np.allclose(np.abs(np.dot(qp, q)), 1.0)


def test_udp_session_roundtrip(cfg):
    hub = protocol.Hub(0, host="127.0.0.1")
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(1.0)
    try:
        s = SimSession(cfg, "empty")
        addr = ("127.0.0.1", hub.port)
        client.sendto(protocol.encode({"v": 1, "type": "cmd", "cmd": "hello"}), addr)
        frames, _ = sim_frames("body_2trk", PROGRAM[:2])
        replies = []
        last = None

        def drain():  # read as we go: hundreds of states would overflow the client's receive buffer
            nonlocal last
            client.setblocking(False)
            while True:
                try:
                    m = protocol.decode(client.recvfrom(65507)[0])
                except (BlockingIOError, ConnectionResetError):
                    break
                replies.append(m["type"])
                last = m if m["type"] == "state" else last

        for k, fr in enumerate(frames):
            client.sendto(protocol.encode(protocol.tracking_from_frame(fr)), addr)
            if k % 3 == 0:  # ~24 Hz ticks are plenty here
                got, cmds = hub.poll()
                for f, _ in got:
                    s.push_frame(f)
                for msg, a in cmds:
                    hub.send(s.command(msg), a)
                s.tick()
                hub.broadcast(s.state_message())
                drain()
        drain()
        assert "scene" in replies and last is not None
        assert last["status"] == "running" and last["profile"] == "body_2trk"
        # robot qpos (38) + the finger overlay joints (22), as listed by the scene message's joint_names
        assert len(last["robot"]["qpos"]) == 7 + len(s.world.joint_names_ext()) + 14 == 74
        assert len(last["objects"]["pos"]) == 1
    finally:
        client.close()
        hub.close()


# -- session: recording, locomotion, replay --------------------------------------------------
def _run(s, frames, until=None, axes=None, offset=0.0):
    t = frame_times(frames)
    end = t[-1] if until is None else until
    tick = 0.0
    i = 0
    while tick <= end:
        while i < len(frames) and t[i] <= tick:
            if axes is not None:
                frames[i].axes = axes
            s.push_frame(frames[i])
            i += 1
        s.tick(now=offset + tick)
        tick += 1 / 50


def test_recording_is_pipeline_compatible(cfg, tmp_path):
    from igris_pipeline.pipeline import stage_correct
    s = SimSession(cfg, "tabletop", record_dir=tmp_path / "raw")
    frames, _ = sim_frames("body_2trk")
    t = frame_times(frames)
    _run(s, [f for f, tt in zip(frames, t) if tt < 3.6])
    assert s.status == "running"
    s.set_recording(True, now=3.6)
    _run(s, [f for f, tt in zip(frames, t) if tt >= 3.6], offset=3.6)
    s.set_recording(False)
    assert len(s.saved) == 1
    ep = Episode.load(s.saved[0])
    assert "raw/body/pos" in ep and ep["sim/robot/qpos"].shape[1] == 38
    assert ep["sim/objects/pos"].shape[1] == len(s.world.dynamic)
    assert json.loads(ep.group_attrs["sim"]["scene_json"])["name"] == "tabletop"
    stage_correct(s.saved[0])  # the offline pipeline accepts it


def test_grips_close_the_fingers_and_are_recorded(cfg, tmp_path):
    from igris_pipeline.hands import MOTOR_IDS, grip_to_command
    from igris_pipeline.pipeline import hand_series, stage_correct

    s = SimSession(cfg, "tabletop", record_dir=tmp_path / "raw")
    frames, _ = sim_frames("body_2trk")
    t = frame_times(frames)
    _run(s, [f for f, tt in zip(frames, t) if tt < 3.6])
    assert s.status == "running"
    rest = [f for f, tt in zip(frames, t) if tt >= 3.6]
    ramp = np.linspace(0.0, 1.0, len(rest))  # left grip squeezed over the take, right released
    for f, g in zip(rest, ramp):
        f.buttons = {**f.buttons, "left_grip": float(g), "right_grip": 0.0}
    s.set_recording(True, now=3.6)
    _run(s, rest, offset=3.6)
    w = s.world
    g = s.latest.buttons["left_grip"]  # the newest frame is the one the last tick used
    assert g > 0.95 and np.allclose(w.hand_cmd, grip_to_command(g, 0.0, s.grip_full))
    # the finger servos follow the command (nothing in the hand) and the hands sit on the robot hands
    assert abs(w.hand_target[3] - s.grip_full * g * np.pi / 2) < 1e-9 and np.all(w.hand_target[11:] == 0)
    assert abs(w.hand_q[3] - w.hand_target[3]) < 0.05 and np.abs(w.hand_q[11:]).max() < 0.01
    assert np.linalg.norm(w.hand_poses()[0, :3] - w.data.xpos[w.hand_body["left"]]) < 0.01
    st = s.state_message()
    assert len(st["robot"]["qpos"]) == 7 + len(s.scene_message()["joint_names"]) + 14 == 74
    assert st["grasp_mode"] == "physics" and len(st["hand_force"]) == 2
    assert st["hand_cmd"][6:] == [round(s.grip_full * g, 3)] * 6 and st["hand_cmd"][:6] == [0.0] * 6
    s.set_recording(False)
    ep = Episode.load(s.saved[0])
    cmd = ep["sim/hand/cmd"]
    assert cmd.shape[1] == 12 and cmd[-1, 6] > 0.5 and cmd[0, 6] < 0.1 and np.all(cmd[:, :6] == 0)
    assert ep["sim/hand/joint_pos"].shape[1] == 22 and json.loads(ep.group_attrs["sim"]["hand_motor_ids"]) == list(MOTOR_IDS)
    assert ep["sim/hand/joint_target"].shape == ep["sim/hand/joint_pos"].shape
    assert ep["sim/hand/pose"].shape[1:] == (2, 7) and ep["sim/hand/force"].shape[1] == 2
    assert ep.group_attrs["sim"]["grasp_mode"] == "physics"
    # offline: the grips in raw/buttons become the hand channel of the motion exports
    stage_correct(s.saved[0])
    series, static = hand_series(Episode.load(s.saved[0]))
    assert series["hand_cmd"].shape[1] == 12 and series["hand_cmd"][-5:, 6].mean() > 0.5
    assert list(static["hand_motor_ids"]) == list(MOTOR_IDS)


def test_hands_can_be_disabled(cfg, weld_cfg):
    w = SimWorld(load_scene("empty"), {**cfg, "hands": {"enabled": False}})
    assert w.overlay is None and len(w.qpos_ext()) == 38 and len(w.joint_names_ext()) == 31
    assert w.grasp_mode == "weld"  # holding by contact needs the physical hands
    w = SimWorld(load_scene("empty"), weld_cfg)  # weld grasp: visual finger overlays
    assert not w.physical and len(w.qpos_ext()) == 60


def test_locomotion_moves_and_turns_about_the_robot(cfg):
    s = SimSession(cfg, "empty")
    frames, _ = sim_frames("body_2trk", [("idle", "upright", "t_pose", 3.0), ("idle", "upright", "neutral", 4.0)])
    t = frame_times(frames)
    _run(s, [f for f, tt in zip(frames, t) if tt < 3.5])
    assert s.status == "running"
    x0 = s.q_target[:2].copy()
    _run(s, [f for f, tt in zip(frames, t) if 3.5 <= tt < 4.5], axes={"left": np.array([0.0, 1.0])}, offset=3.5)
    moved = s.q_target[:2] - x0
    assert 0.6 < moved[0] < 1.3 and abs(moved[1]) < 0.3  # ~1 m forward at full deflection for 1 s
    p0, yaw0 = s.q_target[:2].copy(), s.loco[2]
    _run(s, [f for f, tt in zip(frames, t) if 4.5 <= tt < 5.5], axes={"right": np.array([1.0, 0.0])}, offset=4.5)
    assert s.loco[2] - yaw0 < -np.radians(60)               # stick right turns right (clockwise)
    assert np.linalg.norm(s.q_target[:2] - p0) < 0.15        # ... about the robot, not the world origin


def test_episode_replay_source_roundtrip(tmp_path):
    frames, clip = sim_frames("sparse_trackers", PROGRAM[:2])
    src = SimulatedXRTSource("sparse_trackers", clip=clip, seed=3)
    (path,) = record(src, tmp_path, {"tracker_profile": "sparse_trackers"}, use_buttons=False, on_status=lambda *_: None)
    rep = EpisodeReplaySource(path)
    buf = FrameBuffer()
    while True:
        try:
            buf.append(rep.poll())
        except StopIteration:
            break
    a = buf.to_arrays()
    ep = Episode.load(path)
    for k in ("raw/headset/pos", "raw/controller/left/quat", "raw/tracker/pos"):
        assert np.allclose(a[k], ep[k], atol=1e-9), k
    assert np.array_equal(a["raw/t_ns"], ep["raw/t_ns"])
