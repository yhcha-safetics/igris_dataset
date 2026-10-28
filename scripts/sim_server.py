"""PC simulation server: PICO tracking -> IGRIS-C in a MuJoCo scene -> state to the PICO app.

    python scripts/sim_server.py --source xrt --scene tabletop --viewer              # XRoboToolkit stream
    python scripts/sim_server.py --source udp --scene room --record-dir data/sim01/raw  # PICO app (docs/sim_protocol.md)
    python scripts/sim_server.py --source sim:body_2trk --viewer --show-human         # no headset
    python scripts/sim_server.py --source replay:data/s01/raw/ep_0000.h5 --viewer
    python scripts/sim_server.py --source sim:hmd_3pt --offline --duration 20 --no-network   # benchmark

Stand still for the first seconds (calibration). Controller: A record, B marker, grips grasp,
left stick move, right stick turn, hold Y recalibrate. Commands can also be sent with
scripts/sim_ctl.py (load_scene, set_mode, reset, record, ...).
"""
import argparse
import time

import _common  # noqa: F401
import numpy as np

from igris_sim.protocol import ROBOT_ASSETS, AssetServer, Hub, local_ip
from igris_sim.robot_export import robot_description
from igris_sim.session import SimSession, load_sim_config, summary_line
from igris_sim.sources import PacedSource, make_source
from igris_sim.streaming import _rot_z
from igris_pipeline.skeleton import BONES

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="udp", help="udp | xrt | sim:<profile>[:seed] | replay:<episode.h5>")
ap.add_argument("--scene", default="tabletop", help="name under scenes/ or a scene.json path")
ap.add_argument("--mode", default=None, help="kinematic | harness (default: configs/sim.yaml)")
ap.add_argument("--profile", default="auto", help="auto | body_2trk | body_3trk | sparse_trackers | hmd_3pt")
ap.add_argument("--subject-height", type=float, default=None, help="m; needed for hmd_3pt / sparse_trackers")
ap.add_argument("--body-trackers", type=int, default=None, help="3 if a waist tracker is worn in body mode")
ap.add_argument("--subject", default="anonymous")
ap.add_argument("--record-dir", default=None, help="episodes go here (A toggles recording), e.g. data/sim01/raw")
ap.add_argument("--autorecord", action="store_true", help="start an episode as soon as calibration is done")
ap.add_argument("--config", default=None, help="default: configs/sim.yaml")
ap.add_argument("--port", type=int, default=None)
ap.add_argument("--asset-port", type=int, default=None)
ap.add_argument("--no-network", action="store_true")
ap.add_argument("--viewer", action="store_true", help="desktop MuJoCo viewer")
ap.add_argument("--show-human", action="store_true", help="draw the streamed skeleton in the viewer")
ap.add_argument("--human-offset", type=float, default=1.0, help="viewer: skeleton shift in +Y (m)")
ap.add_argument("--offline", action="store_true", help="run a sim/replay source as fast as possible")
ap.add_argument("--duration", type=float, default=None, help="stop after this many (simulated) seconds")
ap.add_argument("--status-every", type=float, default=2.0)
a = ap.parse_args()

if a.autorecord and not a.record_dir:
    ap.error("--autorecord needs --record-dir")
if a.offline and a.source.split(":")[0] not in ("sim", "replay"):
    ap.error("--offline needs a sim:<profile> or replay:<episode.h5> source")
cfg = load_sim_config(a.config) if a.config else load_sim_config()
net = cfg.get("network", {})
session = SimSession(cfg, a.scene, a.profile, a.mode, a.subject_height, a.body_trackers, a.record_dir, a.subject)
src = make_source(a.source)
if src is not None and not a.offline and not a.source.startswith("xrt"):
    src = PacedSource(src)
hub = assets = None
asset_base = None
if not a.no_network:
    hub = Hub(a.port or int(net.get("port", 9870)), client_timeout=float(net.get("client_timeout_s", 3.0)))
    assets = AssetServer(a.asset_port or int(net.get("asset_port", 9871)))
    assets.add_files(robot_description(session.world)[1], ROBOT_ASSETS)  # the robot is the same in every scene
    asset_base = f"http://{local_ip()}:{assets.port}/"
    print(f"listening on UDP {local_ip()}:{hub.port}, scene assets at {asset_base}")

viewer = None
if a.viewer:
    import mujoco
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(session.world.model, session.world.data)
    viewer_world = session.world


def draw_human(scn):
    import mujoco
    if session.pose is None:
        return
    Rw = _rot_z(session.loco[2])
    p = session.pose.pos @ Rw.T + np.array([session.loco[0], session.loco[1] + a.human_offset, 0.0])
    for pa, c in BONES:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.015, p[pa], p[c])
        scn.ngeom += 1


dt = 1.0 / session.fps
t_start = time.time()
next_tick = time.perf_counter()
last_status = 0.0
pending = None
t_sim = t0_sim = None
ended = False
print(f"scene={session.world.scene.name} mode={session.world.mode} source={a.source} -- stand still to calibrate")
try:
    while True:
        # 1) tracking frames
        if src is not None and not ended:
            try:
                if a.offline:
                    if pending is None:
                        pending = src.poll()
                    t_sim = pending.t_ns * 1e-9 if t_sim is None else t_sim + dt
                    t0_sim = t_sim if t0_sim is None else t0_sim
                    while pending is not None and pending.t_ns * 1e-9 <= t_sim:
                        session.push_frame(pending, rx_time=t_sim)
                        pending = src.poll()
                else:
                    while (fr := src.poll()) is not None:
                        session.push_frame(fr)
            except StopIteration:
                ended = True
        if hub is not None:
            frames, cmds = hub.poll()
            for fr, _ in frames:
                session.push_frame(fr)
            for msg, addr in cmds:
                reply = session.command(msg)
                if reply.get("type") == "scene":
                    # the app may prefer http://<address it reached the server at>:asset_port/
                    reply["assets"], reply["asset_port"] = asset_base, assets.port
                hub.send(reply, addr)
        # 2) simulation + state out
        now = t_sim if a.offline and t_sim is not None else time.time()
        session.tick(now)
        if a.autorecord and session.status == "running" and not session.saved and not session.recorder.active:
            session.set_recording(True, now)
        if hub is not None and hub.clients:  # the state message is only built for listeners
            hub.broadcast(session.state_message())
        if viewer is not None:
            if not viewer.is_running():
                break
            if session.world is not viewer_world:  # scene / mode changed: reopen on the new model
                viewer.close()
                viewer = mujoco.viewer.launch_passive(session.world.model, session.world.data)
                viewer_world = session.world
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                if a.show_human:
                    draw_human(viewer.user_scn)
            viewer.sync()
        wall = time.time()
        if wall - last_status >= a.status_every:
            last_status = wall
            print(summary_line(session) + (f" clients={len(hub.clients)}" if hub is not None else "")
                  + (f" net_errors={hub.errors} ({hub.last_error})" if hub is not None and hub.errors else ""),
                  flush=True)
        if a.duration is not None:
            run_t = (t_sim - t0_sim) if a.offline and t_sim is not None else wall - t_start
            if run_t >= a.duration:
                break
        if ended:
            break
        # 3) pacing (offline runs as fast as possible)
        if not a.offline:
            next_tick += dt
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()  # overrun: do not try to catch up
except KeyboardInterrupt:
    pass
finally:
    session.close()
    print(summary_line(session))
    for p in session.saved:
        print("episode:", p)
    if hub is not None:
        hub.close()
    if assets is not None:
        assets.close()
    if viewer is not None:
        viewer.close()
    if src is not None:
        src.close()
