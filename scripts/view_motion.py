"""리타겟된 IGRIS-C 모션 보기 / 렌더링.

    python scripts/view_motion.py data/s01/motions/ep_0000.npz                 # 인터랙티브 뷰어
    python scripts/view_motion.py data/s01/raw/ep_0000.h5 --mp4 out.mp4        # 동영상 (PATH의 ffmpeg, 없으면 imageio-ffmpeg)
    python scripts/view_motion.py data/s01/raw/ep_0000.h5 --sheet out.png --times 2 10 14 17 23 34
With an episode (.h5) the corrected human skeleton is drawn next to the robot (offset in +Y).
"""
import argparse
import time
from pathlib import Path

import _common  # noqa: F401
import mujoco
import numpy as np
from igris_pipeline.episode import Episode
from igris_pipeline.retarget.robot import IgrisModel
from igris_pipeline.sim.mujoco_eval import load_qpos
from igris_pipeline.skeleton import BONES

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--mp4", default=None)
ap.add_argument("--sheet", default=None, help="write a contact sheet PNG of --times frames")
ap.add_argument("--times", type=float, nargs="*", default=None)
ap.add_argument("--human-offset", type=float, default=1.0, help="draw the human skeleton shifted by this in +Y")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
a = ap.parse_args()

qpos, fps = load_qpos(a.path)
human = None
if a.path.endswith(".h5"):
    ep = Episode.load(a.path)
    human = ep["proc/body/pos"]
robot = IgrisModel()
m, d = robot.model, robot.data
# the offscreen framebuffer must be at least the render size (MJCF default 640x480)
m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)


def add_skeleton(scn, t):
    if human is None:
        return
    i = min(int(t / len(qpos) * len(human)), len(human) - 1)
    p = human[i] + np.array([0.0, a.human_offset, 0.0])
    for pa, c in BONES:
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([0.2, 0.6, 1.0, 1.0], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.015, p[pa], p[c])
        scn.ngeom += 1


def set_frame(t):
    d.qpos[:] = qpos[t]
    mujoco.mj_forward(m, d)


def camera(t):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = qpos[t, :3] + np.array([0, a.human_offset / 2 if human is not None else 0, -0.15])
    cam.distance, cam.azimuth, cam.elevation = 3.2, 160, -12
    return cam


if a.sheet or a.mp4:
    r = mujoco.Renderer(m, a.height, a.width)
    if a.sheet:
        times = a.times or list(np.linspace(0, (len(qpos) - 1) / fps, 6))
        imgs = []
        for ts in times:
            t = min(int(ts * fps), len(qpos) - 1)
            set_frame(t)
            r.update_scene(d, camera(t))
            add_skeleton(r.scene, t)
            imgs.append(r.render().copy())
        cols = 3
        rows = int(np.ceil(len(imgs) / cols))
        H, W = imgs[0].shape[:2]
        sheet = np.full((rows * H, cols * W, 3), 255, np.uint8)
        for k, im in enumerate(imgs):
            sheet[(k // cols) * H:(k // cols + 1) * H, (k % cols) * W:(k % cols + 1) * W] = im
        import matplotlib.pyplot as plt
        plt.imsave(a.sheet, sheet)
        print("wrote", a.sheet)
    if a.mp4:
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:  # fall back to the binary bundled with imageio-ffmpeg (requirements.txt)
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{a.width}x{a.height}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p",
                                 a.mp4], stdin=subprocess.PIPE)
        for t in range(len(qpos)):
            set_frame(t)
            r.update_scene(d, camera(t))
            add_skeleton(r.scene, t)
            proc.stdin.write(r.render().tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed (exit {proc.returncode}); {a.mp4} not written")
        print("wrote", a.mp4)
else:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(m, d) as v:
        t0 = time.time()
        while v.is_running():
            t = int(((time.time() - t0) * fps) % len(qpos))
            set_frame(t)
            with v.lock():
                v.user_scn.ngeom = 0
                add_skeleton(v.user_scn, t)
            v.sync()
            time.sleep(1.0 / fps)
