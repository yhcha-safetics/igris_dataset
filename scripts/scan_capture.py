"""Record a depth-camera capture for scripts/scan_build.py (Intel RealSense D4xx / L515; docs/scan_pipeline.md).

    python scripts/scan_capture.py data/scans/lab01 --seconds 60             # live camera
    python scripts/scan_capture.py data/scans/lab01 --bag rec.bag            # convert a realsense-viewer recording
    python scripts/scan_capture.py --list                                     # connected cameras

Protocol: sweep the camera slowly (< 0.5 m/s, no fast turns) over the scene, floor and the table edges
in view, without touching anything (the scan phase); note the time, then set the camera down or keep
holding it and interact with the objects (the interaction phase). Pass that time to scan_build.py --split.
Depth is aligned to the colour camera; other cameras: write the capture format of igris_scan/rgbd.py.
"""
import argparse
import time

import _common  # noqa: F401
import numpy as np

from igris_scan.rgbd import CaptureWriter, Intrinsics, import_realsense_bag

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?", help="capture folder to create")
ap.add_argument("--bag", default=None, help="convert this .bag instead of recording")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--every", type=int, default=1, help="keep every n-th frame")
ap.add_argument("--countdown", type=float, default=3.0, help="s before recording starts")
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

import pyrealsense2 as rs  # noqa: E402

if a.list:
    devs = rs.context().query_devices()
    for d in devs:
        print(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number),
              "fw", d.get_info(rs.camera_info.firmware_version))
    print(f"{len(devs)} camera(s)")
    raise SystemExit(0)
if not a.out:
    ap.error("give the output folder")
if a.bag:
    out = import_realsense_bag(a.bag, a.out, every=a.every)
    print(f"converted {a.bag} -> {out}")
    raise SystemExit(0)

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
prof = pipe.start(cfg)
dev = prof.get_device()
depth_sensor = dev.first_depth_sensor()
if depth_sensor.supports(rs.option.visual_preset):  # D4xx: high accuracy preset (fewer, better depth points)
    depth_sensor.set_option(rs.option.visual_preset, 3)
scale = depth_sensor.get_depth_scale()
ci = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
intr = Intrinsics(ci.width, ci.height, ci.fx, ci.fy, ci.ppx, ci.ppy)
align = rs.align(rs.stream.color)
name = dev.get_info(rs.camera_info.name)
print(f"{name}: {a.width}x{a.height} @ {a.fps} fps, depth scale {scale * 1000:.3f} mm")
for k in range(int(a.countdown), 0, -1):
    print(f"recording in {k} ...")
    time.sleep(1.0)
n = 0
t_start = None
try:
    with CaptureWriter(a.out, intr, depth_scale=1.0 / scale, source="realsense", device=name) as w:
        print("recording (Ctrl+C to stop)")
        last = time.time()
        while True:
            fs = align.process(pipe.wait_for_frames(5000))
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if not d or not c:
                continue
            t = fs.get_timestamp() * 1e-3
            t_start = t if t_start is None else t_start
            if t - t_start > a.seconds:
                break
            n += 1
            if (n - 1) % a.every:
                continue
            depth = np.asanyarray(d.get_data()).copy()
            w.add(np.asanyarray(c.get_data()).copy(), depth, t)
            if time.time() - last > 1.0:
                last = time.time()
                print(f"  {t - t_start:5.1f} s  {len(w.t)} frames  depth valid {np.mean(depth > 0) * 100:.0f} %")
except KeyboardInterrupt:
    pass
finally:
    pipe.stop()
print(f"saved {a.out}")
