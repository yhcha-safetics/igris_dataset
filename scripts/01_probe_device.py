"""1-1. PICO 연결 확인 및 헤드셋/컨트롤러/모션 트래커 감지 -> 트래커 프로파일 선택.

    python scripts/01_probe_device.py                      # 실기 (XRoboToolkit PC Service)
    python scripts/01_probe_device.py --body-trackers 3 --tracker-hw "PICO Motion Tracker (2024)"
    python scripts/01_probe_device.py --simulate sparse_trackers   # 헤드셋 없이 시뮬레이션 스트림
"""
import argparse

import _common  # noqa: F401
from igris_pipeline.pico.device_probe import probe

ap = argparse.ArgumentParser()
ap.add_argument("--duration", type=float, default=3.0)
ap.add_argument("--body-trackers", type=int, default=None, help="trackers worn in body-tracking mode (2 or 3)")
ap.add_argument("--tracker-hw", default="unknown", help="tracker generation, stored as metadata")
ap.add_argument("--simulate", default=None, help="profile name to simulate instead of the live stream")
ap.add_argument("--out", default=None, help="write the report JSON here")
ap.add_argument("--usb", action="store_true", help="adb reverse tcp:63901 (enter 127.0.0.1 as PC IP in the headset app)")
a = ap.parse_args()

src = None
if a.usb:
    from igris_pipeline.pico.device_probe import usb_tunnel

    print("[usb] adb reverse tcp:63901:", "ok" if usb_tunnel() else "FAILED (headset connected, USB debugging allowed?)")
if a.simulate:
    from igris_pipeline.pico.simulate import SimulatedXRTSource

    src = SimulatedXRTSource(a.simulate)
rep = probe(a.duration, source=src, declared_body_trackers=a.body_trackers, tracker_hw=a.tracker_hw)
print(rep.to_json())
print(f"\n=> tracker profile: {rep.profile}")
if a.out:
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(rep.to_json())
