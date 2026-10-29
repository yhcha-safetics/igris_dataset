"""PICO 4 / 4 Ultra -> IGRIS-C 실시간 뷰어 (MuJoCo). XRoboToolkit 바디 트래킹을 프레임 단위로 리타겟합니다.

    python scripts/live_view.py                              # 실기: PC 서비스 실행, 헤드셋 앱 Full body + Send
    python scripts/live_view.py --simulate body_2trk         # 헤드셋 없이: 합성 스트림을 실시간 속도로 재생
    python scripts/live_view.py --record s01 --subject-height 1.72   # 보면서 녹화 (오른손 A 시작/정지, B 마커)
    python scripts/live_view.py --simulate body_2trk --no-viewer --seconds 20   # 성능 확인 (IK ms/frame)

시작 직후 --calib-s 초 동안 움직일 방향을 보고 T-포즈로 정지하세요(바닥·정면·스케일 캘리브레이션).
녹화 파일은 02_record.py와 같은 data/<session>/raw/ep_XXXX.h5 이고, 데이터셋은 run_pipeline.py(배치 보정)로 만듭니다.
"""
import argparse
import json
import time

import _common
import yaml
from igris_pipeline.live import LiveRetargeter, run_live

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="body_2trk", choices=["body_2trk", "body_3trk"])
ap.add_argument("--calib-s", type=float, default=3.0, help="calibration window at the start (still T-pose)")
ap.add_argument("--simulate", default=None, help="body_2trk | body_3trk: synthetic stream instead of the headset")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
ap.add_argument("--no-viewer", action="store_true", help="headless (prints timing statistics)")
ap.add_argument("--human-offset", type=float, default=1.0, help="draw the human skeleton shifted by this in +Y")
ap.add_argument("--record", default=None, metavar="SESSION", help="also record episodes to data/SESSION/raw")
ap.add_argument("--no-buttons", action="store_true", help="with --record: record the whole run (no A toggling)")
ap.add_argument("--subject", default="anonymous")
ap.add_argument("--subject-height", type=float, default=None)
ap.add_argument("--tracker-hw", default="unknown", help="tracker generation, stored as metadata")
ap.add_argument("--usb", action="store_true", help="adb reverse tcp:63901 (enter 127.0.0.1 as PC IP in the headset app)")
a = ap.parse_args()
if a.usb:
    from igris_pipeline.pico.device_probe import usb_tunnel

    print("[usb] adb reverse tcp:63901:", "ok" if usb_tunnel() else "FAILED (headset connected, USB debugging allowed?)")

cfg = lambda n: yaml.safe_load(open(_common.ROOT / "configs" / n, encoding="utf-8"))  # noqa: E731
live = LiveRetargeter(cfg("retarget_igris_c.yaml"), cfg("tracker_profiles.yaml"), a.profile, a.calib_s)

if a.simulate:
    from igris_pipeline.pico.simulate import SimulatedXRTSource

    source = SimulatedXRTSource(a.simulate, seed=a.seed, realtime=True)
else:
    from igris_pipeline.pico.xrt_source import XRTSource

    source = XRTSource().open()

recorder = None
if a.record:
    from igris_pipeline.pico.recorder import RecordingSession, describe_episode

    meta = {"subject": a.subject, "tracker_profile": a.profile, "tracker_hw": a.tracker_hw,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "recorded_with": "live_view"}
    if a.subject_height:
        meta["subject_height"] = a.subject_height
    recorder = RecordingSession(_common.ROOT / "data" / a.record / "raw", meta,
                                use_buttons=not (a.no_buttons or a.simulate))
try:
    stats = run_live(source, live, viewer=not a.no_viewer, human_offset=a.human_offset, max_seconds=a.seconds,
                     recorder=recorder)
finally:
    source.close()
    if recorder is not None:
        for p in recorder.close():
            print(describe_episode(p))
print("[live]", json.dumps(stats.summary()))
