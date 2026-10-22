"""1. 녹화: XRoboToolkit 스트림 -> data/<session>/raw/ep_XXXX.h5

    python scripts/02_record.py --session s01 --subject-height 1.72            # 오른손 A 버튼으로 시작/정지
    python scripts/02_record.py --session s01 --no-buttons --seconds 60
    python scripts/02_record.py --session sim01 --simulate body_2trk           # 헤드셋 없이 시뮬레이션
"""
import argparse
import json
import time

import _common
from igris_pipeline.pico.device_probe import probe
from igris_pipeline.pico.recorder import describe_episode, record

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
ap.add_argument("--subject", default="anonymous")
ap.add_argument("--subject-height", type=float, default=None, help="m; needed for 3-point / sparse profiles")
ap.add_argument("--profile", default="auto", help="auto | body_2trk | body_3trk | sparse_trackers | hmd_3pt")
ap.add_argument("--body-trackers", type=int, default=None)
ap.add_argument("--tracker-hw", default="unknown")
ap.add_argument("--no-buttons", action="store_true", help="record immediately instead of A-button toggling")
ap.add_argument("--seconds", type=float, default=None)
ap.add_argument("--hands", action="store_true", help="also record 26-joint hand tracking")
ap.add_argument("--simulate", default=None, help="simulate a profile instead of the live stream")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

if a.simulate:
    from igris_pipeline.pico.simulate import SimulatedXRTSource

    def make_source():
        return SimulatedXRTSource(a.simulate, seed=a.seed)
else:
    from igris_pipeline.pico.xrt_source import XRTSource

    live = XRTSource(with_hands=a.hands).open()

    def make_source():
        return live

rep = probe(2.0, source=make_source(), declared_body_trackers=a.body_trackers, tracker_hw=a.tracker_hw)
profile = rep.profile if a.profile == "auto" else a.profile
print(f"[probe] profile={profile} rate={rep.packet_rate_hz:.1f}Hz body={rep.body_tracking} "
      f"trackers={rep.object_trackers} {rep.tracker_serials}")
for n in rep.notes:
    print("  note:", n)
if profile == "none":
    raise SystemExit("no PICO data stream - see notes above")
meta = {"subject": a.subject, "tracker_profile": profile, "device_profile": json.loads(rep.to_json()),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
if a.subject_height:
    meta["subject_height"] = a.subject_height
out = _common.ROOT / "data" / a.session / "raw"
paths = record(make_source(), out, meta, max_seconds=a.seconds, use_buttons=not (a.no_buttons or a.simulate),
               with_hands=a.hands)
for p in paths:
    print(describe_episode(p))
