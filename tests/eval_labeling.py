"""Rules vs. rules+learned labeller on noisy simulated PICO data with random motion programs.

    python tests/eval_labeling.py
Train set: clean synthetic clips (random programs, 3 body scales). Test set: unseen random
programs streamed through the body_2trk simulator and corrected by stage 1-2.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from igris_pipeline.episode import Episode  # noqa: E402
from igris_pipeline.labeling.auto_label import label_motion, load_rules, evaluate  # noqa: E402
from igris_pipeline.labeling.review import train_learned  # noqa: E402
from igris_pipeline.pico.correction import correct_episode  # noqa: E402
from igris_pipeline.pico.recorder import buffer_to_episode  # noqa: E402
from igris_pipeline.pico.simulate import SimulatedXRTSource  # noqa: E402
from igris_pipeline.pico.synthetic import generate, random_program  # noqa: E402
from igris_pipeline.pico.xrt_source import FrameBuffer  # noqa: E402

rules = load_rules(ROOT / "configs/labels.yaml")
prof_cfg = yaml.safe_load(open(ROOT / "configs/tracker_profiles.yaml", encoding="utf-8"))
train = [generate(random_program(100 + s), fps=50, subject_scale=sc) for s, sc in
         zip(range(6), (1.0, 0.92, 1.08, 0.96, 1.04, 1.0))]
model = train_learned([], train)
res = {"rules": [], "rules+learned": []}
for s in range(3):
    clip = generate(random_program(500 + s), fps=72.0, subject_scale=(0.95, 1.0, 1.06)[s])
    src = SimulatedXRTSource("body_2trk", clip=clip, seed=s)
    buf = FrameBuffer()
    while True:
        try:
            fr = src.poll()
        except StopIteration:
            break
        buf.append(fr)
    ep = buffer_to_episode(buf, {"tracker_profile": "body_2trk"})
    out = correct_episode(ep, prof_cfg, "body_2trk")
    a = out["arrays"]
    idx = np.clip(np.round(a["proc/t"] * clip.fps).astype(int), 0, len(clip.t) - 1)
    gt = {k: [v[i] for i in idx] for k, v in clip.labels.items()}
    for name, learned in (("rules", None), ("rules+learned", model)):
        lab = label_motion(a["proc/body/pos"], 50.0, rules, learned=learned, stature=out["calibration"]["stature"])
        res[name].append(evaluate(lab["axes"], gt, 50.0, tolerance_s=0.75))
for name, reps in res.items():
    print(name)
    for axis in ("loco", "lower", "upper"):
        acc = np.mean([r[axis]["acc"] for r in reps])
        f1 = np.mean([r[axis]["macro_f1"] for r in reps])
        print(f"   {axis:6s} acc {acc:.3f}  macro-F1 {f1:.3f}")
