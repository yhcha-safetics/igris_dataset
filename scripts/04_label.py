"""2. 자세/동작 자동 라벨링 + 검수 도구.

    python scripts/04_label.py data/s01                              # 자동 라벨 + PNG 타임라인
    python scripts/04_label.py data/s01/raw/ep_0000.h5 --export-review review.csv
    python scripts/04_label.py data/s01/raw/ep_0000.h5 --import-review review.csv   # 수정본 반영
    python scripts/04_label.py data/s01 --train models/labeler.pkl --with-synthetic # 학습형 라벨러
    python scripts/04_label.py data/s02 --learned models/labeler.pkl               # 규칙+학습 블렌딩
"""
import argparse
from pathlib import Path

import _common
import yaml
from igris_pipeline.episode import Episode
from igris_pipeline.labeling import review
from igris_pipeline.pipeline import stage_label

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--learned", default=None)
ap.add_argument("--export-review", default=None)
ap.add_argument("--import-review", default=None)
ap.add_argument("--train", default=None, help="train LearnedLabeler on reviewed episodes, save to this path")
ap.add_argument("--with-synthetic", action="store_true", help="add synthetic clips to the training set")
ap.add_argument("--overwrite-reviewed", action="store_true", help="also re-label episodes whose labels were reviewed")
a = ap.parse_args()
paths = _common.episode_paths(a.paths)
with open(_common.ROOT / "configs" / "labels.yaml", encoding="utf-8") as f:
    rules = yaml.safe_load(f)

if a.train:
    eps = [Episode.load(p) for p in paths]
    eps = [e for e in eps if e.group_attrs.get("labels", {}).get("reviewed")]
    syn = []
    if a.with_synthetic:
        from igris_pipeline.pico.synthetic import generate, random_program

        # generate()'s seed does not change the motion; vary the program per clip
        syn = [generate(random_program(s), fps=50, subject_scale=sc) for s, sc in ((0, 1.0), (1, 0.92), (2, 1.08))]
    print(f"training on {len(eps)} reviewed episodes + {len(syn)} synthetic clips")
    Path(a.train).parent.mkdir(parents=True, exist_ok=True)
    review.train_learned(eps, syn, a.train)
    print("saved", a.train)
elif a.import_review:
    ep = Episode.load(paths[0])
    review.import_review_csv(ep, a.import_review, rules)
    ep.save(paths[0])
    print("applied", a.import_review, "->", paths[0], "(marked reviewed)")
elif a.export_review:
    ep = Episode.load(paths[0])
    review.export_review_csv(ep, a.export_review)
    print("wrote", a.export_review)
else:
    for p in paths:
        segs = stage_label(p, rules, a.learned, overwrite_reviewed=a.overwrite_reviewed)
        print(f"{p}: {len(segs)} segments")
        for s in segs:
            flag = "  <- review" if s["needs_review"] else ""
            print(f"   {s['t_start']:7.2f}-{s['t_end']:7.2f}s  {s['name']:<28s} p={s['confidence']:.2f}{flag}")
