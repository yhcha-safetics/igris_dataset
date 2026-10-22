"""IK match tables for PICO skeleton -> IGRIS-C.

The tables follow GMR's semantics (robot_body: [human_joint, pos_weight, rot_weight,
pos_offset_xyz, rot_offset_wxyz]) so Robros' own `xrobot_to_igris_c.json` can be reused;
`from_gmr_json()` converts its rotation offsets into this pipeline's joint-frame
convention (see pico/frames.py). Target for a robot body:
    q_target = q_human * rot_offset ;  p_target = p_human_scaled + R(q_target) * pos_offset
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from ..pico.frames import gmr_offset_to_pipeline

GMR_TO_PICO = {
    "Pelvis": "pelvis", "Left_Hip": "left_hip", "Right_Hip": "right_hip", "Spine1": "spine1",
    "Left_Knee": "left_knee", "Right_Knee": "right_knee", "Spine2": "spine2", "Left_Ankle": "left_ankle",
    "Right_Ankle": "right_ankle", "Spine3": "spine3", "Left_Foot": "left_foot", "Right_Foot": "right_foot",
    "Neck": "neck", "Left_Collar": "left_collar", "Right_Collar": "right_collar", "Head": "head",
    "Left_Shoulder": "left_shoulder", "Right_Shoulder": "right_shoulder", "Left_Elbow": "left_elbow",
    "Right_Elbow": "right_elbow", "Left_Wrist": "left_wrist", "Right_Wrist": "right_wrist",
    "Left_Hand": "left_hand", "Right_Hand": "right_hand",
}


def from_gmr_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        g = json.load(f)

    def conv_table(tab):
        out = {}
        for body, (human, wp, wr, po, ro) in tab.items():
            q = gmr_offset_to_pipeline(np.array(ro, dtype=np.float64))
            q = q if q[0] >= 0 else -q
            out[body] = [GMR_TO_PICO[human], float(wp), float(wr), [float(x) for x in po],
                         [round(float(x), 6) for x in q]]
        return out

    return {
        "source": f"converted from {Path(path).name} (robrosinc/GMR); rotation offsets in pipeline convention",
        "robot_root": g["robot_root_name"],
        "human_root": GMR_TO_PICO[g["human_root_name"]],
        "ground_height": float(g.get("ground_height", 0.0)),
        "gmr_scaling": {
            "human_height_assumption": float(g["human_height_assumption"]),
            "scale_table": {GMR_TO_PICO[k]: float(v) for k, v in g["human_scale_table"].items()},
        },
        "stage1": conv_table(g["ik_match_table1"]) if g.get("use_ik_match_table1", True) else {},
        "stage2": conv_table(g["ik_match_table2"]) if g.get("use_ik_match_table2", True) else {},
    }


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
