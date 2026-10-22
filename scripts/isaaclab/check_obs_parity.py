"""Isaac Lab <-> MuJoCo observation parity check (run with the project .venv, MuJoCo side).

1. In Isaac (conda isaaclab env) dump the first observation of env 0 (+ state, policy action and a short
   zero-action rollout):
       C:\\ProgramData\\Anaconda3\\envs\\isaaclab\\python.exe scripts\\isaaclab\\play_eval.py --headless ^
           --checkpoint <run>\\model_X.pt --motion <npz> --dump_obs <run>\\exported\\isaac_obs_dump.npz
2. Compare in MuJoCo:
       .venv\\Scripts\\python.exe scripts\\isaaclab\\check_obs_parity.py ^
           --dump <run>\\exported\\isaac_obs_dump.npz --policy_dir <run>\\exported

Checks:
  [config]   joint order / observation term order of the dump == policy.json
  [state]    MuJoCo state initialised from the same reference frame vs the Isaac state (root, joints, all
             24 articulation bodies -> catches URDF/MJCF kinematic differences)
  [obs t=0]  every observation term, MuJoCo vs Isaac (PASS if max |diff| < --tol)
  [policy]   ONNX / TorchScript action on the Isaac obs vs the Isaac (rsl-rl) action; ONNX on the MuJoCo obs
  [action]   action -> joint-target mapping (absolute / residual) for a random probe action at t=0 and for the
             zero-action rollout targets, MuJoCo vs Isaac joint_pos_target
  [rollout]  zero-action rollout: per-term max |diff| over time (physics divergence, informative only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from igris_pipeline.sim.mujoco_policy import (  # noqa: E402
    MujocoPolicyRunner,
    OnnxPolicy,
    TorchScriptPolicy,
    load_reference,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True)
    p.add_argument("--policy_dir", required=True)
    p.add_argument("--mjcf", default=None)
    p.add_argument("--foot_geom", choices=["urdf", "mjcf"], default="urdf")
    p.add_argument("--joint_dynamics", choices=["isaac", "isaac_static", "mjcf", "none"], default="isaac")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    dump = dict(np.load(args.dump, allow_pickle=False))
    runner = MujocoPolicyRunner(policy_dir=args.policy_dir, mjcf_path=args.mjcf, foot_geom=args.foot_geom,
                                joint_dynamics=args.joint_dynamics, zero_action=True)
    meta = runner.meta
    report: dict = {"dump": os.path.abspath(args.dump), "policy_dir": os.path.abspath(args.policy_dir), "tol": args.tol}
    ok = True

    # ---------------------------------------------------------------- config
    jn = [str(s) for s in dump["joint_names"]]
    tn = [str(s) for s in dump["obs_term_names"]]
    cfg_ok = jn == meta["joint_names"] and tn == meta["observation_order"]
    report["config_ok"] = cfg_ok
    ok &= cfg_ok
    print(f"[config] joint order match={jn == meta['joint_names']}  obs term order match={tn == meta['observation_order']}")

    # ---------------------------------------------------------------- state at t=0
    motion_file = str(dump["motion_file"])
    frame = int(dump["time_step"])
    ref = load_reference(motion_file, meta)
    runner.reset(ref, frame)
    d, m = runner.data, runner.model
    st = {
        "root_pos": np.abs(d.qpos[0:3] - dump["root_pos"]).max(),
        "root_quat": min(np.abs(d.qpos[3:7] - dump["root_quat"]).max(), np.abs(d.qpos[3:7] + dump["root_quat"]).max()),
        "joint_pos": np.abs(d.qpos[runner.qadr] - dump["joint_pos"]).max(),
        "joint_vel": np.abs(d.qvel[runner.dadr] - dump["joint_vel"]).max(),
    }
    body_err = {}
    for i, name in enumerate(str(s) for s in dump["body_names"]):
        body_err[name] = float(np.linalg.norm(d.xpos[m.body(name).id] - dump["body_pos"][i]))
    st["body_pos_max"] = max(body_err.values())
    worst = sorted(body_err.items(), key=lambda kv: -kv[1])[:4]
    report["state_t0"] = {k: float(v) for k, v in st.items()}
    report["state_t0_body_pos_err"] = body_err
    print("[state t=0] " + "  ".join(f"{k}={v:.2e}" for k, v in st.items()))
    print(f"[state t=0] worst bodies: {', '.join(f'{n}={e * 1000:.1f}mm' for n, e in worst)}")

    # ---------------------------------------------------------------- obs t=0, term by term
    terms = runner.observation_terms()
    obs_mj = runner.observation()
    report["obs_t0"] = {}
    print(f"[obs t=0] {'term':22s} {'dim':>4s} {'max|diff|':>10s}  result")
    for t in meta["observation"]["terms"]:
        name = t["name"]
        isaac = dump[f"term/{name}"].reshape(-1)
        mj = np.asarray(terms[name]).reshape(-1)
        diff = float(np.abs(isaac - mj).max())
        passed = diff < args.tol
        ok &= passed
        report["obs_t0"][name] = {"dim": t["dim"], "max_abs_diff": diff, "pass": passed}
        print(f"[obs t=0] {name:22s} {t['dim']:4d} {diff:10.2e}  {'PASS' if passed else 'FAIL'}")
    full = float(np.abs(obs_mj - dump["obs"]).max())
    report["obs_t0_full_vector_max_abs_diff"] = full
    ok &= full < args.tol
    print(f"[obs t=0] full concatenated vector ({obs_mj.shape[0]}): max|diff|={full:.2e}")

    # ---------------------------------------------------------------- policy export
    base = os.path.abspath(args.policy_dir)
    onnx = OnnxPolicy(os.path.join(base, meta["files"]["onnx"]))
    a_isaac = dump["action"]
    a_onnx = onnx(dump["obs"])
    a_onnx_mj = onnx(obs_mj)
    pol = {"onnx_vs_rslrl_on_isaac_obs": float(np.abs(a_onnx - a_isaac).max()),
           "onnx_mujoco_obs_vs_rslrl_isaac_obs": float(np.abs(a_onnx_mj - a_isaac).max())}
    try:
        jit = TorchScriptPolicy(os.path.join(base, meta["files"]["torchscript"]))
        pol["jit_vs_rslrl_on_isaac_obs"] = float(np.abs(jit(dump["obs"]) - a_isaac).max())
    except Exception as e:  # torch missing etc.
        pol["jit_error"] = repr(e)
    report["policy"] = pol
    ok &= pol["onnx_vs_rslrl_on_isaac_obs"] < 1e-4
    print("[policy] " + "  ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in pol.items()))

    # ---------------------------------------------------------------- action -> target mapping
    act = {"action_mode": meta.get("action_mode", "absolute"), "isaac_action_term": str(dump.get("action_term", ""))}
    if "probe_action" in dump:
        runner.reset(ref, frame)
        t_mj = runner.action_to_target(dump["probe_action"])
        act["probe_target_max_abs_diff"] = float(np.abs(t_mj - dump["probe_target"]).max())
        ok &= act["probe_target_max_abs_diff"] < args.tol
    report["action_mapping"] = act
    print("[action] " + "  ".join(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}" for k, v in act.items()))

    # ---------------------------------------------------------------- zero-action rollout (informative)
    seq = dump["zero_action_obs"]
    seq_q = dump["zero_action_joint_pos"]
    seq_t = dump["zero_action_target"] if "zero_action_target" in dump else np.zeros((0, meta["num_joints"]))
    runner.reset(ref, frame)
    rows, qj, tj = [], [], []
    zero = np.zeros(meta["num_joints"])
    for j in range(seq.shape[0]):
        o = runner.observation()
        rows.append(np.abs(o - seq[j]))
        qj.append(float(np.abs(runner.data.qpos[runner.qadr] - seq_q[j]).max()))
        if j < seq_t.shape[0]:  # target the MuJoCo runner applies during step j vs Isaac's joint_pos_target
            tj.append(float(np.abs(runner.action_to_target(zero) - seq_t[j]).max()))
        if j + 1 < seq.shape[0]:
            runner.step(zero)
            runner.k = min(runner.k + 1, ref.length - 1)
    rows = np.stack(rows)
    roll = {}
    for t in meta["observation"]["terms"]:
        a, b = t["slice"]
        roll[t["name"]] = [float(x) for x in rows[:, a:b].max(axis=1)]
    report["zero_action_rollout_max_abs_diff_per_step"] = roll
    report["zero_action_rollout_joint_pos_max_abs_diff"] = qj
    if tj:
        report["zero_action_target_max_abs_diff"] = max(tj)
        ok &= max(tj) < args.tol
        print(f"[action] zero-action joint targets over {len(tj)} steps: max|diff|={max(tj):.2e}")
    n = seq.shape[0] - 1
    show = sorted({0, 1, 2, 5, 10, n} & set(range(n + 1)))
    print(f"[rollout] zero-action, max|diff| per term at steps {show} (physics divergence, informative):")
    for name, v in roll.items():
        print(f"[rollout]   {name:22s} " + " ".join(f"{v[s]:.1e}" for s in show))
    print("[rollout]   joint q (rad)          " + " ".join(f"{qj[s]:.1e}" for s in show))

    report["pass"] = bool(ok)
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.dump)), "parity_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[parity] {'PASS' if ok else 'FAIL'} (tol {args.tol}); report: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
