# IGRIS-C 모션 트래킹 (Stage 4, Isaac Lab 2.3 / Isaac Sim 5.1)

PICO-VR → 보정 → 라벨링 → 리타겟(stage 3, NPZ) → **Isaac Lab RL 트래킹 정책 학습 (이 패키지)** → ONNX/JSON export →
MuJoCo sim2sim 검증 (`igris_pipeline/sim/mujoco_policy.py`).

구조는 BeyondMimic `whole_body_tracking`(MIT, HybridRobotics / Isaac Lab Project Developers)을 참고해 IGRIS-C 용으로
새로 작성했다 (출처는 각 파일 헤더에 표기). Isaac Gym 은 사용하지 않는다 (Windows 미지원).

```
igris_pipeline/isaaclab_ext/
  __init__.py            gym 등록: Igris-C-Tracking-Flat-v0, Igris-C-Tracking-Flat-Play-v0 (문자열 entry point)
  constants.py           순수 파이썬: 경로, 관절/바디 이름, 기본 자세, robros.py 액추에이터 표 (복사본)
  assets.py              IGRIS_C_CFG (UrdfFileCfg → USD 변환 캐시, 지연 PD 액추에이터)
  actuators.py           FastDelayedPDActuator (DelayedPD 와 동일 동작, GPU 동기화 없음)
  motion.py              NPZ 로더 (이름 기반 매핑, 다중 클립, fps 리샘플, 세그먼트 로더)
  mdp/                   MotionCommand, 관측/보상/종료/이벤트 항
  tracking_env_cfg.py    환경 설정 (학습 / PLAY)
  agents/rsl_rl_ppo_cfg.py  PPO 설정 (rsl-rl-lib 3.x)
  export.py              ONNX + TorchScript + policy.json
  metrics.py             numpy 지표 (Isaac 평가와 MuJoCo 평가가 같은 코드 사용)
  make_test_motion.py    합성 테스트 모션 생성기 (.venv, mujoco)
scripts/isaaclab/{train.py, play_eval.py, check_obs_parity.py}
scripts/eval_mujoco_policy.py  → igris_pipeline/sim/mujoco_policy.py
```

## 사용법 (프로젝트 루트 `C:\Users\c\igris_dataset` 에서)

```bat
set ISAAC=C:\ProgramData\Anaconda3\envs\isaaclab\python.exe
set OMNI_KIT_ACCEPT_EULA=YES

:: 0) (선택) 합성 테스트 모션
.venv\Scripts\python.exe -m igris_pipeline.isaaclab_ext.make_test_motion --out data\sample\motions\test_sine.npz

:: 1) 학습  (--motion: 파일 / 디렉터리(재귀) / glob / 콤마 목록 / .txt 목록, 클립은 균등 샘플링)
%ISAAC% scripts\isaaclab\train.py --headless --num_envs 4096 --max_iterations 30000 --motion data\sample\motions ^
       --action_mode residual --run_name <이름>
::    (--action_mode absolute|residual, --action_scale beyondmimic|gmt, --motion_z_offset, --seed, --resume <ckpt>)
::    → logs\isaaclab\igris_c_tracking\<날짜>_<run_name>\model_*.pt, params\, exported\policy.{onnx,pt,json}

:: 2) Isaac 평가 + export (+ 관측 dump)
%ISAAC% scripts\isaaclab\play_eval.py --headless --checkpoint logs\isaaclab\igris_c_tracking\<run>\model_XXXX.pt ^
       --motion data\sample\motions --dump_obs logs\isaaclab\igris_c_tracking\<run>\exported\isaac_obs_dump.npz
::    → <run>\exported\policy.onnx / policy.pt / policy.json,  <run>\eval\metrics_model_XXXX.json

:: 3) Isaac ↔ MuJoCo 관측 일치 검사 (t=0, 항목별)
.venv\Scripts\python.exe scripts\isaaclab\check_obs_parity.py --dump <run>\exported\isaac_obs_dump.npz --policy_dir <run>\exported

:: 4) MuJoCo sim2sim 평가 (같은 지표)
.venv\Scripts\python.exe scripts\eval_mujoco_policy.py --policy_dir <run>\exported --motion data\sample\motions
::    옵션: --backend onnx|jit, --foot_geom urdf|mjcf, --joint_dynamics isaac|isaac_static|mjcf|none,
::          --render, --zero_action, --save_trajectory out.npz

:: 5) A/B 표 (Isaac/MuJoCo 나란히, 클립·세그먼트별 markdown)
.venv\Scripts\python.exe scripts\isaaclab\compare_metrics.py --run A=<runA> --run B=<runB> --clip sim_body2trk_full
```

Isaac 스크립트는 `OMNI_KIT_ACCEPT_EULA=YES` 가 필요하며 Isaac Sim 기동에 ~10 s, USD 첫 변환에 ~5 s 걸린다.
`--checkpoint` 를 생략하면 가장 최근 run 의 마지막 체크포인트, `--motion` 을 생략하면 학습에 쓴 모션 목록
(`params/motions.txt`)을 사용한다. TensorBoard: `%ISAAC% -m tensorboard.main --logdir logs\isaaclab`.

## 설계 요약

* **로봇**: URDF(손목/목 fixed) → 23 DoF. `UrdfFileCfg`(merge_fixed_joints, convex_hull, force drive, 0 gain)로 스폰 시
  USD 변환(`logs/isaaclab/usd_cache/igris_c_v2`, URDF/설정이 바뀔 때만 재변환). 손목·손 링크는 `Link_Elbow_Pitch_*`
  에, 목은 `base_link` 에 병합되어 24 바디. `IGRIS_C_SPAWN=usd` 로 Robros 가 배포한 USD 를 대신 쓸 수 있다.
* **액추에이터**: robros.py 의 `DelayedPDActuatorCfg` 값 그대로(kp/kd/effort/velocity/armature/정마찰/점성마찰,
  학습 시 0–2 physics step 지연). 명시적 PD(τ = clip(kp(q*−q) − kd·q̇))라 MuJoCo 러너와 동일한 법칙. PLAY 는 지연 0.
  자기충돌 off (MJCF 와 동일). 지연 버퍼는 `actuators.FastDelayedPDActuatorCfg` 로 재구현(동작 동일) —
  Isaac Lab 의 `DelayBuffer` 는 substep 마다 `.item()` GPU 동기화를 12번 해서 이 PC(GPU 공유, WDDM)에서
  env.step 이 202 ms → 129 ms 로 줄었다 (64 env 기준, 4096 env 학습 ≈ 18k steps/s).
* **앵커 = `base_link`(토르소, 루트, IMU 바디)**: 루트 상태에서 바로 얻을 수 있어(FK 불필요) 실기/MuJoCo 관측 재현이
  쉽다. 골반(`Link_Waist_Pitch`)은 추적 바디로 포함.
* **추적 바디(12)**: 골반, 좌/우 hip_roll·knee·ankle_roll, base_link, 좌/우 shoulder_roll·elbow.
  **엔드이펙터**: 손(팔꿈치 링크 + 고정 오프셋 (−0.01, ±0.0059, −0.3135); NPZ 의 `Left_Hand/Right_Hand` 를 참조로 사용)
  + 발(ankle_roll).
* **시뮬레이션**: dt 0.005 s(200 Hz), decimation 4(정책 50 Hz), 에피소드 10 s.
* **액션 모드** (`train.py --action_mode`, policy.json `action_mode`):
  * `absolute` (BeyondMimic 방식): q* = q_default + scale·a, scale 프리셋 `beyondmimic` = 0.25·effort/kp
    (팔 0.123, 허리 roll 0.082, 발목 0.44–0.48 rad/unit).
  * `residual`: q* = q_ref(t) + scale·a. q_ref(t) 는 그 step 관측의 `motion_joint_pos` 와 같은 참조 프레임 k
    (Isaac: `mdp.ResidualJointPositionAction`, env.step 시작 시 계산해 decimation 동안 유지; MuJoCo 러너도 동일).
    기본 scale 프리셋 `gmt` = 전 관절 0.25 rad (Robros `igris_c_gmt_public` action.yaml 과 같은 값).
  * `--action_scale beyondmimic|gmt` 로 프리셋 변경. 모드/scale 은 `params/overrides.json` 에 저장되고 play_eval 이
    자동으로 따른다. check_obs_parity 의 `[action]` 항목이 액션→목표각 변환(랜덤 probe 액션, zero-action 롤아웃의
    매 step 목표각)을 Isaac `joint_pos_target` 과 비교한다.
  * 권장 기본값과 A/B 결과는 아래 "액션 모드 A/B" 절 참고.

### 관측 (policy, 130 차원, 이 순서로 concat; 정규화기는 ONNX/TorchScript 내부에 포함 → raw 관측 입력)

| 항 | 차원 | 정의 |
|---|---|---|
| motion_joint_pos | 23 | 참조 관절각 (절대, rad) |
| motion_joint_vel | 23 | 참조 관절속도 |
| motion_anchor_pos_b | 3 | R(q_rob)ᵀ(p_ref − p_rob), 앵커=base_link |
| motion_anchor_ori_b | 6 | rot6(q_rob⁻¹ q_ref) = 회전행렬 첫 두 열, row-major `[R00,R01,R10,R11,R20,R21]` |
| base_ang_vel | 3 | base_link 각속도 (바디 좌표) = MuJoCo freejoint `qvel[3:6]` |
| projected_gravity | 3 | R(q_root)ᵀ(0,0,−1) |
| joint_pos | 23 | q − q_default |
| joint_vel | 23 | q̇ |
| actions | 23 | 직전 raw 액션 |

critic 은 노이즈 없이 위 항 + base_lin_vel + 추적 바디 위치/자세(앵커 좌표)를 받는다. 학습 시 관측 노이즈는
BeyondMimic 값(앵커 위치 ±0.25 등), PLAY 에서는 off. 관절 순서는 PhysX 순서 (`policy.json: joint_names`).
**시간 정렬**: 정책 step k 의 관측은 참조 프레임 k 를 쓰고, step 후 로봇 상태를 프레임 k 와 비교(보상/종료/지표), 그다음 k+=1.

### 보상 (BeyondMimic G1 가중치)

| 항 | 가중치 | 커널 |
|---|---|---|
| anchor 위치 (global) | 0.5 | exp(−‖e‖²/0.3²) |
| anchor 자세 (global) | 0.5 | exp(−θ²/0.4²) |
| 바디 위치 (앵커 xy+yaw 로 재정렬한 참조) | 1.0 | std 0.3 |
| 바디 자세 (재정렬) | 1.0 | std 0.4 |
| 바디 선속도 | 1.0 | std 1.0 |
| 바디 각속도 | 1.0 | std 3.14 |
| action rate L2 | −0.1 | |
| 관절 한계 | −10 | soft limit(0.9) 초과량 |
| 원치 않는 접촉 (발·손 제외 전 바디, >1 N) | −0.1 | |

`mdp.motion_ee_position_error_exp`(손·발)도 구현돼 있으나 기본 비활성.

### 종료

anchor 높이 오차 > 0.25 m, anchor 자세(투영중력 z 차) > 0.8, 손/발 z 오차 > 0.25 m, time-out.
평가(PLAY)에서는 모든 env 가 클립의 프레임 0 에서 시작하고 마지막 프레임에서 hold; T 프레임을 종료 없이 추적하면 성공.

### 이벤트 (학습만, 적당한 수준)

마찰 정/동 (0.4–1.2 / 0.4–1.0), 반발 0–0.3; 관절 기본각 ±0.01 rad(캘리브레이션 오차); base_link CoM ±2.5–3 cm,
질량 −1…+2 kg; 1–3 s 마다 push (선속도 ±0.4 m/s, 각속도 ±0.4/0.6 rad/s); 리셋 시 참조 상태에 자세/속도/관절 노이즈.
샘플링: 클립 균등 + 클립 내부 1 s bin 의 실패 기반 adaptive 샘플링(BeyondMimic).

### 지표 (`metrics.py`, Isaac/MuJoCo 공통 코드)

success_rate, mpkpe_global(추적 바디 world 위치 평균오차, m), mpkpe_root_relative(mjlab R-MPKPE: 참조를 로봇 앵커 xy+yaw
로 재정렬), joint_pos_error(평균 |Δq|, rad), joint_vel_error(RMS, rad/s), ee_pos_error(손+발, 재정렬), anchor_pos/height/ori
오차, anchor_drift_xy_final, completion_ratio, 실패 원인, 클립별 + (NPZ `meta.segments` 가 있으면) 세그먼트별 지표.

## policy.json (sim2sim / 배포 계약)

`joint_names`(정책 순서), `default_joint_pos`, `action_scale`, `kp`, `kd`, `effort_limit`, `velocity_limit`, `armature`,
`joint_friction`(정마찰 effort), `joint_dynamic_friction`, `joint_viscous_friction`, `joint_pos_limits`,
`soft_joint_pos_limits`, `observation.terms`(이름/차원/slice/scale/clip/정의), `anchor_body_name`, `body_names`,
`ee_points`, `terminations`(임계값), `sim_dt`, `decimation`, `motion_fps`, `motion_z_offset`, ONNX 입출력 이름/shape.

## 확인된 sim2sim 주의사항

1. **MJCF ≠ URDF**: 오른쪽 hip pitch 원점 x 가 URDF −0.0155 / MJCF −0.0115 (오른다리 바디 4 mm 오차,
   check_obs_parity 의 `state t=0` 에서 확인됨). 총질량 URDF 58.94 kg / MJCF 57.90 kg. 관절 범위(허리 roll/pitch,
   발목)와 모터 한계(발목 90 Nm vs 163/149 Nm, 허리 60 vs 217/120 Nm)도 다름 → 러너가 JSON 값으로 덮어씀
   (armature, frictionloss=정마찰, damping=점성마찰, jnt_range, 토크 클립; MJCF 모터는 끄고 `qfrc_applied` 로 토크 인가).
2. **발 충돌 형상**: URDF 발 박스 바닥면은 ankle_roll 기준 z=−0.071, MJCF `foot_box` 는 −0.080 (9 mm 차이) + MJCF 는
   발목 메쉬 충돌체가 추가로 있음. 러너 기본값 `--foot_geom urdf` 가 박스를 URDF 치수로 바꾸고 발목 메쉬 충돌을 끈다.
   리타겟터는 이제 URDF 발 박스를 사용한다 (재생성된 sim_body2trk_full: 발목 최저 z = 0.071) → `motion_z_offset 0`.
   예전 MJCF 바닥면 기준 클립(발목 z≈0.080)에는 `--motion_z_offset -0.009` (학습/평가/JSON/MuJoCo 모두 반영).
3. **관절 마찰 모델**: Isaac Sim 5 PhysX 는 정마찰 effort(정지 시 stiction 만) + 동마찰(Robros 설정은 0) + 점성.
   MuJoCo `frictionloss` 는 움직이는 중에도 작용하는 Coulomb 마찰이라 정마찰 값을 넣으면 오히려 어긋난다.
   zero-action 롤아웃(0.5 s)에서 Isaac 대비 관절각 최대오차: frictionloss=정마찰(`--joint_dynamics isaac_static`)
   1.2e-2 rad, 투영중력 2.2e-2 / frictionloss=동마찰(=0, **기본값 `isaac`**) 5.8e-3 rad, 5.2e-3.
   그래서 기본은 armature + damping=점성마찰 + frictionloss=동마찰. (`configs/igris_c_actuators.yaml` 처럼
   friction→frictionloss 로 매핑하는 MuJoCo 평가는 PhysX 보다 마찰이 큰 모델임에 유의.)
4. **속도 한계**: PhysX 는 `velocity_limit_sim` 을 강제하지만 MuJoCo 는 없음 (보통 도달하지 않음).
5. **접촉/솔버**: PhysX TGS(8/4 iter) vs MuJoCo Newton(MJCF 설정: 500 iter, noslip 100). 발 형상을 MJCF 그대로
   (`--foot_geom mjcf`) 두면 9 mm 낙하 때문에 첫 step 의 base 각속도 차가 0.15 rad/s 로 커진다 (URDF 박스: 0.017).
6. **body 속도 정의**: NPZ/보상/관측 모두 바디 *원점* 속도(COM 아님). MuJoCo 에서 `mj_objectVelocity` 는 `mjOBJ_XBODY`
   를 써야 원점 속도 (`mjOBJ_BODY` 는 COM).
7. 손목/목: Isaac 에선 고정, MuJoCo 러너는 PD(kp 100, kd 1)로 0 유지. 참조의 손목 회전은 무시되지만 `Left_Hand`
   참조 위치에는 반영되어 있어 ee 오차에 최대 수 cm 포함될 수 있다.
8. Windows: Isaac Sim 종료(`simulation_app.close()`)가 가끔 멈춤 → 결과 파일은 그 전에 기록되므로 프로세스를 종료해도 됨.
9. **참조 모션 품질**: 첫 sim_body2trk_full 은 어깨 yaw 23.5 rad/s(한계 16), 허리 pitch 한계 초과, base 각속도 13 rad/s
   등이 있었고 리타겟터에서 수정됨 (재생성본: 한계 위반 0, |q̇|/한계 ≤ 0.80, base 각속도 ≤ 4.7 rad/s). 리셋 시 관절은
   soft limit(범위의 90%)로 클립된다. 물리적으로 불가능한 구간은 adaptive 샘플러의 실패 bin 으로 남는다.
10. `check_obs_parity` 는 dump 를 만든 시점의 모션 파일을 다시 읽는다 → NPZ 를 재생성한 뒤에는 dump 도 다시 만들어야 한다
   (옛 dump 로 돌리면 motion_joint_pos 등이 FAIL 로 나오는 것이 정상).

## 액션 모드 A/B (sim_body2trk_full.npz 재생성본, 2025 프레임 / 40.5 s)

조건: 4096 env, 300 iteration, seed 42, 동일 보상/종료/랜덤화, motion_z_offset 0 (두 run 동시 실행, 각 ~17 분).
`absolute` = beyondmimic scale, `residual` = 0.25 rad. 평가: PLAY(노이즈/랜덤화/지연 없음), 프레임 0 시작, model_299.

| 지표 | absolute Isaac | absolute MuJoCo | residual Isaac | residual MuJoCo |
|---|---|---|---|---|
| success | 0 | 0 | 0 | 0 |
| completion (추적한 프레임 비율) | 0.021 | 0.022 | **0.566** | **0.611** |
| MPKPE global [m] | 0.055 | 0.051 | 1.714 | 1.652 |
| MPKPE root-relative [m] | 0.027 | 0.026 | 0.064 | 0.060 |
| joint pos error [rad] | 0.079 | 0.082 | 0.117 | 0.118 |
| 학습 마지막 평균 에피소드 길이 | 14.6 step | | 143.4 step | |

(지표는 살아 있는 동안의 평균이므로 absolute 의 작은 MPKPE 는 처음 43 프레임만 반영된 값이다.)

세그먼트별 (추적 비율 / MPKPE global m, F = 여기서 실패):

| # | 세그먼트 | 프레임 | absolute Isaac | absolute MuJoCo | residual Isaac | residual MuJoCo |
|---|---|---|---|---|---|---|
| 0 | idle | 0-32 | 1.00 / 0.029 | 1.00 / 0.027 | 1.00 / 0.039 | 1.00 / 0.043 |
| 1 | t_pose | 32-137 | 0.10 / 0.128 F | 0.11 / 0.114 F | 1.00 / 0.059 | 1.00 / 0.074 |
| 2 | idle | 137-235 | – | – | 1.00 / 0.067 | 1.00 / 0.088 |
| 3 | walk | 235-446 | – | – | 1.00 / 1.372 | 1.00 / 1.317 |
| 4–11 | idle, wave_right, idle, squat, idle, arms_up, turn, idle | 446-1140 | – | – | 1.00 / 2.29–2.53 | 1.00 / 2.07–2.42 |
| 12 | bend_forward | 1140-1240 | – | – | 0.06 / 2.48 F | 0.97 / 1.91 F |
| 13–21 | walk, walk+wave_left, t_pose, reach_forward, idle, single_leg, idle, run, idle | 1240-2025 | – | – | – | – |

결론 (정직하게):
* **두 모드 모두 아직 클립을 끝까지 추적하지 못한다** (success 0). absolute 는 300 iteration 후에도 T-pose 팔 들기
  (프레임 ~43)에서 손 높이 종료. residual 은 같은 예산으로 23 s(1146 프레임)까지 버티고 bend_forward 에서 종료.
* residual 은 자세(root-relative MPKPE 0.06 m)는 따라가지만 **걷기 구간에서 전진하지 못해 전역 위치가 2–2.7 m
  뒤처진다** (종료 조건이 z 만 보므로 살아남음 → global MPKPE 1.7 m). 앵커 위치 보상(std 0.3 m)이 멀어지면 기울기가
  사라지는 BeyondMimic 설정의 한계로, 학습이 더 진행되거나 xy 드리프트 종료/보상 std 조정이 필요하다.
* Isaac ↔ MuJoCo 일치: 두 모드 모두 t=0 관측/액션→목표각 parity PASS (≤1.6e-7), 같은 세그먼트에서 실패,
  completion 차이 ≤0.045, root-relative MPKPE 차이 ≤0.004 m.
* **권장 기본값**: 리타겟 텔레옵 데이터에는 `--action_mode residual` (scale 0.25). 같은 예산에서 에피소드 길이 10배,
  추적 프레임 27배. absolute 는 BeyondMimic 와의 비교용으로 유지 (코드 기본값은 기존 동작 = absolute).
* **필요 학습량 추정**: BeyondMimic 는 단일 클립에 10k–30k iteration 을 쓴다. 이 40 s 클립(걷기/턴/한발/달리기 포함)은
  residual 기준 최소 3k–5k, 전 구간 성공에는 10k 이상을 예상한다. 이 PC(GPU 단독) ≈ 3.4 s/iter → 3–10 시간.
  재현: `--action_mode residual --max_iterations 10000 --save_interval 500`.

재현 명령:
```bat
%ISAAC% scripts\isaaclab\train.py --headless --num_envs 4096 --max_iterations 300 --save_interval 100 --seed 42 ^
       --motion data\sample\motions\sim_body2trk_full.npz --action_mode absolute --run_name ab_absolute_bmscale_s42
%ISAAC% scripts\isaaclab\train.py --headless --num_envs 4096 --max_iterations 300 --save_interval 100 --seed 42 ^
       --motion data\sample\motions\sim_body2trk_full.npz --action_mode residual --run_name ab_residual_s025_s42
:: 각 run: play_eval --checkpoint <run>\model_299.pt --motion data\sample\motions\sim_body2trk_full.npz --dump_obs ...
::         check_obs_parity / eval_mujoco_policy / compare_metrics (위 사용법 참고)
```
결과 파일: `logs\isaaclab\igris_c_tracking\2026-10-21_17-57-40_ab_absolute_bmscale_s42`,
`...\2026-10-21_17-58-02_ab_residual_s025_s42` (eval\, exported\parity_report.json),
표: `logs\isaaclab\igris_c_tracking\ab_absolute_vs_residual_s42.md`, 학습 로그: `logs\isaaclab\train_ab_*.log`.
