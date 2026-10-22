# PICO 4 → IGRIS-C 모방학습 데이터셋 파이프라인

PICO 4 VR 헤드셋 + 모션 트래커로 사람 동작을 수집해, 로브로스(Robros) IGRIS-C 휴머노이드의
모방학습(모션 트래킹 RL)용 데이터셋을 만들고 평가하는 파이프라인입니다.

```
[1-1] PICO 연결/장비 감지 ─ XRoboToolkit ─> raw/*.h5 (24관절 바디, 헤드셋, 컨트롤러, 트래커)
[1-2] 트래커 구성별 보정   ─ 캘리브레이션, 이상치, 필터, 뼈 길이, 센서 앵커, 발 접지 ─> proc/*
[2]   자동 라벨링          ─ 이동(loco) / 하체(lower) / 상체(upper) 3축 라벨 + 검수 도구 ─> labels/*
[3]   IGRIS-C 리타겟팅     ─ mink 2단계 IK (Robros GMR 테이블 기반) + 균형/접지 ─> motions/*.npz|pkl|csv
[4]   평가                 ─ MuJoCo 물리 타당성 / Isaac Lab 모션 트래킹 RL / MuJoCo sim2sim
```

## 0. 설치

| 용도 | 환경 | 비고 |
|---|---|---|
| 수집·보정·라벨링·리타겟팅·MuJoCo 평가 | `.venv` (Python 3.10) | `pip install -r requirements.txt` |
| Isaac Lab 학습/평가 | conda `isaaclab` (Python 3.11) | Isaac Sim 5.1 + Isaac Lab 2.3.2 (`C:\IsaacLab`) |

```powershell
C:\Users\c\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

외부 자산은 이 저장소에 포함하지 않습니다.
`.venv\Scripts\python scripts\fetch_third_party.py`를 실행하면 검증된 커밋으로 `third_party/`에 clone 됩니다.
코드는 실행하지 않고 모델 파일만 사용합니다.
- `third_party/robros_GMR` — robrosinc/GMR (MIT). IGRIS-C MJCF (`assets/robros_igris_c_v2/igris_c_v2.xml`, 31 DOF)와
  PICO→IGRIS-C IK 테이블 (`ik_configs/xrobot_to_igris_c.json`)
- `third_party/robros_lab_public` — robrosinc/robros_lab_public. Isaac Lab용 URDF (손목·목 고정, 23 DOF)와
  액추에이터 파라미터 (`robros.py`). 이 파라미터를 `configs/igris_c_actuators.yaml`로 옮겨 사용

> **Isaac Gym**: NVIDIA가 지원을 종료했고 Linux 전용(Python ≤3.8)이라 이 Windows 10 PC에서는 쓸 수 없습니다.
> 같은 역할(GPU 병렬 RL)은 Isaac Lab이 대신합니다. Isaac Lab 3.0 / Isaac Sim 6.x는 Windows 11 이상만
> 지원하므로 현재 설치된 Isaac Lab 2.3.2 / Isaac Sim 5.1을 그대로 씁니다.

## 1-1. PICO 4 연결과 장비·트래커 감지

**연결 방법 (XRoboToolkit)**

1. PC: `XRoboToolkit-PC-Service` Windows 빌드를 받아 압축을 풉니다.
   - `VC_redist.x64.exe`를 설치합니다.
   - `install.bat`를 관리자 권한으로 실행합니다(방화벽 규칙 추가).
   - `runService.bat`로 서비스를 실행합니다.
2. 헤드셋: `XRoboToolkit-PICO-1.1.1-SDK29.apk`를 `adb install`로 설치합니다(PICO 4 / Android API 29용).
   - 앱에서 PC IP를 입력합니다(헤드셋 → PC TCP 63901).
   - 바디 트래킹을 쓰려면 Motion Tracker를 **Full body**로 설정합니다.
   - **Send**를 켭니다.
3. 모션 트래커 캘리브레이션: 헤드셋을 켤 때마다 PICO의 Motion Tracker 앱에서 다시 해야 합니다.
4. 감지 실행:

```powershell
.venv\Scripts\python scripts\01_probe_device.py                  # 실기
.venv\Scripts\python scripts\01_probe_device.py --simulate body_2trk   # 헤드셋 없이 시험
```

**감지 항목과 한계** (`igris_pipeline/pico/device_probe.py`)

| 항목 | 감지 방법 |
|---|---|
| 스트림 연결, 패킷 레이트 | `get_time_stamp_ns()` 변화 (헤드셋 시계, ms 해상도) |
| 헤드셋 / 좌·우 컨트롤러 / 핸드트래킹 | 포즈 변화, `get_*_hand_is_active()` |
| 바디 트래킹(24관절) | 관절별 IMU 타임스탬프 변화. SDK의 `is_body_data_available()`은 한번 True가 되면 계속 True라서 쓰지 않음 |
| 오브젝트 모드 트래커 | `num_motion_data_available()` (최대 3개), 시리얼 번호 |
| **바디 모드에서 트래커 개수** | **PC 서비스로는 알 수 없음** → `--body-trackers 3`(허리 트래커 착용 시)으로 지정 |
| **트래커 세대** (Beta Swift = MT_1 / 2024 Motion Tracker = MT_2) | **PC 서비스로는 알 수 없음**(헤드셋 앱 SDK에만 있음) → `--tracker-hw`로 메타데이터에 기록 |
| 헤드셋 모델/OS | `adb`가 있으면 `getprop`으로 조회 |

PICO 바디 트래킹은 **보정된 트래커가 최소 2개** 있어야 동작합니다(트래커 없는 전신 추정은 없음).

감지 결과로 트래커 프로파일을 자동 선택합니다(`configs/tracker_profiles.yaml`).

| 프로파일 | 조건 | 관측되는 부위 |
|---|---|---|
| `body_2trk` | 바디 트래킹, 발목 트래커 2개 | 전신(PICO 솔버). 다리·척추는 추정값 |
| `body_3trk` | 위 조건 + 허리 트래커(`--body-trackers 3`) | 골반·척추 정확도 향상 |
| `sparse_trackers` | 오브젝트 모드 트래커 1–3개 + HMD + 컨트롤러 | 트래커 부착 부위만. 나머지는 자체 IK 솔버로 추정 |
| `hmd_3pt` | 헤드셋 + 컨트롤러만 | 상체 위주. 하체 라벨은 비활성화 |

**녹화**

```powershell
.venv\Scripts\python scripts\02_record.py --session s01 --subject-height 1.72   # 오른손 A: 시작/정지, B: 마커
```

권장 프로토콜: 매 에피소드 시작 시 **3초간 정면 T-포즈로 정지**합니다(캘리브레이션 창).
`hmd_3pt`와 `sparse_trackers`에는 키(`--subject-height`)가 꼭 필요합니다.

알려진 SDK 이슈와 대응:
- 바디 속도·가속도 필드가 라벨이 잘못되어 있고 일부 값이 깨집니다 → 미분값은 포즈에서 다시 계산합니다.
- `get_body_timestamp_ns()`가 0입니다 → 헤드셋 타임스탬프를 사용합니다.

## 1-2. 트래커 종류·개수에 따른 정확도 보정

`igris_pipeline/pico/correction.py`, `scripts/03_correct.py`. 프로파일마다 아래 항목을 다르게 둡니다.
- 관절 그룹별 신뢰도(이후 리타겟팅 IK 가중치로 사용)
- 필터 컷오프(추정 관절일수록 더 강하게 평활)
- 센서 앵커(손 ↔ 컨트롤러, 발목 ↔ 트래커)

1. 균일 레이트(50 Hz)로 리샘플합니다. 0.25 s 이하 결손은 보간하고, 더 긴 결손은 invalid로 둡니다.
2. 이상치(스파이크)를 제거합니다(robust median + 속도 기준).
   `sparse`/`hmd_3pt` 프로파일은 센서 포즈로 24관절을 재구성합니다(`SparseBodySolver`).
3. 정지 구간에서 캘리브레이션합니다. 바닥 높이, 정면 방향, 신장, 뼈 길이, 컨트롤러→손목 오프셋을 구하고,
   PICO 관절 좌표계를 점검합니다.
4. 관절 그룹별 zero-phase 저역통과 필터를 적용합니다.
5. 센서 기준 뼈 길이를 정규화합니다. 정확한 관절(머리, 컨트롤러 기준 손목, 발목)은 고정하고,
   팔·다리는 2-bone IK로 다시 풉니다. 거의 펴진 팔다리는 IK가 노이즈를 크게 증폭하므로(무릎 1 cm 오차 → 9 cm)
   직선 해와 블렌딩합니다.
6. 발 접촉을 검출하고 foot-skate를 정리하며, 바닥 관통을 막습니다.
7. 월드를 정렬합니다(바닥 z=0, 시작 위치 원점, +X 정면). 관절별 신뢰도(0–1)를 함께 저장합니다.

**시뮬레이션 검증** (`tests/eval_correction.py`). 노이즈 모델은 근사이며 실기 검증이 필요합니다.

`body_2trk`, 관절 위치 오차 중앙값(cm), 보정 후 / 보정 전:

| 그룹 | 보정 후 | 보정 전 |
|---|---|---|
| 머리 | 2.0 | 2.0 |
| 손 | 1.5 | 2.5 |
| 팔 | 2.6 | 3.0 |
| 척추 | 3.2 | 3.6 |
| 골반 | 2.9 | 3.8 |
| 다리 | 3.3 | 4.4 |
| 발 | 3.4 | 4.2 |

- p95 오차도 발을 제외하면 모든 그룹에서 줄었습니다. 발은 합성 정답 자체에 발 미끄러짐이 있어서 생긴 차이입니다.
- `body_3trk`에서는 골반 오차가 2.4 → 1.2 cm로 줄었습니다.
- foot skate는 2.8 → 0.6 mm/frame입니다.
- `sparse_trackers`와 `hmd_3pt` 결과는 시뮬레이터의 장착 오프셋이 기본값과 같아서 낙관적입니다.
  실제 컨트롤러 오프셋은 `body_*` 세션의 캘리브레이션 출력(`controller_to_wrist_*`)을
  `tracker_profiles.yaml`의 `sensor_offsets`에 복사해 맞추세요.

## 2. 자세 라벨링 자동화

`igris_pipeline/labeling/`, `configs/labels.yaml`, `scripts/04_label.py`

- **3축 라벨**로 조합 동작(예: `walk+wave_left`)을 별도 규칙 없이 표현합니다.
  - loco: idle / walk / run / turn
  - lower: upright / squat / kneel / bend_forward / single_leg
  - upper: neutral / t_pose / arms_up / wave_l·r / raise_l·r / reach_forward / reach_l·r
- 체형에 무관한 해석 가능한 특징을 씁니다(신장 정규화, 진행 방향 기준 좌표). 예: 골반 높이, 무릎 굴곡,
  몸통 기울기, 손 높이·전방 거리, 손 흔들림 에너지, 속도, 회전 속도.
- 규칙은 YAML로 정의한 soft threshold의 fuzzy AND입니다. 그 뒤 Viterbi 평활(전환 패널티)과 최소 길이 병합을 거쳐
  세그먼트와 확률을 만들고, 확률이 낮은 구간은 **검수 필요**로 표시합니다.
- 학습형 라벨러(HistGradientBoosting)를 규칙 점수와 블렌딩할 수 있습니다.
  검수 완료 에피소드와 합성 데이터로 학습합니다.
- 검수 흐름: `--export-review csv` → 스프레드시트에서 수정 → `--import-review csv`(reviewed 표시) → `--train`.
  타임라인 PNG(`reports/*_labels.png`)도 생성합니다.

정확도는 처음 보는 무작위 동작 시퀀스, 노이즈가 있는 `body_2trk` 데이터, 경계 ±0.75 s 제외 조건에서 측정했습니다
(`tests/eval_labeling.py`).

| 축 | 규칙만 (acc / macro-F1) | 규칙 + 학습 (acc / macro-F1) |
|---|---|---|
| loco | 1.000 / 1.000 | 1.000 / 1.000 |
| lower | 0.965 / 0.932 | 0.977 / 0.950 |
| upper | 0.921 / 0.908 | 0.971 / 0.956 |

## 3. IGRIS-C 리타겟팅

`igris_pipeline/retarget/`, `configs/retarget_igris_c.yaml`, `scripts/05_retarget.py`

- Robros GMR 포크의 `xrobot_to_igris_c.json` IK 테이블을 이 파이프라인 좌표계로 변환해 씁니다.
  변환하면 몸통·다리 오프셋은 항등, 팔은 X축 ±90°가 되며 이를 테스트로 검증했습니다.
- GMR과 같은 2단계 mink IK입니다(daqp QP, 관절 위치·속도 제한).
  GMR 코드에는 `limits` 인자가 `safety_break` 자리에 들어가 속도 제한이 꺼지는 버그가 있는데, 여기서는 고쳤습니다.
- 추가 기능:
  - 1-2단계 **관절 신뢰도를 IK 가중치에 곱하고**, posture 정규화를 둡니다. 관측되지 않은 다리는 기본 자세로 둡니다.
  - **kinematic scaling**: 다리·팔·몸통·어깨폭 비율을 따로 적용합니다. IGRIS-C는 사람보다 다리가 짧습니다(비율 ≈0.84).
  - 로봇 발목을 사람 **발목**에 매칭합니다. Robros 테이블은 발끝 관절에 매칭합니다.
  - **geometry offsets**: 영자세에서 기울어진 분절(허벅지 8.9°, 상완 2.9° 등)을 보정합니다.
  - **접지 발 고정**: 접촉 중인 발은 평평한 발바닥 높이와 yaw-only 방향으로 둡니다.
  - **균형 단계**: 준정적 프레임에서 CoM 투영을 지지 다각형 안으로 당깁니다(mink ComTask).
    보정량은 0.3 m/s로 속도 제한해, 발이 뜨거나 닿을 때 몸통이 튀지 않게 합니다.
  - **Isaac 모델과 일치**:
    - 발바닥은 Robros Isaac URDF의 foot box를 씁니다.
    - 관절 범위는 MJCF와 URDF의 교집합을 씁니다.
    - 관절 속도는 액추에이터 한계의 80%로 제한합니다.
  - **T-포즈 특이점 대책**: 어깨 roll ≈ 90°이면 pitch와 yaw 축이 겹칩니다.
    이때 영공간에서 표류하지 않도록 shoulder pitch/yaw에 posture cost 1.0을 줍니다.
  - **몸통 안정화**:
    - 팔은 orientation 가중치를 낮추고, 팔꿈치·손목 position 목표를 추가합니다.
    - 몸통(`base_link`) orientation 가중치는 50으로 둡니다.
    - 효과: 로봇이 따라 할 수 없는 팔 비틀림 때문에 몸통이 기울던 문제(전방 뻗기에서 36° → 9.5°)를 줄였습니다.
- 출력 형식:
  - `motions/ep_XXXX.npz`: Isaac Lab 트래킹용. 50 fps, joint/body pos·quat(wxyz)·vel.
  - `.pkl`: Robros IGRIS-C canonical GMR 형식.
  - `.csv`: BeyondMimic/mjlab 형식.

시뮬레이션 `body_2trk` 데이터에서의 비교(`tests/eval_retarget.py`, `tests/eval_balance.py`):

| 설정 | 분절 방향 오차 평균 | 발 미끄러짐(접촉 중) | 정적 자세 CoM이 지지면 안에 있는 비율 |
|---|---|---|---|
| Robros 원본 테이블 | 7.98° | 0.119 m/s | — |
| 본 설정(위의 모든 항목) | 7.1° | 0.042 m/s | 80–100% |

- 정적 자세는 서기, 스쿼트, 숙이기, 한발서기입니다. 균형 단계를 넣기 전에는 0–27%였습니다.
- 노이즈 없는 정답 입력에서 분절 방향 오차는 3.6°로, 로봇과 사람의 구조 차이에서 오는 하한입니다.
- 학습용 레퍼런스 품질(샘플):
  - 관절 속도는 한계의 최대 0.80배입니다.
  - 관절 범위 위반은 0 프레임입니다.
  - 베이스 각속도 최대치는 4.7 rad/s입니다(개선 전 12.9).

```powershell
.venv\Scripts\python scripts\view_motion.py data\demo_body2trk\raw\ep_0000.h5            # 뷰어 (사람 스켈레톤 병렬 표시)
.venv\Scripts\python scripts\view_motion.py data\demo_body2trk\raw\ep_0000.h5 --mp4 out.mp4
```

## 4. 평가 (MuJoCo / Isaac Lab)

### 4-1. MuJoCo 물리 타당성 (학습 전 데이터 필터)

`igris_pipeline/sim/mujoco_eval.py`, `scripts/06_eval_mujoco.py`. **라벨 구간별로** 리포트합니다.

- **기구학**: 관절 한계 포화, 액추에이터 속도 한계 초과, 자기충돌 깊이, 바닥 관통.
- **정적 균형**: CoM 투영이 지지 다각형 안에 있는 비율과 여유(cm).
- **하네스 PD 재생**: 200 Hz, implicit PD 서보로 재생하고, 베이스를 가상 스프링으로 기준 궤적에 붙잡습니다.
  하네스 힘(체중 대비 %)이 **"외부 도움이 얼마나 필요한 클립인지"**를 나타냅니다. 토크 포화와 MPKPE도 함께 봅니다.
- **자유 재생**(`--free`): Robros 게인으로 개루프 재생합니다. 균형 제어가 없으므로 정지 자세에서도 약 1.6 s에 넘어지며,
  하한 지표로만 봅니다.

예시(시뮬레이션 세션): 정적 구간은 하네스 5–14% mg, MPKPE 1–2 cm였습니다.
보행 구간은 약 75% mg로 개루프 재생의 한계가 드러나며, 실제 추종은 RL 정책으로 평가합니다.

> `mj_inverse` 역동역학은 쓰지 않습니다. 발바닥이 바닥에 정확히 닿은 기구학 클립에서는 접촉 반력이 0에 가깝게 나와
> 오해를 부르기 때문입니다.

### 4-2. Isaac Lab 모션 트래킹 RL + MuJoCo sim2sim

`igris_pipeline/isaaclab_ext/README.md`에 자세한 내용이 있습니다(명령, 관측/보상/종료 설계, sim2sim 주의점).

- **태스크**: BeyondMimic 방식의 모션 트래킹(`Igris-C-Tracking-Flat-v0`).
  - IGRIS-C 23 DOF(Robros Isaac URDF)를 쓰며, 앵커 바디는 `base_link`입니다.
  - 정책 관측은 130차원입니다.
  - 50 Hz 정책, 200 Hz 물리로 동작합니다.
- **sim2sim**: 학습한 정책을 ONNX/TorchScript와 JSON으로 내보낸 뒤 MuJoCo(`scripts/eval_mujoco_policy.py`)에서 같은 관측·PD로 재생합니다.
  - t=0 관측과 action→target이 Isaac과 1e-7 이내로 일치하는지 `check_obs_parity.py`로 검증합니다.
- **액션 모드**: `--action_mode residual`을 권장합니다(`q_target = q_ref + 0.25·a`).
  - 기본값은 BeyondMimic 방식 `absolute`입니다.

300 iteration A/B 결과(4096 env, 시뮬레이션 샘플 40.5 s 클립, model_299, `logs/isaaclab/igris_c_tracking/ab_absolute_vs_residual_s42.md`):

| | absolute Isaac / MuJoCo | residual Isaac / MuJoCo |
|---|---|---|
| 클립 진행률(completion) | 0.021 / 0.022 | **0.566 / 0.611** |
| root-relative MPKPE | 0.027 / 0.026 m | 0.064 / 0.060 m |
| success | 0 / 0 | 0 / 0 |

- **두 시뮬레이터의 일관성**: 모드와 무관하게 같은 구간에서 실패했습니다.
  진행률 차이는 0.045 이하, root-relative MPKPE 차이는 0.004 m 이하였습니다.
- **absolute 모드**: T-포즈로 팔을 드는 구간에서 실패합니다.
- **residual 모드**: T-포즈부터 회전까지 자세는 따라갑니다. 하지만 걷기에서 앞으로 나아가지 못해 전역 위치가 약 2.3 m 뒤처지고,
  `bend_forward`에서 실패합니다.
- **학습량**: 300 iteration은 짧습니다(GPU가 비어 있을 때 약 17분). 클립 전체를 따라가려면 BeyondMimic 기준 3k–10k+ iteration
  (약 3–10시간)이 필요할 것으로 봅니다.
  전역 표류가 남으면 xy 드리프트 종료 조건을 넣거나 anchor 위치 보상의 std를 넓히는 것을 검토하세요.

```powershell
$env:OMNI_KIT_ACCEPT_EULA="YES"; $ISAAC="C:\ProgramData\Anaconda3\envs\isaaclab\python.exe"
# 샘플 모션(data/는 저장소에 없음) 만들기:
.venv\Scripts\python scripts\run_pipeline.py data\demo_body2trk --simulate body_2trk --subject-height 1.72
mkdir data\sample\motions -Force; copy data\demo_body2trk\motions\ep_0000.npz data\sample\motions\sim_body2trk_full.npz
# --motion: 파일, 폴더(data\<session>\motions), glob, 목록 파일 모두 가능
& $ISAAC scripts\isaaclab\train.py --headless --num_envs 4096 --max_iterations 5000 --motion data\sample\motions\sim_body2trk_full.npz --action_mode residual --run_name long_residual
& $ISAAC scripts\isaaclab\play_eval.py --headless --checkpoint <run>\model_4999.pt --motion data\sample\motions\sim_body2trk_full.npz --dump_obs <run>\exported\isaac_obs_dump.npz
.venv\Scripts\python scripts\isaaclab\check_obs_parity.py --dump <run>\exported\isaac_obs_dump.npz --policy_dir <run>\exported
.venv\Scripts\python scripts\eval_mujoco_policy.py --policy_dir <run>\exported --motion data\sample\motions\sim_body2trk_full.npz
```

## 데이터 포맷

에피소드 HDF5의 구조는 `igris_pipeline/episode.py` 상단 docstring에 있습니다.

- 단계마다 그룹(`raw/`, `proc/`, `labels/`, `retarget/igris_c/`)을 추가하므로 어느 단계부터든 다시 처리할 수 있습니다.
- 좌표계는 Z-up, +X 정면, 쿼터니언은 **wxyz**입니다.
- PICO 원본(Y-up, xyzw)은 기저변환으로 옮깁니다. 정면을 보고 T-포즈를 하면 모든 관절이 항등이 됩니다.

세션 레이아웃:

```
data/<session>/raw/ep_0000.h5      # raw + proc + labels + retarget
data/<session>/motions/ep_0000.{npz,pkl,csv}
data/<session>/reports/ep_0000_{correction,labels,retarget,mujoco}.json, ep_0000_labels.png
data/<session>/manifest.json
```

전체 실행: `.venv\Scripts\python scripts\run_pipeline.py data\s01`
(시뮬레이션 데모: `--simulate body_2trk --subject-height 1.72`). 테스트: `.venv\Scripts\python -m pytest tests -q`

## 한계 / 다음 단계

- **실기 검증 전**입니다. 모든 정량 결과는 합성 모션과 근사 오차 모델로 얻었습니다.
  첫 실측 세션에서 확인할 것:
  - 좌표 변환: PICO 관절이 T-포즈에서 항등인지. 캘리브레이션 리포트의 `rest_orientation_deviation_deg`가 15° 이내인지.
  - 컨트롤러 오프셋.
  - 접촉 검출 임계값.
- `hmd_3pt`의 하체는 추정값입니다. 하체 동작 데이터가 필요하면 트래커 2개 이상을 권장합니다.
- 손가락(H5_C 6 DOF/손)은 아직 리타겟하지 않습니다. 핸드트래킹 26관절은 `--hands`로 기록만 됩니다.
- 합성 보행 데이터는 발이 미끄러지므로, 보행 구간 하네스 수치는 실측 데이터로 다시 봐야 합니다.
