# PICO 4 / 4 Ultra → IGRIS-C 모방학습 데이터셋 파이프라인

PICO 4 / PICO 4 Ultra VR 헤드셋 + 모션 트래커로 사람 동작을 수집해, 로브로스(Robros) IGRIS-C 휴머노이드의
모방학습(모션 트래킹 RL)용 데이터셋을 만들고 평가하는 파이프라인입니다.

```
[1-1] PICO 연결/장비 감지 ─ XRoboToolkit ─> raw/*.h5 (24관절 바디, 헤드셋, 컨트롤러, 트래커)
[1-2] 트래커 구성별 보정   ─ 캘리브레이션, 이상치, 필터, 뼈 길이, 센서 앵커, 발 접지 ─> proc/*
[2]   자동 라벨링          ─ 이동(loco) / 하체(lower) / 상체(upper) 3축 라벨 + 검수 도구 ─> labels/*
[3]   IGRIS-C 리타겟팅     ─ mink 2단계 IK (Robros GMR 테이블 기반) + 균형/접지 ─> motions/*.npz|pkl|csv
[4]   평가                 ─ MuJoCo 물리 타당성 / Isaac Lab 모션 트래킹 RL / MuJoCo sim2sim
[5]   시뮬레이션 텔레오퍼레이션 ─ 실시간 보정·리타게팅 + MuJoCo 가상 환경(접촉 기반 물리 그립) + 녹화 ─> data/<session>/raw/*.h5
[6]   depth 카메라 → 가상 환경 ─ 추적·TSDF 재구성, 바닥·방향 보정, 배경/물체 분할, 상호작용 물체 감지 ─> scenes/<name>/
```

## 0. 설치

| 용도 | 환경 | 비고 |
|---|---|---|
| 수집·보정·라벨링·리타겟팅·MuJoCo 평가·실시간 뷰어 | `.venv` (Python 3.10) | `pip install -r requirements.txt` |
| Isaac Lab 학습/평가 | conda `isaaclab` (Python 3.11, `%USERPROFILE%\anaconda3\envs\isaaclab`) | Isaac Sim 5.1 + Isaac Lab 2.3.2 (`C:\IsaacLab`) |

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe" -m venv .venv   # python.org Python 3.10
.venv\Scripts\python -m pip install -r requirements.txt
```

> `.venv`는 python.org에서 받은 Python 3.10으로 만듭니다. Miniconda/Anaconda의 Python으로 만들면 그 폴더에 있는
> 오래된 `msvcp140.dll`(14.27)이 먼저 로드되어 `import mujoco`가 `WinError 1114`(DLL 초기화 실패)로 실패합니다.

Isaac Lab 환경(Windows 11, RTX 50 시리즈는 torch cu128 빌드 필요):

```powershell
# (관리자 PowerShell 1회) Isaac Sim pip 휠은 Windows 긴 경로 지원이 필요
Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -Type DWord
conda create -y -n isaaclab python=3.11
$ISAAC = "$env:USERPROFILE\anaconda3\envs\isaaclab\python.exe"
& $ISAAC -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
& $ISAAC -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git C:\IsaacLab
# flatdict 4.0.1(Isaac Lab 의존성)은 pkg_resources가 없는 최신 setuptools로는 빌드되지 않음
& $ISAAC -m pip install "setuptools<80"; & $ISAAC -m pip install flatdict==4.0.1 --no-build-isolation
# conda env를 활성화한 셸(또는 CONDA_PREFIX 설정)에서, isaaclab.bat는 전체 경로로 실행
C:\IsaacLab\isaaclab.bat --install rsl_rl
& $ISAAC -m pip show isaaclab   # 핵심 패키지(0.54.x)가 설치됐는지 확인 (없으면: & $ISAAC -m pip install -e C:\IsaacLab\source\isaaclab)
& $ISAAC -m pip install "tensordict<0.10"   # 최신 tensordict는 torch 2.8+ 전용 (torch 2.7에서 rsl_rl import 실패)
# rsl_rl(GitPython)은 git을 PATH에서 찾으므로 학습 셸 PATH에 C:\Program Files\Git\cmd 포함
```
Isaac Sim을 처음 import할 때 NVIDIA Omniverse EULA 동의를 묻습니다. 아래 명령들은 `$env:OMNI_KIT_ACCEPT_EULA="YES"`로 동의한 상태에서 실행합니다.

외부 자산은 이 저장소에 포함하지 않습니다.
`.venv\Scripts\python scripts\fetch_third_party.py`를 실행하면 검증된 커밋으로 `third_party/`에 clone 됩니다
(GMR은 이전 고정 커밋 `e47d4f8`이 upstream에서 사라져 `1281a40`으로 고정. 관절·바디 이름, 31 DOF, IK 테이블 가중치가 코드와 일치함을 확인).
코드는 실행하지 않고 모델 파일만 사용합니다.
- `third_party/robros_GMR` — robrosinc/GMR (MIT). IGRIS-C MJCF (`assets/robros_igris_c_v2/igris_c_v2.xml`, 31 DOF)와
  PICO→IGRIS-C IK 테이블 (`general_motion_retargeting/ik_configs/xrobot_to_igris_c.json`)
- `third_party/robros_lab_public` — robrosinc/robros_lab_public. Isaac Lab용 URDF (손목·목 고정, 23 DOF)와
  액추에이터 파라미터 (`robros.py`). 이 파라미터를 `configs/igris_c_actuators.yaml`로 옮겨 사용

> **Isaac Gym**: NVIDIA가 지원을 종료했고 Linux 전용(Python ≤3.8)이라 Windows에서는 쓸 수 없습니다.
> 같은 역할(GPU 병렬 RL)은 Isaac Lab이 대신합니다(`gymnasium`은 Isaac Lab에 포함).
> 이 PC는 Windows 11이라 Isaac Lab 3.x / Isaac Sim 6.x도 설치할 수 있지만, 코드가 Isaac Lab 2.3 / rsl-rl 3 API
> (TensorDict 관측, `body_link_*`, `obs_groups` 등)에 맞춰져 있어 2.3.2 / 5.1을 그대로 씁니다.

## 1-1. PICO 4 / 4 Ultra 연결과 장비·트래커 감지

**연결 구성 (XRoboToolkit)**

```
헤드셋 XRoboToolkit 앱 ──TCP 63901──> PC RoboticsServiceProcess.exe ──127.0.0.1:60061──> xrobotoolkit_sdk (01_probe / 02_record)
```

| 포트 (`setting.ini`) | 용도 |
|---|---|
| TCP 63901 | 헤드셋 → PC 데이터. 방화벽 인바운드 허용 필요 |
| UDP 29888 | PC 브로드캐스트. 같은 서브넷일 때만 헤드셋에 서버 선택 팝업이 뜸 |
| TCP 127.0.0.1:60061 | 서비스 ↔ Python SDK (로컬 전용) |

**PC: XRoboToolkit-PC-Service**

1. [Releases](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases) v1.0.0의 `XRoboToolkit-PC-Service.win.zip`을 전용 폴더에 풉니다(zip 안에 상위 폴더가 없음).
2. `VC_redist.x64.exe`를 설치합니다.
   `0x80070666`(다른 버전이 이미 설치됨)이 나오면 더 새 VC++ 2015-2022 런타임이 있는 것이므로 건너뜁니다.
3. 관리자 PowerShell에서 압축을 푼 폴더로 이동해 `install.bat`를 실행하고, 방화벽 규칙을 전체 경로로 다시 추가합니다.
   - 탐색기에서 "관리자 권한으로 실행"하면 작업 폴더가 `C:\Windows\System32`가 되어 `setting.ini`·`db` 처리가 엉뚱한 곳에 적용됩니다.
   - 스크립트의 `%CUR_DIR%`가 설정되지 않아 방화벽 규칙(`picobusinesssuit`)이 상대 경로로 등록됩니다.
     이 규칙을 지우고 같은 이름으로 다시 추가합니다. 이름이 같아야 `uninstallScript.bat`가 함께 지웁니다.

   ```powershell
   cd <압축 푼 폴더>
   .\install.bat
   netsh advfirewall firewall delete rule name="picobusinesssuit"
   netsh advfirewall firewall add rule name="picobusinesssuit" dir=in action=allow program="<압축 푼 폴더>\RoboticsServiceProcess.exe" enable=yes profile=any
   ```
4. `runService.bat`로 서비스를 실행합니다(창 없이 백그라운드, 한 번에 하나만 실행).
   - `netstat -ano | findstr "63901 60061"`에서 두 포트가 LISTENING이면 정상입니다.
   - 서비스 로그는 `%LOCALAPPDATA%\PICOBusinessSuitData\log\<서비스를 시작한 날짜>.txt`에 남습니다(PowerShell에서는 `$env:LOCALAPPDATA`).
     서비스를 다시 시작하기 전까지 같은 파일에 이어서 기록되므로 오늘 날짜 파일이 없을 수 있습니다.
     헤드셋 접속은 `new tcp connect device`, 끊김은 `remove tcp`로 기록됩니다.

Qt는 설치하지 않아도 됩니다. 릴리스 zip에 Qt 런타임 DLL이 들어 있고, Qt 설치는 서비스를 소스에서 빌드할 때만 필요합니다.

**헤드셋: XRoboToolkit 앱**

1. 헤드셋에서 개발자 모드와 USB 디버깅을 켭니다(설정 → 일반 → 정보 → 소프트웨어 버전을 여러 번 탭).
   PC에는 Android Platform-Tools(`adb`)를 설치하고 PATH에 추가합니다.
   USB로 처음 연결하면 헤드셋에 뜨는 USB 디버깅 허용 창을 수락합니다. `adb devices`에서 `unauthorized`가 아니라 `device`로 보여야 합니다.
2. 기종에 맞는 APK를 [Releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases) v1.1.1에서 받아 설치합니다.
   기종은 `adb shell getprop ro.product.model`로 확인합니다.

   | 기종 | Android | APK |
   |---|---|---|
   | PICO 4 | 10 (API 29) | `XRoboToolkit-PICO-1.1.1-SDK29.apk` (일반 빌드는 설치되지 않음) |
   | PICO 4 Ultra (`A9210`) | 14 (API 34) | `XRoboToolkit-PICO-1.1.1.apk` (SDK29 빌드도 설치·스트리밍 확인) |

   ```powershell
   adb install -r -g XRoboToolkit-PICO-1.1.1-SDK29.apk   # PICO 4
   adb install -r -g XRoboToolkit-PICO-1.1.1.apk         # PICO 4 Ultra
   ```
3. 라이브러리 → 알 수 없는 출처에서 실행합니다.
   처음 실행할 때는 헤드셋이 인터넷에 연결되어 있어야 합니다(없으면 "No entitlement info").
4. **Network → Enter**에 PC IP를 입력하고 Status가 **WORKING**이 되는지 확인합니다.
   PC와 헤드셋이 다른 서브넷이면 서버 선택 팝업이 뜨지 않으므로 직접 입력합니다. TCP 연결은 라우터를 넘어서도 됩니다.
5. 트래킹 설정:
   - Head, Controller를 켭니다.
   - PICO Motion Tracker Mode: Full body(`body_*`) / Object(`sparse_trackers`) / None(`hmd_3pt`). Num은 착용한 트래커 수입니다.
   - USB로 연결할 때는 `01_probe_device.py` / `02_record.py`에 `--usb`를 주면 `adb reverse tcp:63901 tcp:63901`을 걸고,
     앱에는 **127.0.0.1**을 입력합니다(Wi-Fi·방화벽·헤드셋 절전과 무관, 지연이 작음).
   - 헤드셋을 벗으면 PICO가 앱을 일시정지해 접속이 끊깁니다. 수집·확인 중에는 착용한 상태로 둡니다.
   - PC 쪽 한 번에 준비: `powershell -ExecutionPolicy Bypass -File scripts\pico_start.ps1`
     (서비스 실행, USB면 터널 + 헤드셋 XRoboToolkit 실행, 입력할 PC IP 안내)
   - **Send**를 켭니다.
   - **Switch w/ A Button은 끕니다.** 켜 두면 A 버튼이 Send도 함께 토글합니다(`02_record.py`가 A를 녹화 시작/정지로 씀).
6. 모션 트래커 캘리브레이션: 헤드셋을 켤 때마다 PICO의 Motion Tracker 앱에서 다시 해야 합니다.
   - 캘리브레이션 없이 Full body를 고르면 바디 데이터가 나오지 않고, PC 연결이 끊긴 채 앱에는 WORKING이 그대로 표시될 수 있습니다.
   - 이때 서비스 로그에는 `remove tcp`, 헤드셋 logcat에는 `mode is BodyTracking, but StateCode = 0`이 남습니다.
   - 캘리브레이션을 마친 뒤 앱에서 **Reconnect**를 누릅니다.
7. 감지 실행. 아래 명령은 PowerShell 기준이며, cmd에서는 `#` 이후 주석을 빼고 실행합니다.

```powershell
.venv\Scripts\python scripts\01_probe_device.py --body-trackers 2     # 실기 (Full body, 발목 트래커 2개)
.venv\Scripts\python scripts\01_probe_device.py --body-trackers 3     # 실기 (Full body, 발목 2개 + 허리 1개)
.venv\Scripts\python scripts\01_probe_device.py --simulate body_2trk   # 헤드셋 없이 시험
```

`--body-trackers`는 Full body 모드에서만 쓰입니다. Object / None 모드에서는 무시됩니다.

`No packets received`가 나오면 다음 순서로 확인합니다.
서비스 실행 → 방화벽 TCP 63901 → 앱 Status WORKING(서비스 로그에 `new tcp connect device`) → Send 켜짐 → (Full body이면) 트래커 캘리브레이션.
앱의 WORKING 표시는 연결이 끊겨도 남아 있을 수 있으므로 서비스 로그로 확인합니다.

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

- Full body에서 허리 트래커를 착용했다면 `02_record.py`에도 `--body-trackers 3`을 줍니다. `01_probe_device.py`의 설정은 이어지지 않습니다.
- A/B 버튼은 PC의 `02_record.py`가 읽습니다. 상태(`[rec] recording ...`, `[rec] stopped`, `[rec] saved ...`)는 PC 터미널에만 표시되고 헤드셋에는 표시되지 않습니다.
  10프레임 이하인 에피소드는 저장하지 않으므로 `[rec] stopped` 바로 앞에 `[rec] saved ...`가 없으면 버려진 것입니다. 세션은 Ctrl+C로 끝냅니다.
- 앱의 Data Collection → **Record**는 헤드셋 내부(`/sdcard/Download/trackingData_*.txt`)에 따로 저장하는 기능이며 이 파이프라인에서는 쓰지 않습니다.
  같은 Record 버튼을 다시 누르면 정지합니다.

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

1. 균일 레이트(50 Hz)로 리샘플합니다. 0.25 s 이하 결손은 보간하고, 더 긴 결손은 연속 보간한 뒤 invalid(`proc/valid`)로 둡니다.
   invalid 구간은 신뢰도 0이 되고 3단계 내보내기에서 잘려 나갑니다. 헤드셋 시계가 되돌아가거나 중복된 타임스탬프는 버립니다.
   바디 프로파일인데 바디 데이터가 20% 미만이면(트래커 꺼짐 등) `sparse_trackers`/`hmd_3pt`로 자동 전환합니다.
2. 이상치(스파이크)를 제거합니다(robust median + 속도 기준).
   `sparse`/`hmd_3pt` 프로파일은 센서 포즈로 24관절을 재구성합니다(`SparseBodySolver`).
   이때 실제로 있는(그리고 그 프레임에 유효한) 트래커만 해당 부위의 신뢰도를 올립니다. 발목 트래커가 없으면 하체 라벨을 끕니다.
3. 정지 구간에서 캘리브레이션합니다. 바닥 높이, 정면 방향(±180°에서도 깨지지 않는 원형 중앙값), 신장, 뼈 길이,
   컨트롤러→손목 오프셋을 구하고, PICO 관절 좌표계를 점검합니다(T-포즈 편차 중앙값이 15°를 넘으면 리포트에 경고).
   손에서 15 cm 이상 떨어진 컨트롤러(내려놓은 경우)는 오프셋 추정과 손목 앵커에서 제외합니다.
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
- 검수 흐름: `--export-review csv` → 스프레드시트에서 수정 → `--import-review csv`(reviewed 표시, 세그먼트 재생성) → `--train`.
  - CSV는 UTF-8과 cp949(한국어 Excel의 기본 CSV 저장) 모두 읽습니다.
  - 검수된 에피소드는 `run_pipeline.py`를 다시 돌려도 라벨을 덮어쓰지 않습니다(다시 자동 라벨링하려면 `04_label.py --force`).
  타임라인 PNG(`reports/*_labels.png`)도 생성합니다.
- 특징은 머리 관절 높이로 정규화합니다. 보정 단계의 신장(정수리 기준)은 학습·추론 모두 같은 기준(−0.12 m)으로 바꿔 씁니다.

정확도는 처음 보는 무작위 동작 시퀀스, 노이즈가 있는 `body_2trk` 데이터, 경계 ±0.75 s 제외 조건에서 측정했습니다
(`tests/eval_labeling.py`).

| 축 | 규칙만 (acc / macro-F1) | 규칙 + 학습 (acc / macro-F1) |
|---|---|---|
| loco | 1.000 / 1.000 | 1.000 / 1.000 |
| lower | 0.965 / 0.932 | 0.977 / 0.952 |
| upper | 0.921 / 0.908 | 0.971 / 0.952 |

합성 생성기는 `kneel`, `raise_l/r`, `reach_l/r` 동작을 만들지 못하므로 이 다섯 라벨은 위 정확도에 포함되지 않습니다(규칙만 있고 미검증).
학습형 라벨러도 이 라벨을 본 적이 없어 블렌딩 시 점수를 낮춥니다. 실측 검수 데이터로 학습하기 전에는 `--learned` 없이 쓰는 편이 안전합니다.

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
  - `motions/ep_XXXX.npz`: Isaac Lab 트래킹용. 50 fps, joint/body pos·quat(wxyz)·vel. meta의 라벨 세그먼트는 파일의 fps 기준 프레임입니다.
  - `.pkl`: Robros IGRIS-C canonical GMR 형식.
  - `.csv`: BeyondMimic/mjlab 형식(헤더 없음, MJCF 31관절 순서).
  - 손 채널(`.npz`, `.pkl`): 녹화의 컨트롤러 그립(`raw/buttons/*_grip`)에서 만든 `hand_cmd`(T,12, Robros 정규화 손 명령),
    `hand_joint_pos`(T,22, 손가락 각도), `hand_grip`(T,2)과 레이아웃(`hand_motor_ids`, `hand_joint_names`).
  - 트래킹 결손(`proc/valid`=False)이 있으면 유효 구간별로 `ep_XXXX_p0`, `_p1`… 로 나눠 내보냅니다(2 s 미만 구간은 버림).
    결손이 없으면 지금처럼 `ep_XXXX` 하나입니다. 어떤 구간이 나갔는지는 `reports/*_retarget.json`의 `motion_parts`에 있습니다.
- **Isaac soft 한계 안의 레퍼런스**: Isaac은 `soft_joint_pos_limit_factor=0.9`(URDF 범위 중앙 ±45%) 밖을 매 스텝 감점하고 리셋 때 잘라냅니다.
  그래서 IK의 관절 한계 자체를 이 soft 범위(+0.02 rad 여유)로 둡니다(`postprocess.isaac_soft_limits`).
  예: 무릎 하한 0.02 → 0.114 rad(선 자세에서 약 6.5° 굽힘), 팔꿈치 상한 −0.02 → −0.105 rad, 고관절 신전 상한 0.50 → 0.40 rad.
  사후 클리핑만 하면 접촉 중인 발이 밀려(발 미끄러짐 0.042 → 0.069 m/s) IK 단계에 넣었고, 결과는 0.039 m/s / 방향 오차 6.35°로 이전과 같은 수준입니다.
- 참고: 관절 범위는 여유를 두고 클리핑하고 속도는 0.8배로 제한한 뒤 내보내므로, 아래 "관절 범위 위반 0 / 속도 0.80배"는 구성상 항상 성립합니다.

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
- **정적 균형**: 준정적 프레임(CoM 수평 속도 < 0.15 m/s)에서 CoM 투영이 지지 다각형 안에 있는 비율과 여유(cm).
  보행처럼 동적인 프레임은 원래 지지면 밖에 CoM이 있으므로 제외합니다(전체 프레임 기준 값은 `com_in_support_ratio_all`).
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
  - `--resume <ckpt>`는 그 run의 `params/overrides.json`(액션 모드·스케일·z 오프셋)과 `params/motions.txt`를 이어받습니다.
- **경로**: `params/motions.txt`와 `policy.json`의 `training_motion_files`는 상대경로로 저장하므로, `logs/`와 `data/`를 함께 옮기면 다른 PC에서도 동작합니다.
- **평가 일관성**: 두 평가기 모두 종료(넘어짐) 스텝은 점수에 넣지 않습니다(이전에는 MuJoCo만 넣어 completion이 1/T 높았음).

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
$env:OMNI_KIT_ACCEPT_EULA="YES"; $ISAAC="$env:USERPROFILE\anaconda3\envs\isaaclab\python.exe"
# 샘플 모션(data/는 저장소에 없음) 만들기:
.venv\Scripts\python scripts\run_pipeline.py data\demo_body2trk --simulate body_2trk --subject-height 1.72
mkdir data\sample\motions -Force; copy data\demo_body2trk\motions\ep_0000.npz data\sample\motions\sim_body2trk_full.npz
# --motion: 파일, 폴더(data\<session>\motions), glob, 목록 파일 모두 가능
& $ISAAC scripts\isaaclab\train.py --headless --num_envs 4096 --max_iterations 5000 --motion data\sample\motions\sim_body2trk_full.npz --action_mode residual --run_name long_residual
& $ISAAC scripts\isaaclab\play_eval.py --headless --checkpoint <run>\model_4999.pt --motion data\sample\motions\sim_body2trk_full.npz --dump_obs <run>\exported\isaac_obs_dump.npz
.venv\Scripts\python scripts\isaaclab\check_obs_parity.py --dump <run>\exported\isaac_obs_dump.npz --policy_dir <run>\exported
.venv\Scripts\python scripts\eval_mujoco_policy.py --policy_dir <run>\exported --motion data\sample\motions\sim_body2trk_full.npz
```

## 5. 시뮬레이션 텔레오퍼레이션 (PICO 4 Ultra 앱 + PC 서버)

[Zero-HLM](https://young-bit.github.io/zero-hlm/)처럼 가상 환경에서 로봇을 원격 조작해 데이터를 모으기 위한 부분입니다.
PC 서버(`igris_sim/`, `scripts/sim_server.py`)가 사람 동작을 실시간으로 IGRIS-C에 옮기고, MuJoCo 가상 환경에서 물체와 상호작용하게 합니다.
PICO 4 Ultra 앱([pico_app/](pico_app/README.md))은 화면 표시와 입력을 맡습니다. 앱과의 통신 규격은 [docs/sim_protocol.md](docs/sim_protocol.md)에 있습니다.

```
PICO (XRoboToolkit 또는 PICO 앱) ─UDP─> 스트리밍 보정 ─> 스트리밍 리타게팅 ─> MuJoCo 씬 ─UDP─> PICO 앱 (50 Hz)
                                         (트래커 프로파일별)  (mink IK, 같은 표)   (물체, 잡기)    └─> 녹화 raw/* + sim/*
```

**동작 방식**
- **보정**: 시작 후 3초간 가만히 서 있으면, 오프라인 1-2단계 코드(`correct_episode`)로 그 구간을 보정합니다.
  바닥, 정면, 뼈 길이, 컨트롤러·트래커 오프셋을 구합니다. 이후 프레임은 같은 설정을 인과(causal) 필터로 처리합니다.
  - 트래커 프로파일은 들어오는 데이터로 자동 선택됩니다(`body_2trk` / `body_3trk` / `sparse_trackers` / `hmd_3pt`).
- **리타게팅**: 3단계와 같은 mink IK 표를 프레임마다 풉니다(직전 해에서 시작).
  발 접지 정렬과 관절 속도 제한도 실시간으로 합니다.
- **물리** (`configs/sim.yaml`의 `physics.mode`):
  - `kinematic`: 로봇이 사람을 그대로 따라가고 물체만 시뮬레이션합니다. 벽은 통과합니다.
  - `harness`: 관절 PD와 가상 하네스로 로봇도 물리 시뮬레이션합니다.
- **상호작용 (물리 그립, `configs/sim.yaml`의 `grasp.mode: physics`)**: 물체는 손에 실제로 닿아 생긴 접촉력과 마찰로만 잡힙니다.
  - 손은 물리 강체(손바닥 상자 + 손가락 캡슐)입니다. 로봇 손목을 힘 제한 스프링(3000 N/m, 최대 120 N)으로 따라가므로,
    책상이나 상자에 닿으면 파고들지 않고 표면에서 멈춰 그 힘으로 누릅니다.
  - 손가락은 Robros 손처럼 모터 6개(엄지 굽힘·검지·중지·약지·소지·엄지 회전)가 위치 서보(토크 제한 0.6 N·m)로 움직이고,
    끝마디는 링크처럼 가운데 마디에 묶여 있습니다. 물체에 닿으면 멈춰 그 자리에서 쥡니다.
  - 그래서 물체를 손가락과 엄지 사이(핀치)나 손바닥 사이(양손으로 상자 들기)에 끼워 눌러야 들립니다.
    닿지 않은 물체는 그립을 쥐어도 따라오지 않습니다. 엄지는 그립 0.6 명령에서 검지·중지 끝과 만나도록 맞췄습니다.
  - `grasp` 상태와 이벤트는 손이 1 N 이상 누르면서 물체가 "쥐어진" 경우(마주 보는 접촉, 또는 다른 받침 없이 손에 얹힘)에 생깁니다.
  - 이전 방식(그립을 누르면 손바닥 10 cm 안의 물체가 붙음)은 `grasp.mode: weld`로 쓸 수 있습니다.
  - 합성 시험(`tests/test_sim.py`): 5 cm 큐브 핀치로 들기, 1.5 kg 상자 양손 들기, 2 kg 크레이트 0.4 m 들기,
    닿지 않은 큐브는 움직이지 않음, 책상을 누르면 표면에서 멈춤. 공처럼 둥근 물체는 핀치하면 굴러 빠질 수 있습니다(실제와 같음).
- **손가락(그립 동작)**: 컨트롤러 그립(아날로그)을 쥔 만큼 로봇 손가락이 오므라듭니다(MuJoCo 뷰어와 PICO 앱 모두).
  손 명령은 Robros 손 형식(정규화 12개, 모터 11–16/21–26, 그립 × 0.6, `igris_pipeline/hands.py`)이라 실제 IGRIS-C 손 제어에 그대로 쓸 수 있습니다.
  손가락 모델은 원래 손 메시 크기에 맞춘 간략 모델이며(`igris_sim/hand_model.py`), 관절 이름은 Robros와 같습니다.
  `configs/sim.yaml`의 `hands`에서 끄거나 그립 최대 명령, 손 스프링·모터 세기를 바꿉니다.
- **이동**: 왼쪽 스틱으로 이동, 오른쪽 스틱으로 회전합니다. 실제로 걸은 동작에 더해집니다.
- **녹화**: A로 시작·정지합니다. `raw/*`는 `02_record.py`와 같은 형식이라 `run_pipeline.py`로 다시 처리할 수 있습니다.
  `sim/*`에는 로봇 상태, 물체 포즈, 잡기 상태, 이동 오프셋, 손 명령(`sim/hand/cmd`)과 손가락 각도가 담깁니다.
  물리 그립에서는 손가락 목표 각도(`sim/hand/joint_target`), 손 자세(`sim/hand/pose`), 손이 물체를 누른 힘(`sim/hand/force`)도 기록됩니다.
  `run_pipeline.py`로 다시 처리하면 `motions/*.npz|pkl`에 `hand_cmd`(T,12), `hand_joint_pos`(T,22), `hand_grip`(T,2)이
  모션과 같은 시간축으로 들어갑니다(Isaac 로더는 이 키를 무시).
- **가상 환경**: `scenes/<name>/scene.json`(Unity와 같은 파일). 기본 맵은 세 가지입니다.
  - `empty`: 빈 바닥, 거리 표시, 공
  - `tabletop`: 책상, 집을 물건 4개, 쟁반, 바닥의 통
  - `room`: 6 × 6 m 방, 작업대, 선반, 의자, 상자, 문
  - depth 카메라로 촬영한 실제 공간: 6장(`scripts/scan_build.py`)

```powershell
.venv\Scripts\python scripts\sim_server.py --source xrt --scene tabletop --viewer                # XRoboToolkit 스트림 + 데스크톱 뷰어
.venv\Scripts\python scripts\sim_server.py --source udp --scene room --record-dir data\sim01\raw  # PICO 앱 연결
.venv\Scripts\python scripts\sim_server.py --source sim:body_2trk --viewer --show-human           # 헤드셋 없이 시험
.venv\Scripts\python scripts\sim_server.py --source replay:data\s01\raw\ep_0000.h5 --viewer      # 녹화 재생
.venv\Scripts\python scripts\sim_ctl.py cmd load_scene scene=room                                # 명령 보내기
.venv\Scripts\python scripts\sim_ctl.py relay --source sim:body_2trk                             # 앱 대신 UDP로 트래킹 보내기
```

**성능** (i7-9700K, 합성 데이터, 50 Hz 기준 프레임당):

| 프로파일 | 보정 | IK 평균 / p95 | 사람 관절 오차(루트 기준) |
|---|---|---|---|
| body_2trk | 3.2 ms | 9.6 / 16.6 ms | 4.6 cm |
| body_3trk | 3.2 ms | 9.5 / 15.7 ms | 3.3 cm |
| sparse_trackers | 6.4 ms | 9.6 / 18.1 ms | 2.8 cm |
| hmd_3pt | 5.2 ms | 8.3 / 15.2 ms | 7.3 cm |

물리 계산은 프레임당 약 2–3 ms입니다(물리 손, 2 ms 스텝).
실시간 보정은 인과 필터 때문에 20–45 ms 늦고, 정확도는 오프라인 처리보다 낮습니다. 녹화 데이터는 오프라인으로 다시 처리하세요.

**PICO 4 Ultra 앱** ([pico_app/](pico_app/README.md), Unity 2022.3 + PICO SDK 3.4)
- 헤드셋, 컨트롤러, 모션 트래커(바디 / 오브젝트 모드) 값을 `tracking`으로 보냅니다.
- 서버가 보내는 로봇, 씬, 물체를 표시합니다. 시점은 로봇 머리(1인칭) 또는 로봇 뒤에서 보기 중에서 고릅니다.
- 메뉴에서 씬, 물리 모드, 트래커 모드, 녹화, 다시 보정을 다룹니다.
- 서버는 같은 네트워크에서 자동으로 찾습니다. 로봇 모델과 씬은 접속할 때 서버의 HTTP에서 받으므로, 씬을 추가해도 앱을 다시 빌드하지 않아도 됩니다.
- Unity가 필요 없는 부분(프로토콜, 좌표 변환, FK, 씬·메시 해석)은 `tests/test_app.py`가 C# 컴파일러로 검증합니다.
- 빌드 방법은 [pico_app/README.md](pico_app/README.md)에 있습니다.

## 6. depth 카메라 촬영 → 가상 환경 (`igris_scan/`)

실제 공간을 depth 카메라로 찍어 5장의 시뮬레이션 씬(`scenes/<name>/`)을 만듭니다. 자세한 내용은 [docs/scan_pipeline.md](docs/scan_pipeline.md)에 있습니다.

```
촬영(RGB-D) ─> 카메라 추적 + TSDF 재구성 ─> 보정(중력·바닥·방향·원점·스케일) ─> 분할(바닥/책상/벽/구조물/물체)
            └─ 상호작용 구간: 모델에 정합만 ─> 옮겨진 물체 감지 + 최종 위치 ─────────────────┘─> scene.json + 메시
```

- **촬영**: 먼저 아무것도 건드리지 않고 카메라를 천천히(0.5 m/s 이하) 움직여 공간을 훑습니다(스캔 구간).
  이어서 카메라를 내려놓거나 든 채로 물건을 옮깁니다(상호작용 구간). 두 구간의 경계 시각을 `--split`으로 줍니다.
- **배경과 상호작용한 물건 구분**:
  - 바닥·책상 윗면·벽은 평면으로, 나머지는 덩어리로 나눕니다. 받침 위에 놓인 0.8 m 이하의 덩어리가 물체이고, 나머지는 배경 구조물입니다.
  - 상호작용 구간의 프레임마다 각 물체 표면을 투영해 봅니다. 물체가 있던 자리 너머(배경)가 보이면 "사라짐"으로 칩니다.
    손이나 사람이 앞을 가린 경우는 증거로 쓰지 않습니다. 0.3초 이상 사라진 물체가 상호작용한 물체입니다.
  - 상호작용한 물체만 MuJoCo 자유 강체(질량 = 볼록 껍질 부피 × 밀도)로 내보내고, 나머지는 고정합니다(`--dynamic all`로 모두 움직이게 할 수 있음).
    마지막 1초의 장면에서 새로 생긴 덩어리를 크기·색으로 맞춰 물체가 놓인 최종 위치도 리포트에 남깁니다.
- **결과물**: `scene.json`, 배경 메시(vertex color, 앱에서 색으로 표시), 물체 메시와 충돌용 볼록 껍질,
  `scan_report.json`(보정값, 표면, 물체, 상호작용 증거), `preview.png`(위에서 본 분할 결과).

```powershell
.venv\Scripts\python scripts\scan_capture.py data\scans\lab01 --seconds 60                 # RealSense 촬영 (D4xx)
.venv\Scripts\python scripts\scan_capture.py data\scans\lab01 --bag rec.bag                # realsense-viewer 녹화 변환
.venv\Scripts\python scripts\scan_build.py data\scans\lab01 --name lab01 --split 25         # 스캔 25초 + 상호작용
.venv\Scripts\python scripts\scan_build.py --synthetic data\scans\demo --name scan_demo      # 카메라 없이 합성 데모
.venv\Scripts\python scripts\sim_server.py --scene lab01 --viewer                             # 만든 씬에서 텔레오퍼레이션
```

**합성 촬영 검증** (MuJoCo로 렌더링한 RGB-D, 깊이 노이즈 σ = 1.5 mm × d², `tests/test_scan.py`)
- 카메라 추적 오차 ≤ 1.7 cm(스캔), 사람이 보이는 상호작용 구간 ≤ 1.9 cm, 바닥 평면 잔차 2 mm, 책상 높이 ±1.5 cm
- 물체 4개 모두 검출(위치 ±5 cm 이내, 높이 ±2 cm 이내), 옮긴 물체 1개만 상호작용으로 판정, 이동 거리 ±6 cm 이내
- 640×480·복셀 6 mm에서 물체 크기 오차는 대부분 2 cm 이내입니다. TSDF가 옆면을 부풀리므로 윗면이 평평한 물체는 윗면으로 크기를 잽니다.

## 데이터 포맷

에피소드 HDF5의 구조는 `igris_pipeline/episode.py` 상단 docstring에 있습니다.

- 단계마다 그룹(`raw/`, `proc/`, `labels/`, `retarget/igris_c/`)을 추가하므로 어느 단계부터든 다시 처리할 수 있습니다.
- 시뮬레이션 녹화에는 `sim/` 그룹이 더 있습니다(`igris_sim/recording.py` 참고).
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
- 물리 손은 간략 모델(Robros 손 치수 근사)입니다. 실제 손의 마찰·유연성·모터 토크를 측정하면 `configs/sim.yaml`의 `hands.physics`를 맞추세요.
- 스캔 파이프라인은 합성 촬영으로만 검증했습니다. 실제 카메라에서는 반사·투명 물체(유리, 금속)의 depth가 비고,
  물체 질량은 부피로 추정하므로 `scene.json`에서 고쳐야 할 수 있습니다.
- 합성 보행 데이터는 발이 미끄러지므로, 보행 구간 하네스 수치는 실측 데이터로 다시 봐야 합니다.
