# IGRIS-C 시뮬레이션 서버 프로토콜 (PICO 4 Ultra 앱 ↔ PC)

PICO 4 Ultra 앱(Unity)과 PC 시뮬레이션 서버(`scripts/sim_server.py`, 패키지 `igris_sim/`) 사이의 통신 규격입니다.
앱은 화면 표시와 입력을 맡고, PC는 리타게팅·물리·녹화를 맡습니다.

```
PICO 4 Ultra 앱                                    PC: sim_server.py
  HMD / 컨트롤러 / 바디 트래킹 / 모션 트래커  ──tracking (UDP 9870)──>  스트리밍 보정 → 리타게팅 → MuJoCo
  씬·로봇·물체 렌더링                         <──state  (50 Hz)───────  로봇 qpos, 물체 포즈, 잡기 상태
  메뉴 (씬 선택, 녹화 등)                     ──cmd / <──scene, ack──
  메시 등 씬 파일                             <──HTTP (9871)──────────  scenes/ 폴더 (읽기 전용)
```

## 1. 전송

- **UDP 하나**(기본 9870)로 모든 메시지를 주고받습니다. 데이터그램 하나에 UTF-8 JSON 객체 하나가 들어갑니다.
  모든 메시지에는 `"v": 1`(프로토콜 버전)과 `"type"`이 있습니다.
- 서버는 메시지를 보낸 주소를 클라이언트로 기억하고, 그 주소로 `state`를 50 Hz로 보냅니다.
  `client_timeout_s`(3초) 동안 아무것도 보내지 않은 클라이언트에는 더 보내지 않습니다.
  tracking을 보내지 않는 화면 전용 클라이언트는 1초마다 `hello`를 보내면 됩니다.
- **서버 찾기**: 앱은 서버를 모를 때 `hello` cmd를 `255.255.255.255:9870`으로 1초마다 브로드캐스트합니다.
  처음 답한 서버(`scene` 응답)를 씁니다.
- **HTTP**(기본 9871)는 읽기 전용입니다.
  - `scenes/` 폴더와 로봇 모델(`_robot/`, 5장)을 제공합니다.
  - 주소는 `scene` 메시지의 `assets` 필드에 있습니다. 예: `http://192.168.0.10:9871/tabletop/scene.json`
  - PC에 네트워크 카드가 여러 개면 `assets`의 IP가 헤드셋에서 닿지 않을 수 있습니다. 앱은 UDP로 서버에 닿은 주소와 `asset_port`를 씁니다.
- Windows 방화벽에서 PC의 UDP 9870, TCP 9871 인바운드를 허용해야 합니다.

## 2. 좌표계

| 이름 | 축 | 사용하는 곳 |
|---|---|---|
| native | 오른손 좌표계, Y 위, Z 뒤, 쿼터니언 `[qx, qy, qz, qw]` | `tracking` (XRoboToolkit와 같음) |
| MuJoCo 월드 | 오른손 좌표계, Z 위, +X 앞, +Y 왼쪽, 쿼터니언 `[w, x, y, z]` | `state`, `scene` |
| Unity | 왼손 좌표계, Y 위, Z 앞, X 오른쪽 | 앱 내부 |

변환 공식입니다(Python 기준 구현과 테스트: `igris_sim/protocol.py`, `tests/test_sim.py`).

```csharp
// Unity 트래킹 공간 포즈 -> tracking 메시지용 native pose7
static float[] ToNative(Vector3 p, Quaternion q) => new[] { p.x, p.y, -p.z, -q.x, -q.y, q.z, q.w };

// state / scene의 MuJoCo 위치, 쿼터니언(w,x,y,z) -> Unity
static Vector3 FromMj(IList<float> p) => new Vector3(-p[1], p[2], p[0]);
static Quaternion FromMjQuat(IList<float> q) => new Quaternion(q[2], -q[3], -q[1], q[0]);
```

`tracking`의 포즈는 **트래킹 공간**(XR Origin 기준) 값입니다. 월드 위치를 보내지 않습니다.

PICO SDK(Unity Integration SDK 3.x)에서 읽는 값과 변환은 다음과 같습니다. 구현은 `pico_app`의 `HeadsetInput`, `PicoTracking`입니다.

| 데이터 | 읽는 곳 | SDK가 주는 좌표 | 보낼 때 |
|---|---|---|---|
| 헤드셋, 컨트롤러 | `UnityEngine.XR.InputDevices` | Unity | `ToNative` |
| 바디 트래킹 24관절 | `PXR_MotionTracking.GetBodyTrackingData` | Unity(SDK 안에서 z, qz, qw를 뒤집음) | `ToNative` |
| 오브젝트 트래커 | `PXR_MotionTracking.GetMotionTrackerLocation` | native | 그대로 |

XRoboToolkit도 같은 native 값을 보냅니다. XRoboToolkit 앱은 바디 데이터의 뒤집힌 부호를 다시 되돌립니다.
트래킹 원점은 바닥(Floor)입니다. 서버는 바닥 높이를 데이터에서 추정하므로 원점이 눈높이인 XRoboToolkit 데이터와 함께 써도 됩니다.

## 3. 앱 → 서버

### 3-1. `tracking` (매 프레임, 72–90 Hz)

```json
{"v": 1, "type": "tracking", "t_ns": 1761100000000000000,
 "head": [x, y, z, qx, qy, qz, qw],
 "controllers": {"left": [7], "right": [7]},
 "buttons": {"A": false, "B": false, "X": false, "Y": false,
             "left_trigger": 0.0, "right_trigger": 0.0, "left_grip": 0.0, "right_grip": 0.8,
             "left_menu": false, "right_menu": false},
 "axes": {"left": [x, y], "right": [x, y]},
 "body": [[7] x 24] | null,
 "body_ts": [24개 int] | null,
 "trackers": [{"sn": "PC2310MLL4210362G", "pose": [7]}],
 "hands": {"left": {"active": 1, "joints": [[7] x 26]}, "right": null}}
```

| 필드 | 설명 |
|---|---|
| `t_ns` | 헤드셋 시각(ns). 증가만 하면 되고 PC 시계와 맞출 필요는 없습니다 |
| `head`, `controllers` | 트래킹 공간 포즈. 추적되지 않으면 `[0,0,0,0,0,0,0]` |
| `buttons` | 트리거·그립은 0–1 실수, 나머지는 bool |
| `axes` | 썸스틱. x는 오른쪽 +, y는 위 + |
| `body` | PICO 바디 트래킹 24관절(`BodyTrackerRole` 순서 = SMPL 순서). 새 값이 없는 프레임은 `null` |
| `trackers` | 오브젝트 모드 모션 트래커(최대 3개). 시리얼로 구분합니다 |
| `hands` | 핸드 트래킹 26관절(선택). 현재 서버는 받기만 하고 녹화합니다 |

서버는 처음 들어오는 데이터로 트래커 프로파일을 고릅니다.
`body`가 있으면 `body_2trk`(서버를 `--body-trackers 3`으로 실행하면 `body_3trk`),
`trackers`가 있으면 `sparse_trackers`, 둘 다 없으면 `hmd_3pt`입니다.

### 3-2. `cmd`

```json
{"v": 1, "type": "cmd", "cmd": "load_scene", "scene": "room"}
```

| `cmd` | 인자 | 응답 |
|---|---|---|
| `hello` | | `scene` |
| `list_scenes` | | `scenes` (`names`, `current`) |
| `load_scene` | `scene` | `scene` 또는 `error` |
| `set_mode` | `mode`: `kinematic` / `harness` | `ack` |
| `reset` | | `ack` (스폰 위치로 되돌림, 보정은 유지) |
| `recalibrate` | | `ack` |
| `record` | `on`: true / false | `ack` (`recording`) |
| `marker` | | `ack` |

## 4. 서버 → 앱

### 4-1. `scene` (`hello`, `load_scene`의 응답)

```json
{"v": 1, "type": "scene", "name": "tabletop", "folder": "tabletop", "spec": {scene.json 전체},
 "dynamic": ["cube_red", "cube_green", "can_blue", "ball_yellow", "tray"],
 "joint_names": ["Joint_Waist_Yaw", ...31개, "l_0_joint_thumb_proximal", ...손가락 22개],
 "robot_mjcf": "third_party/robros_GMR/assets/robros_igris_c_v2/igris_c_v2.xml",
 "robot": "_robot/robot.json",
 "assets": "http://192.168.0.10:9871/", "asset_port": 9871, "rate_hz": 50}
```

- `dynamic`: `state.objects`의 순서입니다. 고정 물체는 `spec`에서 한 번만 만들면 됩니다.
- `joint_names`: `state.robot.qpos[7:]`의 관절 순서입니다. 로봇 31개 뒤에 손가락 22개가 옵니다
  (`configs/sim.yaml`의 `hands.enabled`가 꺼져 있으면 로봇 31개만).
- `folder`: 씬 파일(메시)의 위치입니다. 파일 주소는 `<assets><folder>/<file>`입니다. 값이 `null`이면 `scenes/` 밖의 씬이라 메시를 받을 수 없습니다.
- `robot`: 로봇 모델(5장)의 주소이며, `assets` 기준 상대 경로입니다.

### 4-2. `state` (50 Hz)

```json
{"v": 1, "type": "state", "t": 1761100000.123, "status": "running", "message": "", "progress": 1.0,
 "profile": "body_2trk", "mode": "kinematic", "scene": "tabletop",
 "robot": {"qpos": [x, y, z, qw, qx, qy, qz, j0, ..., j30]},
 "objects": {"pos": [[x, y, z], ...], "quat": [[w, x, y, z], ...]},
 "grasp": {"left": "cube_red", "right": null},
 "rig": {"pos": [x, y, z], "quat": [w, x, y, z]},
 "head_cam": {"pos": [x, y, z], "quat": [w, x, y, z]},
 "recording": false, "episodes": 0, "events": [{"t": 1761100000.0, "kind": "grasp", "detail": "left:cube_red"}],
 "harness_pct_mg": null, "timing_ms": {"correct_ms": 3.1, "retarget_ms": 9.6, "physics_ms": 0.8},
 "frames": 1234, "data_age_s": 0.012}
```

| 필드 | 설명 |
|---|---|
| `status` | `calibrating`(가만히 서 있기, `progress` 0–1) / `running`. 보정 실패 사유는 `message`에 표시됩니다 |
| `robot.qpos` | 루트 위치, 루트 쿼터니언(wxyz), 관절각(rad, `joint_names` 순서: 로봇 31개 + 손가락 22개) |
| `hand_cmd` | 손 명령 12개(0–1, Robros 정규화 형식, 모터 순서 오른손 11–16 → 왼손 21–26). 아래 6장 참고 |
| `objects` | `dynamic` 순서의 물체 월드 포즈 |
| `rig` | **트래킹 공간 → 월드** 변환입니다. XR Origin을 `FromMj(rig.pos)`, `FromMjQuat(rig.quat)`에 두면 사용자의 시점이 시뮬레이션 월드와 맞습니다(보정 정렬, 스폰, 썸스틱 이동이 모두 반영됨) |
| `head_cam` | 로봇 머리(눈) 카메라 포즈. 로봇 시점 모드에 씁니다 |
| `harness_pct_mg` | harness 모드에서 가상 하네스가 낸 힘(체중 대비 %) |

## 5. 로봇 모델 (`_robot/robot.json`)

앱은 `state.robot.qpos`로 로봇을 직접 그립니다(FK).
이를 위해 서버는 시작할 때 **컴파일된 MuJoCo 모델**에서 바디 트리와 표시용 메시를 뽑아 HTTP로 제공합니다(`igris_sim/robot_export.py`).
MJCF와 STL을 그대로 쓰지 않는 이유는 MuJoCo가 컴파일하면서 메시를 무게중심 기준으로 다시 놓기 때문입니다.
원본 파일로 그리면 위치가 어긋납니다.

```json
{"format": "igris_robot", "version": 1, "nq": 60, "joint_names": [...31개 + 손가락 22개],
 "bodies": [{"name": "base_link", "parent": -1, "pos": [0,0,0], "quat": [1,0,0,0], "joint": {"type": "free"}},
            {"name": "Link_Waist_Yaw", "parent": 0, "pos": [...], "quat": [...],
             "joint": {"type": "hinge", "qpos": 7, "axis": [0,0,1], "pos": [0,0,0], "ref": 0.0}}, ...],
 "geoms": [{"name": "base_link_visual", "body": 0, "type": "mesh", "mesh": "meshes/Upper_Body.obj",
            "pos": [...], "quat": [...], "rgba": [...]}, ...]}
```

- 바디는 부모가 먼저 나오는 순서이고, 메시 OBJ는 geom 좌표계(MuJoCo 축, m) 값입니다.
- FK 순서는 다음과 같습니다.
  1. free 루트: `qpos[0:7]`를 그대로 씁니다.
  2. 그 밖의 바디: `x = parent * (pos, quat)`.
  3. hinge가 있는 바디: 앵커(`pos`)를 중심으로 `axis` 둘레로 `qpos[i] - ref`만큼 회전합니다.
  4. geom: `body * (pos, quat)`.
- 파이썬 기준 구현은 `robot_export.forward_kinematics`입니다. 테스트가 MuJoCo 결과와 1e-9 이내로 같은지 확인합니다.
- **손가락**: MJCF의 손은 손가락이 펴진 통짜 메시라서, 서버는 손마다 간략화한 관절형 손(손바닥 상자 + 손가락 캡슐,
  `igris_sim/hand_model.py`)을 얹고 원래 손 메시는 내보내지 않습니다. 손가락 바디는 손 바디(`Left_Hand`, `Right_Hand`)의
  자식으로 나오고, 관절의 `qpos` 인덱스는 38 이상(로봇 뒤)입니다. 관절 이름은 Robros 손 레이아웃(`l_0_joint_thumb_proximal`
  … `r_10_joint_little_distal`)과 같습니다. 앱은 다른 관절과 똑같이 FK로 그리면 됩니다.

## 6. 조작

| 입력 | 동작 |
|---|---|
| 오른손 A | 에피소드 녹화 시작 / 정지 (서버를 `--record-dir`로 실행한 경우) |
| 오른손 B | 녹화 중 이벤트 마커 |
| 그립(좌/우, 아날로그) | 그 손의 로봇 손가락을 그립 값만큼 오므림(손 명령 = 그립 × 0.6) + 손바닥 10 cm 안의 가장 가까운 물체 잡기 / 놓기 |
| 왼쪽 스틱 | 머리가 향한 방향 기준으로 이동 (최대 1 m/s) |
| 오른쪽 스틱 좌우 | 로봇 위치를 중심으로 회전 (최대 90°/s) |
| Y 1초 누르기 | 다시 보정 |

설정값은 `configs/sim.yaml`에 있습니다.

## 7. 씬 파일 (`scenes/<name>/scene.json`)

형식 전체는 `igris_sim/scene.py` 상단에 있습니다. 요약입니다.

| 필드 | 설명 |
|---|---|
| `spawn` | `{"pos": [x, y], "yaw_deg": 0}` 로봇과 사용자의 시작 위치 |
| `floor` | `{"rgba": [...]}` 또는 `false`(스캔한 바닥 메시를 쓸 때) |
| `objects[].pos`, `rot` | 바디 원점. `rot`은 `{"euler_deg": [roll, pitch, yaw]}`(고정축 X→Y→Z) 또는 `{"quat": [w,x,y,z]}` |
| `objects[].dynamic`, `graspable`, `mass` | 움직이는 물체 여부, 잡기 가능 여부(기본: dynamic), 질량(kg) |
| `objects[].geoms[]` | 여러 도형으로 된 물체. 없으면 물체 자체가 도형 하나입니다 |
| `shape`, `size` | `box` [x,y,z] 전체 길이 / `sphere` [r] / `cylinder`, `capsule` [r, 높이] |
| `mesh`: `file`, `collision`, `render`, `scale` | MuJoCo용 OBJ/STL, 충돌용 볼록 조각들, 앱 표시용 OBJ/STL(OBJ 정점 색 `v x y z r g b` 지원, 없으면 `file`) |

Unity에서 기본 도형으로 다시 만들 때 `localScale`은 다음과 같습니다(MuJoCo X·Y·Z = Unity Z·X·Y).

| shape | Unity primitive | localScale |
|---|---|---|
| box `[sx, sy, sz]` | Cube | `(sy, sz, sx)` |
| sphere `[r]` | Sphere | `(2r, 2r, 2r)` |
| cylinder `[r, h]` | Cylinder (높이 2) | `(2r, h/2, 2r)` |
| capsule `[r, h]` | Capsule (전체 높이 2) | `(2r, (h+2r)/2, 2r)` |

도형의 축은 MuJoCo 로컬 Z가 Unity 로컬 Y로 바뀝니다. 위치와 회전은 2장의 `FromMj`, `FromMjQuat`을 씁니다.

## 8. 주의할 점

- **kinematic 모드**에서는 로봇이 사람을 그대로 따라가므로 벽·책상을 통과합니다. 물체는 밀립니다.
  물리적으로 막혀야 하면 `harness` 모드를 쓰세요(로봇이 충돌하고 넘어지지 않도록 가상 하네스가 받칩니다).
- 로봇 루트는 사람보다 다리 비율(≈0.84)만큼 덜 움직입니다. 사람 시점(`rig`)과 로봇 머리(`head_cam`)는
  걸을수록 조금씩 어긋나므로, 몰입 모드에는 `head_cam`을 기준으로 한 로봇 시점을 권장합니다.
- 서버 처리 지연은 보정 3–6 ms, IK 6–17 ms, 물리 1 ms입니다(i7-9700K). 여기에 네트워크 지연이 더해집니다.
  앱의 머리 시점은 헤드셋에서 바로 그리므로 이 지연이 없습니다.
- 녹화된 에피소드(`--record-dir`)는 `raw/*`를 그대로 담으므로 `scripts/run_pipeline.py`로 오프라인 고품질
  보정·리타게팅을 다시 할 수 있습니다. `sim/*` 구성은 `igris_sim/recording.py` 상단에 있습니다.
