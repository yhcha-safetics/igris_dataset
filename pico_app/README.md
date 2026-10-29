# IGRIS Sim: PICO 4 Ultra 앱

PICO 4 Ultra 안에서 IGRIS-C 시뮬레이션을 조종하고 데이터를 수집하는 앱(Unity)입니다.
헤드셋, 컨트롤러와 모션 트래커의 움직임을 PC 시뮬레이션 서버(`scripts/sim_server.py`)로 보냅니다.
서버는 그 움직임을 IGRIS-C에 리타게팅하고 MuJoCo로 물리를 계산한 뒤, 로봇과 물체의 상태를 앱으로 돌려줍니다.
통신 규격은 [docs/sim_protocol.md](../docs/sim_protocol.md)에 있습니다.

```
PICO 4 Ultra (이 앱)                          PC (sim_server.py --source udp)
 헤드셋 / 컨트롤러 / 모션 트래커 ── UDP 9870 ──>  보정 → 리타게팅 → MuJoCo 물리 → 녹화(h5)
 로봇, 씬, 물체 렌더링            <── 50 Hz ──   로봇 qpos, 물체 포즈, 잡기 상태
 로봇 메시, 씬 메시              <── HTTP 9871 ─ scenes/ 폴더, 로봇 모델(_robot/)
```

앱에는 씬이나 로봇 모델이 들어 있지 않습니다. 접속할 때 서버에서 받아오므로, `scenes/`에 씬을 추가하면 앱을 다시 빌드하지 않아도 됩니다.

## 1. 준비

| 항목 | 내용 |
|---|---|
| Unity | **Unity Hub**에서 Unity 계정으로 로그인한 뒤 **2022.3 LTS 최신 패치**를 설치합니다. 설치할 때 **Android Build Support**를 선택하고, 그 아래의 **OpenJDK**와 **Android SDK & NDK Tools**도 함께 선택합니다. 2022.3이 목록에 없으면 Hub의 *Install Editor → Archive*에서 찾습니다. 디스크는 약 15 GB가 필요합니다. |
| Git | PICO SDK를 GitHub에서 받아옵니다. `git`이 PATH에 있어야 합니다(이미 설치되어 있음). |
| PICO 4 Ultra | 개발자 모드와 USB 디버깅을 켭니다. `adb devices`에 헤드셋이 보여야 합니다. |
| 네트워크 | 헤드셋과 PC가 같은 공유기(같은 서브넷)에 연결되어 있어야 합니다. |

PC 방화벽에서 서버 포트를 엽니다. 관리자 PowerShell에서 한 번만 실행하면 됩니다.

```powershell
netsh advfirewall firewall add rule name="IGRIS sim UDP" dir=in action=allow protocol=UDP localport=9870
netsh advfirewall firewall add rule name="IGRIS sim HTTP" dir=in action=allow protocol=TCP localport=9871
```

## 2. 빌드

1. Unity Hub에서 *Add → Add project from disk*로 `pico_app` 폴더를 추가하고 엽니다.
   - 처음 열 때 PICO SDK 3.4.0([GitHub](https://github.com/Pico-Developer/PICO-Unity-Integration-SDK))과 XR 패키지를 받습니다. 몇 분 걸립니다.
2. 메뉴에서 **IGRIS → 1. Setup project**를 실행합니다. 다음을 자동으로 설정합니다.
   - Android, IL2CPP, ARM64
   - PICO XR 로더(Multiview)
   - 바디 트래킹 권한, LAN HTTP 허용
   - `Main` 씬
   - *Active Input Handling*이 바뀌었다는 안내가 나오면 에디터를 다시 시작합니다.
3. **IGRIS → 2. Build APK**를 실행하면 `pico_app/Builds/IgrisSim.apk`가 만들어집니다.

검증: Unity 2022.3.16f1(Windows 11, 명령줄 빌드)로 빌드해 PICO 4 Ultra Enterprise(A9210, PICO OS 5.15.7)에서
로봇·씬 표시, 손가락(그립), 상태 창, 녹화까지 확인했습니다. `ProjectSettings/`와 `Packages/packages-lock.json`(PICO SDK 커밋 고정)은
저장소에 들어 있습니다. 이것 없이 처음 열면 Unity가 `manifest.json`에 템플릿 패키지(ads, purchasing 등)를 넣고
APK에 광고·결제 권한이 붙으니, 그런 경우 `manifest.json`을 되돌리고 다시 빌드합니다.

명령줄로 빌드할 수도 있습니다. Unity Hub에서 로그인하고 Personal 라이선스를 활성화한 뒤에 쓸 수 있습니다.
(Unity Hub로 설치한 에디터 경로 예: `%USERPROFILE%\Unity\Hub\Editor\2022.3.16f1\Editor\Unity.exe`)

```powershell
$unity = "C:\Program Files\Unity\Hub\Editor\<버전>\Editor\Unity.exe"
& $unity -batchmode -quit -projectPath pico_app -executeMethod IgrisPico.Editor.ProjectSetup.Run -logFile setup.log
& $unity -batchmode -quit -projectPath pico_app -executeMethod IgrisPico.Editor.Builder.BuildAndroid -logFile build.log
adb install -r -g pico_app\Builds\IgrisSim.apk
```

## 3. 실행

PC에서 서버를 띄웁니다.

```powershell
.venv\Scripts\python scripts\sim_server.py --source udp --scene tabletop --record-dir data\sim01\raw --viewer
```

헤드셋에서 *라이브러리 → 알 수 없는 출처*에 있는 **IGRIS Sim**을 실행합니다.

- 앱은 같은 네트워크의 서버를 자동으로 찾습니다(UDP 브로드캐스트). 찾지 못하면 아래 *문제 해결*을 보세요.
- 연결되면 로봇과 씬을 받아오고, 이어서 **3초 동안 가만히 서 있으면** 보정이 끝납니다(눈앞에 진행률 표시).
- 보정이 끝나면 로봇이 몸을 따라 움직입니다. 기본 시점은 로봇 머리 위치에서 보는 1인칭입니다.

### 조작

| 입력 | 동작 |
|---|---|
| 오른손 A | 에피소드 녹화 시작 / 정지 (서버를 `--record-dir`로 실행한 경우) |
| 오른손 B | 녹화 중 이벤트 마커 |
| 그립(좌/우) | 로봇 손가락이 그립을 쥔 만큼 오므라듭니다. 손 가까이(10 cm)의 물체 잡기 / 놓기. 놓을 때 손을 휘두르면 던질 수 있습니다 |
| 왼쪽 스틱 | 바라보는 방향 기준 이동 |
| 오른쪽 스틱 좌우 | 회전 |
| 왼손 Y 1초 | 다시 보정 |
| 왼손 메뉴 버튼 | 메뉴 열기 / 닫기 |

메뉴는 왼쪽 스틱 위아래로 항목을 고르고, 좌우로 값을 바꾸고, 왼쪽 트리거로 실행합니다. 메뉴가 열려 있는 동안 왼쪽 스틱은 이동에 쓰이지 않습니다.

| 메뉴 | 내용 |
|---|---|
| Scene | 씬 바꾸기(`scenes/` 폴더의 씬, 트리거로 불러오기) |
| Physics | kinematic(로봇이 그대로 따라옴) / harness(로봇이 충돌하고, 가상 하네스가 받침) |
| View | embodied(로봇 머리 시점) / behind(로봇 뒤에서 보기, 트리거로 방향 다시 맞추기) |
| Robot look | solid / ghost(반투명) |
| Trackers | body / objects / none (아래 *모션 트래커* 참고) |
| Object trackers | 오브젝트 모드에서 쓰는 트래커 수(1–3) |
| Motion Tracker calibration | PICO 모션 트래커 앱을 열어 트래커 보정 |
| Start/Stop recording, Recalibrate, Reset to spawn | 녹화, 다시 보정, 시작 위치로 이동 |
| Server | 연결된 서버 주소. 트리거를 누르면 서버를 다시 찾습니다 |

왼쪽 컨트롤러 위의 상태 창에는 다음이 표시됩니다.
- 연결 상태
- 보정/동작 상태, 트래커 프로파일
- 씬, 녹화 여부, 에피소드 수
- 잡은 물체
- 서버 처리 시간, 데이터 지연

### 모션 트래커

트래커 모드는 PICO 시스템의 *모션 트래커* 앱에서 고른 모드와 같아야 합니다. 서버는 들어오는 데이터를 보고 트래커 프로파일을 자동으로 고릅니다.

| 앱 메뉴 Trackers | PICO 모션 트래커 앱 | 서버 프로파일 |
|---|---|---|
| body | 바디 트래킹(발목 2개) | `body_2trk` |
| body | 바디 트래킹(발목 2개 + 허리 1개) | `body_3trk`: 서버를 `--body-trackers 3`으로 실행 |
| objects | 오브젝트 트래킹(허리/발목 등에 1–3개) | `sparse_trackers` |
| none | 트래커 없음 | `hmd_3pt`: 다리는 추정합니다. 서버에 `--subject-height`를 주세요 |

트래커 모드를 바꾼 뒤에는 메뉴에서 **Recalibrate**를 실행하세요. 프로파일은 보정할 때 정해집니다.

### 녹화한 데이터

`--record-dir`로 저장한 에피소드는 기존 h5 형식(`raw/*`)에 시뮬레이션 결과(`sim/*`)가 추가된 것입니다.
손 명령(`sim/hand/cmd`, Robros 정규화 12개)과 손가락 각도(`sim/hand/joint_pos`)도 함께 저장됩니다.
`scripts/run_pipeline.py`로 오프라인 고품질 보정과 리타게팅을 다시 할 수 있습니다.

## 4. 헤드셋 없이 에디터에서 시험

Unity 에디터에서 `Assets/IgrisPico/Scenes/Main.unity`를 열고 Play를 누르면, 키보드와 마우스로 헤드셋을 흉내 냅니다.
서버는 같은 PC에서 `--source udp`로 실행합니다.

| 키 | 동작 |
|---|---|
| 마우스 오른쪽 드래그 | 고개 돌리기 |
| W A S D, Q / E | 왼쪽 스틱(이동), 오른쪽 스틱(회전) |
| Space, B, X, Y | A, B, X, Y 버튼 |
| G / H, F / J | 왼쪽 / 오른쪽 그립, 왼쪽 / 오른쪽 트리거 |
| Tab, 방향키 | 메뉴 열기, 메뉴 이동 |

## 5. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| *Searching for the simulation server* | 서버를 `--source udp`로 실행했는지, PC와 헤드셋이 같은 네트워크인지, 방화벽 UDP 9870이 열렸는지 확인합니다. 공유기가 브로드캐스트를 막으면 아래처럼 서버 주소를 파일로 지정합니다 |
| *Robot download failed* | 방화벽 TCP 9871이 열렸는지 확인합니다 |
| 글자가 한쪽 눈에만 보임 | `Assets/XR/Settings/PXR_Settings`에서 Stereo Rendering Mode를 MultiPass로 바꿉니다 |
| 몸이 따라오지 않음 (`hmd_3pt`로 동작) | 메뉴 Trackers가 PICO 모션 트래커 앱의 모드와 같은지 확인하고, 트래커를 보정한 뒤 Recalibrate를 실행합니다 |
| 상태 창 `trackers body (no body data (BT_MESSAGE_UNKNOWN))` | 트래커 착용 보정이 풀린 상태입니다(헤드셋을 켤 때마다 다시 해야 함, `adb logcat`의 `Swifter_StatusInfo`에서 `WearCalibStatus:0`). 메뉴 *Motion Tracker calibration*으로 보정한 뒤 *Recalibrate* |
| PC와 헤드셋이 다른 서브넷(예: PC는 유선 교내망, 헤드셋은 공유기 Wi-Fi) | 브로드캐스트가 닿지 않으므로 아래처럼 `igris_pico.json`에 PC 주소를 넣습니다. 헤드셋 → PC 방향 UDP/TCP는 공유기를 넘어 동작합니다. USB(`adb reverse`)는 UDP를 넘기지 못해 이 앱에는 쓸 수 없습니다 |

서버 주소를 직접 지정하려면 다음 파일을 만들어 넣습니다.

```powershell
'{"server": "192.168.0.10"}' | Out-File -Encoding ascii igris_pico.json
adb push igris_pico.json /sdcard/Android/data/com.igris.picosim/files/igris_pico.json
```

같은 파일에 `"trackers": "body" | "objects" | "none"`, `"object_trackers": 2`, `"view": "embodied" | "behind"`도 지정할 수 있습니다.

## 6. 구조

| 경로 | 내용 |
|---|---|
| `Assets/IgrisPico/Core/` | Unity에 의존하지 않는 부분: JSON, 좌표 변환, 로봇 FK, 씬 해석, OBJ/STL, 상태 보간. `tests/test_app.py`가 csc로 컴파일해 서버(Python, MuJoCo) 결과와 비교합니다 |
| `Assets/IgrisPico/Runtime/` | 앱: `AppMain`(메인 루프), `SimLink`(UDP), `HeadsetInput`, `PicoTracking`(PICO SDK), `RobotView`, `SceneView`, `RigView`(시점), `Hud`(상태 창과 메뉴) |
| `Assets/IgrisPico/Editor/` | `ProjectSetup`(프로젝트 설정, 씬 생성), `Builder`(APK 빌드) |
| `Assets/IgrisPico/Resources/` | 셰이더(Lit, LitTransparent, Unlit, Text). Multiview를 지원합니다 |
| `CoreTests/` | Core 단위 테스트 프로그램(Unity 밖에서 실행) |

좌표계는 다음 순서로 처리합니다.
1. 헤드셋, 컨트롤러, 바디 트래킹 포즈를 Unity 트래킹 공간(XR Rig의 Camera Offset 기준) 값으로 읽습니다.
2. `Conv.NativeFromUnity`로 native로 바꿔 보냅니다.
3. 오브젝트 트래커는 SDK가 native 값을 주므로 그대로 보냅니다.
4. 서버가 보내는 `rig`(트래킹 공간 → 월드)를 XR Rig에 그대로 적용하면, 실제 몸과 시뮬레이션 월드가 맞습니다. 이 변환식은 `tests/test_app.py`가 검증합니다.

## 7. 라이선스

- 이 폴더의 코드는 저장소의 다른 코드와 같은 조건입니다.
- **PICO Unity Integration SDK**는 PICO의 독점 라이선스입니다(LICENSE.md: 사전 허가 없는 배포와 복제 금지).
  - 이 저장소에는 SDK를 넣지 않았습니다. Unity가 빌드할 때 GitHub에서 받습니다(`Packages/manifest.json`).
  - 빌드한 APK에는 SDK가 포함됩니다. 내부 연구용 설치는 문제가 없지만, 외부에 배포하려면 PICO 개발자 약관을 확인하세요.
