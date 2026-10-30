# depth 카메라 촬영 → 시뮬레이션 가상 환경 (`igris_scan/`)

실제 공간을 RGB-D(depth) 카메라로 찍어, 시뮬레이션 서버와 PICO 앱이 쓰는 씬(`scenes/<name>/scene.json`)을 만듭니다.
배경(바닥, 책상, 벽, 가구)과 물체를 나누고, 촬영 중에 사람이 옮긴 물체(상호작용한 물체)를 찾아 움직일 수 있는 강체로 내보냅니다.

```
scan_capture.py ─> 촬영 폴더(RGB + depth + 시각) ─> scan_build.py
   1 재구성   카메라 추적(키프레임 정합) + TSDF 융합 (스캔 구간만)
   2 보정     중력·바닥(z = 0)·방향(+X)·원점(스폰)·스케일
   3 분할     바닥 / 받침면(책상) / 벽 / 구조물 / 물체
   4 상호작용 상호작용 구간을 모델에 정합 → 사라진 물체 = 옮겨진 물체, 마지막 장면 → 최종 위치
   5 내보내기 scene.json + 배경 메시 + 물체 메시·충돌 껍질 + scan_report.json + preview.png
```

## 1. 촬영

### 장비

- **Intel RealSense D4xx / L515**: `scripts/scan_capture.py`로 바로 촬영합니다(640×480, 30 fps, depth를 컬러 카메라에 정렬).
  realsense-viewer로 녹화한 `.bag`도 `--bag`으로 변환됩니다.
- **다른 카메라**(Azure Kinect, Orbbec, ZED, iPhone LiDAR 앱 등): 아래 폴더 형식으로 저장하면 그대로 처리됩니다.

```
<capture>/
  capture.json    {"format": "igris_rgbd", "version": 1, "depth_scale": 1000.0,
                   "intrinsics": {"width", "height", "fx", "fy", "cx", "cy"}, "fps": 30, "source": "...", "device": "..."}
  frames.csv      index,t_s           (프레임 번호, 첫 프레임부터 초)
  color/000000.png  8비트 RGB (jpg도 됨), depth와 같은 내부 파라미터로 정렬
  depth/000000.png  16비트, 값 / depth_scale = 광축 방향 거리(m), 0 = 측정 없음
  poses.txt       (선택) 프레임마다 카메라→월드 4×4 (16개 수, 행 우선). 외부 트래커가 있으면 추적을 건너뜁니다
```

카메라 좌표는 OpenCV 규약(x 오른쪽, y 아래, z 앞)입니다.

### 촬영 방법 (한 번에 이어서 찍어도 됩니다)

1. **스캔 구간** – 물건을 건드리지 않습니다.
   - 로봇·조작자가 설 자리에서 시작합니다. 첫 카메라 위치의 바닥이 씬의 원점(스폰)이 됩니다.
   - 카메라를 천천히(0.5 m/s 이하, 급회전 없이) 움직여 바닥, 책상 윗면과 모서리, 물체를 여러 방향에서 비춥니다.
   - 카메라를 좌우로 돌리며 훑으면 중력 방향을 카메라 자세에서도 구할 수 있어 보정이 안정됩니다.
2. **상호작용 구간** – 그 시각을 기억해 두었다가 `--split`으로 줍니다.
   - 카메라를 삼각대에 두거나(`--interaction-camera static`) 든 채로 둡니다.
   - 물건을 집어 옮기고, 마지막 1초는 모두 프레임 밖으로 나가거나 가만히 둡니다(최종 위치를 찍기 위해).

```powershell
.venv\Scripts\python scripts\scan_capture.py --list                                    # 연결된 카메라
.venv\Scripts\python scripts\scan_capture.py data\scans\lab01 --seconds 60             # 촬영 (Ctrl+C로 멈춤)
.venv\Scripts\python scripts\scan_capture.py data\scans\lab01 --bag rec.bag            # .bag 변환
```

## 2. 씬 만들기

```powershell
.venv\Scripts\python scripts\scan_build.py data\scans\lab01 --name lab01 --split 25
.venv\Scripts\python scripts\scan_build.py data\scans\lab01 --name lab01 --split 25 --interaction-camera static
.venv\Scripts\python scripts\scan_build.py data\scans\lab01 --name lab01 --support-height 0.74 --set scan.voxel=0.005
.venv\Scripts\python scripts\scan_build.py --synthetic data\scans\demo --name scan_demo       # 카메라 없이 합성 데모
.venv\Scripts\python scripts\sim_server.py --scene lab01 --viewer                             # 만든 씬에서 텔레오퍼레이션
```

설정은 `configs/scan.yaml`에 있고, 어떤 값이든 `--set 섹션.키=값`으로 바꿀 수 있습니다.

### 2-1. 재구성 (`igris_scan/reconstruct.py`)

- depth를 먼저 정리합니다. 범위 밖 값과 물체 경계의 "떠 있는 픽셀"(이웃과 깊이 차가 4 % 이상)을 버립니다.
- 카메라는 키프레임 정합으로 추적합니다.
  - 각 프레임을 현재 키프레임의 depth에 다중 스케일 point-to-plane ICP로 맞춥니다. 시작 자세는 등속 운동 모델로 예측합니다.
  - 키프레임에서 12 cm 또는 10° 이상 움직이면 새 키프레임으로 삼습니다.
  - TSDF 레이캐스트가 아니라 실제 측정 depth에 맞추는 이유: Open3D의 레이캐스트는 비스듬히 보이는 면을
    약 한 복셀(1–3 cm) 멀리 그려, 프레임마다 같은 방향으로 오차가 쌓였습니다(합성 시험에서 10 cm → 1.7 cm로 줄임).
- 정합이 실패한 프레임(일치 비율 25 % 미만, 15 cm·0.15 rad 이상 튐)은 융합하지 않고 리포트에 셉니다.
- 스캔 구간만 TSDF에 넣습니다. 상호작용 구간은 사람과 옮겨지는 물체가 섞이므로 정합만 합니다.

### 2-2. 보정 (`igris_scan/align.py`)

| 항목 | 방법 |
|---|---|
| 중력(위) | RANSAC 평면 중 수평면 후보. 손으로 든 카메라는 거의 굴리지 않으므로 이미지 x축은 늘 수평이고, 훑는 동안 x축들에 모두 수직인 방향이 위입니다(카메라 피치와 무관). 카메라가 돌지 않았으면 이미지 위쪽(-y)을 넓은 허용각으로 씁니다 |
| 바닥 | 첫 카메라보다 0.3 m 이상 아래인 가장 낮은 수평면. 그 평면을 z = 0에 둡니다 |
| 방향 | 첫 카메라가 본 방향을 +X로 하되, 벽·책상 모서리가 이루는 방의 축 중 가장 가까운 것에 맞춥니다(최대 45°) |
| 원점 | 첫 카메라 아래 바닥 = 씬의 스폰(`align.origin_offset`으로 옮김) |
| 스케일 | depth는 미터 단위지만 스테레오 카메라는 1–2 % 틀릴 수 있습니다. 책상의 실제 높이를 `--support-height`로 주면 그 높이에 맞춰 축척합니다 |
| 바닥이 안 보일 때 | 수평면이 하나뿐이고 작으면(1.5 m² 미만) `--support-height`를 주세요. 그 면을 그 높이의 책상으로 둡니다 |

`scan_report.json`의 `alignment`에 변환 행렬, 바닥 잔차, 첫 카메라 높이, 주의사항(`notes`)이 남습니다.

### 2-3. 분할 (`igris_scan/segment.py`)

| 분류 | 기준 | 씬에서 |
|---|---|---|
| 바닥 | z = 0 ± 2 cm | MJCF 바닥(평평, 충돌), 스캔한 바닥 색 |
| 받침면 | 위를 향한 점들의 높이 히스토그램 봉우리 중 0.3 m² 이상 이어진 면(+ 그 위 1.5 cm 층) | 윗면이 측정 높이인 상자(보임 + 충돌) |
| 벽 | 1 m² 이상의 수직 평면 | 벽 뒤쪽의 얇은 상자(충돌만, 보이는 것은 배경 메시) |
| 물체 | 받침면·바닥 위에 놓였고 0.8 m × 1 m 이하, 벽에 붙은 조각·두께 2 cm 미만 조각·책상 다리가 아닌 덩어리 | 물체 메시(vertex color) + 볼록 껍질 충돌체 |
| 구조물 | 나머지 덩어리(가구, 잡동사니) | 배경 메시(보임) + 큰 것은 볼록 껍질 충돌체(안 보임) |

- 0.3 m²보다 작은 수평면(상자 윗면 등)은 받침면이 아니라 그 물체의 일부입니다.
- 물체 크기: TSDF는 옆면을 1–3 cm 부풀립니다. 윗면이 평평하고 충분히 넓으면(상자, 캔) 윗면 윤곽으로 바닥 크기를 정합니다.
- 충돌체: 물체 표면 점과 그 점을 받침 높이로 내린 점의 볼록 껍질입니다. 그래서 물체가 받침에 평평하게 놓입니다.

### 2-4. 상호작용한 물체 (`igris_scan/changes.py`)

- 상호작용 구간의 프레임마다 각 물체 표면점(카메라를 향한 것)을 투영해 측정 depth와 비교합니다.
  - 거의 같음(3 cm + 2 %): 그대로 있음
  - 측정이 더 멂: 물체가 있던 자리 너머가 보임 = 사라짐
  - 측정이 더 가까움: 손이나 사람이 앞을 가림 = 판단하지 않음
- "사라짐" 비율이 0.6을 넘는 상태가 0.3초 이상 이어지면 옮겨진(상호작용한) 물체입니다. 들었다가 제자리에 놓은 경우도 포함됩니다.
- 마지막 1초를 따로 융합해 스캔에 없던 덩어리를 찾고, 옮겨진 물체와 크기·색으로 짝지어 최종 위치를 구합니다.
  물체가 원래 가리던 받침면(높이 3 cm 미만)은 제외합니다.
- 상호작용한 물체만 자유 강체(`dynamic: true`)가 되고 나머지는 고정입니다(`--dynamic all|none`으로 변경).
  질량은 볼록 껍질 부피 × 250 kg/m³(0.05–5 kg로 제한)이므로, 실제 무게를 알면 `scene.json`의 `mass`를 고치세요.

## 3. 결과물 (`scenes/<name>/`)

```
scene.json             igris_sim/scene.py 형식. 스폰은 원점, +X 방향
background.obj         배경 메시(vertex color; 바닥·받침면 윗면·물체는 뺌, 최대 30만 삼각형)
objects/<obj>.obj      물체 표면(vertex color, 물체 좌표계)
objects/<obj>_hull.obj 물체 충돌체(볼록 껍질)
collision/*.obj        큰 구조물의 충돌체(안 보임)
scan_report.json       보정, 표면, 물체(크기, 받침, 상호작용 여부, 사라진 시각, 최종 위치, 질량), 설정, 시간
preview.png            위에서 본 분할 결과(주황: 상호작용한 물체, 화살표: 최종 위치)
```

- 씬은 서버가 HTTP로 앱에 보내므로 앱을 다시 빌드할 필요가 없습니다. 앱은 메시의 vertex color를 그대로 그립니다.
- MuJoCo 뷰어는 vertex color를 그리지 못해 배경은 회색, 물체는 스캔한 대표색으로 보입니다.
- 만든 직후 `igris_sim`의 씬 빌더로 한 번 컴파일해 검사합니다.

## 4. 검증 (합성 촬영, `tests/test_scan.py`)

MuJoCo로 렌더링한 5 × 5 m 방(책상, 상자 2개, 캔, 바닥의 큰 상자, 지나가며 상자를 옮기는 사람)입니다(`igris_scan/synthetic.py`).
depth에 σ = 1.5 mm × d² 노이즈와 1 mm 양자화를 넣었고, 정답은 `gt.json`에 있습니다.

| 항목 | 결과 |
|---|---|
| 카메라 추적 (0.5 m/s 스윕, 240프레임) | 최대 1.3 cm, 프레임 손실 없음 |
| 상호작용 구간 정합 (사람이 화면에 있음) | 최대 1.9 cm |
| 바닥 평면 잔차 / 첫 카메라 높이 | 2 mm / ±3 cm 이내 |
| 책상 높이 | ±1.5 cm |
| 물체 검출 | 4개 모두, 위치 ±5 cm, 높이 ±2 cm |
| 상호작용 판정 | 옮긴 상자만(사라짐 비율 0.92, 나머지 ≤ 0.13), 이동 벡터 ±6 cm |
| 시간 (640×480, 스캔 120프레임 + 상호작용 120프레임) | 약 24초 |

## 5. 한계

- 반사·투명한 물체(유리, 금속, 검은 플라스틱)는 depth가 비어 형태가 빠집니다.
- 한 방향에서만 본 면은 TSDF에 없으므로, 물체 뒤쪽은 볼록 껍질로만 채워집니다. 여러 방향에서 훑으세요.
- 오목한 물체(그릇, 컵 안쪽)는 볼록 껍질 충돌체라 안쪽이 막힙니다.
- 물체끼리 붙어 있으면 한 덩어리로 나옵니다. 촬영 전에 조금 떼어 두세요.
- 실제 카메라 데이터로는 아직 검증하지 않았습니다. 첫 촬영 후 `preview.png`와 `scan_report.json`의 `notes`를 확인하세요.
