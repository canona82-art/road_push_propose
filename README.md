# RoadPush AI · 보행로 점검 관제

**걸으며 찍은 영상 하나로, 실제로 걸을 수 있는 보도 폭과 위험 구간을 지도 위에 보여 주는 AI 서비스입니다.**

스마트폰(Open Camera)으로 보도를 걸으며 영상과 GPS 자막(SRT)을 함께 기록합니다. RoadPush는 그 영상에서 보도 영역과 보행 방해물을 AI로 인식해 프레임마다 **유효 보행폭**과 **위험도(0~100점)** 를 계산합니다. 결과는 100m 구간별 안전 점수와 위험 등급 지도로 보여 줍니다.

## 주요 기능

- **보도 인식**: 보도·차도·자전거도로·화단을 픽셀 단위로 구분 (YOLOv8m-seg, 직접 학습)
- **보행 방해물 탐지**: PM·자전거, 불법 주정차 차량, 볼라드·기둥, 입간판·적치물 (YOLOv8m, 직접 학습)
- **유효 보행폭과 위험도**: 방해물을 피해 남는 가장 좁은 통로 비율로 위험도 산정, 보도 단절 자동 판정
- **GPS 연동 지도**: SRT 좌표로 실제 이동 거리를 계산해 100m 구간마다 점수 집계, 위험 지점을 지도에 표시
- **누적 점검**: 여러 번 촬영한 결과를 MySQL에 쌓고, 같은 거리는 최신 점검을 우선 표시
- **점검 리포트**: 구간별 결과와 위험 프레임을 HTML·CSV 리포트(ZIP)로 저장
- **개인정보 보호**: 영상 속 사람 얼굴 영역 자동 흐림 처리

## 저장소 구성

```
road_push_propose/
├─ roadpush_app/        실행 프로그램 (Flask 웹 대시보드 + 분석 엔진 + 학습된 가중치)
├─ roadpush_ai/         AI 모델 자료 (학습 코드, 학습 설정, 성능 결과, 모델 설명)
└─ 제출자료/            성과보고서, 데모 시연 영상, 점검 리포트 예시
```

| 폴더 | 내용 |
| --- | --- |
| `roadpush_app/app.py` | 웹 서버와 API (Linux용, 기본) |
| `roadpush_app/app_w.py` | 웹 서버와 API (Windows용, 영상 선택창만 다름) |
| `roadpush_app/video_processing.py` | 영상 프레임 추출 → AI 분석 → 100m 구간 집계 |
| `roadpush_app/detecting_ai/roadpush_core_f.py` | AI 분석 엔진 (보도 분할 + 방해물 탐지 + 위험도 계산) |
| `roadpush_app/detecting_ai/roadpush_weights/` | 학습된 가중치 `best.pt` 2개 |
| `roadpush_app/srt_gps.py` | Open Camera SRT 자막에서 GPS 읽기 |
| `roadpush_app/database.py` | MySQL 저장 (테이블 자동 생성) |
| `roadpush_app/reporting.py` | 점검 리포트 생성 |
| `roadpush_app/templates/`, `static/` | 대시보드 화면 |
| `roadpush_ai/` | 모델 학습 자료 → [roadpush_ai/README.md](roadpush_ai/README.md) |
| `제출자료/` | 성과보고서, 데모 시연 영상(`RoadPush_시연영상_최종.mp4`), 실제 분석으로 만든 점검 리포트 예시(`지자체점검리포트/`) |

## 모델 성능 (검증 데이터)

| 모델 | mAP50 | mAP50-95 |
| --- | --- | --- |
| 보도 분할 `sidewalk_seg_kr` (마스크) | 0.771 | 0.590 |
| 장애물 탐지 `obstacle_det_kr` | 0.796 | 0.525 |

학습 데이터, 클래스 구성, 학습 설정은 [roadpush_ai/README.md](roadpush_ai/README.md)에 정리되어 있습니다.

## 데모 시연 영상

`제출자료/RoadPush_시연영상_최종.mp4`에서 주요 기능(영상 분석, 지도 탐색, 위험 지점 확인, 점검 리포트 생성, 누적 점검)을 자막 설명과 함께 볼 수 있습니다.

## 실행 방법

### 준비물

- Python 3.10 이상
- MySQL 8.0 (실행 중이어야 함, 데이터베이스와 테이블은 자동 생성)
- ffmpeg (100m 구간 영상 생성)
- Linux에서는 영상 선택창용 `zenity` (Ubuntu: `sudo apt install zenity`)
- NVIDIA GPU 권장 (CPU로도 동작하지만 느림)

### 설치와 실행

```bash
cd roadpush_app
pip install -r requirements.txt
pip install PyMySQL cryptography
```

운영체제에 맞는 파일로 실행합니다. 두 파일은 "영상 선택" 창을 띄우는 방식만 다르고 나머지 기능은 같습니다.

| 운영체제 | 실행 명령 | 영상 선택창 |
| --- | --- | --- |
| Linux (Ubuntu) | `python app.py` | zenity |
| Windows | `python app_w.py` | Windows 기본 파일 선택창 |

브라우저에서 **http://127.0.0.1:5000** 에 접속합니다. 오른쪽 위에 "MySQL 연결됨"이 보이면 준비가 끝난 것입니다.

MySQL 접속 정보 기본값은 `127.0.0.1:3306`, 사용자 `root`, 비밀번호 `1234`, DB 이름 `road_push`입니다. 다르면 실행 전에 환경변수로 지정합니다.

```bash
export ROADPUSH_DB_PASSWORD='비밀번호'      # Windows cmd: set ROADPUSH_DB_PASSWORD=비밀번호
```

그 밖의 설정: `ROADPUSH_DB_HOST`, `ROADPUSH_DB_PORT`, `ROADPUSH_DB_USER`, `ROADPUSH_DB_NAME`, `ROADPUSH_ANALYSIS_FPS`(초당 분석 프레임 수, 기본 5).

### 사용 순서

1. Open Camera에서 GPS 자막 저장을 켜고 보도를 걸으며 촬영합니다. 영상과 같은 이름의 `.srt`가 함께 생깁니다 (예: `walk01.mp4`, `walk01.srt`).
2. 대시보드에서 **영상 선택**을 누르고 영상을 고릅니다. 같은 폴더의 SRT가 자동으로 연결되고 분석이 시작됩니다.
3. 분석 중에도 지도에 지점이 실시간으로 표시됩니다. 지점을 누르면 그 위치의 AI 분석 프레임과 위험도, 유효 보행폭, GPS를 볼 수 있습니다.
4. **점검 리포트 생성·저장**으로 결과를 내려받습니다.

Windows에서는 프로젝트 폴더 경로에 한글이 없어야 합니다 (OpenCV가 한글 경로의 영상을 열지 못하는 경우가 있음). 업로드한 영상 파일 이름의 한글은 자동으로 영문 이름으로 바뀌어 저장되므로 괜찮습니다.

분석을 시작하면 터미널에 아래 두 줄이 나와야 합니다. `yolov8m.pt`가 나오면 가중치 경로가 잘못된 것입니다.

```
[RoadPush] Seg Model: .../roadpush_app/detecting_ai/roadpush_weights/sidewalk_seg_kr/weights/best.pt
[RoadPush] Det Model: .../roadpush_app/detecting_ai/roadpush_weights/obstacle_det_kr/weights/best.pt
```
