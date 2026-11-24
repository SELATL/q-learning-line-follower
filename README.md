# Line Follower Sim — 개발 인수인계

> 아래 원본 설명서는 보존했습니다. 이 안내는 2026-11-03 이전 준비 시 추가했습니다.

이 프로젝트는 LEGO SPIKE Prime 경험을 바탕으로 Q-learning을 배우기 위한 Python 2D 시뮬레이터입니다. **`main.py`의 자동 주행은 규칙 기반입니다. Q-learning은 별도의 `dashboard.py`/`train.py`(랜덤 트랙·5행동)로 추가했습니다.** `line_follower/env.py`에는 Simulator 래퍼와 거리 오차만 있고 Gymnasium 환경은 아닙니다.

## Q-learning 대시보드 (2026-11-23 추가)

기존 원형 트랙·3행동 시뮬레이터(`main.py`)는 그대로 두고, 별도 실험으로 랜덤 코스 + 5행동 Q-learning을 추가했습니다.

```sh
# Finder에서 Run Dashboard.command 더블클릭 (가상환경이 없으면 자동 생성)
python dashboard.py                        # 빈 Q-table에서 학습 시작, 기본 코스: 미로
python dashboard.py --track curvy          # 부드러운 곡선 코스
python verify.py --exp e0-baseline         # 테스트 + 시드 3개 학습 + 미학습 트랙 비교 -> runs/<exp>/
python verify.py --exp e0-seeds10 --seeds 1 2 3 4 5 6 7 8 9 10
python train.py --episodes 500 --seed 1    # 화면 없이 학습 -> q_table.npy, train_log.csv
```

- 코스: `--track maze`(기본)는 9×6 격자 미로를 무작위 DFS로 만들고 그 안의 한 갈래 길(14~22칸, 약 3.5~6 m)을 선으로 씁니다. 90도 꺾임은 반지름 0.15 m 원호, 회색 선은 그리기 전용 벽입니다. `--track curvy`는 부드러운 곡선 코스입니다. 같은 시드 = 같은 코스.
- 왼쪽: 시뮬레이터, 센서 LED(검정 감지 시 초록), 지금 상태의 의미.
- 오른쪽 위: Q-table. 줄 = "센서가 본 것 × 직전 조향", 칸 = 행동별 Q값, ★ = 그 줄의 최선 행동, 괄호 = 방문 횟수.
- 오른쪽 아래: 학습 진행률(목표 에피소드 대비), 학습 방향(최근 완주율과 직전 구간 비교), 보상·완주율 그래프, 미학습 트랙 평가.
- 버튼: 시드 입력 + 새 학습, 학습/평가/규칙 모드, 일시정지, 한 스텝, 다음 트랙, 빠른 학습 50, 미학습 30개 비교, 속도.
- 키: `t` 학습, `e` 평가, `r` 규칙, `space` 일시정지, `.` 한 스텝, `n` 다음 트랙, `+/-` 속도, `b` 빠른 학습, `x` 미학습 비교, `s` Q-table 저장.
- 학습 결과 파일은 저장소에 포함하지 않습니다. 학습 실험은 사용자가 직접 진행하며 결과는 실험 기록 시트에 남깁니다.

설계 정의(상태·행동·보상·이탈·완주)는 `line_follower/rl_env.py` 맨 위 설명과 `docs/DEVELOPMENT_LOG.md`를 참조하세요. 여러 에이전트로 작업할 때 규칙은 `CLAUDE.md`.

## 새 세션에서 읽을 문서

- [작업 규칙](AGENTS.md)
- [프로젝트 맥락과 Next Session Handoff](docs/PROJECT_CONTEXT.md)
- [아키텍처](docs/ARCHITECTURE.md), [개발 로그](docs/DEVELOPMENT_LOG.md)
- [기술적 결정](docs/DECISIONS.md), [로드맵](docs/ROADMAP.md)
- [이전·검증 결과](docs/VALIDATION.md), [새 세션 초기 프롬프트](docs/SESSION_START.md)

## GitHub 상태

저장소: [SELATL/line-follower-sim](https://github.com/SELATL/line-follower-sim) — **Private**.
사용자 승인 후 저장소 생성과 최초 push를 완료했습니다. 로컬 `main`은 `origin/main`을 추적합니다.
원본 소스와 초기 커밋 이력을 보존했습니다. 후속 push는 해당 작업에 대한 사용자 승인을 확인합니다.

## Windows 설치

아래 명령은 프로젝트 루트에서 실행합니다. Python 3.10 이상과 Git이 필요합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py
```

활성화 없이 실행하므로 PowerShell 실행 정책을 바꿀 필요가 없습니다.

## Mac에서 가져오기

저장소 생성·최초 push가 완료되어 아래 clone을 사용할 수 있습니다. Private 저장소 권한이 있는 계정으로 Git 인증을 먼저 설정합니다. 인증 토큰을 URL이나 소스 파일에 넣지 않습니다.

```sh
git clone https://github.com/SELATL/line-follower-sim.git
cd line-follower-sim
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py
```

`--manual`로 수동 시작, `--snapshot snapshot.png`로 창 없이 PNG 저장이 가능합니다. Mac에서 실제 GUI는 아직 검증하지 않았습니다. `main.py`가 있는 루트에서 실행합니다. 개발 서버·웹 URL·포트는 없습니다.

## Codex Cloud 연결

2026-11-03 확인한 [공식 Cloud 안내](https://learn.chatgpt.com/docs/cloud)와 [환경 설정 안내](https://learn.chatgpt.com/docs/environments/cloud-environments)를 기준으로 합니다. 계정별 UI와 접근 권한은 다를 수 있습니다.

1. GitHub 업로드 후 Codex Cloud에서 GitHub를 연결하고 이 Private 저장소 접근을 허용합니다.
2. Cloud 환경 생성에서 저장소를 선택하고, Python 의존성 설치와 테스트를 요청합니다. Settings > Codex Cloud > Environments에서도 환경을 관리할 수 있습니다.
3. 설치 단계에 `python -m pip install -r requirements.txt`, 검증 단계에 아래 명령을 사용합니다. 환경의 실제 저장소 루트에서 실행하도록 설정합니다.
4. 결과를 검토하고 환경을 Publish한 뒤 새 작업에서 SESSION_START.md 프롬프트를 사용합니다.

```sh
python -m unittest discover -s tests -v
python main.py --snapshot snapshot-cloud.png
```

화면 없는 환경에서는 GUI 창 대신 기존 Agg 기반 snapshot 경로를 사용합니다. 패키지 다운로드가 허용되도록 package manager 네트워크 설정이 필요합니다. 프로젝트 실행에는 API 키·필수 환경변수·외부 서비스가 필요 없으며 `.env.example`도 현재 필요 없습니다. Matplotlib 캐시 쓰기 오류가 생기면 `MPLCONFIGDIR`를 쓰기 가능한 임시 디렉터리로 설정할 수 있습니다. Cloud 실제 설치·실행은 아직 검증하지 않았습니다.

## 두 컴퓨터에서 계속 작업하기

Mac/Windows 모두 새 저장소를 열고 작업 시작 시 `git status`를 확인합니다. 변경이 없다면 `git pull --ff-only`로 갱신합니다. 변경이 있다면 먼저 보존·정리하고 강제 덮어쓰기를 피합니다. 작업 완료 시 코드와 인수인계 문서를 함께 커밋하고, push는 승인 후 진행합니다. 원본 IDE 폴더와 새 저장소를 동시에 편집하지 마세요.

GitHub는 코드·문서·커밋을 공유합니다. 기존 대화 자체를 새 세션에 자동으로 복원하지 않으므로 PROJECT_CONTEXT.md와 SESSION_START.md가 핵심 인수인계입니다.

---

# Python 2D 선 추종 로봇 시뮬레이터

LEGO SPIKE Prime의 두 바퀴·두 컬러 센서 구성을 단순화한 학습용 모델입니다.
강화학습, Q-table, 보상, 학습 과정은 구현하지 않았습니다.

## 설치와 실행

Python 3.10 이상을 설치하고 터미널에서 이 폴더로 이동한 뒤 실행하세요.

```powershell
python -m pip install -r requirements.txt
python main.py
```

기본 실행은 센서값을 비교하는 **규칙 기반 자동 주행**입니다. 기계학습이 아닙니다.
수동으로 시작하려면 `python main.py --manual`을 실행하세요.

- 왼쪽/위쪽/오른쪽 방향키 또는 화면 버튼: 좌회전/직진/우회전. 선택하면 수동 모드로 바뀝니다.
- 선택한 행동은 다음 행동을 선택할 때까지 유지됩니다.
- Space: 일시 정지 또는 재개
- M: 수동/자동 전환
- R: 초기 위치로 복귀, 주행 궤적 삭제
- 창 닫기: 종료

방향키가 반응하지 않으면 트랙 영역을 클릭하세요. Matplotlib 창을 띄울 수 있는 데스크톱 환경에서 실행하세요.
화면 오른쪽 막대와 숫자는 현재 좌우 센서값입니다. 빨간 테두리 점은 센서 위치,
파란 삼각형의 뾰족한 부분은 전방, 얇은 파란 선은 주행 궤적입니다.
수동으로 트랙 밖으로 나가면 R로 복귀할 수 있습니다. 장애물이나 트랙 밖 충돌 판정은 없습니다.
화면의 문구는 기본 글꼴에서 깨지지 않도록 영어로 표시합니다.

## 파일 역할

| 파일 | 역할 |
|---|---|
| `main.py` | 실행 옵션을 읽고 화면 시작 |
| `line_follower/track.py` | 원형 트랙과 위치별 반사광 계산 |
| `line_follower/robot.py` | x, y, theta 상태, 바퀴 속도, 이동 계산 |
| `line_follower/sensors.py` | 좌우 센서 위치와 반사광 값 계산 |
| `line_follower/simulator.py` | 시간 진행, 관측, 초기화, 규칙 제어 |
| `line_follower/visualization.py` | 로봇·센서·궤적 표시, 키보드와 버튼 입력 |
| `tests/test_simulator.py` | 이동, 센서, 초기화, 자동 주행 검증 |
| `requirements.txt` | 필요한 외부 라이브러리 |

## 모델 이해하기

길이는 m, 시간은 s, 각도는 rad입니다. x는 오른쪽, y는 위쪽입니다.
theta=0이면 오른쪽을 보고, 양수 각도는 반시계 방향입니다.
초기 상태는 (0.65, 0, π/2)로 원의 오른쪽에서 위쪽을 봅니다.

검은 선은 반지름 0.65m인 원을 따라 폭 0.035m로 놓입니다.
검정은 반사광 0, 흰색은 100입니다. 선 가장자리 바깥 0.006m 구간은 0~100으로
부드럽게 바뀝니다. 센서는 로봇 중심보다 0.045m 앞쪽에 있고 서로 0.040m 떨어져 있습니다.
로봇이 회전하면 센서 위치도 함께 회전합니다. 여기서는 각 센서가 한 점을 읽습니다.

두 바퀴 간격 L=0.10m, 평균 속도는 0.12m/s입니다.
좌회전은 왼쪽 바퀴 0.08, 오른쪽 0.16m/s, 우회전은 그 반대이며 직진은 모두 0.12m/s입니다.

```text
전진 속도 v = (왼쪽 바퀴 속도 + 오른쪽 바퀴 속도) / 2
각속도 ω = (오른쪽 바퀴 속도 - 왼쪽 바퀴 속도) / L
```

오른쪽 바퀴가 더 빠르면 로봇이 왼쪽으로 돕니다.
`robot.step()`은 한 시간 간격 동안 바퀴 속도가 일정하다고 가정해 원호 이동을 계산합니다.
시간 간격 dt=0.05s이고 애니메이션은 프레임마다 한 번 이동합니다.
느린 컴퓨터에서는 화면의 실제 경과 시간과 시뮬레이션 시간이 다를 수 있습니다.

자동 주행은 왼쪽이 더 어두우면 좌회전, 오른쪽이 더 어두우면 우회전합니다.
차이가 5 이하이면 직진합니다. 두 센서 모두 흰색이면 직진하므로 선을 잃었을 때
스스로 찾아오는 기능은 없습니다. 제공된 트랙과 초기 위치에 맞춘 간단한 예제입니다.
실제 SPIKE의 모터 출력, 조명, 센서 잡음, 미끄러짐, 관성은 모델링하지 않았습니다.

## 테스트와 화면 저장

```powershell
python -m unittest discover -s tests -v
python main.py --snapshot preview.png
```

테스트는 검정/흰색/경계 반사광, 직진 거리, 회전 방향과 대칭성, 센서 좌표변환,
두 센서의 서로 다른 관측, 초기화, 잘못된 입력, 규칙 제어, 100초 트랙 유지 여부를 확인합니다.
화면 저장은 창을 띄우지 않고 10초 주행 후 PNG를 만듭니다.

## 직접 바꿔 보기

1. `robot.py`의 speed를 바꾸고 회전 속도와 주행 결과를 비교하세요.
2. `sensors.py`의 spacing 또는 forward를 바꾸어 선을 얼마나 일찍 읽는지 보세요.
3. `track.py`의 line_width를 바꾸어 좁은 선이 주는 영향을 확인하세요.

나중에 강화학습을 추가할 때는 `Simulator.observe()`, `step(action)`, `reset()`을
사용할 수 있습니다. 현재 observe()는 [왼쪽 반사광, 오른쪽 반사광] NumPy 배열을 반환하고,
step()은 한 번 움직인 뒤 새 센서 배열을 반환합니다. 자세는 sim.robot.x/y/theta로 읽습니다.
