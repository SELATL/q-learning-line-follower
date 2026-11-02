# 아키텍처

## 스택

Python 3.10+, NumPy>=1.24, Matplotlib>=3.7, 표준 라이브러리 unittest·dataclasses·enum·math. Gymnasium·PyTorch·MuJoCo는 현재 의존성이 아니다. requirements.txt는 범위 지정이며 lockfile은 없다. 재현 시 실제 설치 버전을 기록한다.

## 디렉터리

```text
main.py                       실행 옵션 및 Viewer 시작
line_follower/
  __init__.py                 패키지
  robot.py                    Action, Robot, 차동구동 원호 이동
  sensors.py                  센서 좌표변환 및 반사광 읽기
  track.py                    원형 트랙 및 반사광 함수
  simulator.py                초기화·시간 진행·관측·demo_action
  visualization.py            Matplotlib 화면·버튼·키보드·궤적
  env.py                      학습용 래퍼 초안, 거리 오차
tests/test_simulator.py       기존 9개 테스트
preview.png                   원본 제공 정적 미리보기
requirements.txt              실행 의존성
AGENTS.md                     세션 규칙
docs/                         컨텍스트·설계·이력·로드맵·이전 안내
```

## 현재 흐름

`main.py → Viewer → Simulator → Robot.step → sensors.read → Track.reflectance`.
자동 모드는 매 프레임 `demo_action(observe())`로 행동을 고른다. 수동 모드는 선택한 행동을 유지한다. Viewer는 env.py를 사용하지 않는다.

별도 API는 `LineFollowerEnv → Simulator`이며 reset/step 반환은 `[left,right]` 배열이다. Q-learning 연결과 Gymnasium 5개 반환값은 향후 기능이다.

## 모델 수치

좌표 x 오른쪽, y 위, theta=0은 오른쪽, 양수 회전은 반시계. 초기 `(0.65,0,pi/2)`. 원 반지름 0.65m, 선 폭 0.035m, 경계 폭 0.006m. 센서 전방 0.045m, 간격 0.040m. 검정=0, 흰색=100. 점 센서이며 잡음 없음.

바퀴 간격 0.10m, 평균 속도 0.12m/s, 회전 시 양쪽 속도 0.08/0.16m/s, dt=0.05s. 일정 바퀴 속도의 정확한 원호 적분 사용. 규칙 제어는 센서 차이가 5를 초과하면 더 어두운 방향으로 회전한다.

물리 접촉·슬립·관성·실제 모터 출력·장애물·이탈 종료는 없다. 가시화는 시뮬레이션 모델이며 실물 재현을 보장하지 않는다.
