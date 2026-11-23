@AGENTS.md

# 멀티에이전트 추가 규칙 (Hermes · Claude Code · Codex 공통)

위 AGENTS.md가 기본 규칙이다. 아래는 여러 에이전트가 함께 작업할 때만 추가로 지킨다.
AGENTS.md와 충돌하면 AGENTS.md가 우선한다.

## 역할
- 한 변경에는 작성자 1명, 리뷰어 1명. 리뷰어는 작성자와 다른 에이전트여야 한다.
- 리뷰어는 파일을 수정하지 않는다. 지적만 하고, 수정은 작성자가 한다.
- 통과/실패 판정은 에이전트가 아니라 `python verify.py`의 종료 코드로 한다 (0 통과, 1 게이트 실패, 2 테스트 실패).

## 검증
- 코드를 바꾼 뒤에는 `python -m unittest discover -s tests`를 먼저 돌린다.
- 학습·보상·상태·행동·종료 조건(`line_follower/rl_env.py`, `qlearning.py`, `random_track.py`)을 바꿨으면 `python verify.py --exp <짧은-이름>`까지 돌린다. 중간 확인은 `--quick`.
- step 수를 줄인 스모크 테스트로 학습 성공을 판단하지 않는다. 에피소드 하나가 약 750 step이고, 학습 효과는 200~300 에피소드 이후에 나타난다.
- 완주율은 학습에 쓰지 않은 트랙(`EVAL_SEED_BASE`)에서 탐험을 끄고 측정한 값만 성능으로 보고한다. 학습 중 성공률은 성능 근거로 쓰지 않는다.
- 규칙 기반과 같은 트랙에서 비교한다. 규칙 기반 이상이라는 주장은 수치로 확인된 경우에만 한다.

## 실험 기록
- `verify.py`가 `runs/<exp>/`에 `config.json`(하이퍼파라미터·보상 상수·git commit), `metrics.csv`(에피소드별 기록), `summary.json`(판정)을 저장한다.
- 보상·상태·종료 정의를 바꾸면 `rl_env.py` 맨 위 설명과 `docs/DEVELOPMENT_LOG.md`를 함께 고치고, 바꾸기 전과 후의 `runs/<exp>` 이름을 기록한다.
- 서로 다른 보상 설계의 총 보상 크기로 우열을 판단하지 않는다. 미학습 완주율·평균 진행률로 비교한다.

## 금지
- 기존 파일(`robot.py`, `sensors.py`, `track.py`, `simulator.py`, `visualization.py`, `env.py`, `main.py`, `tests/test_simulator.py`)의 동작 변경. 필요하면 새 파일로 확장한다.
- 테스트를 통과시키려고 테스트 기대값이나 `verify.py`의 게이트 기준을 낮추는 것. 기준 변경은 사용자 승인 후 근거와 함께 기록한다.
- 사용자 승인 없는 commit, push, 원격 변경.
