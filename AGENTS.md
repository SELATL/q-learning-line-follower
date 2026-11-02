# Codex 작업 규칙

- 먼저 `docs/PROJECT_CONTEXT.md`, `docs/ROADMAP.md`, `docs/DECISIONS.md`를 읽고 실제 코드와 대조한다.
- 한국어로 설명한다. 사용자는 Python 초급자로, Q-learning 수식은 이해했으나 환경과 코드 연결은 단계별 설명이 필요하다. 한 번에 많은 코드를 추가하지 말고 작은 변경과 확인 명령을 제시한다.
- 구현 사실, 사용자의 실행 보고, 대화에서 제안한 설계를 구분한다. 제안을 완료로 기록하지 않는다.
- 이 저장소는 이전 준비 사본이다. 원본 소스의 내용은 보존했다. 이번 이전 작업에서 알고리즘·서식·기능을 수정하지 않았다.
- 기본 자동 주행은 규칙 기반이며 학습이 아니다. `env.py`는 아직 Gymnasium 환경이 아니다.
- 기본 모델의 단위는 m, s, rad. 행동은 LEFT=0, STRAIGHT=1, RIGHT=2. 순서를 변경하지 않는다.
- 이동 계산은 `robot.py`, 센서는 `sensors.py`, 트랙은 `track.py`, 시간 진행은 `simulator.py`, 학습 환경 어댑터는 `env.py`, 표시는 `visualization.py`에 둔다.
- `Viewer`는 `Simulator`를 직접 사용한다. `env.py`를 바꿔도 화면이 자동으로 새 환경을 사용하는 것은 아니다.
- 향후 Gymnasium 반환 형식 변경은 기존 Simulator와 분리하고 호출부·테스트를 함께 검토한다. 시간 제한과 실제 실패 종료를 구분한다.
- 새 절대경로, 인증 정보, 설치된 라이브러리, 캐시를 저장소에 넣지 않는다. 외부 서비스와 API 키는 현재 필요 없다.
- GitHub 생성·remote 변경·push는 사용자의 사전 승인이 필요하다. 기존 브랜치·커밋을 덮어쓰거나 force push하지 않는다.
- 작업 후 사실에 맞게 컨텍스트·로드맵·개발 로그를 갱신한다. 실험 조건과 검증 한계를 기록한다.

## 실행 및 검증

프로젝트 루트에서 Python 3.10 이상 사용. 실제 검증 환경은 `docs/VALIDATION.md` 참조.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
python main.py --manual
python main.py --snapshot snapshot.png
python -c "from line_follower.env import LineFollowerEnv; e=LineFollowerEnv(); print(e.reset()); print(e.step(1)); print(e.distance_error())"
```

화면 없는 환경에서는 `--snapshot`을 사용한다. 서버와 포트는 없다. OS별 가상환경 명령은 README 참조.
