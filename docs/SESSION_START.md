# 새 Codex 세션 시작 프롬프트

아래 내용을 프로젝트를 연 새 세션에 붙여 넣는다.

```text
이 저장소의 선 추종 Q-learning 학습 프로젝트를 이어서 진행해줘.
먼저 AGENTS.md, docs/PROJECT_CONTEXT.md의 Next Session Handoff,
docs/ROADMAP.md, docs/DECISIONS.md, docs/ARCHITECTURE.md,
docs/DEVELOPMENT_LOG.md, docs/VALIDATION.md를 읽어줘.
git status와 실제 line_follower/env.py를 확인하고 문서와 코드의 차이를 보고해줘.

사용자는 Python 초급자이며 수식과 코드의 연결을 작은 단계로 배우고 있어.
현재 규칙 기반 시뮬레이터와 env 래퍼·거리 오차까지만 확인됐고,
heading_error와 Gymnasium·보상·상태 이산화·Q-learning은 제안 단계였어.
이후 커밋에서 구현됐는지 실제 코드를 확인하고, 미구현 내용을 완료로 가정하지 마.

먼저 기존 테스트와 env smoke 확인을 실행하고 현재 상태를 한국어로 설명해줘.
그다음 로드맵의 첫 미완료 작업 하나를 골라 목적·변경·검증 방법을 설명하며 진행해줘.
기존 이동·센서·규칙 제어를 보존하고 관련 없는 리팩터링은 하지 마.
GitHub 생성·원격 변경·push는 실행 전에 승인을 받아줘.
작업 후 컨텍스트·로드맵·개발 로그를 갱신해줘.
```
