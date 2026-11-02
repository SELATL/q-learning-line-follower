# 이전 준비 및 검증 기록

검증일: 2026-11-03 (Asia/Seoul).

## 보존 범위

실제 IDE에서 개발하던 `outputs/line_follower_sim`의 실행 코드·테스트·requirements·preview를 새 프로젝트 루트에 복사했다. 기존 소스 파일 9개, requirements.txt, preview.png 총 11개를 SHA-256 비교해 모두 원본과 동일함을 확인했다. README는 기존 본문을 유지하고 이전 안내만 앞에 추가했다. 원본의 개인 절대경로는 문서에 복제하지 않았다.

기존 중복 하위 `line_follower_sim/`, 설치 패키지가 있는 `work/`, `__pycache__`, ZIP은 업로드 사본에 포함하지 않았다. 원본 삭제·기능 수정은 하지 않았다.

## Git 및 GitHub

- 현재 채팅 작업 루트와 실제 개발 프로젝트의 확인한 경로에서 Git 저장소가 발견되지 않았다. 보존할 기존 커밋·브랜치·remote는 확인되지 않았다.
- 이전용 사본에서 `git init -b main` 실행. 최초 커밋 `240885a` 생성 완료. remote 없음. 이후 문서 상태 갱신은 후속 커밋으로 보존한다.
- 연결된 GitHub 로그인은 `SELATL`. 앱에 접근 가능한 저장소 목록에는 기존의 다른 프로젝트 하나만 있었다. 후보 `SELATL/line-follower-sim` 조회는 404였다. 권한 밖 저장소의 부재까지 증명하지는 않는다.
- 후보 URL: `https://github.com/SELATL/line-follower-sim` (Private 예정). 생성·remote 변경·push는 사용자 승인 완료. 브라우저 GitHub 로그인 필요로 아직 실행하지 않았다.
- 현재 GitHub connector에는 저장소 생성 도구가 없고 로컬 `gh`도 찾지 못했다. 승인 후 GitHub UI 등 사용 가능한 방법으로 생성하거나 사용자가 제공한 기존 저장소를 사용한다. 토큰을 파일에 저장하지 않는다.

## 테스트

실행 환경: Windows, Python 3.12.10, NumPy 2.5.2, Matplotlib 3.11.2. 이 버전은 검증 환경 기록이며 requirements의 고정 버전이 아니다.

| 검사 | 결과 |
|---|---|
| `python -m unittest discover -s tests -v` | 기존 9개 모두 통과 |
| env reset/직진 smoke | 초기 `[14.91503095,66.8248934]`, 직진 후 `[7.3180869,73.97068683]` |
| 거리 오차 smoke | 초기 0, 직진 후 `2.7691717822131423e-05`m |
| 방향 오차 존재 확인 | `hasattr(e,'heading_error')`는 False. 미구현 확인 |
| `python main.py --snapshot snapshot-validation.png` | 정상 종료, PNG 생성 및 시각 확인. 결과 이미지는 .gitignore로 제외 |

기존 테스트는 반사광·직진·회전 대칭·센서 좌표·초기화·잘못된 입력·규칙 제어·100초 선 근처 유지 확인이다. 100초 검사는 오차 4cm 미만을 기준으로 하며 완주 판정이나 RL 학습 검증이 아니다. env.py의 smoke는 이번 단발 확인이고 자동 테스트로 추가하지 않았다.

## 보안 및 불필요한 파일

- 업로드 사본의 텍스트에서 private key 헤더, 알려진 token 형식, 따옴표로 지정된 secret/password 값, 개인 Windows 절대경로, 이메일 패턴을 검사했다. 실제 비밀 값이나 개인정보 패턴 발견 없음.
- `.env`, 인증 파일, venv, 캐시, 빌드·실험 출력·ZIP은 .gitignore로 제외한다. 토큰 값이나 인증 설정을 출력·복사하지 않았다.
- 원본 preview.png는 프로그램 화면 이미지이며 인증·개인정보 텍스트가 없는지 시각 확인했다. 원본 정적 예제로 보존한다.
- 1MB 초과 업로드 파일 없음. 앱 전체 설정·대화 원문·설치 라이브러리를 업로드하지 않는다.
- 패턴 검사는 비밀 부재의 완전한 증명이 아니다. push 직전에 실제 stage 목록·diff와 보안 검사를 다시 확인한다.

## 검증 한계 및 다음 확인

새 가상환경에서의 의존성 다운로드·설치는 수행하지 않았다. 이미 설치된 환경에서 검증했다. Mac, Linux/Cloud, 실제 Matplotlib 창의 버튼·키보드, GitHub clone/push는 이번에 검증하지 않았다. 소스의 OS 전용 절대경로나 셸 의존은 발견되지 않았지만 각 OS에서 smoke 검증이 필요하다. 환경변수·서버는 필요 없다.

후속 push 시: 중복 저장소 재확인 → 사용자 승인 → Private 설정 → author 정보 검토 → 명시적 파일 stage 및 diff 확인 → 최초 커밋 → remote 설정 → push → Mac/Cloud clone 검증. 기존 저장소를 발견하면 이력과 브랜치를 보존하며 병합 계획을 먼저 검토한다.

최초 stage 점검: 21개 파일을 명시적으로 추가했으며 캐시·환경 파일·snapshot은 제외됨. `git diff --cached --check`는 원본 env.py의 기존 줄 끝 공백 10곳을 보고했다. 소스 보존 원칙에 따라 수정하지 않았다. 실행 오류는 아니다.
