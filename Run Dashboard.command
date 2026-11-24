#!/bin/bash
# 더블클릭하면 Q-learning 대시보드를 실행합니다.
# 빈 Q-table에서 시작해 학습 모드(TRAIN)로 바로 학습이 진행됩니다.
cd "$(dirname "$0")" || exit 1

if [ ! -x ".venv/bin/python" ]; then
  echo "가상환경(.venv)이 없어 새로 만듭니다..."
  python3 -m venv .venv || { echo "python3를 찾을 수 없습니다."; read -r -p "Enter를 누르면 닫힙니다."; exit 1; }
  .venv/bin/python -m pip install -q -r requirements.txt || { echo "패키지 설치 실패"; read -r -p "Enter를 누르면 닫힙니다."; exit 1; }
fi

echo "대시보드 실행 중... (창을 닫으면 종료)"
echo "키: t 학습 | e 평가 | r 규칙 | space 정지 | +/- 속도 | b 빠른 학습 50회 | x 미학습 30개 비교 | s Q-table 저장"
.venv/bin/python dashboard.py "$@"
status=$?
if [ $status -ne 0 ]; then
  read -r -p "오류로 종료됐습니다 (코드 $status). Enter를 누르면 닫힙니다."
fi
