# 실험 데이터와 재현 방법

저장소 루트에서 README의 의존성을 설치한 뒤 명령을 실행합니다. 아래 원자료는 표에 쓰인 설정을 다시 실행해 생성했습니다. 각 `config.json`에 소스 커밋, Python·NumPy 버전, 학습 설정 및 평가 트랙 시드가 기록되어 있습니다.

## 할인율과 학습 길이

각 설정은 학습 시드 1~10과 공통 미학습 평가 트랙 30개를 사용합니다. `summary.json`은 시드별 완주율·평균 진행률과 게이트 결과를 담고, `metrics.csv`는 학습 에피소드별 기록을 담습니다.

| 실험 | 설정 | 결과 파일 |
|---|---|---|
| 기준 | γ = 0.95, 500회 | [설정](baseline-g095/config.json) · [요약](baseline-g095/summary.json) · [학습 기록](baseline-g095/metrics.csv) |
| 할인율 상향 | γ = 0.99, 500회 | [설정](gamma099-500/config.json) · [요약](gamma099-500/summary.json) · [학습 기록](gamma099-500/metrics.csv) |
| 할인율 하향 | γ = 0.90, 500회 | [설정](gamma090-500/config.json) · [요약](gamma090-500/summary.json) · [학습 기록](gamma090-500/metrics.csv) |
| 학습 연장 | γ = 0.99, 2,000회 | [설정](gamma099-2000/config.json) · [요약](gamma099-2000/summary.json) · [학습 기록](gamma099-2000/metrics.csv) |
| 학습 연장 | γ = 0.95, 2,000회 | [설정](gamma095-2000/config.json) · [요약](gamma095-2000/summary.json) · [학습 기록](gamma095-2000/metrics.csv) |

아래 명령은 결과를 각 `runs/<실험 이름>/`에 저장합니다. 평균 완주율이 90% 미만이거나 최저 시드의 완주율이 70% 미만이면, 저장을 마친 뒤 `FAIL`과 종료 코드 `1`을 반환합니다. 종료 코드 `2`는 단위 테스트 실패입니다.

```sh
python verify.py --exp baseline-g095 --gamma 0.95 --episodes 500 --seeds 1 2 3 4 5 6 7 8 9 10
python verify.py --exp gamma099-500 --gamma 0.99 --episodes 500 --seeds 1 2 3 4 5 6 7 8 9 10
python verify.py --exp gamma090-500 --gamma 0.90 --episodes 500 --seeds 1 2 3 4 5 6 7 8 9 10
python verify.py --exp gamma099-2000 --gamma 0.99 --episodes 2000 --seeds 1 2 3 4 5 6 7 8 9 10
python verify.py --exp gamma095-2000 --gamma 0.95 --episodes 2000 --seeds 1 2 3 4 5 6 7 8 9 10
```

## Seed 7 체크포인트

```sh
python reproduce_checkpoints.py
```

학습 시드 7로 500회 연속 학습하며 480·490·499·500회 직후의 Q-table을 복사해 평가합니다. 각 체크포인트는 탐험을 끄고 동일한 트랙 30개에서 평가합니다. 평가에서 Q값 동점 때문에 난수가 소비돼도, 복사한 에이전트만 영향을 받으므로 원본 학습은 같은 순서로 이어집니다.

결과는 `runs/seed7-checkpoints/`에 생성됩니다. [체크포인트별 요약 및 Q-table](seed7-checkpoints/summary.json) · [트랙별 평가 120건](seed7-checkpoints/evaluation.csv)

## 탐색 실험 기록

이전 결과표에 포함되었던 두 실행의 원자료입니다. 학습률 하한을 0으로 설정하면 400회까지의 감쇠 속도와 그 이후의 업데이트 여부가 함께 바뀝니다. 두 실행은 학습 길이도 달라서, 각각의 효과를 이 비교로 분리할 수 없습니다.

| 실험 | 설정 | 결과 파일 |
|---|---|---|
| 기존 학습률 하한 | γ = 0.95, 400회, α 하한 0.01 | [설정](alpha001-g095-400/config.json) · [요약](alpha001-g095-400/summary.json) · [학습 기록](alpha001-g095-400/metrics.csv) |
| 학습률을 0까지 감쇠 | γ = 0.95, 500회, α 하한 0 | [설정](alpha000-g095-500/config.json) · [요약](alpha000-g095-500/summary.json) · [학습 기록](alpha000-g095-500/metrics.csv) |

같은 설정을 재현하려면 위 `verify.py` 명령에서 실험 이름과 에피소드 수를 맞추고 `--alpha-end 0.01` 또는 `--alpha-end 0`을 지정합니다.
