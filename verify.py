"""검증 게이트: 에이전트가 아니라 이 스크립트의 종료 코드가 통과/실패를 판정합니다.

1) 단위 테스트 전체
2) 시드 여러 개로 학습 (병렬)
3) 학습에 쓰지 않은 트랙에서 greedy Q 정책과 규칙 기반을 같은 트랙으로 비교
4) runs/<exp>/ 에 config.json, metrics.csv, summary.json 저장

실행:
    python verify.py                    # 기본: 시드 1,2,3 x 500 에피소드, 미학습 30개 (약 2분)
    python verify.py --exp reward-v2    # 실험 이름 지정
    python verify.py --quick            # 시드 1개 x 500 에피소드 (약 1분, 중간 확인용)
종료 코드: 0 통과, 1 게이트 실패, 2 단위 테스트 실패
"""
import argparse, csv, json, os, platform, subprocess, sys, time, unittest
from datetime import datetime
from multiprocessing import Pool

import numpy as np

# 게이트는 평균 완주율과 최저 시드 완주율을 함께 검사한다. 실행 결과는 summary.json에 기록한다.
MIN_MEAN_SUCCESS = 0.90   # 시드 평균 미학습 완주율
MIN_SEED_SUCCESS = 0.70   # 가장 나쁜 시드의 완주율

# 감시하는 줄: '선이 오른쪽에 · 직전 → 오른쪽'(상태 5). 미로 코너(반지름 0.15 m)는 '강하게 오른쪽'(4)으로만 돌 수 있음
WATCH_STATE, WATCH_BEST = 5, 4
ACTION_KO = ['직진', '약하게 왼쪽', '강하게 왼쪽', '약하게 오른쪽', '강하게 오른쪽']


def train_and_eval(args):
    seed, episodes, n_eval, track, gamma, alpha_end = args
    from line_follower.rl_env import RandomTrackEnv
    from line_follower.qlearning import QAgent, run_episode, evaluate, train_seed
    env, agent, rows = RandomTrackEnv(track=track), QAgent(seed=seed, gamma=gamma, alpha_end=alpha_end), []
    wobble = 0   # 마지막 100 에피소드 중 감시 줄의 ★이 '강하게 오른쪽'이 아니었던 횟수
    for ep in range(episodes):
        eps, alpha = agent.epsilon, agent.alpha
        r = run_episode(env, train_seed(seed, ep), agent.act, agent=agent, learn=True)
        if ep >= episodes - 100 and int(np.argmax(agent.q[WATCH_STATE])) != WATCH_BEST:
            wobble += 1
        rows.append({'seed': seed, 'episode': ep + 1, 'track_seed': r['seed'], 'reward': round(r['reward'], 3),
                     'outcome': r['outcome'], 'steps': r['steps'], 'progress': round(r['progress'], 4),
                     'epsilon': round(eps, 4), 'alpha': round(alpha, 4)})
    ev = evaluate(lambda s: agent.act(s, explore=False), n=n_eval, env=env)
    q = agent.q[WATCH_STATE]
    watch = {'wobble_last100': wobble, 'gap_hard_minus_soft': float(q[4] - q[3]),
             'final_best': int(np.argmax(q))}
    return seed, rows, ev['success_rate'], ev['mean_progress'], agent.q.tolist(), watch


def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', default=datetime.now().strftime('verify-%Y%m%d-%H%M%S'))
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--eval', type=int, default=30)
    p.add_argument('--quick', action='store_true', help='시드 1개만')
    p.add_argument('--track', choices=['maze', 'curvy'], default='maze', help='코스 종류')
    p.add_argument('--alpha-end', type=float, default=0.01, help='α 하한 (기본 0.01, 0이면 400ep 이후 Q 고정)')
    p.add_argument('--gamma', type=float, default=0.95, help='할인율 γ (기본 0.95)')
    a = p.parse_args()
    if a.quick:
        a.seeds = a.seeds[:1]

    print('[1/3] 단위 테스트', flush=True)
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover('tests'))
    if not result.wasSuccessful():
        print('FAIL: 단위 테스트 실패 -> 학습 검증 생략')
        sys.exit(2)

    import line_follower.rl_env as R
    from line_follower.qlearning import QAgent, evaluate, rule_policy, EVAL_SEED_BASE
    out = os.path.join('runs', a.exp)
    os.makedirs(out, exist_ok=True)
    ag = QAgent(alpha_end=a.alpha_end)
    config = {
        'exp': a.exp, 'created': datetime.now().isoformat(timespec='seconds'), 'git_commit': git_commit(),
        'track': a.track, 'gamma': a.gamma, 'seeds': a.seeds, 'episodes': a.episodes, 'eval_tracks': a.eval, 'eval_seed_base': EVAL_SEED_BASE,
        'agent': {'alpha_start': ag.alpha_start, 'alpha_end': ag.alpha_end, 'alpha_decay_episodes': ag.alpha_decay,
                  'gamma': a.gamma, 'eps_start': ag.eps_start, 'eps_end': ag.eps_end,
                  'eps_decay_episodes': ag.eps_decay},
        'env': {k: getattr(R, k) for k in dir(R) if k.isupper() and isinstance(getattr(R, k), (int, float, str))},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__},
    }
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f'\n[2/3] 학습: 코스 {a.track}, 시드 {a.seeds} x {a.episodes} 에피소드 (병렬)', flush=True)
    t0 = time.time()
    with Pool(min(len(a.seeds), os.cpu_count() or 1)) as pool:
        results = pool.map(train_and_eval, [(s, a.episodes, a.eval, a.track, a.gamma, a.alpha_end) for s in a.seeds])
    with open(os.path.join(out, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0][1][0].keys())
        w.writeheader()
        for _, rows, *_rest in results:
            w.writerows(rows)
    for seed, *_, q, _w in results:
        np.save(os.path.join(out, f'q_seed{seed}.npy'), np.array(q))

    print(f'\n[3/3] 미학습 트랙 {a.eval}개 비교 ({time.time() - t0:.0f}s 경과)', flush=True)
    from line_follower.rl_env import RandomTrackEnv
    rule = evaluate(rule_policy, n=a.eval, env=RandomTrackEnv(track=a.track))
    per_seed = {s: {'success': sr, 'mean_progress': mp, **w} for s, _, sr, mp, _, w in results}
    successes = [v['success'] for v in per_seed.values()]
    mean_s, min_s = float(np.mean(successes)), float(np.min(successes))
    passed = mean_s >= MIN_MEAN_SUCCESS and min_s >= MIN_SEED_SUCCESS
    summary = {'passed': passed,
               'gate': {'min_mean_success': MIN_MEAN_SUCCESS, 'min_seed_success': MIN_SEED_SUCCESS},
               'q_learning': {'per_seed': per_seed, 'mean_success': mean_s, 'min_success': min_s},
               'rule_based': {'success': rule['success_rate'], 'mean_progress': rule['mean_progress']}}
    summary['wobble_total'] = sum(v['wobble_last100'] for v in per_seed.values())
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    for s, v in per_seed.items():
        print(f"  Q seed {s}: 완주율 {v['success'] * 100:5.1f}%  평균 진행률 {v['mean_progress'] * 100:5.1f}%  "
              f"| 흔들림 {v['wobble_last100']:3d}/100  최종★ {ACTION_KO[v['final_best']]}")
    print(f"  규칙 기반 : 완주율 {rule['success_rate'] * 100:5.1f}%  평균 진행률 {rule['mean_progress'] * 100:5.1f}%")
    total = summary['wobble_total']
    print(f"  흔들림 합계 (감시 줄 '선이 오른쪽에 · 직전 →', 마지막 100회 중 ★≠강하게 오른쪽): {total}")
    print(f"\n{'PASS' if passed else 'FAIL'}: 평균 {mean_s * 100:.1f}% (기준 {MIN_MEAN_SUCCESS * 100:.0f}%), "
          f"최저 {min_s * 100:.1f}% (기준 {MIN_SEED_SUCCESS * 100:.0f}%)  -> {out}")
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
