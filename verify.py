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

# 게이트 기준 (2026-11-23 실험: 시드 1~7 중 6개 100%, 1개 80% 근거)
MIN_MEAN_SUCCESS = 0.90   # 시드 평균 미학습 완주율
MIN_SEED_SUCCESS = 0.70   # 가장 나쁜 시드의 완주율


def train_and_eval(args):
    seed, episodes, n_eval = args
    from line_follower.rl_env import RandomTrackEnv
    from line_follower.qlearning import QAgent, run_episode, evaluate, train_seed
    env, agent, rows = RandomTrackEnv(), QAgent(seed=seed), []
    for ep in range(episodes):
        eps, alpha = agent.epsilon, agent.alpha
        r = run_episode(env, train_seed(seed, ep), agent.act, agent=agent, learn=True)
        rows.append({'seed': seed, 'episode': ep + 1, 'track_seed': r['seed'], 'reward': round(r['reward'], 3),
                     'outcome': r['outcome'], 'steps': r['steps'], 'progress': round(r['progress'], 4),
                     'epsilon': round(eps, 4), 'alpha': round(alpha, 4)})
    ev = evaluate(lambda s: agent.act(s, explore=False), n=n_eval, env=env)
    return seed, rows, ev['success_rate'], ev['mean_progress'], agent.q.tolist()


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
    ag = QAgent()
    config = {
        'exp': a.exp, 'created': datetime.now().isoformat(timespec='seconds'), 'git_commit': git_commit(),
        'seeds': a.seeds, 'episodes': a.episodes, 'eval_tracks': a.eval, 'eval_seed_base': EVAL_SEED_BASE,
        'agent': {'alpha_start': ag.alpha_start, 'alpha_end': ag.alpha_end, 'alpha_decay_episodes': ag.alpha_decay,
                  'gamma': ag.gamma, 'eps_start': ag.eps_start, 'eps_end': ag.eps_end,
                  'eps_decay_episodes': ag.eps_decay},
        'env': {k: getattr(R, k) for k in dir(R) if k.isupper() and isinstance(getattr(R, k), (int, float, str))},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__},
    }
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f'\n[2/3] 학습: 시드 {a.seeds} x {a.episodes} 에피소드 (병렬)', flush=True)
    t0 = time.time()
    with Pool(min(len(a.seeds), os.cpu_count() or 1)) as pool:
        results = pool.map(train_and_eval, [(s, a.episodes, a.eval) for s in a.seeds])
    with open(os.path.join(out, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0][1][0].keys())
        w.writeheader()
        for _, rows, *_ in results:
            w.writerows(rows)
    for seed, *_, q in results:
        np.save(os.path.join(out, f'q_seed{seed}.npy'), np.array(q))

    print(f'\n[3/3] 미학습 트랙 {a.eval}개 비교 ({time.time() - t0:.0f}s 경과)', flush=True)
    rule = evaluate(rule_policy, n=a.eval)
    per_seed = {s: {'success': sr, 'mean_progress': mp} for s, _, sr, mp, _ in results}
    successes = [v['success'] for v in per_seed.values()]
    mean_s, min_s = float(np.mean(successes)), float(np.min(successes))
    passed = mean_s >= MIN_MEAN_SUCCESS and min_s >= MIN_SEED_SUCCESS
    summary = {'passed': passed,
               'gate': {'min_mean_success': MIN_MEAN_SUCCESS, 'min_seed_success': MIN_SEED_SUCCESS},
               'q_learning': {'per_seed': per_seed, 'mean_success': mean_s, 'min_success': min_s},
               'rule_based': {'success': rule['success_rate'], 'mean_progress': rule['mean_progress']}}
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    for s, v in per_seed.items():
        print(f"  Q seed {s}: 완주율 {v['success'] * 100:5.1f}%  평균 진행률 {v['mean_progress'] * 100:5.1f}%")
    print(f"  규칙 기반 : 완주율 {rule['success_rate'] * 100:5.1f}%  평균 진행률 {rule['mean_progress'] * 100:5.1f}%")
    print(f"\n{'PASS' if passed else 'FAIL'}: 평균 {mean_s * 100:.1f}% (기준 {MIN_MEAN_SUCCESS * 100:.0f}%), "
          f"최저 {min_s * 100:.1f}% (기준 {MIN_SEED_SUCCESS * 100:.0f}%)  -> {out}")
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
