"""화면 없이 학습하고 미학습 트랙에서 Q-learning과 규칙 기반을 비교합니다.

실행 예: python train.py --episodes 500 --seed 1 --eval 30
"""
import argparse, csv, time
from collections import Counter
import numpy as np

from line_follower.rl_env import RandomTrackEnv, ACTION_NAMES, state_label
from line_follower.qlearning import QAgent, run_episode, evaluate, rule_policy, train_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--eval', type=int, default=30, help='미학습 평가 트랙 수')
    p.add_argument('--save', default='q_table.npy')
    p.add_argument('--log', default='train_log.csv')
    a = p.parse_args()

    env = RandomTrackEnv()
    agent = QAgent(seed=a.seed)
    rows, t0 = [], time.time()
    for ep in range(a.episodes):
        eps, alpha = agent.epsilon, agent.alpha
        r = run_episode(env, train_seed(a.seed, ep), agent.act, agent=agent, learn=True)
        rows.append({'episode': ep + 1, 'track_seed': r['seed'], 'reward': round(r['reward'], 3),
                     'outcome': r['outcome'], 'steps': r['steps'], 'progress': round(r['progress'], 4),
                     'epsilon': round(eps, 4), 'alpha': round(alpha, 4)})
        if (ep + 1) % 50 == 0:
            last = rows[-50:]
            print(f"ep {ep+1:4d}  success(last50)={np.mean([x['outcome']=='finish' for x in last]):.2f}  "
                  f"reward(last50)={np.mean([x['reward'] for x in last]):7.1f}  eps={eps:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.save(a.save, agent.q)
    with open(a.log, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    q_eval = evaluate(lambda s: agent.act(s, explore=False), n=a.eval, env=env)
    r_eval = evaluate(rule_policy, n=a.eval, env=env)
    print(f"\n[미학습 트랙 {a.eval}개, 탐험 끔]")
    for name, ev in [('Q-learning', q_eval), ('Rule-based', r_eval)]:
        c = Counter(x['outcome'] for x in ev['results'])
        steps = np.mean([x['steps'] for x in ev['results'] if x['outcome'] == 'finish'] or [np.nan])
        print(f"  {name:10s} 완주율 {ev['success_rate']*100:5.1f}%  평균 진행률 {ev['mean_progress']*100:5.1f}%  "
              f"완주 시 평균 step {steps:6.1f}  {dict(c)}")
    print("\n[학습된 greedy 정책]")
    for s in range(agent.q.shape[0]):
        print(f"  {state_label(s):14s} -> {ACTION_NAMES[int(np.argmax(agent.q[s]))]:8s}  Q={np.round(agent.q[s], 1)}")
    print(f"\n저장: {a.save}, {a.log}")


if __name__ == '__main__':
    main()
