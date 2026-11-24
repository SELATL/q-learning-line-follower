"""Evaluate saved Q-tables along a single, uninterrupted training run."""

import argparse
import csv
import json
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np

from line_follower.qlearning import EVAL_SEED_BASE, QAgent, evaluate, run_episode, train_seed
from line_follower.rl_env import RandomTrackEnv
from verify import ACTION_KO, WATCH_STATE, git_commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[480, 490, 499, 500])
    parser.add_argument('--out', type=Path, default=Path('runs/seed7-checkpoints'))
    args = parser.parse_args()
    checkpoints = sorted(set(args.checkpoints))
    if args.seed < 0 or checkpoints[0] < 1:
        parser.error('--seed must be nonnegative and --checkpoints must be positive')
    if train_seed(args.seed, checkpoints[-1] - 1) >= EVAL_SEED_BASE:
        parser.error('training track seeds must stay below the evaluation seed range')

    commit = git_commit()
    args.out.mkdir(parents=True, exist_ok=True)
    agent = QAgent(seed=args.seed)
    train_env = RandomTrackEnv(track='maze')
    eval_env = RandomTrackEnv(track='maze')
    summary = {
        'git_commit': commit,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__},
        'track': 'maze',
        'seed': args.seed,
        'agent': {
            'gamma': agent.gamma,
            'alpha_start': agent.alpha_start,
            'alpha_end': agent.alpha_end,
            'alpha_decay_episodes': agent.alpha_decay,
            'eps_start': agent.eps_start,
            'eps_end': agent.eps_end,
            'eps_decay_episodes': agent.eps_decay,
        },
        'eval_tracks': 30,
        'eval_seed_base': EVAL_SEED_BASE,
        'watch_state': WATCH_STATE,
        'action_labels': ACTION_KO,
        'checkpoints': [],
    }
    rows = []

    for episode in range(checkpoints[-1]):
        run_episode(train_env, train_seed(args.seed, episode), agent.act, agent=agent, learn=True)
        if agent.episode not in checkpoints:
            continue

        # Greedy tie-breaking can use the RNG. Evaluate a copy so later training
        # sees exactly the same RNG state as a run without checkpoint evaluation.
        snapshot = deepcopy(agent)
        evaluation = evaluate(
            lambda state: snapshot.act(state, explore=False),
            n=summary['eval_tracks'],
            env=eval_env,
        )
        trials = evaluation['results']
        finishes = sum(trial['outcome'] == 'finish' for trial in trials)
        best = int(np.argmax(snapshot.q[WATCH_STATE]))
        summary['checkpoints'].append({
            'episode': agent.episode,
            'finishes': finishes,
            'success_rate': evaluation['success_rate'],
            'mean_progress': evaluation['mean_progress'],
            'watch_best_action': best,
            'q_table': snapshot.q.tolist(),
        })
        rows.extend({'checkpoint': agent.episode, **trial} for trial in trials)
        print(f'{agent.episode}: {finishes}/{len(trials)} 완주, 상태 {WATCH_STATE}의 최대 Q 행동은 {ACTION_KO[best]}')

    with (args.out / 'summary.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write('\n')
    with (args.out / 'evaluation.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
