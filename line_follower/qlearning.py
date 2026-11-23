"""Tabular Q-learning 에이전트, 규칙 기반 비교 정책, 에피소드 실행 함수."""
import numpy as np

from .rl_env import RandomTrackEnv, RLAction, N_STATES, N_ACTIONS

TRAIN_SEED_STRIDE = 100_000    # 학습 트랙 seed = agent_seed * stride + episode
EVAL_SEED_BASE = 10_000_000    # 평가(미학습) 트랙: 학습 시드와 겹치지 않음


class QAgent:
    def __init__(self, alpha=0.1, alpha_end=0.01, alpha_decay_episodes=400, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=300, seed=0):
        self.q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha_start, self.alpha_end, self.alpha_decay = alpha, alpha_end, alpha_decay_episodes
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_episodes
        self.rng = np.random.default_rng(seed)
        self.episode = 0

    @property
    def epsilon(self):
        frac = min(self.episode / max(self.eps_decay, 1), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    @property
    def alpha(self):
        """학습률도 선형 감쇠: 상태가 흑백으로 뭉개져 있어(aliasing) 고정 학습률이면 greedy 정책이 흔들립니다."""
        frac = min(self.episode / max(self.alpha_decay, 1), 1.0)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * frac

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def update(self, s, a, r, s2, terminated):
        """Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a)).
        실패/완주로 끝나면 미래 가치 0, 시간 제한(truncated)은 끝이 아니므로 bootstrap 유지."""
        target = r if terminated else r + self.gamma * self.q[s2].max()
        td = target - self.q[s, a]
        self.q[s, a] += self.alpha * td
        return td


def rule_policy(state):
    """학습 없는 비교 기준: 검은 쪽으로 강하게 꺾고, 둘 다 흰색이면 직전 방향으로 계속 찾습니다."""
    lr, prev = divmod(state, 3)
    lb, rb = divmod(lr, 2)
    if lb and not rb:
        return int(RLAction.HARD_LEFT)
    if rb and not lb:
        return int(RLAction.HARD_RIGHT)
    if lb and rb:
        return int(RLAction.STRAIGHT)
    return {0: int(RLAction.HARD_LEFT), 1: int(RLAction.STRAIGHT), 2: int(RLAction.HARD_RIGHT)}[prev]


def train_seed(agent_seed, episode):
    return agent_seed * TRAIN_SEED_STRIDE + episode


def run_episode(env, seed, policy, agent=None, learn=False):
    """한 에피소드 전체 실행. policy(state)->action. learn=True면 agent를 업데이트합니다."""
    s = env.reset(seed)
    total, outcome = 0.0, 'running'
    while True:
        a = policy(s)
        s2, r, term, trunc, info = env.step(a)
        if learn:
            agent.update(s, a, r, s2, term)
        total += r
        s = s2
        if term or trunc:
            outcome = info.outcome
            break
    if learn:
        agent.episode += 1
    return {'seed': seed, 'reward': total, 'outcome': outcome, 'steps': env.steps,
            'progress': env.s / env.track.length}


def evaluate(policy, n=30, seed_base=EVAL_SEED_BASE, env=None):
    env = env or RandomTrackEnv()
    results = [run_episode(env, seed_base + k, policy) for k in range(n)]
    success = np.mean([r['outcome'] == 'finish' for r in results])
    progress = np.mean([r['progress'] for r in results])
    return {'success_rate': float(success), 'mean_progress': float(progress), 'results': results}
