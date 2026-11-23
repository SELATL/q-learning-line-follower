"""Q-learning 라인트래커 대시보드.

왼쪽: 2D 시뮬레이터 (에피소드마다 새 랜덤 트랙)
오른쪽 위: 실시간 Q-table (12 상태 x 5 행동)
오른쪽 아래: 에피소드 보상 / 성공률 / epsilon 그래프와 수치 지표

실행:
    python dashboard.py                      # 빈 Q-table에서 시작
    python dashboard.py --pretrain 400       # 400 에피소드 미리 학습 후 시작 (면접 시연용)
    python dashboard.py --load q_table.npy --log train_log.csv   # train.py 결과 불러오기 (가장 빠름)
    python dashboard.py --snapshot out.png --pretrain 400   # 화면 없이 PNG 저장

키: t 학습 | e 평가(미학습 트랙, 탐험 끔) | r 규칙 기반 | space 일시정지 | +/- 속도
    b 빠른 학습 50회 | x 미학습 트랙 30개 일괄 비교 | n 다음 에피소드 | s Q-table 저장
"""
import argparse
import numpy as np


SPEEDS = [1, 3, 10, 30, 100, None]   # 프레임당 step 수, None = 에피소드 끝까지 한 번에
MODE_TITLES = {'train': 'TRAIN (epsilon-greedy, learning)',
               'eval': 'EVAL on unseen tracks (greedy, no learning)',
               'rule': 'RULE-BASED on unseen tracks (no learning)'}


class Dashboard:
    def __init__(self, seed=1, pretrain=0, load=None, log=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        from line_follower.rl_env import RandomTrackEnv
        from line_follower.qlearning import QAgent, run_episode, train_seed

        self.plt = plt
        self.seed = seed
        self.env = RandomTrackEnv()
        self.agent = QAgent(seed=seed)
        self.train_hist = []          # 학습 에피소드 기록
        if load:
            self.agent.q = np.load(load)
            if log:   # train.py 로그로 그래프와 에피소드 수(-> epsilon, alpha)를 복원
                import csv
                with open(log) as f:
                    for row in csv.DictReader(f):
                        self.train_hist.append({'seed': int(row['track_seed']), 'reward': float(row['reward']),
                                                'outcome': row['outcome'], 'steps': int(row['steps']),
                                                'progress': float(row['progress']),
                                                'epsilon': float(row['epsilon'])})
                self.agent.episode = len(self.train_hist)
            else:     # 로그가 없으면 학습이 끝난 상태로 간주
                self.agent.episode = max(self.agent.eps_decay, self.agent.alpha_decay)
        self.test = {'eval': [], 'rule': []}   # 미학습 트랙 결과
        self.eval_idx = {'eval': 0, 'rule': 0}
        for ep in range(pretrain):
            if ep % 50 == 0:
                print(f'pretraining {ep}/{pretrain} ...', flush=True)
            eps = self.agent.epsilon
            r = run_episode(self.env, train_seed(seed, self.agent.episode), self.agent.act,
                            agent=self.agent, learn=True)
            r['epsilon'] = eps
            self.train_hist.append(r)

        self.mode = 'train'
        self.speed_i = 2
        self.paused = False
        self.job = None               # 빠른 학습 / 일괄 비교를 한 프레임에 한 에피소드씩 처리
        self.job_text = ''
        self.last_action = None
        self.last_reward = 0.0
        self.last_outcome = '-'

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('Q-Learning Line Follower Dashboard') \
            if hasattr(self.fig.canvas, 'manager') and self.fig.canvas.manager else None
        self._build_sim_axes()
        self._build_qtable_axes()
        self._build_chart_axes()
        self._build_controls(Button)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.start_episode()
        self.redraw_charts()

    # ------------------------------------------------------------ 화면 구성
    def _build_sim_axes(self):
        from line_follower.random_track import RandomTrack
        self.ax = self.fig.add_axes([0.03, 0.24, 0.52, 0.66])
        b = RandomTrack.bounds
        self.ax.set(xlim=(b[0], b[1]), ylim=(b[2], b[3]), aspect='equal')
        self.ax.tick_params(labelsize=7)
        self.ax.set_facecolor('white')
        self.track_patch = None
        self.path, = self.ax.plot([], [], color='tab:blue', lw=1.2, alpha=0.7, zorder=3)
        self.body, = self.ax.plot([], [], color='tab:blue', lw=2.5, zorder=4)
        self.sensor_dots = self.ax.scatter([0, 0], [0, 0], c=[0, 0], s=45, zorder=5, edgecolors='red',
                                           linewidths=1.2, cmap='gray', vmin=0, vmax=100)
        self.start_mark = self.ax.scatter([], [], s=120, marker='o', color='limegreen', zorder=2,
                                          edgecolors='k', label='start')
        self.finish_mark = self.ax.scatter([], [], s=160, marker='s', color='red', zorder=2,
                                           edgecolors='k', label='finish')
        self.ax.legend(loc='upper right', fontsize=8)
        self.sim_text = self.fig.text(0.03, 0.205, '', fontsize=9, family='monospace', va='top')

    def _build_qtable_axes(self):
        from line_follower.rl_env import ACTION_NAMES, N_STATES, state_label
        from matplotlib.patches import Rectangle
        self.qax = self.fig.add_axes([0.63, 0.56, 0.22, 0.38])
        self.qimg = self.qax.imshow(self.agent.q, cmap='viridis', aspect='auto')
        self.qax.set_xticks(range(5), ACTION_NAMES, fontsize=8)
        self.qax.set_yticks(range(N_STATES), [state_label(s) for s in range(N_STATES)],
                            fontsize=8, family='monospace')
        self.qax.xaxis.tick_top()
        self.qax.set_title('Q-table  (row: L/R sensor B/W | previous steer)', fontsize=10, pad=24)
        self.qtexts = [[self.qax.text(a, s, '', ha='center', va='center', fontsize=7.5)
                        for a in range(5)] for s in range(N_STATES)]
        for y in (2.5, 5.5, 8.5):
            self.qax.axhline(y, color='white', lw=1.5)
        self.row_box = Rectangle((-0.5, -0.5), 5, 1, fill=False, edgecolor='red', lw=2)
        self.cell_box = Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor='orange', lw=3)
        self.qax.add_patch(self.row_box)
        self.qax.add_patch(self.cell_box)
        cax = self.fig.add_axes([0.855, 0.56, 0.008, 0.38])
        self.qcbar = self.fig.colorbar(self.qimg, cax=cax)
        self.qcbar.ax.tick_params(labelsize=7)
        self.metrics = self.fig.text(0.885, 0.94, '', fontsize=8.5, family='monospace', va='top')

    def _build_chart_axes(self):
        self.rax = self.fig.add_axes([0.60, 0.32, 0.38, 0.17])
        self.rax.set_ylabel('episode reward', fontsize=8)
        self.rax.tick_params(labelsize=7)
        self.r_dots, = self.rax.plot([], [], '.', color='0.6', ms=3, label='episode')
        self.r_avg, = self.rax.plot([], [], color='tab:blue', lw=1.5, label='moving avg (20)')
        self.rax.legend(fontsize=7, loc='upper left')
        self.sax = self.fig.add_axes([0.60, 0.10, 0.38, 0.17])
        self.sax.set(ylim=(-0.05, 1.05), xlabel='training episode')
        self.sax.set_ylabel('success rate', fontsize=8)
        self.sax.tick_params(labelsize=7)
        self.s_line, = self.sax.plot([], [], color='tab:green', lw=1.5, label='train success (last 20)')
        self.s_test = self.sax.axhline(np.nan, color='tab:purple', ls='--', lw=1.2, label='Q unseen-test')
        self.s_rule = self.sax.axhline(np.nan, color='tab:orange', ls=':', lw=1.5, label='rule unseen-test')
        self.eax = self.sax.twinx()
        self.eax.set_ylim(-0.05, 1.05)
        self.eax.set_ylabel('epsilon', fontsize=8, color='tab:red')
        self.eax.tick_params(labelsize=7, colors='tab:red')
        self.e_line, = self.eax.plot([], [], color='tab:red', lw=1, alpha=0.7, label='epsilon')
        lines = [self.s_line, self.s_test, self.s_rule, self.e_line]
        self.sax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc='center right')

    def _build_controls(self, Button):
        specs = [('Train [t]', lambda e: self.set_mode('train')),
                 ('Eval [e]', lambda e: self.set_mode('eval')),
                 ('Rule [r]', lambda e: self.set_mode('rule')),
                 ('Pause [space]', lambda e: self.toggle_pause()),
                 ('Speed - [-]', lambda e: self.change_speed(-1)),
                 ('Speed + [+]', lambda e: self.change_speed(+1)),
                 ('Fast train 50 [b]', lambda e: self.start_fast_train(50)),
                 ('Test 30 unseen [x]', lambda e: self.start_batch_test(30)),
                 ('Reset Q', lambda e: self.reset_q())]
        self.buttons = []
        w, gap, x0 = 0.098, 0.006, 0.03
        for i, (label, cb) in enumerate(specs):
            btn = Button(self.fig.add_axes([x0 + i * (w + gap), 0.025, w, 0.045]), label)
            btn.label.set_fontsize(8)
            btn.on_clicked(cb)
            self.buttons.append(btn)

    # ------------------------------------------------------------ 에피소드 진행
    def current_seed(self):
        from line_follower.qlearning import train_seed, EVAL_SEED_BASE
        if self.mode == 'train':
            return train_seed(self.seed, self.agent.episode)
        return EVAL_SEED_BASE + self.eval_idx[self.mode]

    def start_episode(self):
        self.ep_seed = self.current_seed()
        self.state = self.env.reset(self.ep_seed)
        self.ep_reward = 0.0
        self.ep_eps = self.agent.epsilon
        self.last_action = None
        self.draw_track()

    def policy(self, s):
        from line_follower.qlearning import rule_policy
        if self.mode == 'train':
            return self.agent.act(s, explore=True)
        if self.mode == 'eval':
            return self.agent.act(s, explore=False)
        return rule_policy(s)

    def sim_step(self):
        """한 step 진행. 에피소드가 끝나면 True."""
        s = self.state
        a = self.policy(s)
        s2, r, term, trunc, info = self.env.step(a)
        if self.mode == 'train':
            self.agent.update(s, a, r, s2, term)
        self.state, self.last_action, self.last_reward = s2, a, r
        self.ep_reward += r
        if term or trunc:
            self.finish_episode(info.outcome)
            return True
        return False

    def finish_episode(self, outcome):
        rec = {'seed': self.ep_seed, 'reward': self.ep_reward, 'outcome': outcome,
               'steps': self.env.steps, 'progress': self.env.s / self.env.track.length}
        self.last_outcome = outcome
        if self.mode == 'train':
            rec['epsilon'] = self.ep_eps
            self.agent.episode += 1
            self.train_hist.append(rec)
        else:
            self.test[self.mode].append(rec)
            self.eval_idx[self.mode] += 1
        self.redraw_charts()

    def update(self, frame=None):
        if self.job is not None:
            try:
                next(self.job)
            except StopIteration:
                self.job = None
                self.start_episode()
        elif not self.paused:
            n = SPEEDS[self.speed_i]
            done = False
            if n is None:
                while not done:
                    done = self.sim_step()
            else:
                for _ in range(n):
                    if self.sim_step():
                        done = True
                        break
            if done:
                self.final_traj = np.array(self.env.trajectory)
                self.start_episode()
        self.draw_state()

    # ------------------------------------------------------------ 버튼 동작
    def set_mode(self, mode):
        if mode != self.mode or self.job is not None:
            self.job = None
            self.mode = mode
            self.start_episode()

    def toggle_pause(self):
        self.paused = not self.paused

    def change_speed(self, d):
        self.speed_i = int(np.clip(self.speed_i + d, 0, len(SPEEDS) - 1))

    def reset_q(self):
        from line_follower.qlearning import QAgent
        self.job = None
        self.agent = QAgent(seed=self.seed)
        self.train_hist.clear()
        self.test = {'eval': [], 'rule': []}
        self.eval_idx = {'eval': 0, 'rule': 0}
        self.mode = 'train'
        self.start_episode()
        self.redraw_charts()

    def start_fast_train(self, n):
        self.mode = 'train'
        def job():
            for i in range(n):
                self.job_text = f'fast training {i + 1}/{n} ...'
                self.start_episode()
                while not self.sim_step():
                    pass
                yield
            self.job_text = ''
        self.job = job()

    def start_batch_test(self, n):
        """Q(greedy)와 규칙 기반을 같은 미학습 트랙 n개에서 비교합니다."""
        self.test = {'eval': [], 'rule': []}
        self.eval_idx = {'eval': 0, 'rule': 0}
        def job():
            for mode in ('eval', 'rule'):
                self.mode = mode
                for i in range(n):
                    self.job_text = f'testing {mode} {i + 1}/{n} ...'
                    self.start_episode()
                    while not self.sim_step():
                        pass
                    yield
            self.mode = 'train'
            self.job_text = ''
        self.job = job()

    def on_key(self, event):
        k = event.key
        if k == 't': self.set_mode('train')
        elif k == 'e': self.set_mode('eval')
        elif k == 'r': self.set_mode('rule')
        elif k == ' ': self.toggle_pause()
        elif k in ('+', '='): self.change_speed(+1)
        elif k == '-': self.change_speed(-1)
        elif k == 'b': self.start_fast_train(50)
        elif k == 'x': self.start_batch_test(30)
        elif k == 'n': self.start_episode()
        elif k == 's':
            np.save('q_table.npy', self.agent.q)
            self.job_text = 'saved q_table.npy'

    # ------------------------------------------------------------ 그리기
    def draw_track(self):
        from matplotlib.patches import Polygon
        tr = self.env.track
        if self.track_patch is not None:
            self.track_patch.remove()
            self.edge_patch.remove()
        edge = Polygon(tr.outline(tr.line_width / 2 + tr.edge_width), closed=True,
                       facecolor='0.6', edgecolor='none', zorder=1)
        core = Polygon(tr.outline(), closed=True, facecolor='black', edgecolor='none', zorder=1.5)
        self.edge_patch = self.ax.add_patch(edge)
        self.track_patch = self.ax.add_patch(core)
        self.start_mark.set_offsets([tr.points[0]])
        self.finish_mark.set_offsets([tr.points[-1]])

    def draw_state(self):
        from line_follower.rl_env import ACTION_NAMES, state_label
        r = self.env.robot
        local = np.array([[0.05, 0], [-0.035, 0.03], [-0.035, -0.03], [0.05, 0]])
        c, s = np.cos(r.theta), np.sin(r.theta)
        pts = local @ np.array([[c, s], [-s, c]]) + [r.x, r.y]
        self.body.set_data(pts[:, 0], pts[:, 1])
        traj = np.array(self.env.trajectory)
        self.path.set_data(traj[:, 0], traj[:, 1])
        self.sensor_dots.set_offsets(self.env.sensors.positions(r))
        self.sensor_dots.set_array(self.env.values)
        speed = SPEEDS[self.speed_i]
        act = ACTION_NAMES[self.last_action] if self.last_action is not None else '-'
        self.ax.set_title(MODE_TITLES[self.mode] + (f'   [{self.job_text}]' if self.job_text else '')
                          + ('   PAUSED' if self.paused else ''), fontsize=11)
        self.sim_text.set_text(
            f"track seed {self.ep_seed}  length {self.env.track.length:.2f} m\n"
            f"step {self.env.steps:4d}/{self.env.max_steps}  progress {self.env.s / self.env.track.length * 100:5.1f}%\n"
            f"sensors L {self.env.values[0]:5.1f}  R {self.env.values[1]:5.1f}   state {state_label(self.state)}\n"
            f"action {act:8s}  reward {self.last_reward:+6.2f}  episode reward {self.ep_reward:8.1f}\n"
            f"dist to line {self.env.dist * 1000:5.1f} mm   speed {'episode' if speed is None else f'{speed} step/frame'}")
        self.draw_qtable()
        self.draw_metrics()
        self.fig.canvas.draw_idle()

    def draw_qtable(self):
        q = self.agent.q
        vmin, vmax = float(q.min()), float(q.max())
        if vmax - vmin < 1e-9:
            vmin, vmax = vmin - 1, vmax + 1
        self.qimg.set_data(q)
        self.qimg.set_clim(vmin, vmax)
        best = np.argmax(q, axis=1)
        mid = (vmin + vmax) / 2
        for s in range(q.shape[0]):
            for a in range(q.shape[1]):
                t = self.qtexts[s][a]
                t.set_text(f'{q[s, a]:.1f}')
                t.set_color('black' if q[s, a] > mid else 'white')
                t.set_fontweight('bold' if a == best[s] and q[s].max() > q[s].min() else 'normal')
        self.row_box.set_y(self.state - 0.5)
        if self.last_action is not None:
            self.cell_box.set_visible(True)
            self.cell_box.set_xy((self.last_action - 0.5, self.state - 0.5))
        else:
            self.cell_box.set_visible(False)

    def draw_metrics(self):
        h = self.train_hist
        last = h[-50:]
        tr_succ = np.mean([x['outcome'] == 'finish' for x in last]) if last else np.nan
        tr_rew = np.mean([x['reward'] for x in last]) if last else np.nan

        def summary(rs):
            if not rs:
                return '   -'
            succ = np.mean([x['outcome'] == 'finish' for x in rs])
            prog = np.mean([x['progress'] for x in rs])
            return f"{succ * 100:5.1f}% (n={len(rs)})\n  avg progress {prog * 100:5.1f}%"

        self.metrics.set_text(
            f"MODE  {self.mode.upper()}\n"
            f"train episodes {self.agent.episode}\n"
            f"epsilon {self.agent.epsilon:.3f}\n"
            f"alpha   {self.agent.alpha:.3f}\n"
            f"gamma   {self.agent.gamma:.2f}\n\n"
            f"-- train (last 50)\n"
            f"success {tr_succ * 100:5.1f}%\n"
            f"reward  {tr_rew:7.1f}\n\n"
            f"-- unseen tracks\n"
            f"Q greedy\n  {summary(self.test['eval'])}\n"
            f"Rule-based\n  {summary(self.test['rule'])}\n\n"
            f"last outcome: {self.last_outcome}")

    def redraw_charts(self):
        h = self.train_hist
        if h:
            x = np.arange(1, len(h) + 1)
            rew = np.array([e['reward'] for e in h])
            succ = np.array([e['outcome'] == 'finish' for e in h], dtype=float)
            k = 20
            avg = np.convolve(rew, np.ones(k), 'full')[:len(h)] / np.minimum(np.arange(1, len(h) + 1), k)
            sr = np.convolve(succ, np.ones(k), 'full')[:len(h)] / np.minimum(np.arange(1, len(h) + 1), k)
            self.r_dots.set_data(x, rew)
            self.r_avg.set_data(x, avg)
            self.s_line.set_data(x, sr)
            self.e_line.set_data(x, [e['epsilon'] for e in h])
            for ax in (self.rax, self.sax):
                ax.set_xlim(0, max(len(h), 10) + 1)
            self.rax.set_ylim(min(rew.min(), 0) * 1.1 - 5, rew.max() * 1.1 + 5)
        else:
            for line in (self.r_dots, self.r_avg, self.s_line, self.e_line):
                line.set_data([], [])
        for key, line in (('eval', self.s_test), ('rule', self.s_rule)):
            rs = self.test[key]
            y = np.mean([x['outcome'] == 'finish' for x in rs]) if rs else np.nan
            line.set_ydata([y, y])

    def show(self):
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self.update, interval=30, cache_frame_data=False)
        self.plt.show()


def main():
    p = argparse.ArgumentParser(description='Q-learning line follower dashboard')
    p.add_argument('--seed', type=int, default=1, help='재현용 시드 (학습 트랙·탐험 난수)')
    p.add_argument('--pretrain', type=int, default=0, help='시작 전 화면 없이 학습할 에피소드 수')
    p.add_argument('--load', help='train.py가 저장한 Q-table (.npy)')
    p.add_argument('--log', help='train.py가 저장한 학습 로그 (.csv), --load와 함께 사용')
    p.add_argument('--snapshot', help='창 대신 PNG 저장')
    a = p.parse_args()
    if a.snapshot:
        import matplotlib
        matplotlib.use('Agg')
    d = Dashboard(seed=a.seed, pretrain=a.pretrain, load=a.load, log=a.log)
    if a.snapshot:
        d.start_batch_test(10)
        while d.job is not None:
            d.update()
        d.set_mode('eval')
        for _ in range(25):
            d.update()
        d.fig.savefig(a.snapshot, dpi=110)
        print('saved', a.snapshot)
    else:
        d.show()


if __name__ == '__main__':
    main()
