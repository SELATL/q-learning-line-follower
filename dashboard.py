"""Q-learning 라인트래커 대시보드 (미로 코스).

왼쪽: 2D 시뮬레이터 + 센서 LED
오른쪽 위: Q-table (상태 해석 라벨, 최선 행동, 방문 횟수)
오른쪽 아래: 학습 진행률·학습 방향 원형 그래프, 보상/완주율 그래프, 미학습 평가

실행:
    python dashboard.py                      # 빈 Q-table에서 학습 시작 (기본: 미로 코스)
    python dashboard.py --track curvy        # 부드러운 곡선 코스
    python dashboard.py --target 800         # 진행률 원형 그래프의 목표 에피소드
    python dashboard.py --load q_table.npy --log train_log.csv   # train.py 결과 불러오기
    python dashboard.py --snapshot out.png --pretrain 100        # 화면 없이 PNG 저장

키: t 학습 | e 평가 | r 규칙 | space 일시정지 | . 한 스텝 | n 다음 트랙 | +/- 속도
    b 빠른 학습 50회 | x 미학습 30개 비교 | s Q-table 저장
"""
import argparse
import numpy as np

SPEEDS = [1, 3, 10, 30, 100, None]   # 프레임당 step 수, None = 에피소드 끝까지 한 번에

# ---------------------------------------------------------------- 색
BG, PANEL, PANEL2 = '#0b1220', '#121b2f', '#1a2540'
TEXT, MUTED, LINE = '#e5e7eb', '#94a3b8', '#2a3757'
BLUE, GREEN, RED, YELLOW, ORANGE, GRAY = '#38bdf8', '#34d399', '#f87171', '#fbbf24', '#fb923c', '#64748b'

ACTION_KO = ['직진', '약하게 왼쪽', '강하게 왼쪽', '약하게 오른쪽', '강하게 오른쪽']
ACTION_ARROW = ['↑', '↖', '←', '↗', '→']
PREV_KO = ['← 왼쪽', '↑ 직진', '→ 오른쪽']
SENSOR_GROUPS = {   # (왼쪽 검정?, 오른쪽 검정?) -> (센서 읽기, 의미)
    (0, 0): ('왼 흰 · 오 흰', '선을 놓침'),
    (0, 1): ('왼 흰 · 오 검', '선이 오른쪽에'),
    (1, 0): ('왼 검 · 오 흰', '선이 왼쪽에'),
    (1, 1): ('왼 검 · 오 검', '선 한가운데'),
}
MODE_TITLES = {'train': '학습 모드 · ε-greedy로 탐험하며 Q 업데이트',
               'eval': '평가 모드 · 처음 보는 트랙, 탐험 끔, Q 고정',
               'rule': '규칙 기반 · 학습 없음 (비교 기준)'}


def setup_korean_font():
    """설치된 한글 폰트를 찾아 matplotlib 기본 폰트로 지정합니다 (Mac: AppleGothic)."""
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ['Apple SD Gothic Neo', 'AppleGothic', 'Malgun Gothic', 'NanumGothic',
                 'Noto Sans CJK KR', 'Noto Sans KR']:
        if cand in names:
            matplotlib.rcParams['font.family'] = [cand, 'DejaVu Sans']
            break
    matplotlib.rcParams['axes.unicode_minus'] = False


def state_parts(state):
    lr, prev = divmod(state, 3)
    lb, rb = divmod(lr, 2)
    return lb, rb, prev


class Dashboard:
    def __init__(self, seed=1, pretrain=0, load=None, log=None, track='maze', target=500, mode=None):
        import matplotlib.pyplot as plt
        from line_follower.rl_env import RandomTrackEnv, N_STATES
        from line_follower.qlearning import QAgent, run_episode, train_seed

        setup_korean_font()
        self.plt = plt
        self.seed = seed
        self.track_kind = track
        self.target = target
        self.env = RandomTrackEnv(track=track)
        self.agent = QAgent(seed=seed)
        self.visits = np.zeros(N_STATES, dtype=int)
        self.train_hist = []
        self.test = {'eval': [], 'rule': []}
        self.eval_idx = {'eval': 0, 'rule': 0}
        if load:
            self.agent.q = np.load(load)
            if log:
                import csv
                with open(log) as f:
                    for row in csv.DictReader(f):
                        self.train_hist.append({'seed': int(row['track_seed']), 'reward': float(row['reward']),
                                                'outcome': row['outcome'], 'steps': int(row['steps']),
                                                'progress': float(row['progress']),
                                                'epsilon': float(row['epsilon'])})
                self.agent.episode = len(self.train_hist)
            else:
                self.agent.episode = max(self.agent.eps_decay, self.agent.alpha_decay)
        for ep in range(pretrain):
            if ep % 50 == 0:
                print(f'pretraining {ep}/{pretrain} ...', flush=True)
            eps = self.agent.epsilon
            r = run_episode(self.env, train_seed(seed, self.agent.episode), self.agent.act,
                            agent=self.agent, learn=True)
            r['epsilon'] = eps
            self.train_hist.append(r)

        # Q-table을 불러왔으면 학습으로 값이 바뀌지 않게 평가 모드로 시작
        self.mode = mode or ('eval' if load else 'train')
        self.speed_i = 2
        self.paused = False
        self.job = None
        self.job_text = ''
        self.last_action = None
        self.last_reward = 0.0
        self.last_outcome = '-'

        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title('Q-Learning 라인트래커 대시보드')
        except AttributeError:
            pass
        self._panels()
        self._build_controls()
        self._build_sim()
        self._build_qtable()
        self._build_metrics()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.start_episode()
        self.redraw_charts()

    # ================================================================ 화면 구성
    def _panel(self, x0, y0, x1, y1):
        from matplotlib.patches import FancyBboxPatch
        self.fig.patches.append(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.006',
                                               transform=self.fig.transFigure, facecolor=PANEL,
                                               edgecolor=LINE, linewidth=1, zorder=-1))

    def _panels(self):
        self.fig.text(0.012, 0.962, 'Q-Learning 라인트래커', fontsize=18, color=TEXT, weight='bold', va='center')
        self.title_info = self.fig.text(0.988, 0.962, '', fontsize=10, color=MUTED, ha='right', va='center')
        self._panel(0.012, 0.03, 0.505, 0.865)
        self._panel(0.515, 0.455, 0.988, 0.865)
        self._panel(0.515, 0.03, 0.988, 0.443)

    def _button(self, rect, label, cb, color=PANEL2):
        from matplotlib.widgets import Button
        b = Button(self.fig.add_axes(rect), label, color=color, hovercolor='#2b3a5e')
        b.label.set_color(TEXT)
        b.label.set_fontsize(9)
        for sp in b.ax.spines.values():
            sp.set_color(LINE)
        b.on_clicked(cb)
        return b

    def _build_controls(self):
        from matplotlib.widgets import TextBox
        y, h = 0.885, 0.045
        self.fig.text(0.012, y + h / 2, '시드', color=MUTED, fontsize=10, va='center')
        self.seed_box = TextBox(self.fig.add_axes([0.047, y, 0.026, h]), '', initial=str(self.seed),
                                color='#0f1a33', hovercolor='#0f1a33', textalignment='center')
        self.seed_box.text_disp.set_color(TEXT)
        self.seed_box.text_disp.set_fontsize(11)
        if hasattr(self.seed_box, 'cursor'):
            self.seed_box.cursor.set_color(TEXT)
        for sp in self.seed_box.ax.spines.values():
            sp.set_color(BLUE)
        self.seed_box.on_submit(lambda text: self.new_training())   # 숫자 입력 후 Enter = 새 학습
        self.fig.canvas.mpl_connect('button_press_event', self._on_click_seed)
        self._button([0.033, y, 0.013, h], '◀', lambda e: self.bump_seed(-1))
        self._button([0.074, y, 0.0125, h], '▶', lambda e: self.bump_seed(+1))
        specs = [('새 학습', lambda e: self.new_training()),
                 ('학습 모드', lambda e: self.set_mode('train')),
                 ('평가 모드', lambda e: self.set_mode('eval')),
                 ('규칙 기반', lambda e: self.set_mode('rule')),
                 ('일시정지', lambda e: self.toggle_pause()),
                 ('한 스텝', lambda e: self.single_step()),
                 ('다음 트랙', lambda e: self.next_track()),
                 ('빠른 학습 50', lambda e: self.start_fast_train(50)),
                 ('미학습 30개 비교', lambda e: self.start_batch_test(30)),
                 ('느리게', lambda e: self.change_speed(-1)),
                 ('빠르게', lambda e: self.change_speed(+1))]
        x0, gap = 0.087, 0.0045
        w = (0.988 - x0 - gap * (len(specs) - 1)) / len(specs)
        self.btn = {}
        for i, (label, cb) in enumerate(specs):
            self.btn[label] = self._button([x0 + i * (w + gap), y, w, h], label, cb)
        self.mode_buttons = {'train': self.btn['학습 모드'], 'eval': self.btn['평가 모드'],
                             'rule': self.btn['규칙 기반']}

    def _build_sim(self):
        self.fig.text(0.025, 0.84, '2D 시뮬레이터', fontsize=13, color=TEXT, weight='bold', va='center')
        self.sim_head = self.fig.text(0.025, 0.812, '', fontsize=10, color=MUTED, va='center')
        # 센서 LED
        self.led_ax = self.fig.add_axes([0.025, 0.745, 0.47, 0.05])
        self.led_ax.set_axis_off()
        self.led_ax.set_xlim(0, 1)
        self.led_ax.set_ylim(0, 1)
        from matplotlib.patches import FancyBboxPatch
        self.led_boxes, self.led_dots, self.led_texts = [], [], []
        for i, name in enumerate(['왼쪽 센서', '오른쪽 센서']):
            x = 0.0 + i * 0.27
            box = FancyBboxPatch((x, 0.05), 0.255, 0.9, boxstyle='round,pad=0,rounding_size=0.05',
                                 facecolor=PANEL2, edgecolor=LINE, transform=self.led_ax.transAxes)
            self.led_ax.add_patch(box)
            dot = self.led_ax.scatter([x + 0.03], [0.5], s=140, zorder=3)
            self.led_ax.text(x + 0.065, 0.68, name, fontsize=9, color=MUTED, va='center')
            t = self.led_ax.text(x + 0.065, 0.3, '', fontsize=10, color=TEXT, va='center', weight='bold')
            self.led_boxes.append(box)
            self.led_dots.append(dot)
            self.led_texts.append(t)
        self.state_text = self.led_ax.text(0.56, 0.5, '', fontsize=10.5, color=TEXT, va='center')
        # 트랙
        self.ax = self.fig.add_axes([0.022, 0.165, 0.473, 0.57])
        self.ax.set_facecolor(PANEL)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.track_artists = []
        self.path, = self.ax.plot([], [], color='#2563eb', lw=1.3, alpha=0.8, zorder=4)
        self.body, = self.ax.fill([0], [0], color='#2563eb', ec='white', lw=1, zorder=6)
        self.sensor_dots = self.ax.scatter([0, 0], [0, 0], c=[0, 0], s=28, zorder=7, edgecolors=ORANGE,
                                           linewidths=1.5, cmap='gray', vmin=0, vmax=100)
        self.sim_foot = self.fig.text(0.025, 0.115, '', fontsize=10.5, color=TEXT, va='top', linespacing=1.6)

    def _build_qtable(self):
        from matplotlib.patches import Rectangle
        from line_follower.rl_env import N_STATES
        self.fig.text(0.528, 0.84, '실시간 Q-Table', fontsize=13, color=TEXT, weight='bold', va='center')
        self.fig.text(0.528, 0.815, 'Q값 = 이 상태에서 이 행동을 하면 앞으로 받을 보상 합의 추정치. '
                                    '각 줄에서 Q가 가장 큰 행동(★)을 고른다.', fontsize=9, color=MUTED, va='center')
        x0, x1, y0, y1 = 0.672, 0.895, 0.495, 0.758
        self.qax = self.fig.add_axes([x0, y0, x1 - x0, y1 - y0])
        self.qimg = self.qax.imshow(self.agent.q, cmap='viridis', aspect='auto')
        self.qax.set_xticks(range(5), [f'{a}\n{k}' for a, k in zip(ACTION_ARROW, ACTION_KO)],
                            fontsize=8, color=TEXT)
        self.qax.xaxis.tick_top()
        self.qax.tick_params(length=0, pad=3)
        self.qax.set_yticks([])
        for sp in self.qax.spines.values():
            sp.set_visible(False)
        self.qtexts = [[self.qax.text(a, s, '', ha='center', va='center', fontsize=8.5)
                        for a in range(5)] for s in range(N_STATES)]
        for y in (2.5, 5.5, 8.5):
            self.qax.axhline(y, color=PANEL, lw=3)
        self.row_box = Rectangle((-0.5, -0.5), 5, 1, fill=False, edgecolor=RED, lw=2.2, zorder=5)
        self.cell_box = Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor=ORANGE, lw=3, zorder=6)
        self.qax.add_patch(self.row_box)
        self.qax.add_patch(self.cell_box)
        # 왼쪽 라벨: 센서 그룹 + 직전 조향
        row_h = (y1 - y0) / N_STATES
        self.prev_labels = []
        for g, (lb, rb) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            yc = y1 - (g * 3 + 1.5) * row_h
            read, meaning = SENSOR_GROUPS[(lb, rb)]
            self.fig.text(0.528, yc + 0.009, meaning, fontsize=10, color=TEXT, weight='bold', va='center')
            self.fig.text(0.528, yc - 0.012, read, fontsize=8.5, color=MUTED, va='center')
            for k in range(3):
                yy = y1 - (g * 3 + k + 0.5) * row_h
                self.prev_labels.append(self.fig.text(0.666, yy, f'직전 {PREV_KO[k]}', fontsize=8.5,
                                                      color=MUTED, va='center', ha='right'))
        self.fig.text(0.528, y1 + 0.014, '센서가 본 것', fontsize=8.5, color=MUTED, va='center')
        self.fig.text(0.666, y1 + 0.014, '직전 조향', fontsize=8.5, color=MUTED, ha='right', va='center')
        # 오른쪽: 최선 행동 + 방문 횟수
        self.fig.text(0.94, y1 + 0.022, '최선 행동 ★', fontsize=8.5, color=MUTED, ha='center', va='center')
        self.fig.text(0.94, y1 + 0.006, '(방문 횟수)', fontsize=8, color=MUTED, ha='center', va='center')
        self.best_texts = [self.fig.text(0.902, y1 - (s + 0.5) * row_h, '', fontsize=8.5, color=TEXT, va='center')
                           for s in range(N_STATES)]
        self.fig.text(0.528, 0.472, '빨간 테두리 = 지금 상태   ·   주황 칸 = 방금 고른 행동   ·   '
                                    '굵은 글씨/★ = 그 줄의 최선 행동   ·   방문 0회인 줄은 아직 배운 게 없음',
                      fontsize=8.5, color=MUTED, va='center')

    def _build_metrics(self):
        self.fig.text(0.528, 0.42, '학습 지표', fontsize=13, color=TEXT, weight='bold', va='center')
        self.donut1 = self.fig.add_axes([0.527, 0.2, 0.105, 0.185])
        self.donut2 = self.fig.add_axes([0.645, 0.2, 0.105, 0.185])
        self.fig.text(0.5795, 0.393, '학습 진행률', fontsize=10, color=TEXT, ha='center', va='center')
        self.fig.text(0.6975, 0.393, '학습 방향', fontsize=10, color=TEXT, ha='center', va='center')
        self.donut1_cap = self.fig.text(0.5795, 0.185, '', fontsize=8.5, color=MUTED, ha='center', va='top')
        self.donut2_cap = self.fig.text(0.6975, 0.185, '', fontsize=8.5, color=MUTED, ha='center', va='top')
        self.metrics = self.fig.text(0.528, 0.125, '', fontsize=9.5, color=TEXT, va='top', linespacing=1.55)
        self.rax = self.fig.add_axes([0.795, 0.255, 0.18, 0.14])
        self.sax = self.fig.add_axes([0.795, 0.06, 0.18, 0.14])
        for ax, title in ((self.rax, '에피소드 보상'), (self.sax, '완주율 (최근 20회) · ε')):
            ax.set_facecolor(PANEL)
            ax.tick_params(labelsize=7, colors=MUTED)
            for sp in ax.spines.values():
                sp.set_color(LINE)
            ax.set_title(title, fontsize=9, color=TEXT, loc='left')
        self.r_dots, = self.rax.plot([], [], '.', color=GRAY, ms=2.5)
        self.r_avg, = self.rax.plot([], [], color=BLUE, lw=1.5)
        self.sax.set_ylim(-0.05, 1.05)
        self.s_line, = self.sax.plot([], [], color=GREEN, lw=1.5, label='완주율')
        self.e_line, = self.sax.plot([], [], color=YELLOW, lw=1, alpha=0.8, label='ε')
        self.s_test = self.sax.axhline(np.nan, color='#c084fc', ls='--', lw=1, label='Q 미학습 평가')
        self.s_rule = self.sax.axhline(np.nan, color=ORANGE, ls=':', lw=1.3, label='규칙 미학습 평가')
        leg = self.sax.legend(fontsize=6.5, loc='center right', ncol=2, facecolor=PANEL, edgecolor=LINE, labelcolor=TEXT)
        leg.set_zorder(10)

    # ================================================================ 에피소드 진행
    def current_seed(self):
        from line_follower.qlearning import train_seed, EVAL_SEED_BASE
        if self.mode == 'train':
            return train_seed(self.seed, self.agent.episode)
        return EVAL_SEED_BASE + self.eval_idx[self.mode]

    def start_episode(self):
        self.ep_seed = self.current_seed()
        self.state = self.env.reset(self.ep_seed)
        self.ep_reward = 0.0
        self.await_next = False
        self.ep_eps = self.agent.epsilon
        self.last_action = None
        self.last_reward = 0.0
        self.draw_track()

    def policy(self, s):
        from line_follower.qlearning import rule_policy
        if self.mode == 'train':
            return self.agent.act(s, explore=True)
        if self.mode == 'eval':
            return self.agent.act(s, explore=False)
        return rule_policy(s)

    def sim_step(self):
        s = self.state
        a = self.policy(s)
        s2, r, term, trunc, info = self.env.step(a)
        if self.mode == 'train':
            self.agent.update(s, a, r, s2, term)
        self.visits[s] += 1
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
                if self.mode != 'train' and self.last_outcome == 'offtrack':
                    # 평가·규칙 모드에서 이탈하면 그 자리에서 멈춰 원인을 볼 수 있게 합니다.
                    self.paused = True
                    self.await_next = True
                else:
                    self.start_episode()
        self.draw_state()

    # ================================================================ 버튼 동작
    def set_mode(self, mode):
        if mode != self.mode or self.job is not None:
            self.job = None
            self.job_text = ''
            self.mode = mode
            self.start_episode()

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused and self.await_next:
            self.start_episode()

    def single_step(self):
        self.paused = True
        if self.await_next:
            self.start_episode()
        elif self.job is None and self.sim_step():
            self.start_episode()
        self.draw_state()

    def next_track(self):
        """지금 에피소드를 기록 없이 버리고 다음 트랙으로 갑니다."""
        if self.mode == 'train':
            self.agent.episode += 1
        else:
            self.eval_idx[self.mode] += 1
        self.start_episode()

    def change_speed(self, d):
        self.speed_i = int(np.clip(self.speed_i + d, 0, len(SPEEDS) - 1))

    def _on_click_seed(self, event):
        """시드 칸을 누르면 비워서 바로 새 숫자를 칠 수 있게 합니다."""
        if event.inaxes is self.seed_box.ax:
            self._set_seed_text('')

    def _set_seed_text(self, text):
        """값만 바꾸고 '입력 완료(새 학습)' 이벤트는 일으키지 않습니다."""
        tb = self.seed_box
        tb.eventson = False
        tb.set_val(text)
        tb.eventson = True
        if hasattr(tb, 'cursor_index'):
            tb.cursor_index = len(text)

    def bump_seed(self, d):
        try:
            cur = int(self.seed_box.text.strip())
        except ValueError:
            cur = self.seed
        self._set_seed_text(str(max(0, cur + d)))

    def new_training(self):
        from line_follower.qlearning import QAgent
        try:
            self.seed = int(self.seed_box.text.strip())
        except ValueError:
            pass
        self._set_seed_text(str(self.seed))
        self.job = None
        self.job_text = ''
        self.agent = QAgent(seed=self.seed)
        self.visits[:] = 0
        self.train_hist.clear()
        self.test = {'eval': [], 'rule': []}
        self.eval_idx = {'eval': 0, 'rule': 0}
        self.mode = 'train'
        self.paused = False
        self.start_episode()
        self.redraw_charts()

    def start_fast_train(self, n):
        self.mode = 'train'
        def job():
            for i in range(n):
                self.job_text = f'빠른 학습 {i + 1}/{n}'
                self.start_episode()
                while not self.sim_step():
                    pass
                yield
            self.job_text = ''
        self.job = job()

    def start_batch_test(self, n):
        self.test = {'eval': [], 'rule': []}
        self.eval_idx = {'eval': 0, 'rule': 0}
        def job():
            for mode, name in (('eval', 'Q 평가'), ('rule', '규칙')):
                self.mode = mode
                for i in range(n):
                    self.job_text = f'미학습 비교: {name} {i + 1}/{n}'
                    self.start_episode()
                    while not self.sim_step():
                        pass
                    yield
            self.mode = 'train'
            self.job_text = ''
        self.job = job()

    def on_key(self, event):
        if self.seed_box.capturekeystrokes:
            return
        k = event.key
        actions = {'t': lambda: self.set_mode('train'), 'e': lambda: self.set_mode('eval'),
                   'r': lambda: self.set_mode('rule'), ' ': self.toggle_pause, '.': self.single_step,
                   'n': self.next_track, '+': lambda: self.change_speed(1), '=': lambda: self.change_speed(1),
                   '-': lambda: self.change_speed(-1), 'b': lambda: self.start_fast_train(50),
                   'x': lambda: self.start_batch_test(30)}
        if k in actions:
            actions[k]()
        elif k == 's':
            np.save('q_table.npy', self.agent.q)
            self.job_text = 'q_table.npy 저장됨'

    # ================================================================ 그리기
    def draw_track(self):
        from matplotlib.patches import Polygon, Rectangle
        tr = self.env.track
        for art in self.track_artists:
            art.remove()
        b = tr.bounds
        self.ax.set_xlim(b[0], b[1])
        self.ax.set_ylim(b[2], b[3])
        self.ax.set_aspect('equal')
        arts = [self.ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], facecolor='#e8edf3',
                                            edgecolor='none', zorder=0))]
        if hasattr(tr, 'walls'):
            from matplotlib.collections import LineCollection
            segs = [[(w[0], w[1]), (w[2], w[3])] for w in tr.walls()]
            arts.append(self.ax.add_collection(LineCollection(segs, colors='#b4c0d0', linewidths=1.4, zorder=1)))
        arts.append(self.ax.add_patch(Polygon(tr.outline(tr.line_width / 2 + tr.edge_width), closed=True,
                                              facecolor='#7c8796', edgecolor='none', zorder=2)))
        arts.append(self.ax.add_patch(Polygon(tr.outline(), closed=True, facecolor='#111111',
                                              edgecolor='none', zorder=2.5)))
        p0, p1 = tr.points[0], tr.points[-1]
        arts.append(self.ax.scatter([p0[0]], [p0[1]], s=90, color=GREEN, edgecolors='k', zorder=3))
        arts.append(self.ax.scatter([p1[0]], [p1[1]], s=110, marker='s', color=RED, edgecolors='k', zorder=3))
        box = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.85)
        arts.append(self.ax.text(p0[0] - 0.05, p0[1] - 0.07, '출발', fontsize=9, color='#0f5132', ha='right',
                                 weight='bold', zorder=8, bbox=box))
        arts.append(self.ax.text(p1[0] + 0.05, p1[1] - 0.07, '도착', fontsize=9, color='#842029', ha='left',
                                 weight='bold', zorder=8, bbox=box))
        self.track_artists = arts

    def draw_state(self):
        from line_follower.rl_env import MOTOR, BLACK_THRESHOLD
        r = self.env.robot
        local = np.array([[0.05, 0], [-0.035, 0.03], [-0.035, -0.03]])
        c, s = np.cos(r.theta), np.sin(r.theta)
        pts = local @ np.array([[c, s], [-s, c]]) + [r.x, r.y]
        self.body.set_xy(pts)
        traj = np.array(self.env.trajectory)
        self.path.set_data(traj[:, 0], traj[:, 1])
        self.sensor_dots.set_offsets(self.env.sensors.positions(r))
        self.sensor_dots.set_array(self.env.values)

        # 센서 LED
        lb, rb, prev = state_parts(self.state)
        for i, (black, val) in enumerate(((lb, self.env.values[0]), (rb, self.env.values[1]))):
            self.led_dots[i].set_color(GREEN if black else '#334155')
            self.led_boxes[i].set_edgecolor(GREEN if black else LINE)
            self.led_texts[i].set_text(f"{'검정 · 선 감지' if black else '흰색'}  ({val:.0f})")
        read, meaning = SENSOR_GROUPS[(lb, rb)]
        self.state_text.set_text(f'지금 상태: {meaning}  ·  직전 조향 {PREV_KO[prev]}   (상태 번호 {self.state})')

        speed = SPEEDS[self.speed_i]
        self.title_info.set_text(f"{'미로' if self.track_kind == 'maze' else '곡선'} 코스  ·  시드 {self.seed}  ·  "
                                 f"속도 {'에피소드 단위' if speed is None else f'{speed} step/프레임'}"
                                 + ('  ·  일시정지' if self.paused else ''))
        self.btn['일시정지'].label.set_text('재생' if self.paused else '일시정지')
        for m, b in self.mode_buttons.items():
            col = '#1d4ed8' if m == self.mode else PANEL2
            b.color = col
            b.ax.set_facecolor(col)
        tr = self.env.track
        head = (f'{MODE_TITLES[self.mode]}   |   트랙 {self.ep_seed} · 길이 {tr.length:.1f} m · '
                f'{self.env.steps}/{self.env.max_steps} step · 진행 {self.env.s / tr.length * 100:.0f}%')
        if self.await_next:
            head = (f'⚠ 이탈! 트랙 {self.ep_seed} · {self.env.steps} step · 진행 {self.env.s / tr.length * 100:.0f}% 에서 '
                    f'선 중심 {self.env.dist * 1000:.0f} mm  →  재생/한 스텝 = 다음 트랙')
            self.sim_head.set_color(RED)
        else:
            self.sim_head.set_color(MUTED)
        self.sim_head.set_text(head + (f'   [{self.job_text}]' if self.job_text else ''))
        if self.last_action is None:
            act = '-'
        else:
            m = MOTOR[self.last_action]
            act = f'{ACTION_ARROW[self.last_action]} {ACTION_KO[self.last_action]}  (모터 {m[0]}, {m[1]})'
        self.sim_foot.set_text(
            f'행동  {act}\n'
            f'보상  {self.last_reward:+.2f}   ·   이번 에피소드 누적  {self.ep_reward:+.1f}   ·   '
            f'선 중심까지 {self.env.dist * 1000:.1f} mm (70 mm 넘으면 이탈)   ·   지난 결과: {self.last_outcome}')
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
        mid = (vmin + vmax) / 2
        for s in range(q.shape[0]):
            learned = q[s].max() > q[s].min()
            best = int(np.argmax(q[s]))
            for a in range(q.shape[1]):
                t = self.qtexts[s][a]
                t.set_text(f'{q[s, a]:.1f}')
                t.set_color('black' if q[s, a] > mid else 'white')
                t.set_fontweight('bold' if learned and a == best else 'normal')
            v = self.visits[s]
            vt = f'{v / 1000:.1f}k' if v >= 1000 else str(v)
            if learned:
                self.best_texts[s].set_text(f'★ {ACTION_ARROW[best]} {ACTION_KO[best]}  ({vt})')
                self.best_texts[s].set_color(TEXT)
            else:
                self.best_texts[s].set_text(f'아직 모름  ({vt})')
                self.best_texts[s].set_color(GRAY)
            self.prev_labels[s].set_color(RED if s == self.state else MUTED)
        self.row_box.set_y(self.state - 0.5)
        if self.last_action is not None:
            self.cell_box.set_visible(True)
            self.cell_box.set_xy((self.last_action - 0.5, self.state - 0.5))
        else:
            self.cell_box.set_visible(False)

    def draw_metrics(self):
        def summary(rs):
            if not rs:
                return '-'
            return (f"{np.mean([x['outcome'] == 'finish' for x in rs]) * 100:.0f}%"
                    f"  (n={len(rs)}, 평균 진행 {np.mean([x['progress'] for x in rs]) * 100:.0f}%)")
        self.metrics.set_text(
            f"미학습 트랙 평가  (x 키 또는 '미학습 30개 비교')\n"
            f"  Q 정책 (탐험 끔):  {summary(self.test['eval'])}\n"
            f"  규칙 기반:  {summary(self.test['rule'])}\n"
            f"α {self.agent.alpha:.3f}   γ {self.agent.gamma:.2f}   ε {self.agent.epsilon:.3f}")

    def _donut(self, ax, frac, color, center, sub):
        ax.clear()
        ax.set_aspect('equal')
        frac = float(np.clip(frac, 0, 1))
        ax.pie([frac, 1 - frac] if frac < 1 else [1], colors=[color, '#24304d'] if frac < 1 else [color],
               startangle=90, counterclock=False, wedgeprops=dict(width=0.26, edgecolor=PANEL))
        ax.text(0, 0.12, center, ha='center', va='center', fontsize=15, color=TEXT, weight='bold')
        ax.text(0, -0.25, sub, ha='center', va='center', fontsize=9, color=color, weight='bold')

    def redraw_charts(self):
        h = self.train_hist
        n = len(h)
        # 1) 진행률
        frac = n / self.target if self.target else 0
        self._donut(self.donut1, frac, BLUE, f'{min(frac, 9.99) * 100:.0f}%', f'{n} / {self.target}')
        self.donut1_cap.set_text(f'목표 {self.target} 에피소드 중\nε {self.agent.epsilon:.2f} = 탐험 확률')
        # 2) 방향: 최근 k회 완주율과 그 직전 k회 비교
        if n < 10:
            self._donut(self.donut2, 0, GRAY, '-', '데이터 부족')
            self.donut2_cap.set_text('10 에피소드 이후 표시\n(최근 완주율 vs 직전)')
        else:
            k = min(50, n // 2)
            succ = np.array([e['outcome'] == 'finish' for e in h], dtype=float)
            recent, before = succ[-k:].mean(), succ[-2 * k:-k].mean()
            delta = recent - before
            if delta > 0.05:
                color, word = GREEN, f'▲ {delta * 100:+.0f}%p 좋아지는 중'
            elif delta < -0.05:
                color, word = RED, f'▼ {delta * 100:+.0f}%p 나빠지는 중'
            elif recent >= 0.9:
                color, word = GREEN, '■ 높은 수준 유지'
            elif recent <= 0.1:
                color, word = GRAY, '■ 아직 변화 없음'
            else:
                color, word = YELLOW, '■ 정체'
            self._donut(self.donut2, recent, color, f'{recent * 100:.0f}%', word.split(' ', 1)[0] +
                        (f' {delta * 100:+.0f}%p' if abs(delta) > 0.05 else ''))
            self.donut2_cap.set_text(f'최근 {k}회 완주율 (탐험 포함)\n{word}')
        # 3) 선 그래프
        if h:
            x = np.arange(1, n + 1)
            rew = np.array([e['reward'] for e in h])
            succ = np.array([e['outcome'] == 'finish' for e in h], dtype=float)
            k = 20
            denom = np.minimum(np.arange(1, n + 1), k)
            self.r_dots.set_data(x, rew)
            self.r_avg.set_data(x, np.convolve(rew, np.ones(k), 'full')[:n] / denom)
            self.s_line.set_data(x, np.convolve(succ, np.ones(k), 'full')[:n] / denom)
            self.e_line.set_data(x, [e['epsilon'] for e in h])
            for ax in (self.rax, self.sax):
                ax.set_xlim(0, max(n, 10) + 1)
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
    p.add_argument('--track', choices=['maze', 'curvy'], default='maze', help='코스 종류')
    p.add_argument('--target', type=int, default=500, help='진행률 그래프의 목표 에피소드 수')
    p.add_argument('--pretrain', type=int, default=0, help='시작 전 화면 없이 학습할 에피소드 수')
    p.add_argument('--load', help='train.py가 저장한 Q-table (.npy)')
    p.add_argument('--log', help='train.py가 저장한 학습 로그 (.csv), --load와 함께 사용')
    p.add_argument('--snapshot', help='창 대신 PNG 저장')
    p.add_argument('--mode', choices=['train', 'eval', 'rule'], help='시작 모드 (기본: 학습, --load면 평가)')
    a = p.parse_args()
    if a.snapshot:
        import matplotlib
        matplotlib.use('Agg')
    d = Dashboard(seed=a.seed, pretrain=a.pretrain, load=a.load, log=a.log, track=a.track, target=a.target,
                  mode=a.mode)
    if a.snapshot:
        for _ in range(30):
            d.update()
        d.fig.savefig(a.snapshot, dpi=110, facecolor=BG)
        print('saved', a.snapshot)
    else:
        d.show()


if __name__ == '__main__':
    main()
