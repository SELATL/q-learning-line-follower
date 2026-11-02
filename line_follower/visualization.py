"""Matplotlib 애니메이션과 키보드/버튼 조작 화면입니다."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from .robot import Action
from .simulator import Simulator, demo_action

class Viewer:
    def __init__(self, manual=False):
        self.sim = Simulator()
        self.manual = manual
        self.action = Action.STRAIGHT
        self.paused = False
        self.history = []
        self.fig, (self.ax, self.sensor_ax) = plt.subplots(1, 2, figsize=(11, 6),
                                                        gridspec_kw={'width_ratios': [3, 1]})
        self.fig.subplots_adjust(bottom=0.23)
        grid = np.linspace(-0.9, 0.9, 700)
        xx, yy = np.meshgrid(grid, grid)
        self.ax.imshow(self.sim.track.reflectance(xx, yy), origin='lower',
                       extent=(-0.9, 0.9, -0.9, 0.9), cmap='gray', vmin=0, vmax=100)
        self.ax.set(xlim=(-0.9, 0.9), ylim=(-0.9, 0.9), xlabel='x (m)', ylabel='y (m)', aspect='equal')
        self.path, = self.ax.plot([], [], color='tab:blue', lw=1, alpha=0.5)
        self.body, = self.ax.plot([], [], color='tab:blue', lw=3)
        self.sensor_dots = self.ax.scatter([], [], c=[], cmap='gray', vmin=0, vmax=100,
                                            edgecolors='red', s=65, zorder=5)
        self.bars = self.sensor_ax.bar(['Left', 'Right'], [0, 0], color=['tab:orange', 'tab:green'])
        self.sensor_ax.set(ylim=(0, 110), ylabel='Reflectance (0=black, 100=white)')
        self.status = self.fig.text(0.06, 0.13, '')
        self.fig.text(0.06, 0.08, 'Arrows: drive | Space: pause | R: reset | M: manual/demo')
        self.buttons = []
        for i, (label, action) in enumerate([('Left', Action.LEFT), ('Straight', Action.STRAIGHT), ('Right', Action.RIGHT)]):
            button = Button(self.fig.add_axes([0.57 + i * 0.12, 0.08, 0.1, 0.06]), label)
            button.on_clicked(lambda event, a=action: self.select(a))
            self.buttons.append(button)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.draw()

    def select(self, action):
        self.manual = True
        self.action = action

    def on_key(self, event):
        actions = {'left': Action.LEFT, 'up': Action.STRAIGHT, 'right': Action.RIGHT}
        if event.key in actions:
            self.select(actions[event.key])
        elif event.key == ' ':
            self.paused = not self.paused
        elif event.key == 'm':
            self.manual = not self.manual
        elif event.key == 'r':
            self.sim.reset()
            self.history.clear()
            self.action = Action.STRAIGHT
        self.draw()

    def draw(self):
        robot = self.sim.robot
        # 로봇 몸체의 삼각형 꼭짓점도 같은 회전변환을 적용합니다.
        local = np.array([[0.06, 0], [-0.04, 0.035], [-0.04, -0.035], [0.06, 0]])
        c, s = np.cos(robot.theta), np.sin(robot.theta)
        points = local @ np.array([[c, s], [-s, c]]) + [robot.x, robot.y]
        self.body.set_data(points[:, 0], points[:, 1])
        self.sensor_dots.set_offsets(self.sim.sensors.positions(robot))
        values = self.sim.observe()
        self.sensor_dots.set_array(values)
        for bar, value in zip(self.bars, values):
            bar.set_height(value)
        self.sensor_ax.set_title(f'L: {values[0]:.1f}  R: {values[1]:.1f}')
        if self.history:
            path = np.array(self.history)
            self.path.set_data(path[:, 0], path[:, 1])
        else:
            self.path.set_data([], [])
        self.ax.set_title('Manual' if self.manual else 'Rule-based demo (no learning)')
        self.status.set_text(f't={self.sim.time:.2f}s  x={robot.x:.3f}  y={robot.y:.3f}  '
                             f'theta={np.degrees(robot.theta):.1f} deg  {self.action.name}'
                             + ('  PAUSED' if self.paused else ''))
        self.fig.canvas.draw_idle()

    def update(self, frame):
        if not self.paused:
            if not self.manual:
                self.action = demo_action(self.sim.observe())
            self.sim.step(self.action)
            self.history.append((self.sim.robot.x, self.sim.robot.y))
            self.history = self.history[-3000:]
        self.draw()

    def show(self):
        # 참조를유지해 애니메이션이 도중에 제거되지 않게 합니다.
        self.animation = FuncAnimation(self.fig, self.update, interval=self.sim.dt * 1000,
                                       cache_frame_data=False)
        plt.show()
