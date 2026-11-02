"""시간 진행, 센서 관측, 초기화를 묶는 환경입니다. 학습은 수행하지 않습니다."""
from dataclasses import dataclass, field
import numpy as np
from .robot import Robot, Action
from .track import Track
from .sensors import ColorSensors

@dataclass
class Simulator:
    track: Track = field(default_factory=Track)
    sensors: ColorSensors = field(default_factory=ColorSensors)
    dt: float = 0.05

    def __post_init__(self):
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt는 유한한 양수여야 합니다.")
        self.reset()

    def reset(self):
        self.robot = Robot(x=self.track.radius)
        self.time = 0.0
        return self.observe()

    def observe(self):
        return self.sensors.read(self.robot, self.track)

    def step(self, action):
        self.robot.step(action, self.dt)
        self.time += self.dt
        return self.observe()

def demo_action(values):
    # 학습 없는 규칙 제어: 더 어두운 센서 쪽으로 회전합니다.
    left, right = values
    if left + 5 < right:
        return Action.LEFT
    if right + 5 < left:
        return Action.RIGHT
    return Action.STRAIGHT
