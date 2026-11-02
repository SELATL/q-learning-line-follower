"""차동 구동 로봇의 자세와 세 가지 행동을 정의합니다."""
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

class Action(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2

@dataclass
class Robot:
    x: float = 0.65
    y: float = 0.0
    theta: float = np.pi / 2  # rad; +x 방향이 0, 반시계 방향이 양수
    wheel_base: float = 0.10
    speed: float = 0.12
    turn_difference: float = 0.08

    def __post_init__(self):
        if self.wheel_base <= 0 or self.speed < 0 or self.turn_difference < 0:
            raise ValueError("바퀴 간격과 속도를 확인하세요.")

    def wheel_speeds(self, action):
        action = Action(action)
        offset = {Action.LEFT: 1, Action.STRAIGHT: 0, Action.RIGHT: -1}[action]
        return (self.speed - offset * self.turn_difference / 2,
                self.speed + offset * self.turn_difference / 2)

    def step(self, action, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt는 유한한 양수여야 합니다.")
        left, right = self.wheel_speeds(action)
        v = (left + right) / 2
        omega = (right - left) / self.wheel_base
        # 일정한 바퀴 속도에서 원호를 따라 이동하는 정확한 해입니다.
        old_theta = self.theta
        angle = omega * dt
        if abs(omega) < 1e-12:
            self.x += v * np.cos(old_theta) * dt
            self.y += v * np.sin(old_theta) * dt
        else:
            self.x += v / omega * (np.sin(old_theta + angle) - np.sin(old_theta))
            self.y += v / omega * (np.cos(old_theta) - np.cos(old_theta + angle))
        self.theta = (old_theta + angle + np.pi) % (2 * np.pi) - np.pi
