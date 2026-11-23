"""랜덤 트랙 + 5행동 + 흑백 센서 상태를 쓰는 Q-learning용 환경입니다.

기존 Simulator/Action(3행동)은 그대로 두고, 이 파일은 별도 실험 환경으로 둡니다.

상태 (12개) = (왼쪽 센서 흑백, 오른쪽 센서 흑백, 직전 조향 방향)
    state = (left_black * 2 + right_black) * 3 + prev_dir      prev_dir: 0=좌, 1=직진, 2=우
행동 (5개) = 모터 출력(좌, 우). 60 -> 0.12 m/s 로 환산합니다.

보상 (한 step에 아래 중 정확히 하나의 경우만 적용, 중복 없음)
    1) 완주:  진행 거리 s >= 트랙 길이 - FINISH_TOL 이고 이탈 아님      -> +50, 종료
    2) 이탈:  로봇 중심과 중심선 거리 > OFFTRACK_DIST                   -> -20, 종료
    3) 그 외: -0.05 (시간)
              + 2 x (이번 step 진행량 / 최대 진행량)   (로봇 중심이 선 폭 안이고 앞으로 갔을 때만)
              - 1 (과도 조향: 직전 좌계열 -> 지금 우계열, 또는 반대로 즉시 반전)
    max_steps 도달은 '시간 제한(truncated)'이며 실패 종료(terminated)와 구분합니다.

이탈은 센서가 아니라 로봇의 실제 위치로 판정합니다. 두 센서가 모두 흰색이어도
선 근처라면 정상일 수 있기 때문입니다(센서 간격 40mm > 선 폭 35mm).
"""
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

from .robot import Robot
from .sensors import ColorSensors
from .random_track import RandomTrack


class RLAction(IntEnum):
    STRAIGHT = 0
    SOFT_LEFT = 1
    HARD_LEFT = 2
    SOFT_RIGHT = 3
    HARD_RIGHT = 4


MOTOR = {  # (왼쪽, 오른쪽) 모터 출력
    RLAction.STRAIGHT: (60, 60),
    RLAction.SOFT_LEFT: (40, 60),
    RLAction.HARD_LEFT: (20, 60),
    RLAction.SOFT_RIGHT: (60, 40),
    RLAction.HARD_RIGHT: (60, 20),
}
MOTOR_TO_MPS = 0.12 / 60
ACTION_NAMES = ['Straight', 'Soft L', 'Hard L', 'Soft R', 'Hard R']
DIR_NAMES = ['L', 'S', 'R']
N_STATES, N_ACTIONS = 12, 5

REWARD_FINISH = 50.0
REWARD_OFFTRACK = -20.0
REWARD_TIME = -0.05
REWARD_STABLE = 2.0
REWARD_OVERSTEER = -1.0
OFFTRACK_DIST = 0.07     # m, 로봇 중심 기준
FINISH_TOL = 0.03        # m
BLACK_THRESHOLD = 50.0   # 반사광 < 50 이면 검정
CENTERING_MODE = 'linear'  # 'linear' | 'band'(초안: 선 폭 안이면 +2)


def action_dir(action):
    a = RLAction(action)
    if a in (RLAction.SOFT_LEFT, RLAction.HARD_LEFT):
        return 0
    if a in (RLAction.SOFT_RIGHT, RLAction.HARD_RIGHT):
        return 2
    return 1


def encode_state(left_black, right_black, prev_dir):
    return (int(left_black) * 2 + int(right_black)) * 3 + int(prev_dir)


def state_label(state):
    lr, prev = divmod(state, 3)
    lb, rb = divmod(lr, 2)
    return f"{'B' if lb else 'W'}{'B' if rb else 'W'} | prev {DIR_NAMES[prev]}"


class MotorRobot(Robot):
    """기존 Robot의 원호 적분을 그대로 쓰고, 바퀴 속도만 5행동 모터 표로 바꿉니다."""

    def wheel_speeds(self, action):
        left, right = MOTOR[RLAction(action)]
        return left * MOTOR_TO_MPS, right * MOTOR_TO_MPS


@dataclass
class StepInfo:
    distance: float
    progress: float
    sensors: np.ndarray
    outcome: str  # 'running' | 'finish' | 'offtrack' | 'timeout'


class RandomTrackEnv:
    def __init__(self, dt=0.05, sensors=None):
        self.dt = dt
        self.sensors = sensors or ColorSensors()
        self.max_progress = 0.12 * dt
        self.track = None

    def reset(self, seed):
        self.track = RandomTrack(seed=seed)
        x, y, th = self.track.start_pose()
        self.robot = MotorRobot(x=x, y=y, theta=th)
        self.prev_dir = 1
        self.steps = 0
        self.max_steps = int(np.ceil(self.track.length / self.max_progress * 2.5))
        self.dist, self.idx, self.s = self.track.nearest(x, y)
        self.sensor_idx = [self.idx, self.idx]
        self.values = self._read_sensors()
        self.trajectory = [(x, y)]
        return self.state()

    def _read_sensors(self):
        pts = self.sensors.positions(self.robot)
        dists = []
        for k, (px, py) in enumerate(pts):
            d, i, _ = self.track.nearest(px, py, hint=self.sensor_idx[k])
            self.sensor_idx[k] = i
            dists.append(d)
        return self.track.reflectance_from_distance(dists)

    def centering(self):
        """중심선에 있으면 1, 이탈 경계(OFFTRACK_DIST)에 가까울수록 0."""
        if CENTERING_MODE == 'band':
            return 1.0 if self.dist <= self.track.line_width / 2 else 0.0
        return max(0.0, 1.0 - self.dist / OFFTRACK_DIST)

    def state(self):
        lb, rb = self.values < BLACK_THRESHOLD
        return encode_state(lb, rb, self.prev_dir)

    def step(self, action):
        """반환: state, reward, terminated, truncated, info"""
        prev_s = self.s
        new_dir = action_dir(action)
        self.robot.step(action, self.dt)
        self.steps += 1
        self.dist, self.idx, self.s = self.track.nearest(self.robot.x, self.robot.y, hint=self.idx)
        self.values = self._read_sensors()
        self.trajectory.append((self.robot.x, self.robot.y))
        progress = self.s - prev_s

        terminated = truncated = False
        if self.dist <= OFFTRACK_DIST and self.s >= self.track.length - FINISH_TOL:
            reward, terminated, outcome = REWARD_FINISH, True, 'finish'
        elif self.dist > OFFTRACK_DIST:
            reward, terminated, outcome = REWARD_OFFTRACK, True, 'offtrack'
        else:
            reward = REWARD_TIME
            if progress > 0:
                reward += REWARD_STABLE * self.centering()
            if {self.prev_dir, new_dir} == {0, 2}:
                reward += REWARD_OVERSTEER
            outcome = 'running'
            if self.steps >= self.max_steps:
                truncated, outcome = True, 'timeout'
        self.prev_dir = new_dir
        info = StepInfo(self.dist, self.s / self.track.length, self.values.copy(), outcome)
        return self.state(), reward, terminated, truncated, info
