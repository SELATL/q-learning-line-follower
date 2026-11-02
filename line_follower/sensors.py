"""로봇 앞쪽의 좌우 센서를 세계 좌표로 변환합니다."""
from dataclasses import dataclass
import numpy as np

@dataclass
class ColorSensors:
    forward: float = 0.045
    spacing: float = 0.040

    def __post_init__(self):
        if self.forward < 0 or self.spacing <= 0:
            raise ValueError("센서 위치와 간격을 확인하세요.")

    def positions(self, robot):
        # 로봇 좌표: x는 전방, y는 왼쪽. 결과 행 순서는 왼쪽, 오른쪽.
        local = np.array([[self.forward, self.spacing / 2],
                          [self.forward, -self.spacing / 2]])
        c, s = np.cos(robot.theta), np.sin(robot.theta)
        rotation = np.array([[c, -s], [s, c]])
        return local @ rotation.T + [robot.x, robot.y]

    def read(self, robot, track):
        points = self.positions(robot)
        return np.asarray(track.reflectance(points[:, 0], points[:, 1]))
