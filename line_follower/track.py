"""원형 검은 띠와 반사광을 계산합니다. 길이 단위는 m입니다."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Track:
    radius: float = 0.65
    line_width: float = 0.035
    edge_width: float = 0.006

    def __post_init__(self):
        if self.radius <= 0 or not 0 < self.line_width < 2 * self.radius or self.edge_width <= 0:
            raise ValueError("트랙 반지름, 선 폭, 경계 폭을 확인하세요.")

    def reflectance(self, x, y):
        # 중심 원으로부터의 거리: 0이면 검은 선의 중앙입니다.
        distance = np.abs(np.hypot(x, y) - self.radius)
        # 선 내부는 0, 바깥은 100. 경계에서는 부드럽게 변합니다.
        return 100.0 * np.clip((distance - self.line_width / 2) / self.edge_width, 0, 1)
