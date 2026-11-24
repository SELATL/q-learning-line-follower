"""시드마다 다른 꼬불꼬불한 열린 트랙(시작점 -> 끝점)을 만듭니다. 길이 단위는 m입니다.

트랙은 직선 구간과 원호 구간을 이어 붙인 중심선(polyline)입니다.
- 최소 곡률 반경(min_radius)을 두어 로봇의 강조향 회전 반경(0.1m)으로 따라갈 수 있게 합니다.
- 서로 멀리 떨어진 구간끼리 clearance 이내로 붙지 않게 해서 지름길·가로지르기를 막습니다.
- 반사광 공식은 기존 Track과 같습니다: 선 안 0, 바깥 100, 경계 폭에서 선형 변화.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RandomTrack:
    seed: int = 0
    bounds: tuple = (-1.0, 1.0, -0.7, 0.7)   # xmin, xmax, ymin, ymax
    target_length: tuple = (3.2, 4.2)        # 트랙 길이 범위
    ds: float = 0.004                        # 중심선 점 간격
    line_width: float = 0.035
    edge_width: float = 0.006
    min_radius: float = 0.22
    max_radius: float = 0.50
    clearance: float = 0.20                  # 멀리 떨어진 구간끼리 최소 거리
    lead_in: float = 0.20                    # 출발 직선 길이
    margin: float = 0.12                     # 벽(화면 경계)과의 여유
    points: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        lo, hi = self.target_length
        for attempt in range(600):
            if attempt and attempt % 150 == 0:   # 계속 실패하면 목표 길이를 조금 줄입니다(결정적).
                lo, hi = lo * 0.9, hi * 0.9
            pts = self._try_generate(rng, (lo, hi))
            if pts is not None and self._valid(pts):
                self.points = pts
                break
        else:
            raise RuntimeError(f"seed={self.seed}: 트랙 생성 실패")
        self._finalize()

    def _finalize(self):
        """중심선 점(self.points)으로 거리 계산용 선분 정보를 만듭니다."""
        seg = np.diff(self.points, axis=0)
        self.seg_start = self.points[:-1]
        self.seg_vec = seg
        self.seg_len2 = np.maximum((seg ** 2).sum(axis=1), 1e-12)
        self.cum_s = np.concatenate([[0.0], np.cumsum(np.sqrt((seg ** 2).sum(axis=1)))])
        self.length = float(self.cum_s[-1])

    # ---------- 생성 ----------
    def _try_generate(self, rng, target_length):
        """구간을 하나씩 붙이되, 후보 구간이 경계를 넘거나 기존 트랙에 붙으면 다른 후보를 뽑습니다."""
        xmin, xmax, ymin, ymax = self.bounds
        x0 = xmin + self.margin + 0.03
        y0 = rng.uniform(ymin + 0.3, ymax - 0.3)
        th0 = rng.uniform(-0.4, 0.4)
        total = rng.uniform(*target_length)
        pts = self._segment(x0, y0, th0, self.lead_in, 0.0)
        th = th0
        s = self.lead_in
        last_dir = 0
        while s < total:
            for _ in range(40):
                if rng.random() < 0.25:
                    length, curvature, direction = rng.uniform(0.08, 0.30), 0.0, 0
                else:
                    radius = rng.uniform(self.min_radius, self.max_radius)
                    angle = np.radians(rng.uniform(40, 150))
                    # 직전과 반대 방향을 더 자주 골라 S자(꼬불꼬불) 형태를 만듭니다.
                    direction = -last_dir if (last_dir and rng.random() < 0.7) else rng.choice([-1, 1])
                    length, curvature = radius * angle, direction / radius
                seg = self._segment(pts[-1, 0], pts[-1, 1], th, length, curvature)[1:]
                if self._fits(seg, pts):
                    pts = np.vstack([pts, seg])
                    th += curvature * length
                    s += length
                    if direction:
                        last_dir = direction
                    break
            else:
                return None
        return pts

    def _segment(self, x, y, th, length, curvature):
        n = max(1, int(round(length / self.ds)))
        ths = th + curvature * self.ds * np.arange(1, n + 1)
        xs = x + np.cumsum(self.ds * np.cos(ths))
        ys = y + np.cumsum(self.ds * np.sin(ths))
        return np.vstack([[x, y], np.column_stack([xs, ys])])

    def _fits(self, seg, pts):
        xmin, xmax, ymin, ymax = self.bounds
        if (seg[:, 0].min() < xmin + self.margin or seg[:, 0].max() > xmax - self.margin
                or seg[:, 1].min() < ymin + self.margin or seg[:, 1].max() > ymax - self.margin):
            return False
        tail = int(np.pi * self.clearance / self.ds)      # 바로 이전 구간은 당연히 가까우므로 제외
        old = pts[:-tail:5] if len(pts) > tail else pts[:0]
        if len(old) == 0:
            return True
        new = seg[::5]
        d = np.hypot(new[:, None, 0] - old[None, :, 0], new[:, None, 1] - old[None, :, 1])
        return d.min() >= self.clearance

    def _valid(self, pts):
        xmin, xmax, ymin, ymax = self.bounds
        if (pts[:, 0].min() < xmin + self.margin - 1e-9 or pts[:, 0].max() > xmax - self.margin + 1e-9
                or pts[:, 1].min() < ymin + self.margin - 1e-9 or pts[:, 1].max() > ymax - self.margin + 1e-9):
            return False
        sub = pts[::5]                      # 2cm 간격으로 검사
        step = 5 * self.ds
        d = np.hypot(sub[:, None, 0] - sub[None, :, 0], sub[:, None, 1] - sub[None, :, 1])
        idx = np.arange(len(sub))
        far_along = np.abs(idx[:, None] - idx[None, :]) * step > np.pi * self.clearance
        return not np.any(far_along & (d < self.clearance))

    # ---------- 질의 ----------
    def nearest(self, x, y, hint=None, window=80):
        """점(x,y)에서 중심선까지 거리, 가장 가까운 선분 번호, 진행 거리 s를 반환합니다.

        hint(직전 선분 번호)가 있으면 그 주변 window개 선분만 검사해 빠르게 계산합니다.
        트랙끼리 clearance 이상 떨어져 있으므로 이탈 판정 거리 안에서는 결과가 같습니다.
        """
        if hint is None:
            lo, hi = 0, len(self.seg_start)
        else:
            lo, hi = max(0, hint - window), min(len(self.seg_start), hint + window)
        a = self.seg_start[lo:hi]
        v = self.seg_vec[lo:hi]
        t = np.clip(((x - a[:, 0]) * v[:, 0] + (y - a[:, 1]) * v[:, 1]) / self.seg_len2[lo:hi], 0, 1)
        px = a[:, 0] + t * v[:, 0]
        py = a[:, 1] + t * v[:, 1]
        d = np.hypot(x - px, y - py)
        k = int(np.argmin(d))
        i = lo + k
        s = self.cum_s[i] + t[k] * np.sqrt(self.seg_len2[i])
        return float(d[k]), i, float(s)

    def reflectance_from_distance(self, d):
        return 100.0 * np.clip((np.asarray(d) - self.line_width / 2) / self.edge_width, 0, 1)

    def start_pose(self):
        p0, p1 = self.points[0], self.points[5]
        return float(p0[0]), float(p0[1]), float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

    def outline(self, half_width=None):
        """그리기용: 선 폭만큼 좌우로 띄운 다각형 꼭짓점."""
        hw = self.line_width / 2 if half_width is None else half_width
        tangent = np.gradient(self.points, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        left = self.points + hw * normal
        right = self.points - hw * normal
        return np.vstack([left, right[::-1]])


@dataclass
class MazeTrack(RandomTrack):
    """격자 미로 속을 지나가는 꼬불꼬불한 한 갈래 길입니다.

    1) cols x rows 격자에 무작위 깊이 우선 탐색(DFS)으로 완전 미로(모든 칸이 한 갈래로 연결)를 만듭니다.
    2) 왼쪽 열의 한 칸에서 출발해, 미로 길을 따라 path_cells칸 떨어진 칸을 도착점으로 고릅니다.
    3) 칸 중심을 잇는 길에서 90도 꺾이는 곳은 반지름 cell/2 원호로 둥글게 만듭니다.
       (U턴은 반지름 cell/2 반원) -> 로봇 강조향 회전 반경 0.10m보다 커서 따라갈 수 있습니다.
    길이 아닌 칸 사이에는 벽(그리기 전용)이 있습니다. 벽은 주행에 영향을 주지 않습니다.
    """
    cols: int = 9
    rows: int = 6
    cell: float = 0.30
    path_cells: tuple = (14, 22)
    bounds: tuple = (-1.4, 1.4, -0.95, 0.95)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.links = self._carve(rng)
        path = self._pick_path(rng)
        self.path_cells_used = path
        self.points = self._build_points(path)
        self._finalize()

    def _carve(self, rng):
        links = {(c, r): set() for c in range(self.cols) for r in range(self.rows)}
        start = (0, int(rng.integers(self.rows)))
        stack, seen = [start], {start}
        while stack:
            c, r = stack[-1]
            nbrs = [(c + dc, r + dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= c + dc < self.cols and 0 <= r + dr < self.rows and (c + dc, r + dr) not in seen]
            if not nbrs:
                stack.pop()
                continue
            nxt = nbrs[int(rng.integers(len(nbrs)))]
            links[(c, r)].add(nxt)
            links[nxt].add((c, r))
            seen.add(nxt)
            stack.append(nxt)
        self.start_cell = start
        return links

    def _pick_path(self, rng):
        parent, order = {self.start_cell: None}, [self.start_cell]
        depth = {self.start_cell: 0}
        for cur in order:                       # 너비 우선으로 출발점에서의 미로 거리 계산
            for n in sorted(self.links[cur]):
                if n not in parent:
                    parent[n] = cur
                    depth[n] = depth[cur] + 1
                    order.append(n)
        lo, hi = self.path_cells
        target = int(rng.integers(lo, hi + 1))
        cands = sorted(c for c, d in depth.items() if d == target)
        if not cands:                            # 그 길이의 칸이 없으면 가장 먼 칸
            far = max(depth.values())
            cands = sorted(c for c, d in depth.items() if d == far)
        end = cands[int(rng.integers(len(cands)))]
        path = [end]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path[::-1]

    def cell_center(self, c, r):
        x0 = (self.bounds[0] + self.bounds[1]) / 2 - self.cols * self.cell / 2
        y0 = (self.bounds[2] + self.bounds[3]) / 2 - self.rows * self.cell / 2
        return np.array([x0 + (c + 0.5) * self.cell, y0 + (r + 0.5) * self.cell])

    def _build_points(self, path):
        h = self.cell / 2
        centers = [self.cell_center(*p) for p in path]
        pts = []
        for i, ctr in enumerate(centers):
            d_in = (ctr - centers[i - 1]) / self.cell if i > 0 else None
            d_out = (centers[i + 1] - ctr) / self.cell if i < len(centers) - 1 else None
            start = ctr if d_in is None else ctr - d_in * h
            end = ctr if d_out is None else ctr + d_out * h
            if d_in is None or d_out is None or np.allclose(d_in, d_out):
                n = max(2, int(np.hypot(*(end - start)) / self.ds) + 1)
                seg = np.linspace(start, end, n)
            else:                                # 90도 꺾임: 칸 모서리를 중심으로 한 사분원
                center = ctr - d_in * h + d_out * h
                a0 = np.arctan2(*(start - center)[::-1])
                a1 = np.arctan2(*(end - center)[::-1])
                da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
                n = max(2, int(abs(da) * h / self.ds) + 1)
                ang = a0 + da * np.linspace(0, 1, n)
                seg = center + h * np.column_stack([np.cos(ang), np.sin(ang)])
            pts.append(seg if not pts else seg[1:])
        return np.vstack(pts)

    def walls(self):
        """그리기용 벽 선분 목록 [(x0,y0,x1,y1), ...]. 연결되지 않은 이웃 칸 사이 + 바깥 테두리."""
        h = self.cell / 2
        out = []
        for c in range(self.cols):
            for r in range(self.rows):
                ctr = self.cell_center(c, r)
                if c + 1 < self.cols and (c + 1, r) not in self.links[(c, r)]:
                    out.append((ctr[0] + h, ctr[1] - h, ctr[0] + h, ctr[1] + h))
                if r + 1 < self.rows and (c, r + 1) not in self.links[(c, r)]:
                    out.append((ctr[0] - h, ctr[1] + h, ctr[0] + h, ctr[1] + h))
        lo = self.cell_center(0, 0) - h
        hi = self.cell_center(self.cols - 1, self.rows - 1) + h
        out += [(lo[0], lo[1], hi[0], lo[1]), (hi[0], lo[1], hi[0], hi[1]),
                (hi[0], hi[1], lo[0], hi[1]), (lo[0], hi[1], lo[0], lo[1])]
        return out


TRACK_TYPES = {'maze': MazeTrack, 'curvy': RandomTrack}


def make_track(kind, seed):
    return TRACK_TYPES[kind](seed=seed)
