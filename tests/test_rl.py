import unittest
import numpy as np

from line_follower.random_track import RandomTrack, MazeTrack
from line_follower import rl_env as R
from line_follower.qlearning import QAgent, rule_policy, run_episode, EVAL_SEED_BASE


class RandomTrackTests(unittest.TestCase):
    def test_same_seed_same_track_different_seed_different_track(self):
        np.testing.assert_array_equal(RandomTrack(seed=5).points, RandomTrack(seed=5).points)
        self.assertFalse(np.array_equal(RandomTrack(seed=5).points[:200], RandomTrack(seed=6).points[:200]))

    def test_track_constraints(self):
        for seed in range(5):
            t = RandomTrack(seed=seed)
            self.assertTrue(t._valid(t.points))
            self.assertGreater(t.length, 2.5)

    def test_reflectance_matches_original_formula(self):
        t = RandomTrack(seed=0)
        self.assertEqual(t.reflectance_from_distance(0.0), 0)
        self.assertEqual(t.reflectance_from_distance(1.0), 100)
        self.assertAlmostEqual(float(t.reflectance_from_distance(t.line_width / 2 + t.edge_width / 2)), 50)


class MazeTrackTests(unittest.TestCase):
    def test_same_seed_same_maze(self):
        np.testing.assert_array_equal(MazeTrack(seed=3).points, MazeTrack(seed=3).points)
        self.assertNotEqual(MazeTrack(seed=3).path_cells_used, MazeTrack(seed=4).path_cells_used)

    def test_path_follows_open_maze_cells(self):
        for seed in range(10):
            m = MazeTrack(seed=seed)
            path = m.path_cells_used
            self.assertEqual(len(set(path)), len(path))             # 같은 칸을 두 번 지나지 않음
            for a, b in zip(path, path[1:]):
                self.assertIn(b, m.links[a])                         # 벽을 뚫지 않음
            gaps = np.hypot(*np.diff(m.points, axis=0).T)
            self.assertLess(gaps.max(), 0.01)                        # 끊김 없는 중심선

    def test_corner_radius_is_followable(self):
        m = MazeTrack(seed=0)
        self.assertGreater(m.cell / 2, 0.10)                          # 강조향 회전 반경 0.10 m보다 큼

    def test_rule_baseline_finishes_maze(self):
        r = run_episode(R.RandomTrackEnv(track='maze'), EVAL_SEED_BASE + 1, rule_policy)
        self.assertEqual(r['outcome'], 'finish')


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.env = R.RandomTrackEnv()
        self.env.reset(0)

    def place(self, offset, along_index=300):
        """중심선의 along_index 지점에서 왼쪽으로 offset(m) 떨어진 곳에 진행 방향으로 로봇을 둡니다."""
        tr = self.env.track
        p, q = tr.points[along_index], tr.points[along_index + 1]
        th = np.arctan2(q[1] - p[1], q[0] - p[0])
        n = np.array([-np.sin(th), np.cos(th)])
        self.env.robot.x, self.env.robot.y = p + offset * n
        self.env.robot.theta = th
        self.env.dist, self.env.idx, self.env.s = tr.nearest(self.env.robot.x, self.env.robot.y)

    def test_state_encoding_is_unique_0_to_11(self):
        states = {R.encode_state(l, r, p) for l in (0, 1) for r in (0, 1) for p in (0, 1, 2)}
        self.assertEqual(states, set(range(12)))

    def test_motor_table(self):
        robot = R.MotorRobot()
        self.assertEqual(robot.wheel_speeds(R.RLAction.STRAIGHT), (0.12, 0.12))
        np.testing.assert_allclose(robot.wheel_speeds(R.RLAction.HARD_LEFT), (0.04, 0.12))
        np.testing.assert_allclose(robot.wheel_speeds(R.RLAction.SOFT_RIGHT), (0.12, 0.08))

    def test_both_white_near_line_is_not_offtrack(self):
        self.place(0.048)                      # 두 센서 모두 흰색이지만 이탈 기준(70mm) 안
        self.env.sensor_idx = [self.env.idx] * 2
        self.assertTrue(np.all(self.env._read_sensors() >= R.BLACK_THRESHOLD))
        _, r, term, trunc, info = self.env.step(R.RLAction.STRAIGHT)
        self.assertFalse(term)
        self.assertEqual(info.outcome, 'running')

    def test_offtrack_reward_is_exactly_minus_20(self):
        self.place(R.OFFTRACK_DIST + 0.01)
        _, r, term, _, info = self.env.step(R.RLAction.STRAIGHT)
        self.assertTrue(term)
        self.assertEqual(info.outcome, 'offtrack')
        self.assertEqual(r, R.REWARD_OFFTRACK)   # 시간·조향 보상이 더해지지 않음

    def test_finish_reward_is_exactly_plus_50(self):
        self.place(0.0, along_index=len(self.env.track.points) - 4)
        self.env.prev_dir = 0
        _, r, term, _, info = self.env.step(R.RLAction.HARD_RIGHT)   # 반전 조향이어도 완주만 적용
        self.assertTrue(term)
        self.assertEqual(info.outcome, 'finish')
        self.assertEqual(r, R.REWARD_FINISH)

    def test_running_reward_components(self):
        self.place(0.0)
        self.env.prev_dir = 1
        _, r, *_ = self.env.step(R.RLAction.STRAIGHT)
        expected = R.REWARD_TIME + R.REWARD_STABLE * self.env.centering()
        self.assertAlmostEqual(r, expected)
        self.env.prev_dir = 0                                         # 좌 -> 우 반전
        _, r2, *_ = self.env.step(R.RLAction.SOFT_RIGHT)
        self.assertAlmostEqual(r2, R.REWARD_TIME + R.REWARD_STABLE * self.env.centering() + R.REWARD_OVERSTEER)

    def test_timeout_is_truncated_not_terminated(self):
        self.env.max_steps = 3
        for _ in range(3):
            _, _, term, trunc, info = self.env.step(R.RLAction.STRAIGHT)
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info.outcome, 'timeout')


class AgentTests(unittest.TestCase):
    def test_q_update_matches_hand_calculation(self):
        ag = QAgent(alpha=0.1, alpha_end=0.1, gamma=0.9)
        ag.q[4] = [0, 0, 0, 10, 0]
        ag.update(0, 2, 1.0, 4, terminated=False)
        self.assertAlmostEqual(ag.q[0, 2], 0.1 * (1.0 + 0.9 * 10))   # 1.0
        ag.update(1, 0, -20.0, 4, terminated=True)                    # 종료: 미래 가치 0
        self.assertAlmostEqual(ag.q[1, 0], -2.0)

    def test_greedy_ignores_epsilon(self):
        ag = QAgent(eps_start=1.0)
        ag.q[3] = [0, 0, 5, 0, 0]
        self.assertTrue(all(ag.act(3, explore=False) == 2 for _ in range(20)))

    def test_rule_baseline_finishes_unseen_track(self):
        r = run_episode(R.RandomTrackEnv(), EVAL_SEED_BASE, rule_policy)
        self.assertEqual(r['outcome'], 'finish')


if __name__ == '__main__':
    unittest.main()
