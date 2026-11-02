import unittest
import numpy as np
from line_follower.track import Track
from line_follower.robot import Robot, Action
from line_follower.sensors import ColorSensors
from line_follower.simulator import Simulator, demo_action

class SimulatorTests(unittest.TestCase):
    def test_track_black_white_and_edge(self):
        track = Track()
        self.assertEqual(track.reflectance(track.radius, 0), 0)
        self.assertEqual(track.reflectance(0, 0), 100)
        self.assertAlmostEqual(track.reflectance(track.radius + track.line_width/2 + track.edge_width/2, 0), 50)

    def test_straight_motion(self):
        robot = Robot(x=0, y=0, theta=0)
        robot.step(Action.STRAIGHT, 1)
        np.testing.assert_allclose([robot.x, robot.y, robot.theta], [0.12, 0, 0], atol=1e-12)

    def test_turn_symmetry(self):
        left = Robot(x=0, y=0, theta=0)
        right = Robot(x=0, y=0, theta=0)
        left.step(Action.LEFT, 0.5)
        right.step(Action.RIGHT, 0.5)
        self.assertGreater(left.theta, 0)
        self.assertLess(right.theta, 0)
        np.testing.assert_allclose([left.x, left.y, left.theta], [right.x, -right.y, -right.theta])

    def test_sensor_rotation(self):
        sensors = ColorSensors()
        robot = Robot(x=1, y=2, theta=np.pi/2)
        np.testing.assert_allclose(sensors.positions(robot), [[0.98, 2.045], [1.02, 2.045]])

    def test_sensors_distinguish_black_and_white(self):
        robot = Robot(x=0.63, y=0, theta=np.pi/2)
        left, right = ColorSensors().read(robot, Track())
        self.assertEqual(left, 100)
        self.assertEqual(right, 0)

    def test_reset(self):
        sim = Simulator()
        initial = sim.observe().copy()
        sim.step(Action.LEFT)
        np.testing.assert_allclose(sim.reset(), initial)
        self.assertEqual(sim.time, 0)
        self.assertEqual(sim.robot.x, sim.track.radius)

    def test_invalid_step(self):
        for dt in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                Robot().step(Action.STRAIGHT, dt)
        with self.assertRaises(ValueError):
            Robot().step(99, 0.05)

    def test_rule_controller(self):
        self.assertEqual(demo_action([0, 100]), Action.LEFT)
        self.assertEqual(demo_action([100, 0]), Action.RIGHT)
        self.assertEqual(demo_action([100, 100]), Action.STRAIGHT)

    def test_demo_stays_near_line(self):
        sim = Simulator()
        for _ in range(2000):  # 100초 동안 여러 바퀴 주행
            sim.step(demo_action(sim.observe()))
            self.assertLess(abs(np.hypot(sim.robot.x, sim.robot.y) - sim.track.radius), 0.04)

if __name__ == '__main__':
    unittest.main()
