"""Run with:  python -m unittest discover tests"""
import math
import unittest

import numpy as np

from driving.sim import DT, World
from driving.track import Track
from driving.vehicle import Bicycle, State


class TrackTests(unittest.TestCase):
    def test_loop_is_closed_and_frenet_roundtrips(self):
        tr = Track()
        x0, y0, _ = tr.pose(0.0)
        xl, yl, _ = tr.pose(tr.length)
        self.assertLess(math.hypot(x0 - xl, y0 - yl), 1e-6)
        for s, d in ((10.0, 0.0), (400.0, 3.7), (900.0, -3.7)):
            x, y, _ = tr.pose(s, d)
            s2, d2 = tr.frenet(float(x), float(y), s)
            self.assertAlmostEqual(s2, s, delta=0.1)
            self.assertAlmostEqual(d2, d, delta=0.1)


class VehicleTests(unittest.TestCase):
    def test_straight_line_and_steering_limit(self):
        st = State(0.0, 0.0, 0.0, 10.0)
        for _ in range(60):
            Bicycle.step(st, 0.0, 0.0, DT)
        self.assertAlmostEqual(st.x, 10.0, delta=0.05)
        self.assertAlmostEqual(st.y, 0.0, delta=1e-9)
        for _ in range(120):
            Bicycle.step(st, 5.0, 0.0, DT)
        self.assertLessEqual(abs(st.delta), Bicycle.MAX_STEER + 1e-9)


class ControllerTests(unittest.TestCase):
    def test_both_controllers_remove_a_lateral_offset(self):
        w = World(seed=0)
        for e in w.egos:
            e.st.x -= math.sin(e.st.psi) * 1.0
            e.st.y += math.cos(e.st.psi) * 1.0
        for _ in range(int(10 / DT)):
            w.step()
        for e in w.egos:
            self.assertLess(abs(e.e_y), 0.15, e.name)

    def test_mpc_tracks_better_than_pure_pursuit_without_collisions(self):
        w = World(seed=0)
        for _ in range(int(120 / DT)):
            w.step()
        self.assertEqual(w.collisions, 0)
        self.assertLess(w.rms(w.mpc), w.rms(w.pp))
        self.assertGreater(w.overtakes, 0)


if __name__ == "__main__":
    unittest.main()
