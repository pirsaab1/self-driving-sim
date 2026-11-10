"""Kinematic bicycle model (reference point = rear axle) with actuator limits."""
import math
from dataclasses import dataclass


@dataclass
class State:
    x: float
    y: float
    psi: float
    v: float
    delta: float = 0.0


class Bicycle:
    L = 2.8              # wheelbase [m]
    LENGTH = 4.6
    WIDTH = 1.9
    REAR_OH = 0.9        # rear overhang behind the rear axle
    MAX_STEER = 0.55     # [rad]
    MAX_STEER_RATE = 0.9  # [rad/s]
    MAX_ACC = 2.5
    MAX_DEC = -6.0

    @classmethod
    def step(cls, s, delta_cmd, a_cmd, dt):
        cmd = max(-cls.MAX_STEER, min(cls.MAX_STEER, delta_cmd))
        r = cls.MAX_STEER_RATE * dt
        s.delta += max(-r, min(r, cmd - s.delta))
        a = max(cls.MAX_DEC, min(cls.MAX_ACC, a_cmd))
        s.x += s.v * math.cos(s.psi) * dt
        s.y += s.v * math.sin(s.psi) * dt
        s.psi += s.v / cls.L * math.tan(s.delta) * dt
        s.v = max(0.0, s.v + a * dt)
        return s
