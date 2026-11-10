"""World: traffic, behaviour planner (ACC + overtaking), and two ego cars tracking one plan."""
import math
import random
from collections import deque

import numpy as np

from .controllers import MPC, PurePursuit, RefPath
from .track import LANE_W, N_LANES, Track, lane_d
from .vehicle import Bicycle, State

DT = 1.0 / 60.0           # physics step
CTRL_EVERY = 2            # controller runs at 30 Hz
PLAN_EVERY = 6            # behaviour planner at 10 Hz
HIST_EVERY = 4            # plot history at 15 Hz
V_CRUISE = 22.0           # [m/s]
A_LAT_MAX = 3.5
LANE_SPEEDS = {0: 13.0, 1: 16.0, 2: 19.0}
LANE_COUNTS = {0: 10, 1: 8, 2: 6}
KINDS = {"sedan": (4.6, 1.9, 1.45), "suv": (4.9, 2.0, 1.8), "truck": (9.6, 2.5, 3.4)}
EGO_CENTER = Bicycle.LENGTH / 2 - Bicycle.REAR_OH      # rear axle -> car centre [m]


def quintic(t):
    t = np.clip(t, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


class Plan:
    """Lane target as a function of (unwrapped) arc length; lane changes are quintic blends."""

    def __init__(self, lane):
        self.lane = lane
        self.change = None            # (s0, length, d_from, d_to, target_lane)
        self.last_change_t = -99.0

    @property
    def target_lane(self):
        return self.change[4] if self.change else self.lane

    def d_ref(self, s):
        s = np.asarray(s, dtype=float)
        if self.change is None:
            return np.full_like(s, lane_d(self.lane))
        s0, ln, d0, d1, _ = self.change
        return d0 + (d1 - d0) * quintic((s - s0) / ln)


class Traffic:
    PALETTE = [(150, 154, 164), (176, 70, 66), (70, 104, 160), (196, 178, 120), (80, 130, 96),
               (210, 212, 218), (40, 44, 52), (120, 70, 140), (200, 120, 50)]

    def __init__(self, vid, lane, s, v, kind, rng):
        self.id, self.lane, self.s, self.v, self.kind = vid, lane, s, v, kind
        self.length, self.width, self.height = KINDS[kind]
        self.color = (230, 232, 236) if kind == "truck" and rng.random() < .5 else rng.choice(self.PALETTE)
        self.phase = rng.uniform(0, 6.28)

    def d(self, t):
        """Lateral position: lane centre plus a small natural wander."""
        return lane_d(self.lane) + 0.12 * math.sin(0.35 * t + self.phase)


class Ego:
    def __init__(self, ctrl, track, lane, s0, v0, color):
        self.ctrl, self.color, self.name = ctrl, color, ctrl.name
        x, y, psi = track.pose(s0, lane_d(lane))
        self.st = State(float(x), float(y), float(psi), v0)
        self.s_total = s0
        self.delta_cmd = 0.0
        self.a_cmd = 0.0
        self.info = {}
        self.trail = deque(maxlen=160)
        self.reset_metrics()

    def reset_metrics(self):
        self.n = 0
        self.sum_e2 = 0.0
        self.max_e = 0.0
        self.sum_dd2 = 0.0
        self.e_y = 0.0
        self.e_psi = 0.0


class World:
    def __init__(self, seed=0, disturb_period=0.0):
        self.rng = random.Random(seed)
        random.seed(seed)
        self.track = Track()
        self.L = self.track.length
        self.t = 0.0
        self.k = 0
        self.disturb_period = disturb_period
        self.next_disturb = disturb_period if disturb_period else None
        self.kicks = deque(maxlen=20)
        self.kick_sign = 1.0
        self.flash = 0.0

        s0 = 40.0
        self.plan = Plan(0)
        self.pp = Ego(PurePursuit(), self.track, 0, s0, 12.0, (70, 150, 255))
        self.mpc = Ego(MPC(), self.track, 0, s0, 12.0, (255, 140, 40))
        self.egos = [self.pp, self.mpc]
        self.traffic = self._spawn_traffic(s0)
        self.v_ref = 12.0
        self.state_name = "CRUISING"
        self.gap = None
        self.lead = None
        self.overtakes = 0
        self.collisions = 0
        self._hit = False
        self.min_gap = 1e9
        self.laps = 0.0
        self.hist = {k: deque(maxlen=450) for k in
                     ("t", "ey_pp", "ey_mpc", "ep_pp", "ep_mpc", "d_pp", "d_mpc", "v_pp", "v_mpc", "vref")}
        self.a_prev = {id(e): 0.0 for e in self.egos}

    # ------------------------------------------------------------ traffic
    def _spawn_traffic(self, s_ego):
        out, vid = [], 0
        for lane, cnt in LANE_COUNTS.items():
            spacing = self.L / cnt
            base = self.rng.uniform(0, spacing)
            placed = []
            for j in range(cnt):
                for _ in range(50):
                    s = (base + j * spacing + self.rng.uniform(-.18, .18) * spacing) % self.L
                    far_from_ego = abs(self.rel_s(s, s_ego)) > 75
                    far_from_lane = all(abs(self.rel_s(s, o)) > 45 for o in placed)
                    if far_from_ego and far_from_lane:
                        break
                else:
                    continue
                placed.append(s)
                r = self.rng.random()
                kind = ("truck" if r < .38 else "suv" if r < .6 else "sedan") if lane == 0 else \
                       ("suv" if r < .35 else "sedan") if lane == 1 else "sedan"
                out.append(Traffic(vid, lane, s, LANE_SPEEDS[lane], kind, self.rng))
                vid += 1
        return out

    def rel_s(self, s_other, s_ref):
        return (s_other - s_ref + self.L / 2) % self.L - self.L / 2

    # ------------------------------------------------------------ planner
    def _lane_of(self, d):
        return int(np.clip(round(d / LANE_W) + 1, 0, N_LANES - 1))

    def _clear(self, lane, s_c, v_e, ahead=35.0, behind=10.0):
        for tv in self.traffic:
            if tv.lane != lane:
                continue
            rel = self.rel_s(tv.s, s_c)
            half = (tv.length + Bicycle.LENGTH) / 2
            closing = max(0.0, v_e - tv.v)
            if -(behind + half) < rel < ahead + half + closing * 3.0:
                return False
        return True

    def _lead(self, lanes, s_c):
        best, best_rel = None, 1e9
        for tv in self.traffic:
            if tv.lane in lanes:
                rel = self.rel_s(tv.s, s_c)
                if 0.0 < rel < best_rel:
                    best, best_rel = tv, rel
        return best, best_rel

    def _plan_step(self):
        e = self.mpc
        s_e, v_e = e.s_total + EGO_CENTER, e.st.v
        pl = self.plan
        _, d_e = self.track.frenet(e.st.x, e.st.y, e.s_total)
        cur = self._lane_of(d_e)

        if pl.change and e.s_total > pl.change[0] + pl.change[1] + 2.0:      # change finished
            pl.lane, pl.change = pl.change[4], None

        lanes = {cur, pl.target_lane}
        lead, rel = self._lead(lanes, s_e)
        self.lead = lead
        self.gap = rel - (lead.length + Bicycle.LENGTH) / 2 if lead else None

        if pl.change is None:
            if lead and self.gap < 45 + 2 * max(0.0, v_e - lead.v) and lead.v < V_CRUISE - 2:
                for tgt in (pl.lane + 1, pl.lane - 1):
                    if 0 <= tgt < N_LANES and self._clear(tgt, s_e, v_e):
                        self._start_change(tgt, s_e, v_e)
                        self.overtakes += 1 if tgt > pl.lane else 0
                        break
            elif pl.lane > 0 and self.t - pl.last_change_t > 4.0 and \
                    (lead is None or self.gap > 60) and self._clear(pl.lane - 1, s_e, v_e, ahead=90, behind=14):
                self._start_change(pl.lane - 1, s_e, v_e)

        # speed target: cruise, curvature limit with braking look-ahead, ACC to the lead car
        look = max(40.0, v_e * v_e / 5.0 + 15.0)
        ss = e.s_total + np.arange(0.0, look, 4.0)
        dd = pl.d_ref(ss)
        kc = self.track.curvature(ss)
        kk = np.abs(kc / np.maximum(1e-3, 1.0 - dd * kc))
        v_curve = math.sqrt(A_LAT_MAX / max(float(kk.max()), 1e-4))
        v_ref = min(V_CRUISE, v_curve)
        if lead is not None:
            d_safe = 6.0 + 1.1 * v_e
            v_ref = min(v_ref, max(0.0, lead.v + 0.6 * (self.gap - d_safe)))
        self.v_ref = v_ref

        if pl.change:
            self.state_name = "OVERTAKING" if pl.change[3] > pl.change[2] else "RETURNING"
        elif lead is not None and self.gap < 40 and lead.v < v_ref + 1:
            self.state_name = "FOLLOWING"
        else:
            self.state_name = "CRUISING"

    def _start_change(self, tgt, s_e, v_e):
        pl = self.plan
        ln = max(45.0, 2.4 * v_e)
        pl.change = (s_e - EGO_CENTER + 2.0, ln, lane_d(pl.lane), lane_d(tgt), tgt)
        pl.last_change_t = self.t

    # ------------------------------------------------------------ control
    def make_ref(self, ego, length=90.0, step=0.5):
        s = ego.s_total - 3.0 + np.arange(0.0, length, step)
        d = self.plan.d_ref(s)
        x, y, _ = self.track.pose(s, d)
        return RefPath(x, y)

    def _control(self, ego):
        ref = self.make_ref(ego)
        delta, info = ego.ctrl.control(ego.st, ref, ego.st.delta)
        ego.delta_cmd, ego.info = delta, info
        a = 1.4 * (self.v_ref - ego.st.v)
        a = float(np.clip(a, Bicycle.MAX_DEC, Bicycle.MAX_ACC))
        prev = self.a_prev[id(ego)]
        a = prev + float(np.clip(a - prev, -6.0 * DT * CTRL_EVERY, 6.0 * DT * CTRL_EVERY))
        self.a_prev[id(ego)] = ego.a_cmd = a
        ego.n += 1
        ey = info["e_y"]
        ego.e_y, ego.e_psi = ey, info["e_psi"]
        ego.sum_e2 += ey * ey
        ego.max_e = max(ego.max_e, abs(ey))
        ego.sum_dd2 += ((delta - ego.st.delta) / (DT * CTRL_EVERY)) ** 2

    # --------------------------------------------------------------- step
    def step(self):
        self.k += 1
        self.t += DT
        for tv in self.traffic:
            tv.s = (tv.s + tv.v * DT) % self.L
        if self.k % PLAN_EVERY == 0:
            self._plan_step()
        for ego in self.egos:
            if self.k % CTRL_EVERY == 0:
                self._control(ego)
            Bicycle.step(ego.st, ego.delta_cmd, ego.a_cmd, DT)
            s_new, _ = self.track.frenet(ego.st.x, ego.st.y, ego.s_total)
            ds = (s_new - ego.s_total % self.L + self.L / 2) % self.L - self.L / 2
            ego.s_total += ds
            if self.k % 3 == 0:
                ego.trail.append((ego.st.x, ego.st.y))
        self.laps = self.mpc.s_total / self.L
        self._collision_check()
        if self.next_disturb and self.t >= self.next_disturb:
            self.disturb()
            self.next_disturb += self.disturb_period
        self.flash = max(0.0, self.flash - DT)
        if self.k % HIST_EVERY == 0:
            h = self.hist
            h["t"].append(self.t)
            h["ey_pp"].append(self.pp.e_y); h["ey_mpc"].append(self.mpc.e_y)
            h["ep_pp"].append(math.degrees(self.pp.e_psi)); h["ep_mpc"].append(math.degrees(self.mpc.e_psi))
            h["d_pp"].append(math.degrees(self.pp.st.delta)); h["d_mpc"].append(math.degrees(self.mpc.st.delta))
            h["v_pp"].append(self.pp.st.v * 3.6); h["v_mpc"].append(self.mpc.st.v * 3.6)
            h["vref"].append(self.v_ref * 3.6)

    def disturb(self):
        """Instant heading kick (like a crosswind gust) applied to both cars."""
        for e in self.egos:
            e.st.psi += 0.10 * self.kick_sign
        self.kick_sign = -self.kick_sign
        self.kicks.append(self.t)
        self.flash = 1.2

    def _collision_check(self):
        e = self.mpc
        s_e, d_e = self.track.frenet(e.st.x, e.st.y, e.s_total)
        s_c = s_e + EGO_CENTER
        hit = False
        for tv in self.traffic:
            rel = self.rel_s(tv.s, s_c)
            dd = abs(d_e - tv.d(self.t))
            if abs(rel) < (tv.length + Bicycle.LENGTH) / 2 and dd < (tv.width + Bicycle.WIDTH) / 2:
                hit = True
            if 0 < rel < 80 and dd < (tv.width + Bicycle.WIDTH) / 2:
                self.min_gap = min(self.min_gap, rel - (tv.length + Bicycle.LENGTH) / 2)
        if hit and not self._hit:
            self.collisions += 1
        self._hit = hit

    # ------------------------------------------------------------ metrics
    def rms(self, ego):
        return math.sqrt(ego.sum_e2 / ego.n) if ego.n else 0.0

    def steer_rate_rms(self, ego):
        return math.degrees(math.sqrt(ego.sum_dd2 / ego.n)) if ego.n else 0.0
