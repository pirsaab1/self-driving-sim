"""Path-tracking controllers: Pure Pursuit and linear MPC (condensed QP)."""
import numpy as np

from .vehicle import Bicycle


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class RefPath:
    """Local reference path sampled ahead of the car (x, y, heading, curvature, arc length)."""

    def __init__(self, x, y):
        self.x, self.y = x, y
        dx, dy = np.gradient(x), np.gradient(y)
        step = np.maximum(np.hypot(dx, dy), 1e-6)
        self.psi = np.unwrap(np.arctan2(dy, dx))
        self.kappa = np.gradient(self.psi) / step
        self.sa = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]

    def nearest(self, x, y):
        n = min(len(self.x), 80)
        i = int(np.argmin((self.x[:n] - x) ** 2 + (self.y[:n] - y) ** 2))
        psi = self.psi[i]
        ey = -(x - self.x[i]) * np.sin(psi) + (y - self.y[i]) * np.cos(psi)
        return i, ey, psi


class PurePursuit:
    name = "Pure Pursuit"

    def __init__(self, k=0.55, ld0=4.0, ld_min=6.0, ld_max=26.0):
        self.k, self.ld0, self.ld_min, self.ld_max = k, ld0, ld_min, ld_max

    def control(self, st, ref, delta_prev=0.0):
        i, ey, psi_r = ref.nearest(st.x, st.y)
        ld = float(np.clip(self.ld0 + self.k * st.v, self.ld_min, self.ld_max))
        j = min(int(np.searchsorted(ref.sa, ref.sa[i] + ld)), len(ref.x) - 1)
        tx, ty = ref.x[j], ref.y[j]
        alpha = np.arctan2(ty - st.y, tx - st.x) - st.psi
        dist = max(np.hypot(tx - st.x, ty - st.y), 1.0)
        delta = float(np.arctan2(2.0 * Bicycle.L * np.sin(alpha), dist))
        return delta, dict(e_y=ey, e_psi=float(wrap(st.psi - psi_r)), target=(tx, ty), pred=None)


class MPC:
    """Linear time-varying MPC on the path-error dynamics.

    State  x = [e_y, e_psi]   (lateral / heading error w.r.t. the reference path)
    Input  u = delta - delta_ff,  delta_ff = atan(L * kappa)   (curvature feed-forward)
        e_y+   = e_y   + dt * v * e_psi
        e_psi+ = e_psi + dt * (v / L) / cos^2(delta_ff) * u
    The horizon is condensed into  min 0.5 U'HU + f'U  s.t. |delta| <= delta_max, which is
    solved in closed form and, if the steering limit is hit, by accelerated projected gradient.
    """
    name = "MPC"

    def __init__(self, dt=0.1, N=20, qy=3.0, qpsi=6.0, r=1.0, rd=12.0):
        self.dt, self.N = dt, N
        self.qy, self.qpsi, self.r, self.rd = qy, qpsi, r, rd
        self.D = np.eye(N) - np.eye(N, k=-1)            # difference operator
        self.DtD = self.D.T @ self.D
        self.tri = np.tril(np.ones((N, N)))
        self.k_idx = np.arange(N)
        self.solve_ms = 0.0

    def control(self, st, ref, delta_prev=0.0):
        import time
        t0 = time.perf_counter()
        N, dt, L = self.N, self.dt, Bicycle.L
        i, ey, psi_r = ref.nearest(st.x, st.y)
        e_psi = float(wrap(st.psi - psi_r))
        v = max(st.v, 3.0)
        s_h = ref.sa[i] + v * dt * self.k_idx
        kap = np.interp(s_h, ref.sa, ref.kappa)
        dff = np.arctan(L * kap)
        b = (v / L) / np.cos(dff) ** 2

        kj = self.k_idx[:, None] - self.k_idx[None, :]
        low = kj >= 0
        Gp = self.tri * (dt * b)[None, :]
        Gy = np.where(low, kj, 0) * dt * dt * v * b[None, :] * low
        y0 = ey + (self.k_idx + 1) * dt * v * e_psi
        p0 = np.full(N, e_psi)

        u_prev = delta_prev - dff[0]
        H = 2 * (self.qy * Gy.T @ Gy + self.qpsi * Gp.T @ Gp + self.r * np.eye(N) + self.rd * self.DtD)
        f = 2 * (self.qy * Gy.T @ y0 + self.qpsi * Gp.T @ p0)
        f[0] -= 2 * self.rd * u_prev
        U = -np.linalg.solve(H, f)

        lo, hi = -Bicycle.MAX_STEER - dff, Bicycle.MAX_STEER - dff
        if np.any(U < lo) or np.any(U > hi):           # steering limit active -> projected gradient
            Uc = np.clip(U, lo, hi)
            z, t = Uc.copy(), 1.0
            step = 1.0 / np.linalg.eigvalsh(H)[-1]
            for _ in range(40):
                Un = np.clip(z - step * (H @ z + f), lo, hi)
                tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
                z = Un + (t - 1) / tn * (Un - Uc)
                Uc, t = Un, tn
            U = Uc
        delta = float(dff[0] + U[0])

        # predicted trajectory (nonlinear rollout) for visualisation
        px, py, pp, pv = st.x, st.y, st.psi, st.v
        pred = [(px, py)]
        for k in range(N):
            dk = float(dff[k] + U[k])
            for _ in range(2):
                px += pv * np.cos(pp) * dt / 2
                py += pv * np.sin(pp) * dt / 2
                pp += pv / L * np.tan(dk) * dt / 2
            pred.append((px, py))
        self.solve_ms = 0.9 * self.solve_ms + 0.1 * (time.perf_counter() - t0) * 1000
        return delta, dict(e_y=ey, e_psi=e_psi, target=None, pred=pred)
