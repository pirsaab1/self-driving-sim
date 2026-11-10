"""Closed three-lane road with Frenet (s, d) helpers.

s = distance along the centre line, d = signed lateral offset (left positive).
Lane i has its centre at d = (i - 1) * LANE_W, so lane 0 is the right-hand (slow) lane.
"""
import numpy as np

LANE_W = 3.7
N_LANES = 3


def lane_d(i):
    return (i - 1) * LANE_W


def _smooth(a, w):
    k = np.ones(w) / w
    ext = np.r_[a[-w:], a, a[:w]]
    return np.convolve(ext, k, mode="same")[w:-w]


class Track:
    def __init__(self, a=320.0, b=130.0, n=2.2, ds=0.5):
        th = np.linspace(0, 2 * np.pi, 60001)[:-1]            # superellipse "stadium" loop
        c, s = np.cos(th), np.sin(th)
        x = a * np.sign(c) * np.abs(c) ** (2.0 / n)
        y = b * np.sign(s) * np.abs(s) ** (2.0 / n)
        xe, ye = np.r_[x, x[0]], np.r_[y, y[0]]
        cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(xe), np.diff(ye)))]
        self.length = cum[-1]
        self.n_pts = int(round(self.length / ds / 24)) * 24    # multiple of 24 -> stable dashes
        self.ds = self.length / self.n_pts
        sn = np.arange(self.n_pts) * self.ds
        self.X = _smooth(np.interp(sn, cum, xe), 9)
        self.Y = _smooth(np.interp(sn, cum, ye), 9)
        dx = (np.roll(self.X, -1) - np.roll(self.X, 1)) / (2 * self.ds)
        dy = (np.roll(self.Y, -1) - np.roll(self.Y, 1)) / (2 * self.ds)
        psi = np.arctan2(dy, dx)
        self.COS, self.SIN = np.cos(psi), np.sin(psi)
        dpsi = np.angle(np.exp(1j * (np.roll(psi, -1) - np.roll(psi, 1)))) / (2 * self.ds)
        self.KAP = _smooth(dpsi, 21)

    # ------------------------------------------------------------------
    def _interp(self, arr, s):
        u = (np.asarray(s, dtype=float) % self.length) / self.ds
        i = np.floor(u).astype(int)
        f = u - i
        return arr[i % self.n_pts] * (1 - f) + arr[(i + 1) % self.n_pts] * f

    def pose(self, s, d=0.0):
        """World (x, y, heading) of the point at (s, d)."""
        x, y = self._interp(self.X, s), self._interp(self.Y, s)
        c, sn = self._interp(self.COS, s), self._interp(self.SIN, s)
        n = np.hypot(c, sn)
        c, sn = c / n, sn / n
        return x - d * sn, y + d * c, np.arctan2(sn, c)

    def curvature(self, s):
        return self._interp(self.KAP, s)

    def frenet(self, x, y, s_hint, window=40.0):
        """Project a world point onto the centre line near s_hint -> (s in [0, L), d)."""
        i0 = int(round((s_hint % self.length) / self.ds))
        k = int(window / self.ds)
        idx = np.arange(i0 - k, i0 + k + 1) % self.n_pts
        j = int(np.argmin((self.X[idx] - x) ** 2 + (self.Y[idx] - y) ** 2))
        i = idx[j]
        t = (x - self.X[i]) * self.COS[i] + (y - self.Y[i]) * self.SIN[i]
        d = -(x - self.X[i]) * self.SIN[i] + (y - self.Y[i]) * self.COS[i]
        return (i * self.ds + t) % self.length, d
