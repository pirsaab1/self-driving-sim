"""3D chase-camera renderer: sky, hills, trees, guardrails, signs, 3D traffic, live plots.

Everything is drawn with plain pygame polygons through a small pinhole-camera projection
(no OpenGL needed). Camera modes: CHASE, HIGH (drone-style), COCKPIT.
"""
import math
import random

import numpy as np
import pygame

from .track import LANE_W, N_LANES, lane_d
from .vehicle import Bicycle

EGO_FWD = Bicycle.LENGTH / 2 - Bicycle.REAR_OH            # rear axle -> car centre

UHD = (3840, 2160)
FOG, FOG_DIST, NEAR, FAR = (184, 204, 222), 270.0, 1.6, 330.0
STEP = 3.0                                        # road sampling step [m]
GRASS, GRASS_B = (82, 128, 72), (74, 118, 66)
SHOULDER, ASPHALT = (118, 116, 112), (66, 68, 74)
WHITE = (232, 234, 238)
SUN = np.array([0.55, 0.35, 0.78]); SUN = SUN / np.linalg.norm(SUN)
PP_COL, MPC_COL = (70, 150, 255), (255, 140, 40)
REF_COL = (120, 235, 235)
TEXT, DIM = (225, 230, 240), (130, 138, 156)
HUD_BG, PLOT_BG, GRID = (20, 23, 33), (14, 16, 24), (44, 50, 70)

# (x0, x1, y0, y1, z0, z1, role) in car-centred coordinates (x forward, y left, z up)
CAR_SPECS = {
    "sedan": ([(-2.3, 2.3, -0.95, 0.95, 0.30, 0.95, "body"), (-1.4, 0.9, -0.80, 0.80, 0.95, 1.42, "cabin")],
              [(-1.4, 0.32, 0.64), (1.4, 0.32, 0.64)], 0.95),
    "suv": ([(-2.45, 2.45, -1.0, 1.0, 0.32, 1.15, "body"), (-2.0, 1.1, -0.90, 0.90, 1.15, 1.78, "cabin")],
            [(-1.5, 0.36, 0.72), (1.5, 0.36, 0.72)], 1.0),
    "truck": ([(-4.8, 4.8, -0.9, 0.9, 0.30, 0.62, "chassis"), (-4.8, 2.3, -1.25, 1.25, 0.62, 3.4, "trailer"),
               (2.6, 4.8, -1.2, 1.2, 0.5, 2.2, "body"), (3.2, 4.8, -1.1, 1.1, 2.2, 3.05, "cabin")],
              [(3.4, 0.0, 0.5), (-1.0, 0.0, 0.5), (-2.3, 0.0, 0.5), (-3.6, 0.0, 0.5)], 1.25),
}
FACES = (("rear", (0, 1, 3, 2)), ("front", (4, 5, 7, 6)), ("right", (0, 4, 5, 1)),
         ("left", (2, 6, 7, 3)), ("top", (1, 5, 7, 3)))


def shade(c, f):
    return tuple(max(0, min(255, int(v * f))) for v in c)


def fogc(c, z):
    t = 1.0 - math.exp(-max(z, 0.0) / FOG_DIST)
    return tuple(int(a + (b - a) * t) for a, b in zip(c, FOG))


def lerp(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def put(surf, font, txt, color, pos, center=False):
    s = font.render(txt, True, color)
    surf.blit(s, s.get_rect(center=pos) if center else pos)
    return s


class Renderer:
    MODES = [("CHASE", 12.0, 5.4, 9.0), ("HIGH", 22.0, 17.0, 26.0), ("COCKPIT", -0.9, 1.25, 1.2)]

    def __init__(self, world, show_window=True, resolution=(1920, 1080)):
        pygame.init()
        pygame.font.init()
        self.w = world
        W, H = resolution
        self.ui = H / 1080.0
        ui = self.ui
        self.hud_w = int(560 * ui)
        self.main_w = W - self.hud_w
        self.H = H
        self.size = (W, H)
        mono = "consolas,menlo,dejavusansmono"
        sf = pygame.font.SysFont
        self.f_s, self.f_m = sf(mono, int(13 * ui)), sf(mono, int(15 * ui))
        self.f_b, self.f_xl = sf(mono, int(20 * ui), bold=True), sf(mono, int(44 * ui), bold=True)
        self.f_sign = sf(mono, int(34 * ui), bold=True)

        self.f = (self.main_w / 2) / math.tan(math.radians(35.0))     # focal length [px], 70 deg HFOV
        self.cx, self.cyc = self.main_w / 2, H * 0.5
        self.mode = 0
        self.yaw = None
        self._bg = {}
        self.half = N_LANES * LANE_W / 2

        if show_window:
            try:
                self.screen = pygame.display.set_mode(self.size, pygame.SCALED | pygame.RESIZABLE)
            except pygame.error:
                self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Self-Driving Sim - Pure Pursuit vs MPC")
        else:
            self.screen = pygame.Surface(self.size)
        self._make_trees()
        self._make_signs()
        self._mini = self._make_minimap()
        self._vig = self._make_vignette()

    def to_bytes(self):
        return pygame.image.tostring(self.screen, "RGB")

    def save_png(self, path):
        pygame.image.save(self.screen, path)

    def cycle_camera(self):
        self.mode = (self.mode + 1) % len(self.MODES)

    # --------------------------------------------------------------- camera
    def _set_camera(self, pos, yaw, pitch):
        c, s, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        self.cpos = np.array(pos, dtype=float)
        self.fwd = np.array([c * cp, s * cp, -sp])
        self.right = np.array([s, -c, 0.0])
        self.up = np.array([c * sp, s * sp, cp])
        self.hy = self.cyc - self.f * math.tan(pitch)

    def project(self, P):
        rel = np.asarray(P, dtype=float) - self.cpos
        xc, yc, zc = rel @ self.right, rel @ self.up, rel @ self.fwd
        z = np.maximum(zc, 1e-3)
        return self.cx + self.f * xc / z, self.cyc - self.f * yc / z, zc

    def _p1(self, x, y, z):
        sx, sy, zc = self.project(np.array([[x, y, z]]))
        return float(sx[0]), float(sy[0]), float(zc[0])

    # ----------------------------------------------------------- static data
    def _make_trees(self):
        tr, rng = self.w.track, random.Random(3)
        sub_x, sub_y = tr.X[::6], tr.Y[::6]
        out = []
        while len(out) < 2600:
            s = rng.uniform(0, tr.length)
            side = rng.choice((-1, 1))
            d = side * (self.half + rng.uniform(6.0, 95.0))
            x, y, _ = tr.pose(s, d)
            if float(np.min((sub_x - x) ** 2 + (sub_y - y) ** 2)) < (abs(d) * 0.88) ** 2:
                continue
            out.append((float(x), float(y), rng.random() < 0.55, rng.uniform(0.8, 1.7), rng.uniform(0.85, 1.15)))
        self._tx = np.array([[t[0], t[1]] for t in out])
        self._tinfo = out

    def _make_signs(self):
        L = self.w.track.length
        self._signs = [(L * 0.07 + i * L / 3.0, "speed", 1) for i in range(3)] + \
                      [(L * 0.22 + i * L / 3.0, "gantry", 0) for i in range(3)]
        self._gantry_txt = [self.f_sign.render(t, True, WHITE) for t in ("EXIT 4  >", "CITY  12 km", "AIRPORT  >")]
        self._speed_txt = self.f_sign.render("80", True, (20, 20, 24))

    def _make_vignette(self):
        sw, sh = 40, 26
        small = pygame.Surface((sw, sh), pygame.SRCALPHA)
        for i in range(sw):
            for j in range(sh):
                dx, dy = (i + .5) / sw * 2 - 1, (j + .5) / sh * 2 - 1
                d = min(1.0, math.hypot(dx, dy) / 1.3)
                small.set_at((i, j), (0, 0, 0, int(max(0.0, d - .55) / .45 * 90)))
        return pygame.transform.smoothscale(small, (self.main_w, self.H))

    def _make_minimap(self):
        tr, ui = self.w.track, self.ui
        mw, mh = int(260 * ui), int(150 * ui)
        surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
        surf.fill((10, 12, 18, 205))
        pad = 12 * ui
        sx = (mw - 2 * pad) / (tr.X.max() - tr.X.min())
        sy = (mh - 2 * pad) / (tr.Y.max() - tr.Y.min())
        self._ms = min(sx, sy)
        self._mo = (mw / 2 - (tr.X.max() + tr.X.min()) / 2 * self._ms,
                    mh / 2 + (tr.Y.max() + tr.Y.min()) / 2 * self._ms)
        pts = [(self._mo[0] + x * self._ms, self._mo[1] - y * self._ms) for x, y in zip(tr.X[::8], tr.Y[::8])]
        pygame.draw.lines(surf, (90, 96, 112), True, pts, max(4, int(6 * ui)))
        pygame.draw.lines(surf, (40, 44, 54), True, pts, max(2, int(3 * ui)))
        self._mini_size = (mw, mh)
        return surf

    # ------------------------------------------------------------ background
    def _background(self, pitch):
        key = round(pitch, 3)
        if key in self._bg:
            return self._bg[key]
        W, H, f = self.main_w, self.H, self.f
        hy = int(self.cyc - f * math.tan(pitch))
        ground = pygame.Surface((W, H))
        ground.fill(FOG)
        for y in range(max(hy, 0), H):
            t = (y - hy) / max(1, H - hy)
            pygame.draw.line(ground, lerp(FOG, GRASS, min(1.0, t ** 0.5)), (0, y), (W, y))
        Wp, Hp = int(round(2 * math.pi * f)), max(hy + 2, 10)
        pano = pygame.Surface((Wp, Hp))
        for y in range(Hp):
            pygame.draw.line(pano, lerp((84, 134, 208), FOG, (y / (Hp - 1)) ** 1.15), (0, y), (Wp, y))
        sx, sy = int(Wp * 0.07), int(hy * 0.40)
        for r in range(int(hy * 0.55), 14, -5):
            t = 1 - r / (hy * 0.55)
            pygame.draw.circle(pano, lerp((150, 190, 232), (255, 248, 226), t ** 1.4), (sx, sy), r)
        pygame.draw.circle(pano, (255, 252, 240), (sx, sy), 16)
        rng = random.Random(7)
        clouds = pygame.Surface((Wp, Hp), pygame.SRCALPHA)
        for _ in range(26):
            cx0, cy0 = rng.uniform(0, Wp), rng.uniform(0.12, 0.62) * hy
            for _ in range(6):
                w_, h_ = rng.uniform(60, 190), rng.uniform(14, 34)
                pygame.draw.ellipse(clouds, (255, 255, 255, 150),
                                    (cx0 + rng.uniform(-110, 110), cy0 + rng.uniform(-12, 12), w_, h_))
        pano.blit(clouds, (0, 0))
        for amp, col, f1, f2, f3, base in ((1.0, (150, 172, 198), 3, 7, 19, 0.07), (0.55, (104, 138, 112), 5, 13, 31, 0.035)):
            pts = [(0, hy)]
            for u in range(0, Wp + 1, 8):
                a = 2 * math.pi * u / Wp
                h = hy * (base + 0.05 * amp * (math.sin(f1 * a + 0.7) * 0.9 + math.sin(f2 * a + 2.1) * 0.5
                                              + math.sin(f3 * a + 4.0) * 0.25 + 1.6))
                pts.append((u, hy - h))
            pts.append((Wp, hy))
            pygame.draw.polygon(pano, col, pts)
        self._bg[key] = (ground, pano, hy)
        return self._bg[key]

    # ------------------------------------------------------------------ road
    def _ring(self, d, z=0.0):
        k = (round(d, 3), z)
        r = self._rc.get(k)
        if r is None:
            x, y, _ = self.w.track.pose(self._ss, d)
            rel = np.column_stack((x, y, np.full_like(x, z))) - self.cpos
            xc, yc, zc = rel @ self.right, rel @ self.up, rel @ self.fwd
            zz = np.maximum(zc, 1e-3)
            r = self._rc[k] = (self.cx + self.f * xc / zz, self.cyc - self.f * yc / zz, zc, xc, yc)
        return r

    @staticmethod
    def _clip_near(pts):
        """Sutherland-Hodgman clip of a camera-space polygon against the near plane."""
        out, n = [], len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ina, inb = a[2] >= NEAR, b[2] >= NEAR
            if ina:
                out.append(a)
            if ina != inb:
                t = (NEAR - a[2]) / (b[2] - a[2])
                out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, NEAR))
        return out

    def _strip(self, d0, d1, base, G, jit=0.0, mod=None, on=0, extend=True):
        a0, a1 = self._ring(d0), self._ring(d1)
        ks = self._ks
        gall = ks // G
        todo = []
        for g in np.unique(gall):
            if mod is not None and g % mod != on:
                continue
            idx = np.nonzero(gall == g)[0]
            lo, hi = int(idx[0]), int(idx[-1]) + (1 if extend else 0)
            hi = min(hi, len(ks) - 1)
            if hi - lo < 1:
                continue
            sl = slice(lo, hi + 1)
            if (a0[2][sl] < NEAR).any() or (a1[2][sl] < NEAR).any():
                if (a0[2][sl] < 0).all() and (a1[2][sl] < 0).all():
                    continue
                cam = [(a0[3][i], a0[4][i], a0[2][i]) for i in range(lo, hi + 1)] + \
                      [(a1[3][i], a1[4][i], a1[2][i]) for i in range(hi, lo - 1, -1)]
                cl = self._clip_near(cam)
                if len(cl) < 3:
                    continue
                pts = [(self.cx + self.f * p[0] / p[2], self.cyc - self.f * p[1] / p[2]) for p in cl]
                zm = float(np.maximum(a0[2][sl], NEAR).mean())
            else:
                pts = list(zip(a0[0][sl].tolist(), a0[1][sl].tolist())) + \
                      list(zip(a1[0][sl][::-1].tolist(), a1[1][sl][::-1].tolist()))
                zm = float(a0[2][sl].mean())
            todo.append((zm, int(g), pts))
        for z, g, pts in sorted(todo, key=lambda t: -t[0]):
            col = shade(base, 1.0 + jit * (((g * 7919) % 13) / 13.0 - 0.5) * 2) if jit else base
            pygame.draw.polygon(self.screen, fogc(col, z), pts)

    def _ground(self):
        h, scr = self.half, self.screen
        sh_out, sh_in = h + 1.5, h
        self._strip(-(sh_out + 45), -sh_out, GRASS, 8, jit=0.0, mod=2, on=0)
        self._strip(-(sh_out + 45), -sh_out, GRASS_B, 8, mod=2, on=1)
        self._strip(sh_out, sh_out + 45, GRASS, 8, mod=2, on=0)
        self._strip(sh_out, sh_out + 45, GRASS_B, 8, mod=2, on=1)
        self._strip(-sh_out, -sh_in, SHOULDER, 4)
        self._strip(sh_in, sh_out, SHOULDER, 4)
        self._strip(-sh_in, sh_in, ASPHALT, 4, jit=0.035)
        for sgn in (-1, 1):
            self._strip(sgn * (sh_in - 0.55), sgn * (sh_in - 0.25), WHITE, 4)
        for d in (-LANE_W / 2, LANE_W / 2):
            self._strip(d - 0.09, d + 0.09, WHITE, 2, mod=3, on=0)

    # --------------------------------------------------------------- objects
    def _objects(self):
        objs, h = [], self.half
        rd = h + 2.4
        for side in (-1, 1):
            d = side * rd
            b, t = self._ring(d, 0.45), self._ring(d, 0.95)
            p0, p1 = self._ring(d, 0.0), self._ring(d, 1.05)
            for k in range(len(self._ks) - 1):
                if min(b[2][k], b[2][k + 1]) < NEAR:
                    continue
                pts = [(b[0][k], b[1][k]), (b[0][k + 1], b[1][k + 1]), (t[0][k + 1], t[1][k + 1]), (t[0][k], t[1][k])]
                objs.append((float(b[2][k]), "rail", (pts, (self._ks[k] // 2) % 2)))
                if self._ks[k] % 4 == 0:
                    objs.append((float(p0[2][k]), "post", (p0[0][k], p0[1][k], p1[1][k], p0[2][k])))
            tl = self._ring(d, 9.0)
            for k in range(len(self._ks) - 1):
                if self._ks[k] % 12 == 0 and ((self._ks[k] // 12) % 2 == (0 if side < 0 else 1)):
                    if p0[2][k] < NEAR + 1 or tl[2][k] < NEAR + 1:
                        continue
                    ax, ay, _ = self.w.track.pose(self._ss[k], d - side * 2.6)
                    asx, asy, az = self._p1(float(ax), float(ay), 9.0)
                    if az > NEAR:
                        objs.append((float(p0[2][k]), "lamp", (p0[0][k], p0[1][k], tl[0][k], tl[1][k], asx, asy, p0[2][k])))
        # trees
        cp = self.cpos
        fx, fy = self.fwd[0], self.fwd[1]
        n = math.hypot(fx, fy)
        fx, fy = fx / n, fy / n
        rel = self._tx - cp[:2]
        zc = rel[:, 0] * fx + rel[:, 1] * fy
        xc = rel[:, 0] * self.right[0] + rel[:, 1] * self.right[1]
        m = np.nonzero((zc > 4) & (zc < FAR) & (np.abs(xc) < zc * 1.4 + 18))[0]
        if len(m):
            P = np.column_stack((self._tx[m], np.zeros(len(m))))
            sx, sy, zz = self.project(P)
            for j, i in enumerate(m):
                objs.append((float(zz[j]), "tree", (float(sx[j]), float(sy[j]), float(zz[j]), self._tinfo[i])))
        # signs / gantries
        L = self.w.track.length
        for s_sign, kind, side in self._signs:
            rel_s = (s_sign - self._ss[0]) % L
            if 8 < rel_s < FAR - 10:
                objs.append((rel_s - 14, kind, (s_sign, side)))
        # cars
        for tv in self.w.traffic:
            x, y, psi = self.w.track.pose(tv.s, tv.d(self.w.t))
            objs.append((float((np.array([x, y, 1.0]) - cp) @ self.fwd), "car",
                         (float(x), float(y), float(psi), tv.kind, tv.color, False, 0)))
        return objs

    # ----------------------------------------------------------------- draws
    def _draw_rail(self, pts, par, z):
        col = (178, 184, 192) if par == 0 else (140, 146, 156)
        pygame.draw.polygon(self.screen, fogc(col, z), pts)
        pygame.draw.line(self.screen, fogc((222, 226, 232), z), pts[3], pts[2], 1)

    def _draw_tree(self, sx, sy, z, info):
        _, _, conifer, size, tint = info
        k = self.f / z * size
        scr = self.screen
        if k < 1.2 or sx < -300 or sx > self.main_w + 300:
            return
        trunk = fogc((86, 64, 44), z)
        pygame.draw.polygon(scr, trunk, [(sx - .22 * k, sy), (sx + .22 * k, sy), (sx + .16 * k, sy - 2.4 * k), (sx - .16 * k, sy - 2.4 * k)])
        if conifer:
            for i, (wd, y0, hh) in enumerate(((2.1, 1.6, 3.0), (1.6, 3.4, 2.7), (1.05, 5.0, 2.4))):
                col = fogc(shade((34, 86, 56), tint * (1 + i * .1)), z)
                pygame.draw.polygon(scr, col, [(sx - wd * k, sy - y0 * k), (sx + wd * k, sy - y0 * k), (sx, sy - (y0 + hh) * k)])
        else:
            for dx, dy, r, f in ((0, 3.6, 2.2, 1.0), (-1.1, 3.0, 1.6, .9), (1.2, 3.1, 1.5, .95), (0.2, 4.4, 1.5, 1.12)):
                col = fogc(shade((64, 132, 70), tint * f), z)
                pygame.draw.circle(scr, col, (sx + dx * k, sy - dy * k), r * k)

    def _draw_lamp(self, x0, y0, x1, y1, ax, ay, z):
        w = max(1, int(self.f / z * 0.1))
        col = fogc((92, 96, 104), z)
        pygame.draw.line(self.screen, col, (x0, y0), (x1, y1), w)
        pygame.draw.line(self.screen, col, (x1, y1), (ax, ay), max(1, w - 1))
        pygame.draw.circle(self.screen, fogc((250, 244, 210), z), (ax, ay), max(2, int(self.f / z * 0.28)))

    def _draw_sign(self, s_sign, kind, side):
        tr, h = self.w.track, self.half
        if kind == "speed":
            d = side * (h + 3.6)
            x, y, _ = tr.pose(s_sign, d)
            bx, by, bz = self._p1(float(x), float(y), 0.0)
            tx_, ty_, _ = self._p1(float(x), float(y), 2.7)
            if bz < NEAR + 2:
                return
            k = self.f / bz
            w = max(1, int(.1 * k))
            pygame.draw.line(self.screen, fogc((120, 124, 132), bz), (bx, by), (tx_, ty_), w)
            r = 0.62 * k
            pygame.draw.circle(self.screen, fogc((200, 30, 36), bz), (tx_, ty_), r)
            pygame.draw.circle(self.screen, fogc((246, 246, 248), bz), (tx_, ty_), r * .78)
            if r > 9:
                ts = pygame.transform.smoothscale(self._speed_txt, (int(r * 1.0), int(r * 0.7)))
                self.screen.blit(ts, ts.get_rect(center=(int(tx_), int(ty_))))
        else:
            idx = int(round(s_sign / (tr.length / 3.0))) % 3
            corners = [(-6.2, 4.7), (6.2, 4.7), (6.2, 6.4), (-6.2, 6.4)]
            pts = []
            for dd, zz in corners:
                x, y, _ = tr.pose(s_sign, dd)
                px, py, pz = self._p1(float(x), float(y), zz)
                if pz < 4:
                    return
                pts.append((px, py, pz))
            z = sum(p[2] for p in pts) / 4
            for sgn in (-1, 1):
                x, y, _ = tr.pose(s_sign, sgn * (h + 2.9))
                a = self._p1(float(x), float(y), 0.0)
                b = self._p1(float(x), float(y), 6.5)
                pygame.draw.line(self.screen, fogc((96, 100, 108), z), a[:2], b[:2], max(2, int(.35 * self.f / z)))
            P2 = [(p[0], p[1]) for p in pts]
            pygame.draw.polygon(self.screen, fogc((22, 112, 74), z), P2)
            pygame.draw.polygon(self.screen, fogc(WHITE, z), P2, max(1, int(.06 * self.f / z)))
            wpx = abs(P2[1][0] - P2[0][0])
            if wpx > 40:
                src = self._gantry_txt[idx]
                sc = min(0.78 * wpx / src.get_width(), 0.55 * abs(P2[2][1] - P2[1][1]) / src.get_height())
                ts = pygame.transform.smoothscale(src, (max(2, int(src.get_width() * sc)), max(2, int(src.get_height() * sc))))
                c = ((P2[0][0] + P2[1][0]) / 2, (P2[0][1] + P2[2][1]) / 2)
                self.screen.blit(ts, ts.get_rect(center=(int(c[0]), int(c[1]))))

    # ------------------------------------------------------------------ cars
    def _draw_car(self, cx, cy, psi, kind, color, braking=False, blink=0, wire=None):
        boxes, wheels, tail_h = CAR_SPECS[kind]
        c, s = math.cos(psi), math.sin(psi)
        cp = self.cpos
        zc0 = float((np.array([cx, cy, 1.0]) - cp) @ self.fwd)
        if zc0 < NEAR + 2.5 or zc0 > FAR:
            return

        def W(x, y, z):
            return (cx + x * c - y * s, cy + x * s + y * c, z)

        scr = self.screen
        normals = {"rear": np.array([-c, -s, 0.0]), "front": np.array([c, s, 0.0]),
                   "left": np.array([-s, c, 0.0]), "right": np.array([s, -c, 0.0]), "top": np.array([0.0, 0.0, 1.0])}
        if wire is None:
            ln = boxes[0][1] - boxes[0][0] if kind != "truck" else 9.6
            wd = boxes[0][3] - boxes[0][2] if kind != "truck" else 2.5
            sx, sy, sz = [], [], []
            for lx, ly in ((-ln / 2 - 0.3, -wd / 2 - 0.2), (ln / 2 - 0.3, -wd / 2 - 0.2), (ln / 2 - 0.3, wd / 2 - 0.2), (-ln / 2 - 0.3, wd / 2 - 0.2)):
                x, y, _ = W(lx, ly, 0.0)
                sx.append(x - 0.45); sy.append(y - 0.3)
            P = np.array([[x, y, 0.02] for x, y in zip(sx, sy)])
            px, py, pz = self.project(P)
            if (pz > NEAR).all():
                pygame.draw.polygon(scr, fogc((38, 40, 44), zc0), list(zip(px.tolist(), py.tolist())))

        for (x0, x1, y0, y1, z0, z1, role) in boxes:
            loc = [(xx, yy, zz) for xx in (x0, x1) for yy in (y0, y1) for zz in (z0, z1)]
            P = np.array([W(*p) for p in loc])
            sx, sy, zc = self.project(P)
            if (zc < NEAR).any():
                continue
            if wire is not None:
                for a, b in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
                    pygame.draw.line(scr, wire, (sx[a], sy[a]), (sx[b], sy[b]), max(2, int(self.ui * 2)))
                continue
            faces = []
            for name, idx in FACES:
                n = normals[name]
                ctr = P[list(idx)].mean(axis=0)
                if n @ (cp - ctr) <= 0:
                    continue
                faces.append((float(zc[list(idx)].mean()), name, idx, float(n @ SUN)))
            for z, name, idx, lit in sorted(faces, key=lambda f: -f[0]):
                L = 0.55 + 0.5 * max(0.0, lit)
                if role in ("body", "cab"):
                    col = shade(color, L)
                elif role == "cabin":
                    col = shade(color, L * 0.9) if name == "top" else shade((30, 40, 54), 0.7 + L * 0.5)
                elif role == "trailer":
                    col = shade((228, 230, 234), L)
                else:
                    col = shade((46, 48, 54), L)
                pts = [(sx[i], sy[i]) for i in idx]
                pygame.draw.polygon(scr, fogc(col, z), pts)
                if name == "top":
                    pygame.draw.polygon(scr, fogc(shade(col, 1.18), z), pts, 1)
            if role in ("body", "chassis") or (role == "trailer"):
                pass
        # wheels on the visible side
        left_vis = (np.array([-s, c, 0.0]) @ (cp - np.array([cx, cy, 0.0]))) > 0
        sgn = 1 if left_vis else -1
        wy = (boxes[0][3] if kind != "truck" else 0.95) * sgn
        if wire is None:
            for wx, w0, w1 in wheels:
                loc = [(wx - 0.36, wy, 0.0), (wx + 0.36, wy, 0.0), (wx + 0.36, wy, w1), (wx - 0.36, wy, w1)]
                P = np.array([W(*p) for p in loc])
                sx, sy, zc = self.project(P)
                if (zc > NEAR).all():
                    pygame.draw.polygon(scr, fogc((20, 20, 24), zc0), list(zip(sx.tolist(), sy.tolist())))
            # tail / head lights
            rear_vis = (normals["rear"] @ (cp - np.array([cx, cy, 0.0]))) > 0
            front_vis = (normals["front"] @ (cp - np.array([cx, cy, 0.0]))) > 0
            ln2 = boxes[0][1] if kind != "truck" else boxes[1][0] * -1 + 0.0
            xr = (boxes[0][0] if kind != "truck" else -4.8) - 0.01
            xf = (boxes[0][1] if kind != "truck" else 4.8) + 0.01
            half_w = boxes[0][3] if kind != "truck" else 1.2
            for (xx, vis, lit_col, dark_col) in ((xr, rear_vis, (255, 36, 30) if braking else (200, 24, 24), (150, 20, 20)),
                                                    (xf, front_vis, (255, 248, 214), (225, 220, 190))):
                if not vis:
                    continue
                for sgn2 in (-1, 1):
                    loc = [(xx, sgn2 * (half_w - 0.08), tail_h - 0.2), (xx, sgn2 * (half_w - 0.48), tail_h - 0.2),
                           (xx, sgn2 * (half_w - 0.48), tail_h - 0.02), (xx, sgn2 * (half_w - 0.08), tail_h - 0.02)]
                    P = np.array([W(*p) for p in loc])
                    sx, sy, zc = self.project(P)
                    if (zc > NEAR).all():
                        pygame.draw.polygon(scr, lit_col, list(zip(sx.tolist(), sy.tolist())))
                if xx == xr:
                    loc = [(xx, -0.3, 0.42), (xx, 0.3, 0.42), (xx, 0.3, 0.55), (xx, -0.3, 0.55)]
                    P = np.array([W(*p) for p in loc])
                    sx, sy, zc = self.project(P)
                    if (zc > NEAR).all():
                        pygame.draw.polygon(scr, (236, 236, 230), list(zip(sx.tolist(), sy.tolist())))
            if blink and rear_vis:
                for ssd in ((-blink),):
                    x, y, z = W(xr, ssd * (half_w - 0.12), tail_h - 0.38)
                    px, py, pz = self._p1(x, y, z)
                    if pz > NEAR:
                        pygame.draw.circle(scr, (255, 176, 0), (px, py), max(2, int(self.f / pz * 0.1)))

    # ----------------------------------------------------------- ground marks
    def _ground_overlays(self, show):
        w, tr, scr = self.w, self.w.track, self.screen
        e = w.mpc
        s_ref = e.s_total + np.arange(-4.0, 90.0, 2.0)
        x, y, _ = tr.pose(s_ref, w.plan.d_ref(s_ref))
        sx, sy, zc = self.project(np.column_stack((x, y, np.full_like(x, 0.03))))
        for j in range(0, len(sx) - 1, 2):
            if zc[j] > NEAR and zc[j + 1] > NEAR:
                pygame.draw.line(scr, fogc(REF_COL, zc[j]), (sx[j], sy[j]), (sx[j + 1], sy[j + 1]), max(2, int(self.ui * 3)))
        if show & 2 and w.mpc.info.get("pred"):
            P = np.array([[px, py, 0.05] for px, py in w.mpc.info["pred"]])
            sx, sy, zc = self.project(P)
            ok = zc > NEAR
            pts = list(zip(sx[ok].tolist(), sy[ok].tolist()))
            if len(pts) > 1:
                pygame.draw.lines(scr, MPC_COL, False, pts, max(3, int(self.ui * 6)))
        if show & 1 and w.pp.info.get("target"):
            tx_, ty_ = w.pp.info["target"]
            ang = np.linspace(0, 2 * math.pi, 24)
            P = np.column_stack((tx_ + 1.3 * np.cos(ang), ty_ + 1.3 * np.sin(ang), np.full_like(ang, 0.05)))
            sx, sy, zc = self.project(P)
            if (zc > NEAR).all():
                pygame.draw.lines(scr, PP_COL, True, list(zip(sx.tolist(), sy.tolist())), max(3, int(self.ui * 4)))
                a = self._p1(w.pp.st.x, w.pp.st.y, 0.05)
                b = self._p1(tx_, ty_, 0.05)
                pygame.draw.line(scr, PP_COL, a[:2], b[:2], max(2, int(self.ui * 2)))

    # ----------------------------------------------------------------- draw
    def draw(self, alpha=1.0, paused=False, speed=1.0, show=3):
        w, tr, scr = self.w, self.w.track, self.screen
        name, back, height, pitch_deg = self.MODES[self.mode]
        ego = w.pp if show == 1 else w.mpc
        rh = float(tr.pose(ego.s_total + 14.0)[2])
        vx = 0.5 * math.cos(ego.st.psi) + 0.5 * math.cos(rh)
        vy = 0.5 * math.sin(ego.st.psi) + 0.5 * math.sin(rh)
        target = math.atan2(vy, vx) if self.mode != 2 else ego.st.psi
        if self.yaw is None:
            self.yaw = target
        self.yaw += ((target - self.yaw + math.pi) % (2 * math.pi) - math.pi) * (0.12 if self.mode != 2 else 0.35)
        pitch = math.radians(pitch_deg)
        yc, ys = math.cos(self.yaw), math.sin(self.yaw)
        base_x = ego.st.x + (EGO_FWD if self.mode == 2 else 0.0) * math.cos(ego.st.psi)
        base_y = ego.st.y + (EGO_FWD if self.mode == 2 else 0.0) * math.sin(ego.st.psi)
        self._set_camera((base_x - back * yc, base_y - back * ys, height), self.yaw, pitch)
        ground, pano, hy = self._background(pitch)

        self._rc = {}
        s0 = ego.s_total - max(back, 0.0) - 8.0
        k0 = int(math.floor(s0 / STEP))
        self._ks = k0 + np.arange(int(FAR / STEP) + 1)
        self._ss = self._ks * STEP

        scr.blit(ground, (0, 0))
        x0 = int((self.cx + self.f * self.yaw) % pano.get_width()) - pano.get_width()
        scr.blit(pano, (x0, 0))
        scr.blit(pano, (x0 + pano.get_width(), 0))
        self._ground()
        self._ground_overlays(show)

        objs = self._objects()
        for ego_, on, color in ((w.pp, show == 1, PP_COL), (w.mpc, show & 2, MPC_COL)):
            if not on or (self.mode == 2 and ego_ is ego):
                continue
            st = ego_.st
            ccx, ccy = st.x + EGO_FWD * math.cos(st.psi), st.y + EGO_FWD * math.sin(st.psi)
            blink = 0
            if w.plan.change and int(w.t * 3) % 2 == 0:
                blink = 1 if w.plan.change[3] > w.plan.change[2] else -1
            objs.append((float((np.array([ccx, ccy, 1.0]) - self.cpos) @ self.fwd), "car",
                         (ccx, ccy, st.psi, "sedan", color, ego_.a_cmd < -0.5, blink)))
        objs.sort(key=lambda o: -o[0])
        for z, kind, data in objs:
            if kind == "rail":
                self._draw_rail(data[0], data[1], z)
            elif kind == "post":
                sx, sy0, sy1, zz = data
                pygame.draw.line(scr, fogc((120, 124, 132), zz), (sx, sy0), (sx, sy1), max(1, int(self.f / zz * 0.1)))
            elif kind == "lamp":
                self._draw_lamp(*data)
            elif kind == "tree":
                self._draw_tree(*data)
            elif kind in ("speed", "gantry"):
                self._draw_sign(data[0], kind, data[1])
            elif kind == "car":
                self._draw_car(*data)

        if show == 3 and not (self.mode == 2):                  # Pure Pursuit shown as a wireframe ghost
            st = w.pp.st
            ccx, ccy = st.x + EGO_FWD * math.cos(st.psi), st.y + EGO_FWD * math.sin(st.psi)
            self._draw_car(ccx, ccy, st.psi, "sedan", PP_COL, wire=PP_COL)
        for ego_, on, col, label in ((w.pp, show & 1, PP_COL, "PP"), (w.mpc, show & 2, MPC_COL, "MPC")):
            if not on or (self.mode == 2 and ego_ is ego):
                continue
            st = ego_.st
            px, py, pz = self._p1(st.x + EGO_FWD * math.cos(st.psi), st.y + EGO_FWD * math.sin(st.psi), 2.5)
            if pz > NEAR + 2 and 0 < px < self.main_w:
                lbl = self.f_s.render(label, True, (14, 14, 22))
                r = lbl.get_rect(center=(int(px), int(py))).inflate(int(12 * self.ui), int(6 * self.ui))
                pygame.draw.rect(scr, col, r)
                scr.blit(lbl, lbl.get_rect(center=(int(px), int(py))))
        scr.blit(self._vig, (0, 0))
        self._draw_overlay(paused, speed)
        self._draw_hud(show)

    # ------------------------------------------------------------- overlays
    def _draw_overlay(self, paused, speed):
        w, ui, scr = self.w, self.ui, self.screen
        pad = int(18 * ui)
        panel = pygame.Surface((int(300 * ui), int(150 * ui)), pygame.SRCALPHA)
        panel.fill((10, 12, 18, 190))
        scr.blit(panel, (pad, pad))
        put(scr, self.f_xl, f"{w.mpc.st.v * 3.6:3.0f}", TEXT, (pad + 14 * ui, pad + 8 * ui))
        put(scr, self.f_m, "km/h", DIM, (pad + 118 * ui, pad + 34 * ui))
        col = {"OVERTAKING": (255, 190, 70), "RETURNING": (255, 190, 70),
               "FOLLOWING": (255, 120, 110), "CRUISING": (120, 230, 160)}[w.state_name]
        put(scr, self.f_b, w.state_name, col, (pad + 14 * ui, pad + 70 * ui))
        gap = f"{w.gap:5.1f} m" if w.gap is not None and w.gap < 200 else "  --  "
        put(scr, self.f_s, f"LANE {w.plan.target_lane + 1}/3   GAP {gap}", TEXT, (pad + 14 * ui, pad + 100 * ui))
        put(scr, self.f_s, f"LAP {w.laps:4.2f}  OVERTAKES {w.overtakes}  CAM {self.MODES[self.mode][0]}", DIM, (pad + 14 * ui, pad + 120 * ui))
        if paused:
            put(scr, self.f_b, "PAUSED", (255, 200, 90), (self.main_w / 2, 40 * ui), center=True)
        if w.flash > 0:
            put(scr, self.f_b, "DISTURBANCE!", (255, 90, 90), (self.main_w / 2, 80 * ui), center=True)
        mw, mh = self._mini_size
        mx, my = self.main_w - mw - pad, pad
        scr.blit(self._mini, (mx, my))
        tr = w.track
        for tv in w.traffic:
            x, y, _ = tr.pose(tv.s, 0.0)
            pygame.draw.circle(scr, (150, 156, 170), (mx + self._mo[0] + x * self._ms, my + self._mo[1] - y * self._ms), max(2, int(3 * ui)))
        for ego in (w.pp, w.mpc):
            pygame.draw.circle(scr, ego.color, (mx + self._mo[0] + ego.st.x * self._ms, my + self._mo[1] - ego.st.y * self._ms), max(3, int(5 * ui)))

    # ------------------------------------------------------------------ HUD
    def _plot(self, rect, title, unit, series, ymin, ymax, symmetric=True, kicks=()):
        scr, ui, h = self.screen, self.ui, self.w.hist
        pygame.draw.rect(scr, PLOT_BG, rect)
        pygame.draw.rect(scr, GRID, rect, 1)
        put(scr, self.f_s, f"{title} [{unit}]", TEXT, (rect.x + 8 * ui, rect.y + 4 * ui))
        t = np.array(h["t"])
        if len(t) < 2:
            return
        win = 30.0
        vals = [np.array(h[k]) for k, _ in series]
        lo, hi = ymin, ymax
        m = max(float(np.max(np.abs(v))) for v in vals if len(v))
        if symmetric:
            hi = max(ymax, m * 1.15)
            lo = -hi
        else:
            lo, hi = min(ymin, min(float(v.min()) for v in vals) - 1), max(ymax, max(float(v.max()) for v in vals) + 1)
        def Y(v):
            return rect.bottom - (v - lo) / (hi - lo) * rect.h
        if lo < 0 < hi:
            pygame.draw.line(scr, GRID, (rect.x, Y(0)), (rect.right, Y(0)), 1)
        put(scr, self.f_s, f"{hi:.1f}", DIM, (rect.right - 44 * ui, rect.y + 4 * ui))
        put(scr, self.f_s, f"{lo:.1f}", DIM, (rect.right - 44 * ui, rect.bottom - 18 * ui))
        now = t[-1]
        for kt in kicks:
            if now - kt < win:
                x = rect.right - (now - kt) / win * rect.w
                pygame.draw.line(scr, (255, 90, 90), (x, rect.y), (x, rect.bottom), 1)
        for (key, color), v in zip(series, vals):
            x = rect.right - (now - t) / win * rect.w
            ok = x >= rect.x
            pts = list(zip(x[ok].tolist(), np.clip(Y(v[ok]), rect.y, rect.bottom).tolist()))
            if len(pts) > 1:
                pygame.draw.lines(scr, color, False, pts, max(2, int(2 * ui)))

    def _draw_hud(self, show):
        w, ui, scr = self.w, self.ui, self.screen
        x0 = self.main_w
        pygame.draw.rect(scr, HUD_BG, (x0, 0, self.hud_w, self.H))
        pygame.draw.line(scr, GRID, (x0, 0), (x0, self.H), max(1, int(2 * ui)))
        pad = int(18 * ui)
        y = int(14 * ui)
        put(scr, self.f_b, "SELF-DRIVING SIM", TEXT, (x0 + pad, y))
        y += int(30 * ui)
        put(scr, self.f_s, "Same road, same traffic, same lane-change plan", DIM, (x0 + pad, y))
        y += int(26 * ui)
        colw = (self.hud_w - 2 * pad) // 3
        fh = self.f_m.get_height()
        for j, (name, col) in enumerate((("Pure Pursuit", PP_COL), ("MPC", MPC_COL))):
            pygame.draw.rect(scr, col, (x0 + pad + colw * (j + 1), y + 2, int(12 * ui), int(12 * ui)))
            put(scr, self.f_s, name, col, (x0 + pad + colw * (j + 1) + int(16 * ui), y))
        y += int(24 * ui)
        rows = [("RMS error", lambda e: f"{w.rms(e):.3f} m"),
                ("Max error", lambda e: f"{e.max_e:.3f} m"),
                ("Steer rate", lambda e: f"{w.steer_rate_rms(e):.1f} d/s")]
        for label, fn in rows:
            put(scr, self.f_m, label, DIM, (x0 + pad, y))
            for j, e in enumerate((w.pp, w.mpc)):
                put(scr, self.f_m, fn(e), TEXT, (x0 + pad + colw * (j + 1), y))
            y += fh + int(4 * ui)
        put(scr, self.f_m, "Collisions", DIM, (x0 + pad, y))
        put(scr, self.f_m, str(w.collisions), (255, 120, 120) if w.collisions else TEXT, (x0 + pad + colw * 2, y))
        put(scr, self.f_s, f"MPC solve {w.mpc.ctrl.solve_ms:.2f} ms", DIM, (x0 + pad + colw * 2 + int(40 * ui), y + 2))
        y += fh + int(14 * ui)

        foot = int(70 * ui)
        avail = self.H - y - foot
        n = 4
        gap = int(10 * ui)
        ph = (avail - (n - 1) * gap) // n
        pw = self.hud_w - 2 * pad
        kicks = list(w.kicks)
        specs = [("Cross-track error", "m", (("ey_pp", PP_COL), ("ey_mpc", MPC_COL)), 0.5, 0.5, True),
                 ("Heading error", "deg", (("ep_pp", PP_COL), ("ep_mpc", MPC_COL)), 3.0, 3.0, True),
                 ("Steering angle", "deg", (("d_pp", PP_COL), ("d_mpc", MPC_COL)), 5.0, 5.0, True),
                 ("Speed (grey = target)", "km/h", (("vref", (110, 118, 140)), ("v_mpc", MPC_COL)), 0.0, 90.0, False)]
        for s in specs:
            self._plot(pygame.Rect(x0 + pad, y, pw, ph), s[0], s[1], s[2], s[3], s[4], s[5], kicks)
            y += ph + gap
        hints = "SPACE pause  +/- speed  D disturb  1/2/3 car  C camera  R reset"
        put(scr, self.f_s, hints, DIM, (x0 + pad, self.H - int(26 * ui)))
        put(scr, self.f_s, "Red lines = disturbance (heading kick)", (255, 120, 120), (x0 + pad, self.H - int(48 * ui)))
