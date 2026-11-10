# Self-Driving Sim: Pure Pursuit vs MPC with Lane Changes

A real-time 3D simulator where a self-driving car laps a three-lane highway with dense traffic,
follows a planned path, overtakes slower cars and trucks with smooth lane changes, and is controlled
by **two path-tracking controllers running side by side**: **Pure Pursuit** and
**Model Predictive Control (MPC)**. Live plots show how each one performs, and you
can kick both cars with a "crosswind" disturbance to watch them recover.

![demo](demo.gif)

> Record your own demo: `python main.py --save demo.mp4` (or `demo.gif`)

## Features

- **Kinematic bicycle model** with steering-angle, steering-rate and acceleration limits
- **Pure Pursuit** with speed-adaptive look-ahead
- **Linear MPC** (20-step, 2 s horizon) with curvature feed-forward, steering-limit constraints, and a hand-written condensed QP solver (closed form + accelerated projected gradient), no external optimisation library
- **Behaviour planner**: adaptive cruise control, overtaking and returning to the slow lane; lane changes are quintic (jerk-minimising) blends in the Frenet frame
- **Live error plots**: cross-track error, heading error, steering angle and speed for both controllers, with disturbance markers
- **Dense traffic**: 24 vehicles (sedans, SUVs and trucks) in three lanes at different speeds, with natural lateral wander, a collision monitor and a curvature-limited speed profile
- **Realistic 3D environment** drawn with a software pinhole camera (no OpenGL): sky with sun and clouds, hills, forest, guardrails, street lamps, speed signs, overhead gantries, distance fog, shadows, lit 3D cars with tail lights and turn signals
- **Three camera angles** (`C` to switch): chase, high drone-style, and cockpit, plus the predicted MPC trajectory, Pure Pursuit look-ahead point, reference path and a minimap
- MP4 / GIF export (no screen recorder needed), headless benchmark, unit tests

## Quick start

```bash
git clone https://github.com/pirsaab1/self-driving-sim.git
cd self-driving-sim
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python main.py
```

| Command | What it does |
|---|---|
| `python main.py --res 720p` | smaller window (`720p`, `1080p`, `1440p`, `4k`) |
| `python main.py --camera high` | start with the drone-style camera |
| `python main.py --save demo.mp4` | render a 60 s MP4 with automatic disturbances |
| `python main.py --save demo.gif --seconds 15` | render a GIF |
| `python main.py --bench 300 --disturb 14` | headless benchmark of both controllers |
| `python -m unittest discover tests` | run tests |

Controls: `SPACE` pause, `+` / `-` sim speed, `D` disturbance kick, `C` camera angle, `1` / `2` / `3` show Pure Pursuit / MPC / both, `R` reset, `ESC` quit.

## How it works

**Vehicle.** Rear-axle kinematic bicycle: `x' = v cos(psi)`, `y' = v sin(psi)`, `psi' = v tan(delta) / L`, with a rate-limited steering actuator.

**Road and planning.** The road is a closed centre line with Frenet coordinates `(s, d)`. The planner decides the target lane; a lane change is `d(s) = d0 + (d1 - d0) * (10t^3 - 15t^4 + 6t^5)`, `t = (s - s0) / length`, which has zero lateral velocity and acceleration at both ends. Speed follows `min(cruise, curvature limit, ACC gap controller)`.

**Pure Pursuit.** Picks a point one look-ahead distance `Ld = 4 + 0.55 v` ahead on the path and steers along the arc to it: `delta = atan(2 L sin(alpha) / Ld)`.

**MPC.** Error dynamics around the reference path, with `u = delta - delta_ff` and `delta_ff = atan(L kappa)`:

```
e_y   <- e_y   + dt * v * e_psi
e_psi <- e_psi + dt * (v / L) / cos^2(delta_ff) * u
```

The horizon is condensed into `min 0.5 U'HU + f'U` with cost weights on lateral error, heading error, steering effort and steering change, subject to the steering limit. The solve takes about 0.25 ms.

## Results

300 s per run (about 3.9 laps and 9 to 10 overtakes through 24 vehicles), averaged over 5 random traffic seeds, 0 collisions in every run (minimum bumper gap 20 m). Cross-track error measured at the rear axle.

| Scenario | Controller | RMS error | Max error | Steering rate (RMS) |
|---|---|---:|---:|---:|
| Lane changes only | Pure Pursuit | 0.075 m | 0.39 m | 1.2 deg/s |
| Lane changes only | **MPC** | **0.006 m** | **0.02 m** | 1.2 deg/s |
| + heading kick every 14 s | Pure Pursuit | 0.148 m | 0.65 m | 3.6 deg/s |
| + heading kick every 14 s | **MPC** | **0.036 m** | **0.29 m** | 16.1 deg/s |

MPC tracks about 4 to 12 times more accurately; the price is a much more aggressive steering reaction to disturbances.

**Limitations.** The plant is the same kinematic model the MPC uses (no tyre slip, delay, or sensor noise), so the MPC advantage is an upper bound. See the roadmap.

## Project structure

```
main.py                 CLI, window loop, MP4/GIF export, benchmark
driving/track.py        closed road, lanes, Frenet transforms
driving/vehicle.py      kinematic bicycle model with limits
driving/controllers.py  Pure Pursuit and MPC
driving/sim.py          traffic, behaviour planner, ego cars, metrics
driving/render.py       3D chase-camera renderer, environment, plots, HUD
tests/                  unit tests
```

## Roadmap

- [ ] Dynamic bicycle model with tyre slip, plus sensor noise and actuation delay
- [ ] Nonlinear MPC and comparison with Stanley and LQR controllers
- [ ] Obstacle avoidance with lattice or optimisation-based local planner
- [ ] Multi-vehicle interaction (other cars react to the ego car)
- [ ] Hills and banked road, rain and night lighting

## License

MIT
