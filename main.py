"""Self-Driving Sim - entry point.

  python main.py                       interactive window (1080p)
  python main.py --res 4k              4K canvas
  python main.py --camera high         start with the high (drone-style) camera; press C to switch
  python main.py --save demo.mp4       render an MP4 (no window, 60 s by default)
  python main.py --save demo.gif       render a GIF instead
  python main.py --bench 300           headless benchmark: Pure Pursuit vs MPC
"""
import argparse
import os
import sys

from driving.sim import CTRL_EVERY, DT, World

RES = {"4k": (3840, 2160), "1440p": (2560, 1440), "1080p": (1920, 1080), "720p": (1280, 720)}


def parse():
    ap = argparse.ArgumentParser(description="Pure Pursuit vs MPC lane-change simulator")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", choices=list(RES), default=None, help="canvas size (default 1080p)")
    ap.add_argument("--disturb", type=float, default=None,
                    help="seconds between automatic heading kicks (default: 14 for --save, off otherwise)")
    ap.add_argument("--camera", choices=["chase", "high", "cockpit"], default="chase", help="camera angle")
    ap.add_argument("--save", metavar="FILE", help="export demo.mp4 or demo.gif and exit")
    ap.add_argument("--seconds", type=float, default=None, help="length of the export (default 60 mp4 / 20 gif)")
    ap.add_argument("--bench", type=float, metavar="SECONDS", help="headless benchmark")
    return ap.parse_args()


def bench(args):
    w = World(seed=args.seed, disturb_period=args.disturb or 0.0)
    for _ in range(int(args.bench / DT)):
        w.step()
    print(f"time={args.bench:.0f}s laps={w.laps:.2f} overtakes={w.overtakes} "
          f"collisions={w.collisions} min_gap={w.min_gap:.1f} m")
    for e in w.egos:
        print(f"{e.name:13s} rms={w.rms(e):.3f} m  max={e.max_e:.3f} m  "
              f"steer_rate_rms={w.steer_rate_rms(e):.1f} deg/s")
    print(f"MPC solve time: {w.mpc.ctrl.solve_ms:.2f} ms")


def save_video(args):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from driving.render import Renderer
    mp4 = args.save.lower().endswith(".mp4")
    secs = args.seconds or (60.0 if mp4 else 20.0)
    res = RES[args.res or ("1080p" if mp4 else "720p")]
    w = World(seed=args.seed, disturb_period=args.disturb if args.disturb is not None else 14.0)
    rend = Renderer(w, show_window=False, resolution=res)
    rend.mode = ["chase", "high", "cockpit"].index(args.camera)
    fps = 30 if mp4 else 15
    steps_per_frame = int(round(1.0 / DT / fps))
    frames = int(secs * fps)
    W, H = rend.size
    if mp4:
        try:
            import imageio.v2 as imageio
            import numpy as np
        except ImportError:
            sys.exit("MP4 export needs:  pip install imageio imageio-ffmpeg")
        writer = imageio.get_writer(args.save, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=2, pixelformat="yuv420p")
    else:
        from PIL import Image
        imgs = []
    for f in range(frames):
        for _ in range(steps_per_frame):
            w.step()
        rend.draw()
        if mp4:
            writer.append_data(np.frombuffer(rend.to_bytes(), dtype=np.uint8).reshape(H, W, 3))
        else:
            imgs.append(Image.frombytes("RGB", rend.size, rend.to_bytes())
                        .convert("P", palette=Image.ADAPTIVE, colors=96))
        print(f"\rrecording {f + 1}/{frames}", end="")
    if mp4:
        writer.close()
    else:
        imgs[0].save(args.save, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / fps), loop=0, optimize=True)
    print(f"\nsaved {args.save}")


def run_window(args):
    import pygame
    from driving.render import Renderer
    w = World(seed=args.seed, disturb_period=args.disturb or 0.0)
    rend = Renderer(w, show_window=True, resolution=RES[args.res or "1080p"])
    rend.mode = ["chase", "high", "cockpit"].index(args.camera)
    clock = pygame.time.Clock()
    speed, acc, paused, show = 1.0, 0.0, False, 3
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_UP):
                    speed = min(8.0, speed * 1.5)
                elif e.key in (pygame.K_MINUS, pygame.K_DOWN):
                    speed = max(0.25, speed / 1.5)
                elif e.key == pygame.K_d:
                    w.disturb()
                elif e.key == pygame.K_c:
                    rend.cycle_camera()
                elif e.key == pygame.K_1:
                    show = 1
                elif e.key == pygame.K_2:
                    show = 2
                elif e.key == pygame.K_3:
                    show = 3
                elif e.key == pygame.K_r:
                    w = World(seed=args.seed + int(w.t) + 1, disturb_period=args.disturb or 0.0)
                    rend.w = w
        dt = clock.tick(60) / 1000.0
        if not paused:
            acc += min(dt, 0.1) * speed
            while acc >= DT:
                acc -= DT
                w.step()
        rend.draw(paused=paused, speed=speed, show=show)
        pygame.display.flip()


if __name__ == "__main__":
    a = parse()
    if a.bench:
        bench(a)
    elif a.save:
        save_video(a)
    else:
        run_window(a)
