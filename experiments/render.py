"""Render the machine with MuJoCo's offscreen renderer.

    python experiments/render.py            -> media/launch.gif, media/knuckleballs.gif

launch.gif: a curveball going through the three wheels, about 250x slow motion.
knuckleballs.gif: five knuckleballs from the catcher's view, identical machine
settings, ball dropped into the chute a different way each time.
"""

import imageio.v2 as iio
import mujoco
import numpy as np
from _style import MEDIA, ROOT
from PIL import Image, ImageDraw, ImageFont

from pitching_machine.aero import Aero, random_rotations
from pitching_machine.constants import MPH, PLATE_DISTANCE, RPM
from pitching_machine.flight import fly
from pitching_machine.machine import Machine
from pitching_machine.pitch import preset_config, throw
from plate_scatter import chute_composer

W, H = 640, 360


def font(size):
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def caption(img, lines, size=17):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = font(size)
    y = 10
    for line in lines:
        d.text((13, y + 1), line, font=f, fill=(0, 0, 0))
        d.text((12, y), line, font=f, fill=(255, 255, 255))
        y += size + 5
    return np.asarray(im)


def save_gif(frames, name, fps):
    MEDIA.mkdir(exist_ok=True)
    path = MEDIA / name
    iio.mimsave(path, frames, duration=1000 / fps, loop=0)
    print(f"saved {path.relative_to(ROOT)} ({len(frames)} frames, {path.stat().st_size / 1e6:.1f} MB)")


def render_launch():
    cfg = preset_config("curveball")
    m = Machine(cfg, scene=True)
    r = mujoco.Renderer(m.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, cfg.release_height]
    cam.distance, cam.azimuth, cam.elevation = 1.0, 205.0, -10.0  # in front of the machine, off to the 3B side
    every = 20  # 0.2 ms of sim time per frame
    frames, after = [], None
    m.reset()
    while m.data.time < cfg.max_time:
        for _ in range(every):
            m.step()
        pos, vel, omega, _ = m.ball_state()
        if after is None and not m.contacts and pos[0] > 0:
            after = m.data.time
        if after is not None and m.data.time - after > 0.004:
            break
        r.update_scene(m.data, cam)
        spin = omega @ (m.R_head @ cfg.spin_axis_head) / RPM
        frames.append(caption(r.render(), [
            f"t = {1e3 * m.data.time:5.1f} ms   (~250x slow motion)",
            f"ball {np.linalg.norm(vel) / MPH:5.1f} mph   spin {spin:+5.0f} rpm",
            "wheels " + " / ".join(f"{rpm:.0f}" for rpm in cfg.wheel_rpm) + " rpm",
        ]))
    frames += [frames[-1]] * 20  # hold the last frame
    save_gif(frames, "launch.gif", fps=20)


def render_knuckleballs(n=5, seed=4):
    cfg = preset_config("knuckleball")
    res = throw(cfg).launch
    chutes = random_rotations(n, np.random.default_rng(seed))
    R_exit = chute_composer(res)(chutes)
    f = fly(np.tile(res.position, (n, 1)), np.tile(res.velocity, (n, 1)), np.tile(res.omega, (n, 1)), R_exit, Aero())

    m = Machine(res.config, scene=True)
    d = m.data
    r = mujoco.Renderer(m.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [PLATE_DISTANCE - 5.5, 0.0, 1.0]
    cam.distance, cam.azimuth, cam.elevation = 8.0, 180.0, -3.0  # behind home plate, looking out
    colors = np.array([[0.93, 0.63, 0.0, 1], [0.92, 0.41, 0.2, 1], [0.16, 0.47, 0.84, 1], [0.11, 0.69, 0.48, 1],
                       [0.91, 0.48, 0.64, 1]], dtype=np.float32)
    frames = []
    t_end = np.nanmax(f.plate_time)
    for t in np.arange(0.0, t_end + 0.12, 0.012):  # 12 ms per frame at 25 fps: about 1/3 speed
        k = min(np.searchsorted(f.t, t), len(f.t) - 1)
        for i in m.wheels:
            d.qpos[m.qadr[i]] = m.wheel_targets[i] * t
        # Park the model's own ball out of sight; draw all five as scene spheres.
        d.qpos[m.ball_qadr : m.ball_qadr + 3] = [0.0, 0.0, -5.0]
        mujoco.mj_forward(m.model, d)
        r.update_scene(d, cam)
        scn = r.scene
        for b in range(n):
            reached = ~np.isnan(f.plate_time[b]) and t > f.plate_time[b]
            kb = np.searchsorted(f.t, f.plate_time[b]) if reached else k
            trail = f.pos[: kb + 1 : 4, b]
            for j, p in enumerate(trail):
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.014, 0, 0]),
                                    p, np.eye(3).ravel(), colors[b] * np.array([1, 1, 1, 0.55], dtype=np.float32))
                scn.ngeom += 1
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0369, 0, 0]),
                                    f.pos[kb, b], f.R[kb, b].ravel(), np.array([0.97, 0.96, 0.92, 1], dtype=np.float32))
                scn.ngeom += 1
        frames.append(caption(r.render(), [
            "5 knuckleballs, identical machine settings",
            f"only the ball's seams in the chute differ   t = {1e3 * t:3.0f} ms",
        ], size=16))
    frames += [frames[-1]] * 25
    save_gif(frames, "knuckleballs.gif", fps=25)


if __name__ == "__main__":
    render_launch()
    render_knuckleballs()
