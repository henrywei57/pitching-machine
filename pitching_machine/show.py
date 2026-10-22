"""Play several pitches at the same time, each ball and trail in its own color.

Every pitch is already simulated (its launch and its flight are recorded), so
this only animates them on one shared clock: all the balls go into the wheels
together in slow motion, then fly to the plate at half speed. The machine on
screen shows the first pitch's wheel speeds and aim.

    play([a, b, c], ["orange", "blue", "green"])        # MuJoCo viewer (mjpython on macOS)
    save_gif("pitches.gif", [a, b], ["red", "blue"])    # offscreen, no window needed
"""

import time

import mujoco
import numpy as np

from .constants import BALL_RADIUS, FOOT, INCH, MPH, RPM
from .machine import Machine

COLORS = {
    "orange": (0.92, 0.41, 0.20), "blue": (0.16, 0.47, 0.84), "green": (0.11, 0.69, 0.48),
    "yellow": (0.93, 0.63, 0.00), "pink": (0.91, 0.48, 0.64), "purple": (0.38, 0.31, 0.84),
    "red": (0.89, 0.29, 0.28), "white": (0.97, 0.96, 0.92), "black": (0.08, 0.08, 0.08),
    "gray": (0.55, 0.55, 0.55),
}
# Handed out in this order to pitches that don't pick a color.
COLOR_ORDER = ["orange", "blue", "green", "yellow", "pink", "purple", "red", "white", "gray", "black"]

SLOWMO = 250.0  # launch played this many times slower than real
FLIGHT_SPEED = 0.5  # flight played at this fraction of real speed


def rgb(color):
    """A color name ('red') or hex ('#ff8800') -> (r, g, b) in 0..1."""
    if isinstance(color, str) and color.lower() in COLORS:
        return COLORS[color.lower()]
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    names = ", ".join(COLOR_ORDER)
    raise ValueError(f"unknown color {color!r}: use one of {names}, or a hex code like '#ff8800'")


def pick_colors(wanted):
    """Fill in the colors nobody asked for, skipping ones already taken."""
    taken = {w.lower() for w in wanted if isinstance(w, str)}
    spare = iter(c for c in COLOR_ORDER if c not in taken)
    return [w if w is not None else next(spare, "white") for w in wanted]


def timeline(pitches, fps=60.0):
    """Sim times to draw, one per frame: slow motion until the last ball leaves
    the wheels, then the flights at FLIGHT_SPEED until the last ball reaches the plate."""
    t_launch = max(p.launch.exit_time for p in pitches)
    t_flight = max((p.flight_time if np.isfinite(p.flight_time) else p.flight.t[-1]) for p in pitches)
    slow = np.arange(0.0, t_launch, 1.0 / fps / SLOWMO)
    fast = np.arange(t_launch, t_launch + t_flight + 0.15, FLIGHT_SPEED / fps)
    return np.r_[slow, fast], t_launch


def ball_at(p, t):
    """Where pitch p's ball is at sim time t (seconds after its ball was fed in)."""
    L = p.launch
    if t <= L.exit_time:
        ts = np.r_[0.0, L.log["t"]]
        start = L.log["pos"][0] - L.log["t"][0] * (L.log["pos"][1] - L.log["pos"][0]) / max(L.log["t"][1] - L.log["t"][0], 1e-12)
        ps = np.vstack([start, L.log["pos"]])
        return np.array([np.interp(t, ts, ps[:, i]) for i in range(3)])
    return _flight_point(p, t - L.exit_time)


def _flight_point(p, tf):
    f = p.flight
    end = p.flight_time if np.isfinite(p.flight_time) else f.t[-1]
    if tf >= end:
        return f.plate_pos[0] if np.isfinite(p.flight_time) else f.pos[-1, 0]
    return np.array([np.interp(tf, f.t, f.pos[:, 0, i]) for i in range(3)])


def trail_points(p, t, every=0.008):
    """Points behind the ball since it left the wheels, one every `every` seconds of flight."""
    tf = t - p.launch.exit_time
    if tf <= 0:
        return []
    return [_flight_point(p, s) for s in np.arange(0.0, tf, every)]


def draw(scn, pitches, colors, t, start=0):
    """Add every ball and trail at time t to an mjvScene, starting at geom index `start`."""
    scn.ngeom = start
    eye = np.eye(3).ravel()
    for p, c in zip(pitches, colors):
        r, g, b = rgb(c)
        for q in trail_points(p, t):
            if scn.ngeom >= scn.maxgeom:
                return
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.014, 0, 0]), q, eye,
                                np.array([r, g, b, 0.6], dtype=np.float32))
            scn.ngeom += 1
        if scn.ngeom < scn.maxgeom:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([BALL_RADIUS, 0, 0]),
                                ball_at(p, t), eye, np.array([r, g, b, 1.0], dtype=np.float32))
            scn.ngeom += 1


def legend(pitches, colors):
    """(left, right) overlay text: one line per pitch."""
    left, right = ["PITCHES"], [""]
    for p, c in zip(pitches, colors):
        L = p.launch
        wheels = " / ".join(f"{r:.0f}" for r in L.config.wheel_rpm)
        where = (f"{p.plate_height / FOOT:.2f} ft high, {abs(p.plate_side) / INCH:.1f} in toward "
                 f"{'1B' if p.plate_side >= 0 else '3B'}") if np.isfinite(p.flight_time) else "hit the ground"
        left.append(c)
        right.append(f"{wheels} rpm  ->  {L.speed / MPH:.1f} mph, {L.signed_spin / RPM:+.0f} rpm  ->  {where}")
    return "\n".join(left), "\n".join(right)


def _stage(pitches):
    """The machine to draw: the first pitch's wheels and aim, its own ball hidden."""
    m = Machine(pitches[0].launch.config, scene=True)
    m.data.qpos[m.ball_qadr : m.ball_qadr + 3] = [0.0, 0.0, -5.0]
    return m


def _pose_machine(m, t):
    for i in m.wheels:
        m.data.qpos[m.qadr[i]] = m.wheel_targets[i] * t
    m.data.qpos[m.ball_qadr : m.ball_qadr + 3] = [0.0, 0.0, -5.0]
    mujoco.mj_forward(m.model, m.data)


def play(pitches, colors, camera="watcher"):
    """Loop the pitches in MuJoCo's viewer until the window is closed."""
    import mujoco.viewer

    from .view import CAMERAS

    m = _stage(pitches)
    times, t_launch = timeline(pitches)
    left, right = legend(pitches, colors)
    keys = ["CONTROLS", "[  ]", "Esc", "mouse", "close window"]
    what = ["", "change camera", "free camera", "drag to orbit, right-drag to pan, scroll to zoom",
            "go on with the rest of the script"]
    with mujoco.viewer.launch_passive(m.model, m.data, show_left_ui=False, show_right_ui=False) as v:
        with v.lock():
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = m.model.camera(camera).id
        while v.is_running():
            for t in list(times) + [times[-1]] * 60:  # hold the end for a second
                start = time.time()
                phase = "launch (250x slow motion)" if t < t_launch else "flight (half speed)"
                fixed = v.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED and 0 <= v.cam.fixedcamid < len(CAMERAS)
                cam = CAMERAS[v.cam.fixedcamid] if fixed else "free"
                with v.lock():
                    _pose_machine(m, t)
                    draw(v.user_scn, pitches, colors, t)
                    v.opt.geomgroup[:] = 1
                v.set_texts([
                    (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, left + f"\n\n{phase}", right + "\n\n"),
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     "\n".join(keys), "\n".join(what[:1] + [f"change camera (now: {cam})"] + what[2:])),
                ])
                v.sync()
                if not v.is_running():
                    return
                time.sleep(max(0.0, 1.0 / 60.0 - (time.time() - start)))


def save_gif(path, pitches, colors, camera="watcher", size=(640, 360), fps=25):
    """Render the same animation offscreen and save it as a GIF."""
    import imageio.v2 as iio

    m = _stage(pitches)
    times, _ = timeline(pitches, fps=fps)
    r = mujoco.Renderer(m.model, size[1], size[0])
    frames = []
    for t in list(times) + [times[-1]] * fps:
        _pose_machine(m, t)
        r.update_scene(m.data, camera=camera)
        draw(r.scene, pitches, colors, t, start=r.scene.ngeom)
        frames.append(r.render())
    iio.mimsave(path, frames, duration=1000 / fps, loop=0)
    return path
