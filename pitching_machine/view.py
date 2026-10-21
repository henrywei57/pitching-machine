"""Watch a pitch in MuJoCo's interactive viewer: the launch in slow motion,
then the flight to the plate, on a loop.

    mjpython -m pitching_machine.view curveball      # macOS (mjpython ships with mujoco)
    python -m pitching_machine.view knuckleball      # Linux / Windows
    mjpython -m pitching_machine.view --rpm 3200 2600 2000   # top, lower 3B, lower 1B

Drag to orbit, right-drag to pan, scroll to zoom. Close the window to quit.
"""

import argparse
import time

import mujoco
import numpy as np

from .aero import Aero
from .flight import fly
from .machine import THREE_WHEEL, TWO_WHEEL, Machine, mat_to_quat
from .pitch import PRESETS, preset_config, rpm_config, throw

FRAME = 1.0 / 60.0  # s of wall clock per frame


def replay(machine: Machine, slowmo=250.0, flight_speed=0.5):
    """Drive `machine` through one pitch, yielding once per frame."""
    m = machine
    m.reset()
    steps = max(1, round(FRAME / slowmo / m.cfg.timestep))
    while m.data.time < m.cfg.max_time:
        for _ in range(steps):
            m.step()
        yield
        if not m.contacts and m.ball_state()[0][0] > 0.05:
            break

    pos, vel, omega, quat = m.ball_state()
    f = fly(pos, vel, omega, m.data.xmat[m.ball_body].reshape(1, 3, 3), Aero())
    t_end = np.nanmax(f.plate_time) if np.isfinite(f.plate_time).any() else f.t[-1]
    t0 = m.data.time
    wheel_q = [m.data.qpos[m.qadr[i]] for i in m.wheels]
    wheel_w = m.wheel_targets
    a = m.ball_qadr
    for t in np.arange(0.0, t_end + 0.1, FRAME * flight_speed):
        k = min(np.searchsorted(f.t, t), len(f.t) - 1)
        m.data.qpos[a : a + 3] = f.pos[k, 0]
        m.data.qpos[a + 3 : a + 7] = mat_to_quat(f.R[k, 0])
        m.data.qvel[:] = 0.0
        for i in m.wheels:
            m.data.qpos[m.qadr[i]] = wheel_q[i] + wheel_w[i] * t
        m.data.time = t0 + t
        mujoco.mj_forward(m.model, m.data)
        yield


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS), help="named pitch (default: knuckleball)")
    ap.add_argument("--rpm", type=float, nargs="+", metavar="RPM",
                    help="set each wheel yourself, in order: top, lower 3B, lower 1B (with --wheels 2: top, bottom)")
    ap.add_argument("--wheels", type=int, choices=(2, 3), default=3, help="three-wheel (default) or two-wheel machine")
    a = ap.parse_args(argv)
    angles = THREE_WHEEL if a.wheels == 3 else TWO_WHEEL
    if a.rpm is not None:
        if a.preset:
            ap.error("give a preset or --rpm, not both")
        try:
            cfg = rpm_config(a.rpm, wheel_angles_deg=angles)
        except ValueError as e:
            ap.error(str(e))
    else:
        cfg = preset_config(a.preset or "knuckleball", wheel_angles_deg=angles)

    import mujoco.viewer

    try:
        cfg = throw(cfg).launch.config  # aimed at the zone
    except RuntimeError as e:
        ap.exit(1, f"{e}\n")
    machine = Machine(cfg, scene=True)
    with mujoco.viewer.launch_passive(machine.model, machine.data) as v:
        v.cam.lookat[:] = [2.0, 0.0, 1.2]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 5.0, 135.0, -12.0
        while v.is_running():
            for _ in replay(machine):
                start = time.time()
                v.sync()
                if not v.is_running():
                    return
                time.sleep(max(0.0, FRAME - (time.time() - start)))
            time.sleep(1.0)


if __name__ == "__main__":
    main()
