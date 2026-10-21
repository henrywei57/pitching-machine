"""Watch a pitch in MuJoCo's interactive viewer: the launch in slow motion,
then the flight to the plate, on a loop.

    mjpython -m pitching_machine.view curveball      # macOS (mjpython ships with mujoco)
    python -m pitching_machine.view knuckleball      # Linux / Windows

Drag to orbit, right-drag to pan, scroll to zoom. Close the window to quit.
"""

import argparse
import time

import mujoco
import numpy as np

from .aero import Aero
from .flight import fly
from .machine import WHEELS, Machine, mat_to_quat
from .pitch import PRESETS, preset_config, throw

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
    wheel_q = {w: m.data.qpos[m.qadr[w]] for w in WHEELS}
    wheel_w = m.wheel_targets
    a = m.ball_qadr
    for t in np.arange(0.0, t_end + 0.1, FRAME * flight_speed):
        k = min(np.searchsorted(f.t, t), len(f.t) - 1)
        m.data.qpos[a : a + 3] = f.pos[k, 0]
        m.data.qpos[a + 3 : a + 7] = mat_to_quat(f.R[k, 0])
        m.data.qvel[:] = 0.0
        for w in WHEELS:
            m.data.qpos[m.qadr[w]] = wheel_q[w] + wheel_w[w] * t
        m.data.time = t0 + t
        mujoco.mj_forward(m.model, m.data)
        yield


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("preset", nargs="?", default="knuckleball", choices=sorted(PRESETS))
    a = ap.parse_args(argv)

    import mujoco.viewer

    cfg = throw(preset_config(a.preset)).launch.config  # aimed at the zone
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
