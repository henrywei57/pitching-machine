"""Watch the machine throw in MuJoCo's interactive viewer, and change the wheels while you watch.

    mjpython -m pitching_machine.view                          # macOS (mjpython ships with mujoco)
    python -m pitching_machine.view curveball                  # Linux / Windows
    mjpython -m pitching_machine.view --rpm 3200 2600 2000 --camera catcher

In the window:
  Wheel speeds   In the panel on the right, open "Control" and drag the
                 top_rpm, lower_3B_rpm and lower_1B_rpm sliders. The next pitch
                 uses them; the machine re-aims at the middle of the zone first.
  Cameras        [ and ] step through watcher, catcher, batter, side, machine and
                 follow_ball. Esc gives a free camera: drag to orbit, right-drag
                 to pan, scroll to zoom.
  Quit           close the window.
"""

import argparse
import contextlib
import time

import mujoco
import numpy as np

from .aero import Aero
from .constants import FOOT, INCH, MPH, RPM
from .flight import fly
from .machine import THREE_WHEEL, TWO_WHEEL, Machine, MachineConfig, cameras, mat_to_quat
from .pitch import PRESETS, preset_config, rpm_config, throw

FRAME = 1.0 / 60.0  # s of wall clock per frame
CAMERAS = [*cameras(MachineConfig()), "follow_ball"]


def replay(machine: Machine, slowmo=250.0, flight_speed=0.5, lock=contextlib.nullcontext):
    """Drive `machine` through one pitch, yielding once per frame.

    Yields ("launch", None) while the ball is in the wheels, then
    ("flight", (flight, k)) with the flight and the current sample index.
    `lock` guards each frame's changes to the model data (the viewer's lock).
    """
    m = machine
    with lock():
        m.reset()
    steps = max(1, round(FRAME / slowmo / m.cfg.timestep))
    while m.data.time < m.cfg.max_time:
        with lock():
            for _ in range(steps):
                m.step()
            done = not m.contacts and m.ball_state()[0][0] > 0.05
        yield "launch", None
        if done:
            break

    with lock():
        pos, vel, omega, quat = m.ball_state()
        R0 = m.data.xmat[m.ball_body].reshape(1, 3, 3).copy()
        wheel_q = [m.data.qpos[m.qadr[i]] for i in m.wheels]
        t0 = m.data.time
    f = fly(pos, vel, omega, R0, Aero())
    t_end = np.nanmax(f.plate_time) if np.isfinite(f.plate_time).any() else f.t[-1]
    wheel_w = m.wheel_targets
    a = m.ball_qadr
    for t in np.arange(0.0, t_end + 0.1, FRAME * flight_speed):
        k = min(np.searchsorted(f.t, t), len(f.t) - 1)
        with lock():
            m.data.qpos[a : a + 3] = f.pos[k, 0]
            m.data.qpos[a + 3 : a + 7] = mat_to_quat(f.R[k, 0])
            m.data.qvel[:] = 0.0
            for i in m.wheels:
                m.data.qpos[m.qadr[i]] = wheel_q[i] + wheel_w[i] * t
            m.data.time = t0 + t
            mujoco.mj_forward(m.model, m.data)
        yield "flight", (f, k)


def overlay(pitch, phase=""):
    """(left column, right column) text for the viewer: wheels, what came out, where it went."""
    c = pitch.launch.config
    left = [f"{lab} wheel" for lab in c.labels]
    right = [f"{rpm:.0f} rpm" for rpm in c.wheel_rpm]
    L = pitch.launch
    left += ["", "out of the machine", "spin"]
    right += ["", f"{L.speed / MPH:.1f} mph", f"{L.signed_spin / RPM:+.0f} rpm, axis {L.spin_axis_deg:+.0f} deg"]
    if np.isfinite(pitch.flight_time):
        side = pitch.plate_side / INCH
        left += ["at the plate", "", "break"]
        right += [f"{pitch.plate_speed / MPH:.1f} mph, {pitch.plate_height / FOOT:.2f} ft high",
                  f"{abs(side):.1f} in toward {'1B' if side >= 0 else '3B'}",
                  f"{pitch.horizontal_break:+.1f} in side, {pitch.vertical_break:+.1f} in up"]
    else:
        left += ["at the plate"]
        right += ["hit the ground first"]
    if phase:
        left += ["", phase]
        right += ["", ""]
    return "\n".join(left), "\n".join(right)


HELP = ("change wheel speeds: right panel > Control > *_rpm sliders",
        "[ ]  next camera     Esc  free camera     close window to quit")


def _trail(scn, f, k, every=8):
    scn.ngeom = 0
    for p in f.pos[: k + 1 : every, 0]:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.015, 0, 0]), p,
                            np.eye(3).ravel(), np.array([0.92, 0.41, 0.2, 0.7], dtype=np.float32))
        scn.ngeom += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="In the window:" + __doc__.split("In the window:")[1])
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS), help="named pitch (default: knuckleball)")
    ap.add_argument("--rpm", type=float, nargs="+", metavar="RPM",
                    help="starting wheel speeds, in order: top, lower 3B, lower 1B (with --wheels 2: top, bottom)")
    ap.add_argument("--wheels", type=int, choices=(2, 3), default=3, help="three-wheel (default) or two-wheel machine")
    ap.add_argument("--camera", choices=CAMERAS, default="watcher", help="starting view (default: watcher)")
    a = ap.parse_args(argv)
    angles = THREE_WHEEL if a.wheels == 3 else TWO_WHEEL
    if a.rpm is not None:
        if a.preset:
            ap.error("give a preset or --rpm, not both")
        try:
            cfg = rpm_config(a.rpm, wheel_angles_deg=angles)
            cfg.check_rpm()
        except ValueError as e:
            ap.error(str(e))
    else:
        cfg = preset_config(a.preset or "knuckleball", wheel_angles_deg=angles)

    import mujoco.viewer

    try:
        pitch = throw(cfg)  # aimed at the middle of the zone
    except RuntimeError as e:
        ap.exit(1, f"{e}\n")
    machine = Machine(pitch.launch.config, scene=True)
    m = machine.model
    print(__doc__.split("In the window:")[1].rstrip())
    print("\n" + pitch.summary())

    with mujoco.viewer.launch_passive(m, machine.data, show_left_ui=False) as v:
        with v.lock():
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = m.camera(a.camera).id

        def show(phase=""):
            left, right = overlay(pitch, phase)
            v.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right),
                         (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, HELP[0], HELP[1])])

        stuck_at = None
        while v.is_running():
            # The sliders write straight into the motor commands, which are in rpm.
            with v.lock():
                asked = tuple(float(np.round(x)) for x in machine.data.ctrl)
            if np.abs(np.subtract(asked, machine.cfg.wheel_rpm)).max() > 0.5:
                if asked == stuck_at:
                    v.sync()
                    time.sleep(0.1)
                    continue
                v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "aiming for the new wheel speeds...",
                             " / ".join(f"{r:.0f}" for r in asked) + " rpm"))
                v.sync()
                try:
                    pitch = throw(machine.cfg.with_(wheel_rpm=asked))
                except RuntimeError:
                    stuck_at = asked
                    v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                 "the ball never left the wheels", "turn the wheels up"))
                    continue
                stuck_at = None
                with v.lock():
                    machine.retarget(pitch.launch.config)
                print("\n" + pitch.summary())

            with v.lock():
                v.user_scn.ngeom = 0
            for phase, info in replay(machine, lock=v.lock):
                start = time.time()
                show("launch (250x slow motion)" if phase == "launch" else "flight (half speed)")
                if info is not None:
                    with v.lock():
                        _trail(v.user_scn, *info)
                v.sync()
                if not v.is_running():
                    return
                time.sleep(max(0.0, FRAME - (time.time() - start)))
            # Pause on the result; keep listening to the sliders.
            for _ in range(int(1.0 / FRAME)):
                v.sync()
                time.sleep(FRAME)


if __name__ == "__main__":
    main()
