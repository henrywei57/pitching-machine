"""Machine + air: throw a pitch from wheel settings and see where it crosses the plate.

    python -m knuckle.pitch curveball
    python -m knuckle.pitch --speed 85 --spin -2000 --tilt 15 --calibrate --plot pitch.png
"""

import argparse
from dataclasses import dataclass

import numpy as np

from .aero import Aero
from .constants import FOOT, INCH, MPH, PLATE_DISTANCE, RPM
from .flight import Flight, fly, movement
from .machine import LaunchResult, MachineConfig, _ry, _rz, calibrate, launch, quat_to_mat, settings_for

# Target speed (mph), spin (rpm, + topspin / - backspin) and head tilt (deg).
# A positive tilt turns the spin axis so the ball also breaks toward first base,
# the glove side of a right-handed pitcher.
PRESETS = {
    "fastball": dict(speed_mph=90, spin_rpm=-2200, tilt_deg=0),
    "curveball": dict(speed_mph=78, spin_rpm=2500, tilt_deg=30),
    # Same dial settings as the curveball. The "knuckle" is a better grip, and on
    # a machine the grip is the friction between tire and leather.
    "knuckle-curve": dict(speed_mph=78, spin_rpm=2500, tilt_deg=30, friction=1.0),
    "knuckleball": dict(speed_mph=67, spin_rpm=0, tilt_deg=0),
}

ZONE_MIDDLE = 2.5 * FOOT  # aim point above the plate


@dataclass
class Pitch:
    launch: LaunchResult
    flight: Flight
    horizontal_break: float  # in, + toward first base
    vertical_break: float  # in, + up (induced vertical break)

    @property
    def plate_height(self):
        return float(self.flight.plate_pos[0, 2])

    @property
    def plate_side(self):
        return float(self.flight.plate_pos[0, 1])

    @property
    def plate_speed(self):
        return float(np.linalg.norm(self.flight.plate_vel[0]))

    @property
    def flight_time(self):
        return float(self.flight.plate_time[0])

    def summary(self):
        lines = [self.launch.summary()]
        c = self.launch.config
        lines.append(f"aim          {c.elevation_deg:+.2f} deg up   {c.azimuth_deg:+.2f} deg toward 1B")
        if np.isnan(self.flight_time):
            lines.append("ball hit the ground before the plate")
        else:
            lines.append(
                f"at the plate {self.plate_speed / MPH:5.1f} mph after {self.flight_time:.3f} s   "
                f"height {self.plate_height / FOOT:.2f} ft   side {self.plate_side / INCH:+.1f} in"
            )
            lines.append(
                f"break        {self.horizontal_break:+.1f} in horizontal (+ toward 1B)   "
                f"{self.vertical_break:+.1f} in vertical (vs. gravity + drag only)"
            )
        return "\n".join(lines)


def _exit_state(res: LaunchResult, elevation_deg, azimuth_deg):
    """Exit state if the whole machine were re-aimed, rotating about the barrel origin."""
    c = res.config
    A = _rz(np.radians(azimuth_deg)) @ _ry(-np.radians(elevation_deg))
    A0 = _rz(np.radians(c.azimuth_deg)) @ _ry(-np.radians(c.elevation_deg))
    D = A @ A0.T
    origin = np.array([0.0, 0.0, c.release_height])
    return origin + D @ (res.position - origin), D @ res.velocity, D @ res.omega, D @ quat_to_mat(res.quat)


def aim(res: LaunchResult, height=ZONE_MIDDLE, side=0.0, iterations=6):
    """Elevation and azimuth that put an average ball (seams ignored) at (side, height).

    An operator aims a knuckleball machine at the middle of the zone and lets the
    seams do what they do, so the aim ignores the seam force.
    """
    model = Aero(seam=False, crisis_shift=0.0)
    el, az = res.config.elevation_deg, res.config.azimuth_deg
    h = 0.05  # deg, finite-difference step
    for _ in range(iterations):
        states = [_exit_state(res, e, a) for e, a in ((el, az), (el + h, az), (el, az + h))]
        p, v, w, R = (np.array(x) for x in zip(*states))
        f = fly(p, v, w, R, model, record_every=10**9)
        y = f.plate_pos[:, 1:3]
        if np.isnan(y).any():  # landed short: just aim higher and retry
            el += 2.0
            continue
        err = y[0] - np.array([side, height])
        J = np.stack([(y[1] - y[0]) / h, (y[2] - y[0]) / h], axis=1)
        step = np.linalg.solve(J, -err)
        el, az = el + step[0], az + step[1]
        if np.abs(err).max() < 0.002:
            break
    return el, az


def throw(cfg: MachineConfig, aero: Aero = None, aim_at=ZONE_MIDDLE) -> Pitch:
    """Launch, optionally aim, launch again aimed, and fly to the plate."""
    aero = aero or Aero()
    res = launch(cfg)
    if not res.exited:
        raise RuntimeError("the ball never left the wheels:\n" + res.summary())
    if aim_at is not None:
        el, az = aim(res, height=aim_at)
        res = launch(cfg.with_(elevation_deg=el, azimuth_deg=az))
    R0 = quat_to_mat(res.quat)
    h, v, f = movement(res.position, res.velocity, res.omega, R0, aero)
    return Pitch(res, f, float(h[0]), float(v[0]))


def preset_config(name, calibrated=False, **overrides):
    p = dict(PRESETS[name])
    p.update(overrides)
    speed, spin = p.pop("speed_mph"), p.pop("spin_rpm")
    if calibrated:
        return calibrate(speed, spin, **p).config
    return settings_for(speed, spin, **p)


def plot_pitch(pitch: Pitch, path):
    import matplotlib.pyplot as plt

    f = pitch.flight
    x, y, z = (f.pos[:, 0, i] for i in range(3))
    keep = x <= PLATE_DISTANCE + 0.05
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(x[keep] / FOOT, z[keep] / FOOT)
    ax[0].set(xlabel="distance from machine (ft)", ylabel="height (ft)", title="side view")
    ax[1].plot(x[keep] / FOOT, y[keep] / INCH)
    ax[1].set(xlabel="distance from machine (ft)", ylabel="toward 1B (in)", title="top view")
    for a in ax:
        a.axvline(PLATE_DISTANCE / FOOT, color="0.6", lw=1, ls="--")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Throw a pitch with the two-wheel machine.")
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS), help="named pitch")
    ap.add_argument("--speed", type=float, help="target speed, mph")
    ap.add_argument("--spin", type=float, help="target spin, rpm (+ topspin, - backspin)")
    ap.add_argument("--tilt", type=float, default=0.0, help="head tilt, deg (+ breaks toward 1B)")
    ap.add_argument("--friction", type=float, help="tire-leather friction coefficient")
    ap.add_argument("--calibrate", action="store_true", help="adjust wheel rpm until the radar gun agrees")
    ap.add_argument("--seed", type=int, help="random seam orientation in the chute (default: fixed)")
    ap.add_argument("--plot", help="save a side/top view to this file")
    a = ap.parse_args(argv)

    over = {} if a.friction is None else {"friction": a.friction}
    if a.preset:
        cfg = preset_config(a.preset, calibrated=a.calibrate, **over)
    elif a.speed is not None and a.spin is not None:
        args = dict(tilt_deg=a.tilt, **over)
        cfg = calibrate(a.speed, a.spin, **args).config if a.calibrate else settings_for(a.speed, a.spin, **args)
    else:
        ap.error("give a preset, or both --speed and --spin")
    if a.seed is not None:
        q = np.random.default_rng(a.seed).normal(size=4)
        cfg = cfg.with_(ball_quat=tuple(q / np.linalg.norm(q)))

    p = throw(cfg)
    print(p.summary())
    want = PRESETS[a.preset] if a.preset else dict(speed_mph=a.speed, spin_rpm=a.spin)
    got_v, got_s = p.launch.speed / MPH, p.launch.signed_spin / RPM
    if a.calibrate and (abs(got_v - want["speed_mph"]) > 0.5 or abs(got_s - want["spin_rpm"]) > 50):
        print(f"note: the machine can't reach {want['speed_mph']:.0f} mph / {want['spin_rpm']:+.0f} rpm; "
              "this is the closest it gets. More squeeze or grip (--friction) raises the ceiling.")
    if a.plot:
        plot_pitch(p, a.plot)
        print(f"saved {a.plot}")


if __name__ == "__main__":
    main()
