"""Ball flight from release to the plate.

Integrates many balls at once (RK4 on position and velocity). A sphere in
torque-free flight keeps its angular velocity, so orientation is exact:
R(t) = Rot(omega * t) R0. Spin decay over a half-second flight is small and
poorly pinned down by data, so it is left out.
"""

from dataclasses import dataclass

import numpy as np

from .aero import Aero, acceleration
from .constants import BALL_RADIUS, G, INCH, PLATE_DISTANCE

GRAVITY = np.array([0.0, 0.0, -G])


def rotations(axis_angle):
    """Rodrigues: (N, 3) rotation vectors -> (N, 3, 3) matrices."""
    th = np.linalg.norm(axis_angle, axis=1)
    k = axis_angle / np.maximum(th, 1e-12)[:, None]
    K = np.zeros((len(k), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


@dataclass
class Flight:
    t: np.ndarray  # (T,)
    pos: np.ndarray  # (T, N, 3)
    vel: np.ndarray  # (T, N, 3)
    acc_aero: np.ndarray  # (T, N, 3), what an accelerometer at the ball's center feels
    R: np.ndarray  # (T, N, 3, 3), body -> world
    omega: np.ndarray  # (N, 3)
    # At the front edge of home plate (NaN if the ball landed first):
    plate_time: np.ndarray  # (N,)
    plate_pos: np.ndarray  # (N, 3)
    plate_vel: np.ndarray  # (N, 3)

    @property
    def n(self):
        return self.pos.shape[1]


def fly(position, velocity, omega, R0, aero: Aero = None, dt=1e-3, plate_x=PLATE_DISTANCE, t_max=1.5, record_every=1,
        stop_at_ground=True):
    """Fly a batch of balls until each reaches the plate or the ground.

    dt = 1 ms is converged to better than 0.01 in at the plate, even for a
    2500 rpm curveball whose seam force flickers with every turn (see
    tests/test_flight.py)."""
    aero = aero or Aero()
    p = np.array(position, dtype=float).reshape(-1, 3)
    v = np.array(velocity, dtype=float).reshape(-1, 3)
    w = np.array(omega, dtype=float).reshape(-1, 3)
    R0 = np.array(R0, dtype=float).reshape(-1, 3, 3)
    n = len(p)

    def accel(t, vel):
        R = rotations(w * t) @ R0
        return acceleration(vel, w, R, aero), R

    ts, P, V, A, Rs = [], [], [], [], []
    plate_t = np.full(n, np.nan)
    plate_p = np.full((n, 3), np.nan)
    plate_v = np.full((n, 3), np.nan)
    done = np.zeros(n, dtype=bool)

    t, step = 0.0, 0
    a, R = accel(t, v)
    while t < t_max and not done.all():
        if step % record_every == 0:
            ts.append(t), P.append(p.copy()), V.append(v.copy()), A.append(a), Rs.append(R)
        k1v, k1p = a + GRAVITY, v
        k2v = accel(t + dt / 2, v + dt / 2 * k1v)[0] + GRAVITY
        k2p = v + dt / 2 * k1v
        k3v = accel(t + dt / 2, v + dt / 2 * k2v)[0] + GRAVITY
        k3p = v + dt / 2 * k2v
        k4v = accel(t + dt, v + dt * k3v)[0] + GRAVITY
        k4p = v + dt * k3v
        p_new = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

        crossed = ~done & (p[:, 0] < plate_x) & (p_new[:, 0] >= plate_x)
        if crossed.any():
            f = ((plate_x - p[:, 0]) / (p_new[:, 0] - p[:, 0]))[crossed]
            plate_t[crossed] = t + f * dt
            plate_p[crossed] = p[crossed] + f[:, None] * (p_new[crossed] - p[crossed])
            plate_v[crossed] = v[crossed] + f[:, None] * (v_new[crossed] - v[crossed])
            done |= crossed
        if stop_at_ground:
            done |= p_new[:, 2] < BALL_RADIUS  # hit the dirt first

        p, v, t, step = p_new, v_new, t + dt, step + 1
        a, R = accel(t, v)
    ts.append(t), P.append(p.copy()), V.append(v.copy()), A.append(a), Rs.append(R)
    return Flight(np.array(ts), np.array(P), np.array(V), np.array(A), np.array(Rs), w, plate_t, plate_p, plate_v)


def movement(position, velocity, omega, R0, aero: Aero = None, **kw):
    """Break at the plate in inches, Statcast style: where the ball crosses
    minus where the same release would cross with only gravity and drag.
    Returns (horizontal, vertical); horizontal + toward first base."""
    aero = aero or Aero()
    real = fly(position, velocity, omega, R0, aero, **kw)  # the whole path, for plots and replays
    ref_aero = Aero(drag=aero.drag, cd_constant=aero.cd_constant, lift=False, seam=False,
                    crisis_shift=aero.crisis_shift, rho=aero.rho)
    # The reference can dive into the dirt before the plate (a big-backspin pitch
    # aimed low); let it pass through the ground so the break is still defined.
    ref = fly(position, velocity, omega, R0, ref_aero, record_every=10**9, stop_at_ground=False, **kw)
    d = (real.plate_pos - ref.plate_pos) / INCH
    return d[:, 1], d[:, 2], real
