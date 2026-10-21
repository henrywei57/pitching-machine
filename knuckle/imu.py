"""A virtual instrumented baseball.

Noel Perkins's lab at Michigan put a wireless inertial measurement unit (a
3-axis accelerometer and 3-axis gyro) inside a baseball to measure pitches
directly (McGinnis & Perkins, Sensors 12, 2012). This module does the same
thing to a simulated pitch, so the simulation and the "measurement" can be
compared the way the real ball would let you.

The useful fact: an accelerometer at the center of a ball in flight cannot
feel gravity (everything falls together). It reads only the aerodynamic force
divided by mass, in the ball's own frame. Combine it with the gyro to rotate
that reading into the world, and the ball reports the drag, lift and seam
forces acting on it at every instant, with no model of the air at all.
"""

from dataclasses import dataclass

import numpy as np

from .aero import Aero, acceleration
from .constants import BALL_AREA, BALL_MASS, G, RHO_AIR
from .flight import GRAVITY, Flight, rotations


@dataclass(frozen=True)
class IMUSpec:
    """Sensor characteristics. Defaults are in the range of small MEMS parts."""

    rate_hz: float = 1000.0
    gyro_noise: float = np.radians(0.3)  # rad/s per sample (white)
    gyro_bias: float = np.radians(0.5)  # rad/s, fixed per ball, random direction
    gyro_range: float = np.radians(20000.0)  # rad/s (~3300 rpm) before it clips
    accel_noise: float = 0.02 * G  # m/s^2 per sample
    accel_bias: float = 0.01 * G
    accel_range: float = 200.0 * G


@dataclass
class IMURecord:
    t: np.ndarray  # (T,)
    gyro: np.ndarray  # (T, 3) body frame, rad/s
    accel: np.ndarray  # (T, 3) body frame, m/s^2 (specific force)


def record(flight: Flight, ball=0, spec=IMUSpec(), seed=0) -> IMURecord:
    """What the IMU inside ball `ball` of a simulated flight would log."""
    rng = np.random.default_rng(seed)
    t = flight.t
    keep = np.r_[True, np.diff(np.floor(t * spec.rate_hz)) > 0]  # resample to the IMU rate
    t = t[keep]
    R = flight.R[keep, ball]  # body -> world
    w_body = np.einsum("tji,j->ti", R, flight.omega[ball])
    a_body = np.einsum("tji,tj->ti", R, flight.acc_aero[keep, ball])

    def unit():
        u = rng.normal(size=3)
        return u / np.linalg.norm(u)

    gyro = w_body + spec.gyro_bias * unit() + spec.gyro_noise * rng.normal(size=w_body.shape)
    accel = a_body + spec.accel_bias * unit() + spec.accel_noise * rng.normal(size=a_body.shape)
    return IMURecord(t, np.clip(gyro, -spec.gyro_range, spec.gyro_range), np.clip(accel, -spec.accel_range, spec.accel_range))


@dataclass
class Reconstruction:
    t: np.ndarray
    R: np.ndarray  # (T, 3, 3) estimated orientation
    vel: np.ndarray  # (T, 3) estimated velocity
    acc_aero: np.ndarray  # (T, 3) measured aerodynamic acceleration, world frame
    cd: np.ndarray  # (T,) measured drag coefficient
    omega: np.ndarray  # (T, 3) measured spin, world frame


def reconstruct(rec: IMURecord, R0, v0, rho=RHO_AIR) -> Reconstruction:
    """Turn raw IMU samples into world-frame aerodynamics.

    Needs the starting orientation and velocity, which a real system gets from
    the ball sitting still before the throw and a radar gun (here, the machine).
    """
    n = len(rec.t)
    R = np.empty((n, 3, 3))
    R[0] = R0
    for i in range(1, n):
        dt = rec.t[i] - rec.t[i - 1]
        w_mid = 0.5 * (rec.gyro[i] + rec.gyro[i - 1])  # body-frame rate
        R[i] = R[i - 1] @ rotations((w_mid * dt)[None])[0]
    a_world = np.einsum("tij,tj->ti", R, rec.accel)
    vel = np.empty((n, 3))
    vel[0] = v0
    for i in range(1, n):
        dt = rec.t[i] - rec.t[i - 1]
        vel[i] = vel[i - 1] + dt * (0.5 * (a_world[i] + a_world[i - 1]) + GRAVITY)
    speed = np.linalg.norm(vel, axis=1)
    vhat = vel / speed[:, None]
    q_over_m = 0.5 * rho * BALL_AREA * speed**2 / BALL_MASS
    cd = -np.einsum("ti,ti->t", a_world, vhat) / q_over_m
    omega = np.einsum("tij,tj->ti", R, rec.gyro)
    return Reconstruction(rec.t, R, vel, a_world, cd, omega)


def model_residual(recon: Reconstruction, model: Aero):
    """Measured aerodynamic acceleration minus what `model` predicts for the
    same velocity, spin and orientation. The difference is the physics the
    model is missing."""
    pred = acceleration(recon.vel, recon.omega, recon.R, model)
    return recon.acc_aero - pred, pred
