import numpy as np
import pytest

from pitching_machine.aero import Aero, random_rotations
from pitching_machine.constants import INCH, MPH
from pitching_machine.flight import fly
from pitching_machine.imu import IMUSpec, model_residual, reconstruct, record
from pitching_machine.machine import Machine
from pitching_machine.pitch import ZONE_MIDDLE, preset_config, throw
from pitching_machine.view import replay

PERFECT = IMUSpec(gyro_noise=0, gyro_bias=0, accel_noise=0, accel_bias=0)


def knuckleball_flight():
    R0 = random_rotations(1, np.random.default_rng(5))
    w = np.array([0.0, 0.0, 2.0])  # a lazy 20 rpm
    return fly([0.05, 0, 1.75], [29.0, 0, 1.0], w, R0, Aero()), R0[0]


def test_perfect_imu_recovers_the_aerodynamic_force():
    f, R0 = knuckleball_flight()
    rec = record(f, spec=PERFECT)
    rc = reconstruct(rec, R0, f.vel[0, 0])
    k = np.searchsorted(f.t, rc.t)
    assert np.abs(rc.acc_aero - f.acc_aero[k, 0]).max() < 0.02  # m/s^2
    assert np.abs(rc.vel - f.vel[k, 0]).max() < 0.01  # m/s


def test_residual_against_the_full_model_is_just_noise():
    f, R0 = knuckleball_flight()
    rc = reconstruct(record(f, spec=PERFECT), R0, f.vel[0, 0])
    residual, _ = model_residual(rc, Aero())
    assert np.abs(residual).max() < 0.05


def test_presets_are_aimed_at_the_zone():
    p = throw(preset_config("curveball"))
    assert p.plate_height == pytest.approx(ZONE_MIDDLE, abs=0.5 * INCH)
    assert abs(p.plate_side) < 0.5 * INCH
    assert p.vertical_break < -10  # a curveball drops
    assert 60 < p.plate_speed / MPH < 80


def test_viewer_replay_runs_headless():
    m = Machine(preset_config("knuckleball"), scene=True)
    frames = sum(1 for _ in replay(m))
    assert frames > 50
    assert m.ball_state()[0][0] > 15  # ended near the plate
