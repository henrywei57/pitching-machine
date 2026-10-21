import numpy as np
import pytest

from pitching_machine.aero import Aero, random_rotations
from pitching_machine.constants import G, INCH
from pitching_machine.flight import fly, rotations


def test_rotation_vectors():
    R = rotations(np.array([[0.0, 0.0, np.pi / 2]]))[0]
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_vacuum_flight_is_a_parabola():
    v0 = np.array([30.0, 1.0, 2.0])
    f = fly([0, 0, 1.8], v0, [0, 0, 0], np.eye(3), Aero(drag="none", lift=False, seam=False))
    t = f.plate_time[0]
    expected = np.array([0, 0, 1.8]) + v0 * t + 0.5 * np.array([0, 0, -G]) * t**2
    assert np.allclose(f.plate_pos[0], expected, atol=1e-9)


@pytest.mark.parametrize("spin", [0.0, 2500.0])
def test_one_millisecond_steps_are_converged(spin):
    rng = np.random.default_rng(0)
    n = 8
    R0 = random_rotations(n, rng)
    p = np.tile([0.05, 0.0, 1.75], (n, 1))
    v = np.tile([33.0, 0.0, 1.5], (n, 1))
    w = np.tile([0.0, spin * 2 * np.pi / 60 * 0.866, spin * 2 * np.pi / 60 * 0.5], (n, 1))
    coarse = fly(p, v, w, R0, dt=1e-3, record_every=10**9).plate_pos
    fine = fly(p, v, w, R0, dt=1e-4, record_every=10**9).plate_pos
    assert np.abs(coarse - fine).max() < 0.01 * INCH


def test_topspin_drops_and_backspin_rides():
    p, v = [0.05, 0.0, 1.75], [35.0, 0.0, 2.0]
    model = Aero(seam=False)
    top = fly(p, v, [0, 230, 0], np.eye(3), model).plate_pos[0, 2]
    none = fly(p, v, [0, 0, 0], np.eye(3), model).plate_pos[0, 2]
    back = fly(p, v, [0, -230, 0], np.eye(3), model).plate_pos[0, 2]
    assert top < none < back
