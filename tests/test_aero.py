import numpy as np
import pytest

from knuckle import aero
from knuckle.aero import Aero, acceleration, drag_coefficient, kernel, random_rotations, seam_terms
from knuckle.constants import BALL_AREA, BALL_MASS, G, RHO_AIR
from knuckle.machine import _rz
from knuckle.seams import seam_points


def test_seam_lies_on_the_sphere():
    pts, w = seam_points(500)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0)
    assert w.sum() == pytest.approx(1.0)


def test_seam_symmetry_quarter_turn_plus_mirror():
    pts, _ = seam_points(400)
    mapped = pts @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]).T  # (x, y, z) -> (-y, x, -z)
    d = np.linalg.norm(mapped[:, None, :] - pts[None, :, :], axis=2).min(axis=1)
    assert d.max() < 1e-9


def test_drag_opposes_velocity_and_backspin_lifts():
    v = np.array([[35.0, 0.0, 0.0]])
    backspin = np.array([[0.0, -230.0, 0.0]])
    _, parts = acceleration(v, backspin, np.eye(3)[None], Aero(seam=False), details=True)
    assert parts["drag"][0, 0] < 0
    assert parts["lift"][0, 2] > 0


def test_seam_force_is_perpendicular_to_the_flow():
    rng = np.random.default_rng(0)
    R = random_rotations(50, rng)
    vhat = rng.normal(size=(50, 3))
    vhat /= np.linalg.norm(vhat, axis=1, keepdims=True)
    C, _ = seam_terms(R, vhat, *seam_points(180))
    assert np.abs(np.einsum("ni,ni->n", C, vhat)).max() < 1e-12


def test_mirror_symmetric_poses_have_no_sideways_force():
    vhat = np.tile([1.0, 0.0, 0.0], (2, 1))
    R = np.array([_rz(t) @ _rz(-np.pi / 4) for t in (0.0, np.pi / 4)])
    C, _ = seam_terms(R, vhat, *seam_points(720))
    assert np.abs(C[:, 1]).max() < 1e-3


def test_vectorized_seam_force_matches_brute_force():
    rng = np.random.default_rng(1)
    R = random_rotations(20, rng)
    vhat = np.tile([0.0, 0.6, 0.8], (20, 1))
    pts, w = seam_points(180)
    C, _ = seam_terms(R, vhat, pts, w)
    S = np.einsum("nij,mj->nmi", R, pts)
    mu = S @ vhat[0]
    perp = S - mu[..., None] * vhat[0]
    perp /= np.linalg.norm(perp, axis=-1, keepdims=True)
    C_ref = np.einsum("nm,m,nmi->ni", kernel(np.degrees(np.arccos(mu))), w, perp)
    assert np.allclose(C, C_ref, atol=1e-10)


def test_model_reproduces_the_wind_tunnel_curve():
    theta = np.radians(np.arange(0, 180, 3.0))
    R = np.array([_rz(t) @ _rz(-np.pi / 4) for t in theta])
    C, _ = seam_terms(R, np.tile([1.0, 0.0, 0.0], (len(theta), 1)), *seam_points(360))
    q = 0.5 * RHO_AIR * aero.SEAM_REF_SPEED**2 * BALL_AREA
    measured = 0.5 * (np.sin(4 * theta) - 0.25 * np.sin(8 * theta) + 0.08 * np.sin(12 * theta) - 0.025 * np.sin(16 * theta))
    model = C[:, 1] * q / (BALL_MASS * G)
    assert np.sqrt(np.mean((model - measured) ** 2)) < 0.01 * np.sqrt(np.mean(measured**2))


def test_drag_crisis_limits():
    assert drag_coefficient(5.0) == pytest.approx(aero.CD_LAMINAR, abs=0.01)
    assert drag_coefficient(80.0) == pytest.approx(aero.CD_TURBULENT, abs=0.01)
    assert 0.2 < drag_coefficient(40.0) < 0.4  # ~90 mph: the familiar 0.3 to 0.35


def test_seam_force_tapers_with_spin():
    R = random_rotations(1, np.random.default_rng(2))
    v = np.array([[30.0, 0.0, 0.0]])
    still = acceleration(v, np.zeros((1, 3)), R, Aero(lift=False, drag="none"))
    spun = acceleration(v, np.array([[0.0, 0.0, 250.0]]), R, Aero(lift=False, drag="none"))
    assert np.linalg.norm(spun) < 0.3 * np.linalg.norm(still)
