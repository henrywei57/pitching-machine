import numpy as np
import pytest

from pitching_machine.constants import MPH, RPM
from pitching_machine.machine import (TWO_WHEEL, Machine, MachineConfig, calibrate, launch, nip_angle, quat_to_mat,
                                      settings_for)


def parked_ball_stiffness(contact_model, squeeze=0.008):
    """Normal force / squash with the ball sitting still in the pinch."""
    cfg = MachineConfig(wheel_rpm=(0, 0, 0), feed_speed=0, squeeze=squeeze, contact_model=contact_model)
    m = Machine(cfg)
    m.model.opt.gravity[:] = 0
    m.data.qpos[m.ball_qadr : m.ball_qadr + 3] = [0, 0, cfg.release_height]
    for _ in range(20000):
        m.step()
    fn, _, squash, _ = m.contacts[0]
    return fn / squash, cfg.contact_stiffness


@pytest.mark.parametrize("contact_model", ["coulomb", "mujoco"])
def test_contact_spring_is_what_the_config_says(contact_model):
    k, want = parked_ball_stiffness(contact_model)
    assert k == pytest.approx(want, rel=0.02)


@pytest.mark.parametrize("wheels", [None, TWO_WHEEL])
def test_settings_for_inverts_the_ideal_kinematics(wheels):
    kw = {} if wheels is None else {"wheel_angles_deg": wheels}
    cfg = settings_for(84.0, -1900.0, 25.0 if wheels is None else 0.0, **kw)
    assert cfg.ideal_speed / MPH == pytest.approx(84.0)
    assert cfg.ideal_spin / RPM == pytest.approx(-1900.0)
    assert cfg.spin_axis_deg == pytest.approx(25.0 if wheels is None else 0.0)


def test_equal_wheels_throw_almost_no_spin():
    r = launch(settings_for(67, 0))
    assert r.exited
    assert abs(r.signed_spin / RPM) < 10


def test_faster_top_wheel_means_topspin_and_vice_versa():
    assert launch(settings_for(78, 2500)).signed_spin > 0
    assert launch(settings_for(78, -2500)).signed_spin < 0


@pytest.mark.parametrize("wheels", [None, TWO_WHEEL])
def test_ball_leaves_slower_than_the_rims(wheels):
    kw = {} if wheels is None else {"wheel_angles_deg": wheels}
    r = launch(settings_for(90, -2200, **kw))
    assert 0.85 < r.speed_efficiency < 1.0


def test_same_launch_is_bit_identical():
    a, b = launch(settings_for(78, 2500)), launch(settings_for(78, 2500))
    assert np.array_equal(a.velocity, b.velocity)
    assert np.array_equal(a.omega, b.omega)
    assert np.array_equal(a.quat, b.quat)


def test_seam_orientation_in_the_chute_does_not_change_the_launch():
    """The tires see a sphere, so the chute orientation only rotates the result."""
    base = settings_for(78, 2500)
    q = np.array([0.3, -0.5, 0.7, 0.4])
    other = base.with_(ball_quat=tuple(q / np.linalg.norm(q)))
    a, b = launch(base), launch(other)
    assert np.allclose(a.velocity, b.velocity, atol=1e-9)
    assert np.allclose(a.omega, b.omega, atol=1e-6)
    Phi = quat_to_mat(a.quat) @ quat_to_mat(base.ball_quat).T
    assert np.allclose(Phi @ quat_to_mat(other.ball_quat), quat_to_mat(b.quat), atol=1e-8)


def test_slick_tires_spit_the_ball_back_out_of_the_nip():
    cfg = settings_for(78, 0, squeeze=0.018, friction=0.3)
    assert cfg.friction < np.tan(nip_angle(cfg))
    assert not launch(cfg).exited


@pytest.mark.parametrize("axis_deg", [-60, -20, 30, 75])
def test_three_wheels_point_the_spin_axis_with_speeds_alone(axis_deg):
    cfg = settings_for(78, 2200, axis_deg)
    assert cfg.tilt_deg == 0
    r = launch(cfg)
    assert r.spin_axis_deg == pytest.approx(axis_deg, abs=1.5)


def test_two_wheels_have_to_tilt_the_head():
    cfg = settings_for(78, 2500, 30, wheel_angles_deg=TWO_WHEEL)
    assert cfg.tilt_deg == 30
    w = launch(cfg).omega
    assert np.degrees(np.arctan2(w[2], w[1])) == pytest.approx(30, abs=1)


def test_calibrate_hits_a_reachable_target():
    r = calibrate(80, 2000)
    assert r.speed / MPH == pytest.approx(80, abs=0.3)
    assert r.signed_spin / RPM == pytest.approx(2000, abs=20)


def test_mujoco_sensors_see_the_launch():
    r = Machine(settings_for(78, 2500)).launch(record_every=1)
    peak_g = np.linalg.norm(r.log["accel"], axis=1).max() / 9.81
    assert peak_g > 300  # hundreds of g while the tires grab the ball
    assert np.linalg.norm(r.log["gyro"][-1]) == pytest.approx(r.spin, rel=0.02)


def test_pybullet_port_uses_the_same_spring():
    pytest.importorskip("pybullet")
    from pitching_machine.pybullet_machine import static_stiffness

    cfg = MachineConfig()
    assert static_stiffness(cfg) == pytest.approx(cfg.contact_stiffness, rel=0.02)
