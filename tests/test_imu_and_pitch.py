import mujoco
import numpy as np
import pytest

from pitching_machine.aero import Aero, random_rotations
from pitching_machine.constants import INCH, MPH
from pitching_machine.flight import fly
from pitching_machine.imu import IMUSpec, model_residual, reconstruct, record
from pitching_machine.machine import Machine
from pitching_machine.pitch import ZONE_MIDDLE, main, preset_config, rpm_config, throw
from pitching_machine.view import CAMERAS, KEY_BACKSPACE, KEY_ENTER, WheelEntry, overlay, parse_command, replay

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


def test_cli_sets_each_wheel_by_hand(capsys):
    main(["--rpm", "3200", "2600", "2000"])
    out = capsys.readouterr().out
    assert "top 3200   lower 3B 2600   lower 1B 2000 rpm" in out
    assert "axis +30 deg" in out  # top fastest, lower 1B slowest: a curveball breaking toward 1B


def test_cli_wants_one_rpm_per_wheel(capsys):
    with pytest.raises(SystemExit):
        main(["--rpm", "3000", "2000"])
    assert "top, lower 3B, lower 1B" in capsys.readouterr().err


def test_retarget_matches_a_freshly_built_machine():
    a = preset_config("curveball")
    q = np.array([0.2, 0.9, -0.3, 0.1])
    b = a.with_(wheel_rpm=(3000.0, 2400.0, 2100.0), elevation_deg=3.0, azimuth_deg=-1.0,
                ball_quat=tuple(q / np.linalg.norm(q)))
    m = Machine(a, scene=True)
    m.retarget(b)
    got, want = m.launch(), Machine(b).launch()
    assert np.allclose(got.velocity, want.velocity, atol=1e-9)
    assert np.allclose(got.omega, want.omega, atol=1e-7)
    assert np.allclose(got.position, want.position, atol=1e-12)


def test_retarget_refuses_a_different_machine():
    m = Machine(preset_config("curveball"))
    with pytest.raises(ValueError):
        m.retarget(m.cfg.with_(squeeze=0.008))


def test_wheel_rpm_has_to_fit_the_sliders():
    with pytest.raises(ValueError):
        Machine(rpm_config([7000, 2000, 2000]))
    with pytest.raises(ValueError):
        Machine(rpm_config([-100, 2000, 2000]))


def test_viewer_has_the_cameras_and_rpm_sliders():
    m = Machine(preset_config("knuckleball"), scene=True).model
    assert [m.camera(i).name for i in range(m.ncam)] == CAMERAS
    assert [m.actuator(i).name for i in range(m.nu)] == ["top_rpm", "lower_3B_rpm", "lower_1B_rpm"]
    assert list(m.actuator_ctrlrange[0]) == [0.0, 6000.0]


def test_viewer_overlay_shows_wheels_and_result():
    p = throw(rpm_config([3200, 2600, 2000]))
    left, right = overlay(p, "flight")
    assert left.splitlines()[:3] == ["top wheel", "lower 3B wheel", "lower 1B wheel"]
    assert right.splitlines()[:3] == ["3200 rpm", "2600 rpm", "2000 rpm"]
    assert "axis +30 deg" in right


@pytest.mark.parametrize("typed, want", [
    ("3500 2600 2000", (3500, 2600, 2000)),
    ("top 3500", (3500, 2600, 2000)),
    ("3b 2400", (3200, 2400, 2000)),
    ("lower 1B 1800", (3200, 2600, 1800)),
    ("top 3500, 1b 2100", (3500, 2600, 2100)),
])
def test_typed_wheel_speeds(typed, want):
    assert parse_command(typed, rpm_config([3200, 2600, 2000])) == want


@pytest.mark.parametrize("typed", ["3000 2000", "top", "hello 5", "top 9000", ""])
def test_typed_nonsense_gets_a_hint(typed):
    with pytest.raises(ValueError):
        parse_command(typed, rpm_config([3200, 2600, 2000]))


def test_typing_a_preset_loads_its_wheels():
    assert parse_command("knuckleball", rpm_config([3200, 2600, 2000])) == preset_config("knuckleball").wheel_rpm


def _type_into(entry, keys, current=(3200, 2600, 2000)):
    """Feed a string to WheelEntry: digits, '\n' for Enter, '<' for Backspace."""
    out = None
    for ch in keys:
        code = {"\n": KEY_ENTER, "<": KEY_BACKSPACE}.get(ch, ord(ch))
        got = entry.key(code, current)
        out = got if got is not None else out
    return out


def test_typing_in_the_window_sets_each_wheel():
    e = WheelEntry(["top", "lower 3B", "lower 1B"])
    assert _type_into(e, "3500\n2400\n2100\n") == (3500, 2400, 2100)
    assert not e.active


def test_empty_enter_keeps_a_wheel_and_backspace_fixes_typos():
    e = WheelEntry(["top", "lower 3B", "lower 1B"])
    assert _type_into(e, "35009\n") is None  # only 4 digits fit
    assert _type_into(e, "\n18<9\n") == (3500, 2600, 19)


def test_typing_too_fast_is_refused_on_screen():
    e = WheelEntry(["top", "lower 3B", "lower 1B"])
    assert _type_into(e, "9000\n") is None
    assert "6000" in e.box((3200, 2600, 2000))[1]
    assert e.active and e.wheel == 0


def test_backspace_from_the_viewer_is_noticed():
    m = Machine(preset_config("curveball"), scene=True)
    assert not m.was_reset
    mujoco.mj_resetData(m.model, m.data)
    assert m.was_reset


# -- PitchScript --------------------------------------------------------------------

def run_script(src):
    from pitching_machine.script import Interpreter

    return Interpreter().run(src)


def test_script_makes_a_pitch_and_reads_it(capsys):
    v = run_script('x = pitch(3200, 2600, 2000)\nprint(f"{x.speed:.1f}", x.axis)\nr = x.result()\n')
    out = capsys.readouterr().out
    assert out.split() == [f"{v['x'].speed:.1f}", str(v["x"].axis)]
    assert "lower 1B 2000" in v["r"]


def test_script_loops_lists_and_presets():
    v = run_script(
        "ps = []\n"
        "for top in range(3000, 3401, 400):\n"
        "    ps.append(pitch(top, 2600, 2000))\n"
        "k = pitch('knuckleball', color='red', seed=2)\n"
        "faster = ps[1].speed > ps[0].speed\n"
    )
    assert len(v["ps"]) == 2 and v["faster"] is True
    assert v["k"].color == "red"
    assert [p.color for p in v["ps"]] == ["orange", "blue"]  # handed out in order


def test_script_speed_and_spin_form():
    v = run_script("p = pitch(speed=80, spin=2200, axis=30, calibrate=True)\n")
    assert abs(v["p"].speed - 80) < 0.5 and abs(v["p"].spin - 2200) < 50


@pytest.mark.parametrize("src, words", [
    ("x = pitch(3200, 2600)", "needs 3 wheel rpms"),
    ("import os", "doesn't have import"),
    ("def f():\n    pass", "doesn't have making your own functions"),
    ("print(nope)", "don't know what 'nope'"),
    ("x = pitch('curveball')\nprint(x._p)", "doesn't have '._p'"),
    ("x = pitch(9000, 2600, 2000)", "between 0 and 6000"),
    ("x = pitch('curveball', colour='red')", "no option called colour="),
    ("x = pitch('curveball', color='plaid')", "unknown color"),
    ("print(1 / 0)", "dividing by zero"),
    ("while True:\n    pass", "is it stuck"),
])
def test_script_mistakes_say_what_went_wrong(src, words):
    from pitching_machine.script import ScriptError

    with pytest.raises(ScriptError) as e:
        run_script(src)
    assert words in str(e.value)
    assert e.value.line is not None


def test_script_saves_a_gif(tmp_path):
    out = tmp_path / "two.gif"
    run_script(f'a = pitch("curveball")\nb = pitch("knuckleball")\ngif("{out}", a, b, camera="side")\n')
    assert out.stat().st_size > 10_000


def test_examples_parse():
    import ast
    from pathlib import Path

    files = sorted(Path(__file__).parents[1].glob("examples/*.pitch"))
    assert len(files) >= 4
    for f in files:
        ast.parse(f.read_text())
