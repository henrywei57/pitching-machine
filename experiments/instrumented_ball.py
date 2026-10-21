"""What if the ball itself is the instrument?

Throw a knuckleball with the full model standing in for reality. Put a
virtual IMU in it (knuckle/imu.py), with noise and bias. From nothing but the
IMU's samples, the starting orientation and the radar speed, recover the
aerodynamic force on the ball at every instant. Then compare that to what a
simple model (constant drag, no seams) predicts. The simulation predicts, the
ball answers, and the difference is the physics the simple model is missing.

    python experiments/instrumented_ball.py
"""

import numpy as np
from _style import INK, INK_2, MUTED, PITCH_COLOR, SERIES, label_end, save, setup

import matplotlib.pyplot as plt
from knuckle.aero import Aero, acceleration
from knuckle.constants import FOOT, INCH
from knuckle.flight import fly
from knuckle.imu import IMUSpec, model_residual, reconstruct, record
from knuckle.machine import quat_to_mat
from knuckle.pitch import preset_config, throw

NAIVE = Aero(drag="constant", cd_constant=0.35, seam=False)


def smooth(y, n=25):
    k = np.ones(n) / n
    return np.convolve(np.pad(y, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")


def main():
    setup()
    # A chute orientation that gives this knuckleball a good dance.
    q = np.random.default_rng(11).normal(size=4)
    cfg = preset_config("knuckleball").with_(ball_quat=tuple(q / np.linalg.norm(q)))
    res = throw(cfg).launch
    R0 = quat_to_mat(res.quat)
    truth = fly(res.position, res.velocity, res.omega, R0, Aero())
    naive = fly(res.position, res.velocity, res.omega, R0, NAIVE)

    spec = IMUSpec()
    rec = record(truth, spec=spec, seed=3)
    rc = reconstruct(rec, R0, res.velocity)
    residual, _ = model_residual(rc, NAIVE)

    # Ground truth to grade the measurement against (same time samples).
    idx = np.searchsorted(truth.t, rec.t)
    _, parts = acceleration(truth.vel[idx, 0], truth.omega[[0] * len(idx)], truth.R[idx, 0], Aero(), details=True)

    # Dead-reckon the path from the IMU alone.
    pos = np.empty_like(rc.vel)
    pos[0] = res.position
    for i in range(1, len(pos)):
        pos[i] = pos[i - 1] + 0.5 * (rc.vel[i] + rc.vel[i - 1]) * (rc.t[i] - rc.t[i - 1])

    def at_plate(t, p):
        k = np.searchsorted(p[:, 0], truth.plate_pos[0, 0])
        f = (truth.plate_pos[0, 0] - p[k - 1, 0]) / (p[k, 0] - p[k - 1, 0])
        return p[k - 1] + f * (p[k] - p[k - 1])

    plate_true = truth.plate_pos[0]
    plate_naive = naive.plate_pos[0]
    plate_imu = at_plate(rc.t, pos)
    miss = np.linalg.norm((plate_naive - plate_true)[1:]) / INCH
    imu_err = np.linalg.norm((plate_imu - plate_true)[1:]) / INCH
    print(res.summary().splitlines()[2])
    print(f"simple model misses the real plate location by {miss:.1f} in")
    print(f"dead-reckoning the IMU lands within {imu_err:.2f} in of it")
    in_flight = rc.t < truth.plate_time[0]
    print(f"measured C_D ranges {smooth(rc.cd)[in_flight].min():.2f} to {smooth(rc.cd)[in_flight].max():.2f} "
          f"during this one pitch (simple model assumes {NAIVE.cd_constant})")

    t_ms = 1e3 * rec.t
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.4))
    end = 1e3 * truth.plate_time[0]

    ax = axs[0, 0]
    # Body-frame axes on a neutral ramp: blue/orange mean world sideways/vertical below.
    for i, (axis, c) in enumerate(zip("xyz", ("#b8b7b1", "#7d7c77", INK))):
        ax.plot(t_ms, rec.accel[:, i], color=c, lw=0.9)
        label_end(ax, t_ms[-1], smooth(rec.accel[:, i])[-1], f"{axis} (ball's own axes)")
    ax.set(title="Raw accelerometer, 1 kHz", ylabel="m/s^2", xlabel="ms after release")

    ax = axs[0, 1]
    ax.plot(t_ms, rc.cd, color=MUTED, lw=0.6, alpha=0.6)
    ax.plot(t_ms, smooth(rc.cd), color=PITCH_COLOR["knuckleball"], lw=2)
    ax.plot(t_ms, parts["cd"], color=INK, lw=1.2, ls="--")
    ax.axhline(NAIVE.cd_constant, color=INK_2, lw=1, ls=":")
    label_end(ax, t_ms[-1], smooth(rc.cd)[-1], "measured by the ball", dy=8)
    label_end(ax, t_ms[-1], parts["cd"][-1], "truth", dy=-8)
    label_end(ax, t_ms[-1], NAIVE.cd_constant, "simple model's guess")
    ax.set(title="Drag coefficient, as the ball reports it", ylabel="C_D", xlabel="ms after release")

    ax = axs[1, 0]
    for i, (axis, c) in enumerate((("sideways", SERIES[0]), ("vertical", SERIES[1])), start=1):
        ax.plot(t_ms, smooth(residual[:, i]), color=c, lw=2)
        ax.plot(t_ms, parts["seam"][:, i], color=INK, lw=1, ls="--")
        label_end(ax, t_ms[-1], smooth(residual[:, i])[-1], axis)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(title="Measured minus simple model = the missing physics", ylabel="m/s^2",
           xlabel="ms after release (dashed: the seam force, from the truth)")

    ax = axs[1, 1]
    keep = truth.pos[:, 0, 0] <= plate_true[0]
    ax.plot(truth.pos[keep, 0, 0] / FOOT, truth.pos[keep, 0, 1] / INCH, color=INK)
    keep_n = naive.pos[:, 0, 0] <= plate_true[0]
    ax.plot(naive.pos[keep_n, 0, 0] / FOOT, naive.pos[keep_n, 0, 1] / INCH, color=INK_2, ls=":")
    keep_i = pos[:, 0] <= plate_true[0]
    ax.plot(pos[keep_i, 0] / FOOT, pos[keep_i, 1] / INCH, color=PITCH_COLOR["knuckleball"], lw=2.5, alpha=0.6)
    label_end(ax, plate_true[0] / FOOT, plate_true[1] / INCH, "what happened\n(and the IMU's\ndead reckoning)")
    label_end(ax, plate_naive[0] / FOOT, plate_naive[1] / INCH, "simple model")
    ax.set_xlim(0, plate_true[0] / FOOT + 12)
    ax.set(title="Top view: the gap at the plate", ylabel="toward 1B (in)", xlabel="distance from machine (ft)")

    for ax in axs[:, 0]:
        ax.set_xlim(0, end + 160)
    axs[0, 1].set_xlim(0, end + 160)
    fig.tight_layout()
    save(fig, "instrumented_ball.png")


if __name__ == "__main__":
    main()
