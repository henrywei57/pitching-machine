"""Fit the seam-force kernel to the knuckleball wind-tunnel curve, and show
what the fitted model says about drag.

Watts & Sawyer (1975) measured the sideways force on a non-spinning baseball
in a wind tunnel as they turned it about a vertical axis. Giordano &
Nakanishi fit their data with

    F_lat / mg = 0.5 [sin 4t - 0.25 sin 8t + 0.08 sin 12t - 0.025 sin 16t]

The model in pitching_machine/aero.py instead puts a kernel on every bit of seam
(pitching_machine/seams.py) depending on how far it sits from the front of the ball.
This script finds the smallest kernel that makes the 3-D model reproduce
that curve when the ball is turned the same way, then prints it for
SEAM_KERNEL.

    python experiments/fit_seam_model.py
"""

import numpy as np
from _style import INK, INK_2, MUTED, SERIES, save, setup

import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pitching_machine import aero
from pitching_machine.aero import Aero, drag_coefficient, kernel, random_rotations, seam_terms
from pitching_machine.constants import BALL_AREA, BALL_MASS, G, MPH, RHO_AIR
from pitching_machine.machine import _rz
from pitching_machine.seams import seam_points


def giordano(theta):
    return 0.5 * (np.sin(4 * theta) - 0.25 * np.sin(8 * theta) + 0.08 * np.sin(12 * theta) - 0.025 * np.sin(16 * theta))


def turned_ball(theta):
    """Orientations of a ball turned by theta about the vertical, starting from
    the mirror-symmetric pose (flow along (1, 1, 0)/sqrt 2 in the seam frame)."""
    return np.array([_rz(t) @ _rz(-np.pi / 4) for t in theta])


def main():
    setup()
    theta = np.radians(np.arange(0.0, 180.0, 1.0))
    R = turned_ball(theta)
    vhat = np.tile([1.0, 0.0, 0.0], (len(theta), 1))
    q = 0.5 * RHO_AIR * aero.SEAM_REF_SPEED**2 * BALL_AREA
    target = giordano(theta) * BALL_MASS * G / q  # as a coefficient at the reference speed
    pts, w = seam_points(360)

    def model(p):
        saved = aero.SEAM_KERNEL
        aero.SEAM_KERNEL = (tuple(p[:3]), tuple(p[3:]))
        try:
            return seam_terms(R, vhat, pts, w)[0]
        finally:
            aero.SEAM_KERNEL = saved

    # The sideways force alone doesn't pin the kernel down: many kernels fit it
    # to 0.03%. A small ridge penalty on the amplitudes picks the smallest one
    # that still fits (misfit ~0.05%), and makes the fit repeatable.
    ridge = 1e-3

    def residual(p):
        return np.r_[model(p)[:, 1] - target, ridge * p[[0, 3]]]

    best = None
    for start in ([-5, 90, 10, -1, 115, 8], [-8, 95, 12, -2, 120, 6], [-3, 100, 15, -3, 120, 10]):
        r = least_squares(residual, start, bounds=([-50, 40, 2] * 2, [50, 150, 40] * 2))
        best = r if best is None or r.cost < best.cost - 1e-12 else best
    p = best.x
    C = model(p)
    rms = np.sqrt(np.mean((C[:, 1] - target) ** 2)) / np.sqrt(np.mean(target**2))
    print("SEAM_KERNEL = ((%.3f, %.3f, %.3f), (%.3f, %.3f, %.3f))" % tuple(p))
    print(f"rms misfit: {100 * rms:.2f}% of the measured curve")
    in_package = np.array(aero.SEAM_KERNEL).ravel()
    if np.abs(in_package - p).max() > 0.01:
        print(f"note: pitching_machine/aero.py has {tuple(map(tuple, aero.SEAM_KERNEL))}")

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))

    # Where on the ball a seam matters.
    ax = axs[0]
    alpha = np.linspace(0, 180, 721)
    ax.plot(alpha, kernel(alpha), color=INK)
    ax.axvspan(80, 125, color=SERIES[0], alpha=0.08, lw=0)
    ax.text(128, kernel(np.array([100.0]))[0] * 0.75, "seams here,\nwhere the flow\nleaves the ball,\npush the ball\naway from them", color=INK_2, fontsize=8, ha="left")
    ax.set(title="Seam kernel", xlabel="angle from the front of the ball (deg)", ylabel="K (force per unit seam)")
    ax.set_xticks([0, 45, 90, 135, 180])

    # The fit.
    ax = axs[1]
    deg = np.degrees(theta)
    to_mg = q / (BALL_MASS * G)
    ax.plot(deg, giordano(theta), color=MUTED, lw=4, alpha=0.6, label="wind tunnel (Giordano fit)")
    ax.plot(deg, C[:, 1] * to_mg, color=SERIES[0], label="model, sideways")
    ax.plot(deg, C[:, 2] * to_mg, color=SERIES[1], lw=1.2, label="model, vertical (prediction)")
    ax.set_ylim(-0.62, 0.9)
    ax.legend(loc="upper left", ncol=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(title="Knuckleball force vs. ball angle", xlabel="ball turned about vertical (deg)", ylabel="force / weight")
    ax.set_xticks([0, 45, 90, 135, 180])

    # Drag is not one number for a ball that doesn't spin.
    ax = axs[2]
    m = Aero()
    rng = np.random.default_rng(0)
    Rr = random_rotations(2000, rng)
    _, trip = seam_terms(Rr, np.tile([1.0, 0.0, 0.0], (len(Rr), 1)), *m._seam)
    speeds = np.linspace(15, 50, 141)
    cds = np.array([drag_coefficient(np.full_like(trip, v), trip, m) for v in speeds])
    lo, hi = np.percentile(cds, [5, 95], axis=1)
    mph = speeds / MPH
    ax.fill_between(mph, lo, hi, color=SERIES[0], alpha=0.25, lw=0, label="middle 90% of seam orientations")
    ax.plot(mph, drag_coefficient(speeds, None, m), color=INK, label="spinning-ball curve (Adair / Giordano)")
    ax.legend(loc="lower left")
    ax.axvspan(60, 72, color=MUTED, alpha=0.12, lw=0)
    ax.text(66, 0.535, "knuckleball\nspeeds", color=INK_2, fontsize=8, ha="center")
    ax.set(title="Drag coefficient, non-spinning ball", xlabel="speed (mph)", ylabel="C_D")
    at_67 = np.array([drag_coefficient(np.full_like(trip, 67 * MPH), trip, m)])[0]
    print(f"C_D at 67 mph over seam orientations: {at_67.min():.2f} to {at_67.max():.2f} (median {np.median(at_67):.2f})")

    fig.tight_layout()
    save(fig, "seam_model.png")


if __name__ == "__main__":
    main()
