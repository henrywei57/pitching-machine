"""Identical setups, different paths.

Same machine, same wheel speeds, same aim. The only thing that changes from
pitch to pitch is how the ball happened to sit in the chute, which no real
machine controls. Spinning pitches don't care. The knuckleball does.

Also checks the solver: the same pitch thrown twice is bit-for-bit identical,
so the scatter is the model (the seams), not numerical noise.

    python experiments/plate_scatter.py
"""

import numpy as np
from _style import INK_2, MUTED, PITCH_COLOR, SURFACE, label_end, save, setup

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from knuckle.aero import random_rotations
from knuckle.constants import FOOT, INCH
from knuckle.flight import fly
from knuckle.machine import _rx, head_rotation, launch, quat_to_mat
from knuckle.pitch import PRESETS, preset_config, throw

N = 300


def chute_composer(res):
    """The launch never looks at seam orientation (the ball is a sphere to the
    tires), so the ball's turn during launch is the same whatever way it sat in
    the chute: R_exit = Phi @ R_start. Returns a function chute -> R_exit."""
    cfg = res.config
    R_head = head_rotation(cfg)
    Phi = quat_to_mat(res.quat) @ (R_head @ quat_to_mat(cfg.ball_quat)).T
    return lambda chute: Phi @ R_head @ chute


def fly_batch(res, R_exit):
    n = len(R_exit)
    return fly(np.tile(res.position, (n, 1)), np.tile(res.velocity, (n, 1)), np.tile(res.omega, (n, 1)), R_exit,
               record_every=10**9)


def main():
    setup()
    rng = np.random.default_rng(7)
    chutes = random_rotations(N, rng)

    # 1. Determinism, and the chute shortcut checked against two full launches.
    cfg = preset_config("knuckleball")
    a, b = throw(cfg), throw(cfg)
    print("same knuckleball twice, max difference at the plate:",
          np.abs(a.flight.plate_pos - b.flight.plate_pos).max(), "m")
    q = rng.normal(size=4)
    other = launch(a.launch.config.with_(ball_quat=tuple(q / np.linalg.norm(q))))
    predicted = chute_composer(a.launch)(quat_to_mat(other.config.ball_quat))
    print("chute shortcut vs. a full re-launch, max rotation-matrix error:",
          np.abs(predicted - quat_to_mat(other.quat)).max())

    # 2. Where 300 pitches of each kind cross the plate.
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.4), sharex=True, sharey=True)
    spreads = {}
    for ax, name in zip(axs, PRESETS):
        res = throw(preset_config(name)).launch
        f = fly_batch(res, chute_composer(res)(chutes))
        side, height = f.plate_pos[:, 1] / INCH, f.plate_pos[:, 2] / FOOT
        spreads[name] = (np.nanstd(side), np.nanstd(height) * 12)
        ax.add_patch(Rectangle((-8.5, 1.5), 17, 2.0, fill=False, ec=MUTED, lw=1))
        ax.scatter(side, height, s=14, color=PITCH_COLOR[name], ec=SURFACE, lw=0.6, zorder=3)
        ax.set_title(name)
        ax.text(0.03, 0.03, f"spread (1 sd)\n{spreads[name][0]:.1f} in side\n{spreads[name][1]:.1f} in height",
                transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
        ax.set_aspect(12)  # side in inches, height in feet: same physical scale
        ax.set_xlabel("side (in), catcher's view")
        print(f"{name:14s} sd {spreads[name][0]:.1f} in side, {spreads[name][1]:.1f} in height, "
              f"{np.isnan(side).sum()} landed short")
    axs[0].set_ylabel("height at the plate (ft)")
    axs[0].set_xlim(-36, 36)
    axs[0].set_ylim(0.0, 5.0)
    fig.suptitle(f"{N} pitches each, same machine settings, ball dropped in the chute any which way",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "plate_scatter.png")

    # 3. It isn't noise: plate location as a smooth function of the chute angle.
    angles = np.radians(np.arange(0.0, 360.0, 0.5))
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for name in ("curveball", "knuckleball"):
        res = throw(preset_config(name)).launch
        R = chute_composer(res)(np.array([_rx(t) for t in angles]))  # turn the ball about the barrel
        f = fly_batch(res, R)
        deg = np.degrees(angles)
        for ax, y, unit in ((axs[0], f.plate_pos[:, 1] / INCH, "in"), (axs[1], f.plate_pos[:, 2] / INCH, "in")):
            y = y - np.nanmean(y)
            ax.plot(deg, y, color=PITCH_COLOR[name])
            k = len(y) - 1 if name == "curveball" else int(np.argmax(y))
            label_end(ax, deg[k], y[k], name, dx=4 if name == "curveball" else 8, dy=0 if name == "curveball" else 4)
        slope = np.nanmax(np.abs(np.gradient(f.plate_pos[:, 1] / INCH, np.degrees(angles))))
        print(f"{name}: up to {slope:.2f} in of side-to-side change per degree of chute rotation")
    axs[0].set(title="Side at the plate", ylabel="in from average", xlabel="ball's rotation in the chute (deg)")
    axs[1].set(title="Height at the plate", ylabel="in from average", xlabel="ball's rotation in the chute (deg)")
    for ax in axs:
        ax.set_xticks(range(0, 361, 45))
        ax.set_xlim(0, 410)
    fig.tight_layout()
    save(fig, "chute_sensitivity.png")


if __name__ == "__main__":
    main()
