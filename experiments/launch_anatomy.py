"""Eight milliseconds inside the machine: how a curveball gets its speed and spin.

    python experiments/launch_anatomy.py
"""

import numpy as np
from _style import INK, INK_2, MUTED, SERIES, label_end, save, setup

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pitching_machine.constants import G, MPH, RPM
from pitching_machine.pitch import preset_config
from pitching_machine.machine import Machine


def main():
    setup()
    cfg = preset_config("curveball")
    res = Machine(cfg).launch(record_every=1)
    L = res.log
    wheels = range(cfg.n_wheels)
    touching = np.any([L[f"normal_{i}"] > 0 for i in wheels], axis=0)
    t0 = L["t"][touching][0]
    t = (L["t"] - t0) * 1e3  # ms since first touch
    print(res.summary())

    fig, axs = plt.subplots(3, 2, figsize=(10.5, 8.2), sharex=True)
    colors = SERIES[: cfg.n_wheels]

    ax = axs[0, 0]
    for i, c in zip(wheels, colors):
        ax.plot(t, L[f"rim_{i}"] / MPH, color=c)
        label_end(ax, t[-1], L[f"rim_{i}"][-1] / MPH, f"{cfg.labels[i]} rim")
    ax.plot(t, L["speed"] / MPH, color=INK)
    k = np.searchsorted(t, 3.5)
    label_end(ax, t[k], L["speed"][k] / MPH, "ball", dx=6, dy=-4)
    ax.axhline(cfg.ideal_speed / MPH, color=MUTED, lw=1, ls="--")
    ax.text(t[0], cfg.ideal_speed / MPH + 2, "rigid, no-slip ideal", color=INK_2, fontsize=8)
    ax.set(title="Speed", ylabel="mph")

    ax = axs[0, 1]
    ax.plot(t, L["spin"] / RPM, color=INK)
    ax.axhline(cfg.ideal_spin / RPM, color=MUTED, lw=1, ls="--")
    ax.text(t[0], cfg.ideal_spin / RPM + 80, "rigid, no-slip ideal", color=INK_2, fontsize=8)
    ax.set(title=f"Spin about the {cfg.spin_axis_deg:+.0f} deg axis (topspin +)", ylabel="rpm")

    ax = axs[1, 0]
    for i, c in zip(wheels, colors):
        ax.plot(t, L[f"normal_{i}"], color=c)
        ax.plot(t, L[f"friction_{i}"], color=c, ls=":", lw=1.4)
        ax.plot(t, cfg.friction * L[f"normal_{i}"], color=c, lw=0.8, alpha=0.5)
    k = np.argmax(L["normal_0"])
    label_end(ax, t[k], L["normal_0"][k], "normal (squash) force", dx=6, dy=6)
    k = np.argmax(L["friction_0"])
    label_end(ax, t[k], L["friction_0"][k] * 0.55, "friction (dotted),\ncapped at mu N (faint)", dx=-118, dy=0)
    ax.set(title="Contact forces", ylabel="N")

    ax = axs[1, 1]
    for i, c in zip(wheels, colors):
        ax.plot(t, 1e3 * L[f"squash_{i}"], color=c)
    ax.set(title="Squash (tire and ball overlap)", ylabel="mm")

    ax = axs[2, 0]
    for i, c in zip(wheels, colors):
        ax.plot(t, L[f"slip_{i}"] / MPH, color=c)
    ax.set(title="Slip at each contact", ylabel="mph", xlabel="ms since the ball first touched the tires")

    ax = axs[2, 1]
    acc = np.linalg.norm(L["accel"], axis=1) / G
    ax.plot(t, acc, color=INK)
    k = np.argmax(acc)
    label_end(ax, t[k], acc[k], f"peak {acc[k]:.0f} g", dx=6)
    ax.set(title="What an IMU inside the ball feels", ylabel="g", xlabel="ms since the ball first touched the tires")

    for ax in axs.flat:
        ax.set_xlim(t[0] - 1, t[-1] + 3.4)
    handles = [Line2D([], [], color=c) for c in colors] + [Line2D([], [], color=INK)]
    fig.legend(handles, [f"{lab} wheel" for lab in cfg.labels] + ["ball"], loc="upper left", ncol=4,
               bbox_to_anchor=(0.01, 0.955))
    fig.suptitle(
        f"A curveball launch: {res.speed / MPH:.1f} mph and {res.signed_spin / RPM:.0f} rpm out of "
        + " / ".join(f"{rpm:.0f}" for rpm in cfg.wheel_rpm) + " rpm wheels",
        x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "launch_anatomy.png")


if __name__ == "__main__":
    main()
