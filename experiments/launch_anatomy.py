"""Nine milliseconds inside the machine: how a curveball gets its speed and spin.

    python experiments/launch_anatomy.py
"""

import numpy as np
from _style import INK, INK_2, MUTED, SERIES, label_end, save, setup

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from knuckle.constants import G, MPH, RPM
from knuckle.pitch import preset_config
from knuckle.machine import Machine


def main():
    setup()
    cfg = preset_config("curveball")
    res = Machine(cfg).launch(record_every=1)
    L = res.log
    touching = (L["normal_top"] > 0) | (L["normal_bottom"] > 0)
    t0 = L["t"][touching][0]
    t = (L["t"] - t0) * 1e3  # ms since first touch
    print(res.summary())

    fig, axs = plt.subplots(3, 2, figsize=(10.5, 8.2), sharex=True)
    top_c, bot_c = SERIES[0], SERIES[1]

    ax = axs[0, 0]
    ax.plot(t, L["top_speed"] / MPH, color=top_c)
    ax.plot(t, L["bottom_speed"] / MPH, color=bot_c)
    ax.plot(t, L["speed"] / MPH, color=INK)
    label_end(ax, t[-1], L["top_speed"][-1] / MPH, "top rim")
    label_end(ax, t[-1], L["bottom_speed"][-1] / MPH, "bottom rim")
    label_end(ax, t[-1], L["speed"][-1] / MPH, "ball")
    ax.axhline(cfg.ideal_speed / MPH, color=MUTED, lw=1, ls="--")
    ax.text(t[0], cfg.ideal_speed / MPH + 2, "rigid, no-slip ideal", color=INK_2, fontsize=8)
    ax.set(title="Speed", ylabel="mph")

    ax = axs[0, 1]
    ax.plot(t, L["spin"] / RPM, color=INK)
    ax.axhline(cfg.ideal_spin / RPM, color=MUTED, lw=1, ls="--")
    ax.text(t[0], cfg.ideal_spin / RPM + 80, "ideal: rim-speed difference / ball diameter", color=INK_2, fontsize=8)
    ax.set(title="Spin (topspin +)", ylabel="rpm")

    ax = axs[1, 0]
    for k, c in (("top", top_c), ("bottom", bot_c)):
        ax.plot(t, L[f"normal_{k}"], color=c)
        ax.plot(t, L[f"friction_{k}"], color=c, ls=":", lw=1.4)
        ax.plot(t, cfg.friction * L[f"normal_{k}"], color=c, lw=0.8, alpha=0.5)
    i = np.argmax(L["normal_top"])
    label_end(ax, t[i], L["normal_top"][i], "normal (squash) force", dx=6, dy=6)
    j = np.argmax(L["friction_bottom"])
    label_end(ax, t[j], L["friction_bottom"][j] * 0.55, "friction (dotted),\ncapped at mu N (faint)", dx=-118, dy=0)
    ax.set(title="Contact forces", ylabel="N")

    ax = axs[1, 1]
    for k, c in (("top", top_c), ("bottom", bot_c)):
        ax.plot(t, 1e3 * L[f"squash_{k}"], color=c)
    ax.set(title="Squash (tire and ball overlap)", ylabel="mm")

    ax = axs[2, 0]
    for k, c in (("top", top_c), ("bottom", bot_c)):
        ax.plot(t, L[f"slip_{k}"] / MPH, color=c)
    ax.set(title="Slip at each contact", ylabel="mph", xlabel="ms since the ball first touched the tires")

    ax = axs[2, 1]
    acc = np.linalg.norm(L["accel"], axis=1) / G
    ax.plot(t, acc, color=INK)
    k = np.argmax(acc)
    label_end(ax, t[k], acc[k], f"peak {acc[k]:.0f} g", dx=6)
    ax.set(title="What an IMU inside the ball feels", ylabel="g", xlabel="ms since the ball first touched the tires")

    for ax in axs.flat:
        ax.set_xlim(t[0] - 1, t[-1] + 2.6)
    handles = [Line2D([], [], color=top_c), Line2D([], [], color=bot_c), Line2D([], [], color=INK)]
    fig.legend(handles, ["top wheel", "bottom wheel", "ball"], loc="upper left", ncol=3, bbox_to_anchor=(0.01, 0.955))
    fig.suptitle(
        f"A curveball launch: {res.speed / MPH:.1f} mph and {res.signed_spin / RPM:.0f} rpm out of "
        f"{cfg.top_rpm:.0f} / {cfg.bottom_rpm:.0f} rpm wheels",
        x=0.01, ha="left", fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "launch_anatomy.png")


if __name__ == "__main__":
    main()
