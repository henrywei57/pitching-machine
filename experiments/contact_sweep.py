"""How squash and grip decide what comes out of the machine.

Left two panels: a 90 mph / 2200 rpm fastball dialed in with the textbook
(rigid, no-slip) settings, swept over tire squeeze for three friction
coefficients. Right panel: the knuckle-curve question. Same curveball wheel
speeds, more grip.

    python experiments/contact_sweep.py
"""

import numpy as np
from _style import INK, INK_2, MUTED, PITCH_COLOR, label_end, save, setup

import matplotlib.pyplot as plt
from knuckle.constants import RPM
from knuckle.machine import launch, nip_angle, settings_for
from knuckle.pitch import PRESETS, preset_config


# Ordinal ramps: friction on the blue ramp (light = slick), squeeze on a neutral one.
MU_COLOR = {0.4: "#86b6ef", 0.7: "#2a78d6", 1.0: "#104281"}
SQUEEZE_COLOR = {0.008: "#b8b7b1", 0.010: "#7d7c77", 0.0127: INK}
GRID_LINE = "#c9c8c3"


def main():
    setup()
    squeezes = np.linspace(0.004, 0.020, 17)
    mm = 1e3 * squeezes
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))

    for mu, color in MU_COLOR.items():
        sp, sn = [], []
        for sq in squeezes:
            r = launch(settings_for(90, -2200, squeeze=sq, friction=mu))
            sp.append(100 * r.speed_efficiency if r.exited else np.nan)
            sn.append(100 * r.spin_efficiency if r.exited else np.nan)
        sp, sn = np.array(sp), np.array(sn)
        axs[0].plot(mm, sp, color=color, marker="o", ms=3.5, label=f"mu = {mu}")
        axs[1].plot(mm, sn, color=color, marker="o", ms=3.5)
        # Label each spin curve where it climbs through 50%.
        k = np.nanargmin(np.abs(sn - 50))
        label_end(axs[1], mm[k], sn[k], f"mu {mu}", dx=-6, dy=0, ha="right")
        if np.isnan(sp).any():
            first = mm[np.isnan(sp)][0]
            tan_nip = np.tan(nip_angle(settings_for(90, -2200, squeeze=first / 1e3)))
            print(f"mu {mu}: spat back out of the nip from {first:.1f} mm squeeze (tan(nip angle) = {tan_nip:.2f})")
            axs[0].annotate(f"mu {mu}: spat back out\n(mu < tan nip angle)", (mm[np.isfinite(sp)][-1], sp[np.isfinite(sp)][-1]),
                            xytext=(8, -26), textcoords="offset points", fontsize=8, color=INK_2,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    axs[0].legend(loc="lower right")
    axs[0].set(title="Exit speed, fastball settings", xlabel="squeeze per tire (mm)", ylabel="% of rigid no-slip speed")
    axs[1].set(title="Exit spin, fastball settings", xlabel="squeeze per tire (mm)", ylabel="% of rigid no-slip spin")
    for ax in axs[:2]:
        ax.axhline(100, color=MUTED, lw=1, ls="--")
        ax.axvline(12.7, color=GRID_LINE, lw=1)
        ax.text(12.9, ax.get_ylim()[0] + 2, "default", color=INK_2, fontsize=8)

    # Grip: the same curveball dial settings, friction from slick to tacky.
    ax = axs[2]
    mus = np.round(np.arange(0.3, 1.31, 0.05), 2)
    for sq, color in SQUEEZE_COLOR.items():
        spins = []
        for mu in mus:
            r = launch(preset_config("curveball", friction=mu, squeeze=sq))
            spins.append(r.signed_spin / RPM if r.exited else np.nan)
        ax.plot(mus, spins, color=color, marker="o", ms=2.5)
        k = np.nanargmin(np.abs(np.array(spins) - 1500))  # label where the curves are apart
        label_end(ax, mus[k], spins[k], f"{1e3 * sq:g} mm", dx=7, dy=-3)
        if sq == 0.008:
            gain = np.interp(1.0, mus, spins) - np.interp(0.7, mus, spins)
            print(f"8 mm squeeze: knuckle-curve grip adds {gain:.0f} rpm over the curveball")
    ideal = preset_config("curveball").ideal_spin / RPM
    ax.axhline(ideal, color=MUTED, lw=1, ls="--")
    ax.text(1.31, ideal - 120, "rigid no-slip ideal", color=INK_2, fontsize=8, ha="right")
    for name, mu in (("curveball", 0.7), ("knuckle-curve", PRESETS["knuckle-curve"]["friction"])):
        ax.axvline(mu, color=PITCH_COLOR[name], lw=1.2)
        ax.text(mu + 0.015, 0.04, f"{name} grip", color=INK_2, fontsize=8, ha="left", va="bottom", transform=ax.get_xaxis_transform())
    ax.text(0.04, 0.70, "squeeze per tire:", color=INK_2, fontsize=8, ha="left", transform=ax.transAxes)
    ax.set(title="Grip: same curveball wheels, more friction", xlabel="tire-leather friction coefficient", ylabel="exit spin (rpm)")

    fig.tight_layout()
    save(fig, "contact_sweep.png")


if __name__ == "__main__":
    main()
