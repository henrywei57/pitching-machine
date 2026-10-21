"""Same machine, different solvers.

Each row launches the same three pitches. If a row disagrees with the others,
that's the solver (or its contact model) talking, not the physics. Every engine
is also run twice to check it gives bit-identical answers.

    python experiments/compare_engines.py      (writes figures/engine_comparison.md)
"""

import numpy as np
from _style import FIGURES, ROOT

from knuckle.constants import MPH, RPM
from knuckle.machine import launch, settings_for

PITCHES = {"fastball": (90, -2200), "curveball": (78, 2500), "knuckleball": (67, 0)}


def mujoco(contact_model, dt):
    def run(cfg):
        r = launch(cfg.with_(contact_model=contact_model, timestep=dt))
        return r.exited, r.speed, r.signed_spin, (r.velocity, r.omega)
    return run


def bullet(dt):
    from knuckle.pybullet_machine import launch as bl

    def run(cfg):
        r = bl(cfg, timestep=dt)
        return r.exited, r.speed, r.signed_spin(), (r.velocity, r.omega)
    return run


def ideal(cfg):
    return True, cfg.ideal_speed, cfg.ideal_spin, None


ENGINES = [
    ("rigid, no slip (the textbook formula)", ideal),
    ("MuJoCo + Coulomb contact law, dt 10 us (this repo's default)", mujoco("coulomb", 1e-5)),
    ("MuJoCo + Coulomb contact law, dt 50 us", mujoco("coulomb", 5e-5)),
    ("MuJoCo's own soft contact, dt 10 us", mujoco("mujoco", 1e-5)),
    ("PyBullet compliant contact, dt 10 us", bullet(1e-5)),
    ("PyBullet compliant contact, dt 100 us", bullet(1e-4)),
]


def main():
    rows = []
    for name, run in ENGINES:
        cells, repeatable = [], True
        for pitch, (v, s) in PITCHES.items():
            cfg = settings_for(v, s)
            ok, speed, spin, raw = run(cfg)
            if raw is not None:
                again = run(cfg)[3]
                repeatable &= all(np.array_equal(a, b) for a, b in zip(raw, again))
            cells.append(f"{speed / MPH:.1f} mph, {spin / RPM:+.0f} rpm" if ok else "stuck")
        rows.append((name, cells, "n/a" if raw is None else ("yes" if repeatable else "**no**")))
        print(name, cells, rows[-1][2])

    head = "| engine | " + " | ".join(f"{p} ({v} mph, {s:+d} rpm dialed)" for p, (v, s) in PITCHES.items()) + " | repeatable |"
    lines = [head, "|" + "---|" * (len(PITCHES) + 2)]
    lines += [f"| {n} | " + " | ".join(c) + f" | {r} |" for n, c, r in rows]
    out = FIGURES / "engine_comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
