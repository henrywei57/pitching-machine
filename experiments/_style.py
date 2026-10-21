"""Shared figure style and paths for the experiment scripts."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
MEDIA = ROOT / "media"
sys.path.insert(0, str(ROOT))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
# Categorical slots, always assigned in this order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PITCH_COLOR = {"fastball": SERIES[0], "curveball": SERIES[1], "knuckle-curve": SERIES[2], "knuckleball": SERIES[3]}


def setup():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "text.color": INK,
            "font.size": 9.5,
        }
    )


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")
    return path


def label_end(ax, x, y, text, color=INK_2, dx=4, dy=0, **kw):
    """Direct label at the end of a line, in text ink (the line carries the color)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=8.5, color=color, **kw)
