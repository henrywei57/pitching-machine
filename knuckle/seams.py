"""Baseball seam geometry.

The seam is modeled with the classic "tennis ball curve", a closed curve that
lies exactly on the unit sphere and splits it into two congruent halves:

    x = a cos t + b cos 3t
    y = a sin t - b sin 3t
    z = 2 sqrt(ab) sin 2t,        a + b = 1

Its symmetries are the ones a real baseball has: a 180 degree turn about each
coordinate axis maps the seam onto itself, and a 90 degree turn about z combined
with a mirror through the equator does too. That second symmetry is why the
knuckleball force repeats every 90 degrees of ball rotation.
"""

import numpy as np

# Shape parameter. b = 0.3 puts the U-turns of the seam at about 66 degrees of
# latitude, which looks like a baseball. (b -> 0 is a great circle, b = 0.5
# pinches the two halves together at the poles.)
SEAM_B = 0.3


def seam_points(n=360, b=SEAM_B):
    """Return (points, weights) for n samples along the seam.

    points  -- (n, 3) unit vectors in the ball's body frame
    weights -- (n,) arc-length fraction carried by each sample, sums to 1
    """
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    a = 1.0 - b
    c = 2.0 * np.sqrt(a * b)
    pts = np.stack(
        [a * np.cos(t) + b * np.cos(3 * t), a * np.sin(t) - b * np.sin(3 * t), c * np.sin(2 * t)],
        axis=1,
    )
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    w = 0.5 * (seg + np.roll(seg, 1))
    return pts, w / w.sum()


def seam_length(b=SEAM_B, radius=1.0, n=4000):
    pts, _ = seam_points(n, b)
    return radius * np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum()


# Body frame in which the knuckleball reference angle theta = 0 is defined:
# airflow along (1, 1, 0)/sqrt(2) sees a seam that is mirror-symmetric about
# the vertical plane containing the flow, so there is no sideways force.
SYMMETRIC_FLOW_DIR = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
