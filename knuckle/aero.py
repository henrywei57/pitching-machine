"""Forces of the air on a baseball.

Three pieces, all quasi-steady (each depends only on the ball's current
velocity, spin and seam orientation):

Drag. A smooth sphere has a "drag crisis": above a critical speed the boundary
layer turns turbulent, stays attached longer, and drag drops by more than
half. The baseline curve is the fit to Adair's baseball data from Giordano &
Nakanishi, *Computational Physics* (2006), section 2.2:

    F_D / m = B2/m * v^2,   B2/m = 0.0039 + 0.0058 / (1 + exp((v - 35)/5))   [1/m]

Seams trip the boundary layer, so where the seams sit relative to the flow
moves the crisis. A ball that isn't spinning keeps one seam picture for the
whole flight, so its drag is not one number: the crisis speed here slides by
up to +-CRISIS_SHIFT m/s with the fraction of seam lying near the separation
line.

Magnus lift. Lift coefficient vs spin parameter S = r*omega_perp/v from Alan
Nathan's trajectory calculator: C_L = 1 / (2.32 + 0.4/S).

Seam force. When a ball barely spins, the seams sit still near the line where
the flow leaves the ball, so one side separates differently from the other and
the wake shifts sideways. (This is the knuckleball, and in spinning balls the
"seam-shifted wake".) The model sums a kernel over every bit of seam:

    C_seam = sum_i  w_i K(alpha_i) n_i

alpha_i is the seam point's angle from the front stagnation point, n_i the
unit vector from the flow axis toward that point, and K a pair of Gaussians in
alpha. K was fit (experiments/fit_seam_model.py) so that a ball turned about a
vertical axis reproduces the knuckleball lateral force Watts & Sawyer measured
in a wind tunnel (Am. J. Phys. 43, 960, 1975), in the form Giordano &
Nakanishi fit to it:

    F_lat / mg = 0.5 [sin 4t - 0.25 sin 8t + 0.08 sin 12t - 0.025 sin 16t]

The vertical part of the seam force and every orientation off that one axis
are the model's own prediction, not data.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .constants import BALL_AREA, BALL_MASS, BALL_RADIUS, RHO_AIR
from .seams import seam_points

# Seam-force kernel K(alpha) = sum A_j exp(-1/2 ((alpha - mu_j)/sigma_j)^2), alpha in degrees.
# Produced by experiments/fit_seam_model.py; calibrated at SEAM_REF_SPEED.
SEAM_KERNEL = ((-2.237, 98.676, 9.205), (-1.103, 116.276, 8.548))
SEAM_REF_SPEED = 30.0  # m/s (67 mph), a typical knuckleball

# The wind-tunnel calibration is for a ball that isn't spinning. On a spinning
# ball the moving surface energizes the boundary layer and the seams sweep
# through the separation region every few milliseconds, so the seam picture
# matters less. How much less is an open research question (the "seam-shifted
# wake" on spinning pitches); this is a placeholder taper, C_seam / (1 + S/S_t),
# chosen so seam orientation moves a fastball by a few inches, not a foot.
# Knuckleballs have S ~ 0.002 and are unaffected.
SEAM_SPIN_TAPER = 0.05

CRISIS_SPEED = 35.0  # m/s
CRISIS_WIDTH = 5.0  # m/s
CRISIS_SHIFT = 5.0  # m/s, max shift of the crisis speed with seam orientation
# Convert Giordano's B2/m (1/m) to a drag coefficient: C_D = 2 m (B2/m) / (rho A).
_B2M_TO_CD = 2.0 * BALL_MASS / (RHO_AIR * BALL_AREA)
CD_TURBULENT = 0.0039 * _B2M_TO_CD  # ~0.22
CD_LAMINAR = (0.0039 + 0.0058) * _B2M_TO_CD  # ~0.55


def random_rotations(n, rng):
    """Uniformly random rotation matrices (via random unit quaternions)."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return quats_to_mats(q)


def quats_to_mats(q):
    q = np.atleast_2d(q)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )


@dataclass(frozen=True)
class Aero:
    """Switchable aerodynamic model. The defaults are the full model."""

    drag: str = "crisis"  # "crisis" (seam- and speed-dependent), "constant", or "none"
    cd_constant: float = 0.35  # used when drag == "constant"
    lift: bool = True
    seam: bool = True
    seam_scale: float = 1.0
    seam_spin_taper: float = SEAM_SPIN_TAPER  # see below; None disables the taper
    crisis_shift: float = CRISIS_SHIFT
    rho: float = RHO_AIR
    n_seam: int = 180

    def __post_init__(self):
        object.__setattr__(self, "_seam", seam_points(self.n_seam))
        object.__setattr__(self, "_trip_range", _trip_range(self.n_seam))


def kernel(alpha_deg):
    K = np.zeros_like(alpha_deg)
    for A, mu, sig in SEAM_KERNEL:
        K = K + A * np.exp(-0.5 * ((alpha_deg - mu) / sig) ** 2)
    return K


def seam_terms(R, vhat, pts, w):
    """Seam force coefficient vector and trip fraction for a batch of balls.

    R    -- (N, 3, 3) body -> world rotations
    vhat -- (N, 3) unit velocity
    returns C_seam (N, 3), trip (N,)

    The trip fraction is the arc-length share of seam inside the main kernel
    band, i.e. sitting where it can trip the boundary layer.
    """
    S = R @ pts.T  # seam points in world, (N, 3, M)
    mu = np.clip(np.einsum("ni,nim->nm", vhat, S), -1.0, 1.0)  # cos(alpha)
    alpha = np.degrees(np.arccos(mu))
    (A1, m1, s1), (A2, m2, s2) = SEAM_KERNEL
    band = np.exp(-0.5 * ((alpha - m1) / s1) ** 2)
    K = A1 * band + A2 * np.exp(-0.5 * ((alpha - m2) / s2) ** 2)
    # Unit vector from the flow axis toward each seam point is (S - mu v) / sin(alpha).
    g = K * w / np.sqrt(1.0 - mu**2 + 1e-12)
    C = (S @ g[..., None])[..., 0] - vhat * (g * mu).sum(axis=1, keepdims=True)
    return C, band @ w


@lru_cache(maxsize=None)
def _trip_range(n_seam, n=4000, seed=0):
    """Mean and spread of the trip fraction over random orientations (for normalizing)."""
    pts, w = seam_points(n_seam)
    rng = np.random.default_rng(seed)
    R = random_rotations(n, rng)
    vhat = np.tile([1.0, 0.0, 0.0], (n, 1))
    _, trip = seam_terms(R, vhat, pts, w)
    return float(trip.mean()), float(trip.max() - trip.min())


def drag_coefficient(speed, trip=None, model: Aero = None):
    model = model or Aero()
    speed = np.asarray(speed, dtype=float)
    if model.drag == "none":
        return np.zeros_like(speed)
    if model.drag == "constant":
        return np.full_like(speed, model.cd_constant)
    v_c = CRISIS_SPEED
    if trip is not None and model.crisis_shift:
        mean, span = model._trip_range
        # More seam on the separation line -> earlier transition -> lower crisis speed.
        v_c = CRISIS_SPEED - 2.0 * model.crisis_shift * (np.asarray(trip) - mean) / span
    return CD_TURBULENT + (CD_LAMINAR - CD_TURBULENT) / (1.0 + np.exp((speed - v_c) / CRISIS_WIDTH))


def lift_coefficient(S):
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(S > 1e-9, 1.0 / (2.32 + 0.4 / np.maximum(S, 1e-9)), 0.0)


def acceleration(vel, omega, R, model: Aero = None, details=False):
    """Aerodynamic acceleration (N, 3) for a batch of balls (gravity not included).

    vel   -- (N, 3) m/s
    omega -- (N, 3) rad/s, world frame
    R     -- (N, 3, 3) ball orientation, body -> world
    """
    model = model or Aero()
    vel = np.atleast_2d(vel)
    omega = np.atleast_2d(omega)
    speed = np.linalg.norm(vel, axis=1)
    vhat = vel / np.maximum(speed, 1e-9)[:, None]
    q_over_m = 0.5 * model.rho * BALL_AREA * speed**2 / BALL_MASS  # dynamic pressure * A / m

    wxv = np.cross(omega, vhat)
    w_perp = np.linalg.norm(wxv, axis=1)
    S = BALL_RADIUS * w_perp / np.maximum(speed, 1e-9)  # spin parameter

    C_seam, trip = np.zeros_like(vel), None
    if model.seam or (model.drag == "crisis" and model.crisis_shift):
        C_seam, trip = seam_terms(R, vhat, *model._seam)
    if model.seam:
        # A coefficient, so the force scales with v^2 through q.
        C_seam = model.seam_scale * C_seam
        if model.seam_spin_taper:
            C_seam = C_seam / (1.0 + S / model.seam_spin_taper)[:, None]
    else:
        C_seam = np.zeros_like(vel)

    cd = drag_coefficient(speed, trip, model)
    a_drag = -(q_over_m * cd)[:, None] * vhat

    a_lift = np.zeros_like(vel)
    if model.lift:
        cl = lift_coefficient(S)
        a_lift = (q_over_m * cl / np.maximum(w_perp, 1e-12))[:, None] * wxv

    a_seam = q_over_m[:, None] * C_seam
    total = a_drag + a_lift + a_seam
    if details:
        return total, {"drag": a_drag, "lift": a_lift, "seam": a_seam, "cd": cd, "trip": trip}
    return total
