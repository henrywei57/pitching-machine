"""A three-wheel pitching machine in MuJoCo.

Three tires sit 120 degrees apart around the ball's path, like a Hack Attack
style machine: one on top, two lower ones on the third-base and first-base
sides. Nothing here hands the ball a spin rate. The wheels are driven by
speed-controlled motors, the ball is fed into the gap between them, and
whatever speed and spin it leaves with comes out of the contact: the tires and
ball squash into each other, friction can only push as hard as mu times the
squash force, and the wheels have real inertia, so they slow down a little as
they give up energy to the ball.

With three wheels the rigid, no-slip kinematics fix the ball's speed, spin
rate and spin direction from the three rim speeds, so a curveball that breaks
sideways comes from the wheel speeds alone. A two-wheel machine (set
wheel_angles_deg=TWO_WHEEL) can only spin the ball about one axis and has to
rotate its head (tilt_deg) for anything else.

Contact models
--------------
"coulomb" (default): an explicit tire-ball contact law. The normal force is a
spring-damper on the overlap (the squash). Friction is solved every step as
Coulomb friction: the contact impulses that would make every contact stick,
projected back onto the friction cone |F_t| <= mu F_n when that asks for too
much. MuJoCo integrates the wheels, motors and ball.

"mujoco": MuJoCo's own soft-constraint contact. It is a good contact model for
robots, but its normal force is not a spring: the solver trades normal force
against friction, so N comes out larger than k * squash whenever the ball is
being accelerated hard, and friction "creeps" on the damping time scale. It is
kept for comparison (experiments/compare_engines.py).

Frames
------
World: x points from the machine toward home plate, z is up, y is the
first-base side (the pitcher's left when facing the plate).

Head: the wheel assembly. Its x axis is the barrel (the ball's path through
the wheels). A wheel at angle theta sits in direction (0, cos theta, sin theta)
from the barrel: 90 is the top, 210 lower on the 3B side, 330 lower on the 1B
side, 270 the bottom.
"""

from dataclasses import dataclass, field, fields, replace

import mujoco
import numpy as np

from .constants import BALL_INERTIA, BALL_MASS, BALL_RADIUS, INCH, MPH, PLATE_DISTANCE, RPM
from .seams import seam_points


THREE_WHEEL = (90.0, 210.0, 330.0)  # top, lower 3B side, lower 1B side
MAX_RPM = 6000.0  # motor limit; also the range of the viewer's rpm sliders
TWO_WHEEL = (90.0, 270.0)  # top and bottom


def wheel_label(theta_deg):
    t = theta_deg % 360
    if abs(t - 90) < 1:
        return "top"
    if abs(t - 270) < 1:
        return "bottom"
    vert = "upper" if np.sin(np.radians(t)) > 0 else "lower"
    side = "1B" if np.cos(np.radians(t)) > 0 else "3B"
    return f"{vert} {side}"


def _wrap_axis(phi, s):
    """Spin direction phi (deg) and signed rate s -> phi in (-90, 90], s carrying the sign."""
    if phi > 90:
        phi, s = phi - 180, -s
    elif phi <= -90:
        phi, s = phi + 180, -s
    return phi, s


@dataclass
class MachineConfig:
    wheel_rpm: tuple = (1800.0, 1800.0, 1800.0)
    wheel_angles_deg: tuple = THREE_WHEEL
    tilt_deg: float = 0.0  # head rotation about the barrel (only a two-wheel machine needs it)
    elevation_deg: float = 0.0  # barrel angle above horizontal
    azimuth_deg: float = 0.0  # barrel angle toward 1B (+y)
    release_height: float = 1.75  # m, barrel height above the floor

    # Wheels: roughly a 10 in pneumatic tire on a steel hub.
    wheel_radius: float = 0.127
    wheel_width: float = 0.07
    wheel_mass: float = 3.0
    wheel_inertia: float = 0.035  # kg m^2 about the axle (tire mass sits near the rim)
    bearing_damping: float = 2e-4  # N m s/rad

    # Motors: velocity servo with a torque ceiling (a ~1/2 hp DC motor).
    motor_kv: float = 0.5  # N m per rad/s of speed error
    motor_torque: float = 2.0  # N m

    # Contact between tire and ball.
    contact_model: str = "coulomb"  # or "mujoco"
    squeeze: float = 0.0127  # m (half an inch) of interference per tire
    friction: float = 0.7  # sliding friction, rubber tire on leather
    contact_stiffness: float = 1.3e5  # N/m per contact; tire and ball squashing together
    contact_damping: float = 60.0  # N s/m; hysteresis in the rubber and the ball

    feed_speed: float = 1.5  # m/s; how fast the chute rolls the ball into the wheels
    ball_quat: tuple = (1.0, 0.0, 0.0, 0.0)  # seam orientation in the chute (w, x, y, z), head frame

    timestep: float = 1e-5  # s; contact lasts ~10 ms, so it needs small steps
    max_time: float = 0.06  # s; give up if the ball never comes out

    def __post_init__(self):
        if len(self.wheel_rpm) != len(self.wheel_angles_deg):
            raise ValueError("need one wheel_rpm per wheel angle")

    def check_rpm(self):
        if min(self.wheel_rpm) < 0 or max(self.wheel_rpm) > MAX_RPM:
            raise ValueError(f"wheel rpm must be between 0 and {MAX_RPM:.0f}, got {self.wheel_rpm}")

    def with_(self, **kw):
        return replace(self, **kw)

    @property
    def n_wheels(self):
        return len(self.wheel_angles_deg)

    @property
    def labels(self):
        return [wheel_label(t) for t in self.wheel_angles_deg]

    # Ideal (rigid, no-slip) kinematics -------------------------------------------
    @property
    def surface_speeds(self):
        return np.array(self.wheel_rpm) * RPM * self.wheel_radius

    def _ideal(self):
        """(speed, spin direction in deg, signed spin in rad/s) for a rigid ball that
        doesn't slip. Each tire's rim speed must equal the ball's surface speed
        where it touches: V_i = v + r (w_y sin theta_i - w_z cos theta_i).
        Two wheels leave w_z free; the minimum-norm answer sets it to zero."""
        th = np.radians(self.wheel_angles_deg)
        A = np.stack([np.ones_like(th), BALL_RADIUS * np.sin(th), -BALL_RADIUS * np.cos(th)], axis=1)
        v, wy, wz = np.linalg.lstsq(A, self.surface_speeds, rcond=None)[0]
        s = np.hypot(wy, wz)
        if s < 1e-9:
            return v, 0.0, 0.0
        return (v, *_wrap_axis(np.degrees(np.arctan2(wz, wy)), s))

    @property
    def ideal_speed(self):
        return self._ideal()[0]

    @property
    def ideal_spin(self):
        """Signed spin (rad/s) about spin_axis_head: + topspin-like, - backspin-like."""
        return self._ideal()[2]

    @property
    def spin_axis_deg(self):
        """Direction of the ideal spin axis in the head's y-z plane: 0 is pure top/backspin,
        + turns it so the ball also breaks toward first base."""
        return self._ideal()[1]

    @property
    def spin_axis_head(self):
        phi = np.radians(self.spin_axis_deg)
        return np.array([0.0, np.cos(phi), np.sin(phi)])


def settings_for(speed_mph, spin_rpm, axis_deg=0.0, wheel_angles_deg=THREE_WHEEL,
                 wheel_radius=MachineConfig.wheel_radius, tilt_deg=0.0, **kw):
    """Wheel rpms a naive operator would dial in for a target speed, spin and spin axis.

    Positive spin_rpm is topspin, negative is backspin; axis_deg turns the spin
    axis so the ball also breaks toward first base (+) or third base (-). This
    inverts the ideal kinematics, so the real machine will come up short.
    Three wheels make any axis from speeds alone; two wheels tilt the head.
    """
    v = speed_mph * MPH
    rs = spin_rpm * RPM * BALL_RADIUS
    th = np.radians(wheel_angles_deg)
    spans = np.linalg.matrix_rank(np.stack([np.ones_like(th), np.sin(th), np.cos(th)], axis=1)) == 3
    phi = np.radians(axis_deg) if spans else 0.0
    if not spans:
        tilt_deg = tilt_deg + axis_deg
    V = v + rs * np.sin(th - phi)
    return MachineConfig(wheel_rpm=tuple(V / (RPM * wheel_radius)), wheel_angles_deg=tuple(wheel_angles_deg),
                         wheel_radius=wheel_radius, tilt_deg=tilt_deg, **kw)


# ----------------------------------------------------------------------------------
# Rotations
# ----------------------------------------------------------------------------------


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def head_rotation(cfg):
    """Head frame -> world frame."""
    return _rz(np.radians(cfg.azimuth_deg)) @ _ry(-np.radians(cfg.elevation_deg)) @ _rx(np.radians(cfg.tilt_deg))


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).ravel())
    return q


def quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def _fmt(v):
    return " ".join(f"{x:.9g}" for x in np.ravel(v))


def _cross(a, b):
    # np.cross is ~10x slower than this for single 3-vectors, and the contact
    # law calls it a few hundred thousand times per pitch.
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ----------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------


def wheel_dirs(cfg):
    """Unit vectors (head frame) from the barrel to each wheel, and each wheel's axle.
    The axle d x x makes the wheel turn the "same way" for every wheel: a negative
    angular velocity about it drives the tire surface toward the plate."""
    th = np.radians(cfg.wheel_angles_deg)
    d = np.stack([np.zeros_like(th), np.cos(th), np.sin(th)], axis=1)
    a = np.cross(d, [1.0, 0.0, 0.0])
    return d, a


def wheel_offset(cfg):
    """Distance from the barrel axis to each wheel's axle."""
    return cfg.wheel_radius + BALL_RADIUS - cfg.squeeze


def entry_x(cfg):
    """Head-frame x where the ball first touches the tires."""
    rsum = cfg.wheel_radius + BALL_RADIUS
    return -np.sqrt(rsum**2 - wheel_offset(cfg) ** 2)


def nip_angle(cfg):
    """Angle between the contact normal and the barrel's cross-section at first touch.
    A rolling mill only grabs its stock if mu > tan(nip angle); so does this."""
    return np.arccos(wheel_offset(cfg) / (cfg.wheel_radius + BALL_RADIUS))


def ball_start(cfg):
    """Where the ball sits in the chute (world position and orientation quaternion)."""
    R_head = head_rotation(cfg)
    pos = np.array([0, 0, cfg.release_height]) + R_head @ np.array([entry_x(cfg) - 0.002, 0, 0])
    return pos, mat_to_quat(R_head @ quat_to_mat(cfg.ball_quat))


def motor_names(cfg):
    """Actuator names, which the viewer shows next to its rpm sliders."""
    names = [wheel_label(t).replace(" ", "_") + "_rpm" for t in cfg.wheel_angles_deg]
    return [n if names.count(n) == 1 else f"{n}{i}" for i, n in enumerate(names)]


def _lookat(pos, target):
    """MJCF xyaxes for a camera at pos looking at target, with world z up."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return _fmt(np.r_[right, np.cross(right, f)])


FOVY = {"watcher": 50, "side": 55}


# Named views for the viewer and renderer. "watcher" is someone standing next to
# the machine, a little behind it, watching it throw.
def cameras(cfg, plate_distance=PLATE_DISTANCE):
    h, xp = cfg.release_height, plate_distance
    return {
        "watcher": ((-2.6, -0.95, 1.75), (xp * 0.6, 0.15, 1.05)),
        "catcher": ((xp + 1.2, 0.0, 0.95), (0.0, 0.0, h)),
        "batter": ((xp + 0.45, -0.75, 1.55), (0.0, 0.0, h)),
        "side": ((xp * 0.5, -13.0, 1.3), (xp * 0.5, 0.0, 1.0)),
        "machine": ((-0.9, -0.42, h + 0.17), (0.0, 0.0, h)),
    }


def build_mjcf(cfg, scene=False, plate_distance=PLATE_DISTANCE):
    """MJCF text for the machine and ball.

    scene=True adds the field dressing (mound, plate, strike zone) and the
    machine's frame and motors, used for rendering; none of it collides.
    """
    R_head = head_rotation(cfg)
    c = wheel_offset(cfg)
    dirs, axles = wheel_dirs(cfg)
    Rw, ww = cfg.wheel_radius, cfg.wheel_width
    n = cfg.n_wheels
    # Inertia about a diameter: half the axle inertia (thin ring) plus the width term.
    # Nothing in the launch depends on it; it just has to be physically valid.
    Ip = 0.5 * cfg.wheel_inertia + cfg.wheel_mass * ww**2 / 12.0
    nocollide = 'contype="0" conaffinity="0"'

    pairs = ""
    if cfg.contact_model == "mujoco":
        # With a constant impedance of 0.5 and solref = (-k, -b), a ball pinched
        # between fixed axles feels 2 m (k r + b v) at each contact when nothing
        # slides (measured; tests/test_machine.py checks it), so divide by 2m.
        k, b = cfg.contact_stiffness / (2 * BALL_MASS), cfg.contact_damping / (2 * BALL_MASS)
        attrs = (
            f'condim="3" friction="{cfg.friction} {cfg.friction} 0.005 0.0001 0.0001" '
            f'solref="{-k:.9g} {-b:.9g}" solimp="0.5 0.5 0.001 0.5 2"'
        )
        pairs = "\n  <contact>\n" + "".join(
            f'    <pair geom1="ball" geom2="tire{i}" {attrs}/>\n' for i in range(n)) + "  </contact>"
    elif cfg.contact_model != "coulomb":
        raise ValueError(f"unknown contact_model {cfg.contact_model!r}")

    # Red stitches so you can see the ball turn (visual only: no mass, no collisions).
    stitches = "\n      ".join(
        f'<geom type="sphere" size="0.0035" pos="{_fmt(p * BALL_RADIUS * 0.985)}" '
        f'rgba="0.8 0.1 0.1 1" {nocollide} group="1" mass="0"/>'
        for p in seam_points(72)[0]
    )

    frame = field = ""
    if scene:
        # A motor on each axle, struts joining the motors, and the feed chute.
        steel = 'rgba="0.15 0.25 0.55 1"'
        parts = []
        ends = []
        for d, a in zip(dirs, axles):
            hub = c * d
            parts.append(f'<geom type="cylinder" fromto="{_fmt(hub + a * (ww / 2 + 0.005))} {_fmt(hub + a * (ww / 2 + 0.09))}" '
                         f'size="0.032" rgba="0.22 0.22 0.24 1" {nocollide}/>')
            ends.append(hub + a * (ww / 2 + 0.05) + d * 0.05)
        for i in range(n if n > 2 else 1):
            p0, p1 = ends[i], ends[(i + 1) % n]
            parts.append(f'<geom type="capsule" fromto="{_fmt(p0)} {_fmt(p1)}" size="0.016" {steel} {nocollide}/>')
        parts.append(f'<geom type="capsule" fromto="-0.35 0 {-BALL_RADIUS - 0.004} {entry_x(cfg) - 0.01} 0 {-BALL_RADIUS - 0.004}" '
                     f'size="0.004" rgba="0.6 0.6 0.6 1" {nocollide}/>')
        frame = "\n      " + "\n      ".join(parts)
        xp = plate_distance
        zw, zt, zb = 8.5 * INCH, 3.5 * 0.3048, 1.5 * 0.3048  # zone half-width, top, bottom
        zone = [((xp, -zw, zb), (xp, -zw, zt)), ((xp, zw, zb), (xp, zw, zt)),
                ((xp, -zw, zt), (xp, zw, zt)), ((xp, -zw, zb), (xp, zw, zb))]
        zone = "\n    ".join(
            f'<geom type="capsule" fromto="{_fmt(a)} {_fmt(b)}" size="0.006" rgba="1 1 1 0.55" {nocollide}/>' for a, b in zone
        )
        low = cfg.release_height + min(c * d[2] for d in dirs) - Rw - 0.02
        field = f"""
    <geom name="mound" type="ellipsoid" pos="0 0 -0.05" size="2.7 2.7 0.30" rgba="0.62 0.42 0.27 1" {nocollide}/>
    <geom name="dirt" type="cylinder" pos="{xp + 0.2} 0 0.001" size="2.0 0.001" rgba="0.62 0.42 0.27 1" {nocollide}/>
    <geom name="plate" type="box" pos="{xp + 8.5 * INCH} 0 0.004" size="{8.5 * INCH} {8.5 * INCH} 0.004" rgba="1 1 1 1" {nocollide}/>
    <geom name="post" type="cylinder" fromto="-0.05 0 0 -0.05 0 {low}" size="0.035" rgba="0.2 0.2 0.22 1" {nocollide}/>
    {zone}""" + "".join(
            f'\n    <camera name="{name}" pos="{_fmt(pos)}" xyaxes="{_lookat(pos, target)}" fovy="{FOVY.get(name, 40)}"/>'
            for name, (pos, target) in cameras(cfg, plate_distance).items())

    ball_pos, ball_quat = ball_start(cfg)
    # Rides along with the ball but doesn't turn with it.
    follow = (f'\n      <camera name="follow_ball" mode="trackcom" pos="-1.1 -0.75 0.35" '
              f'xyaxes="{_lookat((-1.1, -0.75, 0.35), (0, 0, 0))}" fovy="45"/>') if scene else ""

    def wheel(i, d, a):
        # Wheel body frame: x along the barrel, y along the axle, z toward the wheel.
        q = mat_to_quat(np.stack([[1.0, 0.0, 0.0], a, d], axis=1))
        return f"""
      <body name="wheel{i}" pos="{_fmt(c * d)}" quat="{_fmt(q)}">
        <joint name="axle{i}" type="hinge" axis="0 1 0" damping="{cfg.bearing_damping}"/>
        <inertial pos="0 0 0" mass="{cfg.wheel_mass}" diaginertia="{Ip} {cfg.wheel_inertia} {Ip}"/>
        <geom name="tire{i}" type="cylinder" size="{Rw} {ww / 2}" euler="1.5707963 0 0" material="tire" {nocollide}/>
        <geom type="box" size="{Rw * 0.75} {ww / 2 + 0.002} 0.012" material="hub" {nocollide} mass="0"/>
        <geom type="box" size="0.012 {ww / 2 + 0.002} {Rw * 0.75}" material="hub" {nocollide} mass="0"/>
      </body>"""

    wheels = "".join(wheel(i, d, a) for i, (d, a) in enumerate(zip(dirs, axles)))
    # Motor commands are in rpm. The gear turns the axle's rad/s (negative drives the
    # tire toward the plate) into rpm, and kv and the force limit are rescaled so
    # the torque on the axle is still kv * (target - speed), capped at motor_torque.
    gear = -1.0 / RPM
    motors = "\n    ".join(
        f'<velocity name="{name}" joint="axle{i}" gear="{gear:.12g}" kv="{cfg.motor_kv / gear**2:.12g}" '
        f'ctrlrange="0 {MAX_RPM:g}" forcerange="{-cfg.motor_torque * RPM:.12g} {cfg.motor_torque * RPM:.12g}"/>'
        for i, name in enumerate(motor_names(cfg)))

    return f"""
<mujoco model="{n}_wheel_pitching_machine">
  <compiler angle="radian" autolimits="true"/>
  <size nuserdata="1"/>
  <option timestep="{cfg.timestep}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1280" offheight="720" fovy="40"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.45 0.62 0.85" rgb2="0.85 0.9 0.97" width="256" height="256"/>
    <texture name="grass" type="2d" builtin="checker" rgb1="0.20 0.48 0.20" rgb2="0.23 0.53 0.23" width="512" height="512"/>
    <material name="grass" texture="grass" texrepeat="12 12" reflectance="0"/>
    <material name="tire" rgba="0.08 0.08 0.08 1" specular="0.2"/>
    <material name="hub" rgba="0.75 0.75 0.78 1" specular="0.6"/>
  </asset>
  <worldbody>
    <light pos="5 -4 12" dir="-0.2 0.3 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="40 12 0.1" material="grass" {nocollide}/>{field}
    <body name="head" pos="0 0 {cfg.release_height}" quat="{_fmt(mat_to_quat(R_head))}">{frame}{wheels}
    </body>
    <body name="ball" pos="{_fmt(ball_pos)}" quat="{_fmt(ball_quat)}">
      <freejoint name="ball"/>
      <inertial pos="0 0 0" mass="{BALL_MASS}" diaginertia="{BALL_INERTIA} {BALL_INERTIA} {BALL_INERTIA}"/>
      <geom name="ball" type="sphere" size="{BALL_RADIUS}" rgba="0.96 0.95 0.9 1" {nocollide}/>
      <site name="imu" pos="0 0 0" size="0.005"/>{follow}
      {stitches}
    </body>
  </worldbody>{pairs}
  <actuator>
    {motors}
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
  </sensor>
</mujoco>
"""


# ----------------------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------------------

LOG_KEYS = ("t", "x", "speed", "spin", "gyro", "accel")
# Per wheel i: rim_i, normal_i, friction_i, squash_i, slip_i.
CONTACT_KEYS = ("normal", "friction", "squash", "slip")


@dataclass
class LaunchResult:
    config: MachineConfig
    exited: bool
    # Exit state, world frame.
    position: np.ndarray
    velocity: np.ndarray
    omega: np.ndarray  # rad/s
    quat: np.ndarray  # ball orientation (w, x, y, z), body -> world
    exit_time: float
    # Time series during the launch, one row per recorded step.
    log: dict = field(repr=False)

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    @property
    def spin(self):
        return float(np.linalg.norm(self.omega))

    @property
    def spin_rpm(self):
        return self.spin / RPM

    @property
    def signed_spin(self):
        """Spin (rad/s) about the axis the wheel speeds ask for: + topspin-like, - backspin-like."""
        return float(self.omega @ (head_rotation(self.config) @ self.config.spin_axis_head))

    @property
    def spin_axis_deg(self):
        """Direction the ball actually spins about, in the same convention as the config."""
        w = head_rotation(self.config).T @ self.omega
        if np.hypot(w[1], w[2]) < 1e-6:
            return 0.0
        return _wrap_axis(np.degrees(np.arctan2(w[2], w[1])), 1.0)[0]

    @property
    def speed_efficiency(self):
        return self.speed / self.config.ideal_speed

    @property
    def spin_efficiency(self):
        s = self.config.ideal_spin
        return self.signed_spin / s if abs(s) > 1e-9 else np.nan

    def summary(self):
        c = self.config
        rpms = "   ".join(f"{lab} {rpm:.0f}" for lab, rpm in zip(c.labels, c.wheel_rpm))
        rims = " / ".join(f"{v / MPH:.1f}" for v in c.surface_speeds)
        axis = f"   axis {c.spin_axis_deg:+.0f} deg" if abs(c.ideal_spin) > 100 * RPM and abs(c.spin_axis_deg) > 0.5 else ""
        lines = [
            f"wheels       {rpms} rpm   (rims {rims} mph)",
            f"ideal exit   {c.ideal_speed / MPH:6.1f} mph   {c.ideal_spin / RPM:+6.0f} rpm{axis}",
        ]
        if not self.exited:
            lines.append("ball never left the wheels (stalled, or spat back out of the nip)")
            return "\n".join(lines)
        eff = f"{100 * self.speed_efficiency:.1f}% of ideal speed"
        if np.isfinite(self.spin_efficiency) and abs(c.ideal_spin) > 100 * RPM:
            eff += f", {100 * self.spin_efficiency:.1f}% of ideal spin"
        axis = f"   axis {self.spin_axis_deg:+.0f} deg" if axis else ""
        L = self.log
        lines += [
            f"actual exit  {self.speed / MPH:6.1f} mph   {self.signed_spin / RPM:+6.0f} rpm{axis}   ({eff})",
            f"contact      {1e3 * L['contact_time']:.2f} ms   peak squash {1e3 * L['peak_squash']:.1f} mm   "
            f"peak normal force {L['peak_normal']:.0f} N   slipping {100 * L['slip_fraction']:.0f}% of the time   "
            f"wheels slowed {L['wheel_drop_pct']:.1f}%",
        ]
        return "\n".join(lines)


class Machine:
    """The compiled MuJoCo model for one configuration, plus the contact law."""

    def __init__(self, cfg: MachineConfig, scene=False):
        cfg.check_rpm()
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(cfg, scene=scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.wheels = range(cfg.n_wheels)
        self.ball_body = m.body("ball").id
        self.ball_geom = m.geom("ball").id
        self.tire_geoms = {m.geom(f"tire{i}").id: i for i in self.wheels}
        self.wheel_bodies = [m.body(f"wheel{i}").id for i in self.wheels]
        self.dadr = [m.jnt_dofadr[m.joint(f"axle{i}").id] for i in self.wheels]
        self.qadr = [m.jnt_qposadr[m.joint(f"axle{i}").id] for i in self.wheels]
        self.ball_qadr = m.jnt_qposadr[m.joint("ball").id]
        self.ball_dadr = m.jnt_dofadr[m.joint("ball").id]
        self.head_body = m.body("head").id
        self.R_head = head_rotation(cfg)
        self.contacts = {}
        self.reset()

    # Things that can change without rebuilding the model.
    RETARGETABLE = {"wheel_rpm", "elevation_deg", "azimuth_deg", "tilt_deg", "ball_quat"}

    def retarget(self, cfg: MachineConfig):
        """Switch to new wheel speeds, aim or chute orientation in place, so an open
        viewer keeps working. Anything else (wheel layout, squeeze, ...) needs a new Machine."""
        changed = {f.name for f in fields(cfg) if getattr(cfg, f.name) != getattr(self.cfg, f.name)}
        if changed - self.RETARGETABLE:
            raise ValueError(f"can't change {sorted(changed - self.RETARGETABLE)} in place; build a new Machine")
        cfg.check_rpm()
        self.cfg = cfg
        self.R_head = head_rotation(cfg)
        m = self.model
        m.body_quat[self.head_body] = mat_to_quat(self.R_head)
        pos, quat = ball_start(cfg)
        m.qpos0[self.ball_qadr : self.ball_qadr + 3] = pos
        m.qpos0[self.ball_qadr + 3 : self.ball_qadr + 7] = quat
        self.reset()

    @property
    def wheel_targets(self):
        """Wheel angular velocity about its axle (rad/s); negative drives the tire toward the plate."""
        return -np.array(self.cfg.wheel_rpm) * RPM

    def reset(self):
        """Ball in the chute, wheels at speed."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        tgt = self.wheel_targets
        for i in self.wheels:
            d.qvel[self.dadr[i]] = tgt[i]  # wheels already up to speed, as on a real machine
        d.ctrl[:] = self.cfg.wheel_rpm  # motor commands are in rpm
        d.userdata[0] = 1.0  # mj_resetData zeroes this, so a reset from outside (the viewer) is detectable
        d.qvel[self.ball_dadr : self.ball_dadr + 3] = self.R_head @ np.array([self.cfg.feed_speed, 0, 0])
        mujoco.mj_forward(m, d)
        self.contacts = {}

    @property
    def was_reset(self):
        """True if something other than reset() wiped the data (MuJoCo's viewer does on Backspace)."""
        return self.data.userdata[0] == 0.0

    def ball_state(self):
        d = self.data
        a, b = self.ball_qadr, self.ball_dadr
        pos = d.qpos[a : a + 3].copy()
        quat = d.qpos[a + 3 : a + 7].copy()
        vel = d.qvel[b : b + 3].copy()
        # Free-joint angular velocity is in the body frame; rotate to world.
        omega = quat_to_mat(quat) @ d.qvel[b + 3 : b + 6]
        return pos, vel, omega, quat

    # -- explicit contact law ---------------------------------------------------------

    def _coulomb_step(self):
        """Compute tire-ball forces for this step and load them into xfrc/qfrc_applied."""
        m, d, cfg = self.model, self.data, self.cfg
        dt = m.opt.timestep
        pos, vel, omega, _ = self.ball_state()
        d.xfrc_applied[self.ball_body] = 0.0
        d.qfrc_applied[:] = 0.0
        self.contacts = {}

        cons = []
        for k in self.wheels:
            wb = self.wheel_bodies[k]
            cw = d.xpos[wb]
            axis = d.xmat[wb].reshape(3, 3)[:, 1].copy()
            rel = pos - cw
            radial = rel - (rel @ axis) * axis
            dist = np.sqrt(radial @ radial)
            squash = cfg.wheel_radius + BALL_RADIUS - dist
            if squash <= 0.0:
                continue
            n = radial / dist  # from the wheel toward the ball
            p = pos - n * (BALL_RADIUS - 0.5 * squash)  # middle of the overlap
            fn = max(0.0, cfg.contact_stiffness * squash - cfg.contact_damping * (vel @ n))
            w = d.qvel[self.dadr[k]]
            r_b, r_w = p - pos, p - cw
            u = _cross(axis, r_w)  # tire surface velocity per unit wheel speed
            slip = vel + _cross(omega, r_b) - w * u
            t1 = _cross(axis, n)  # along the direction of travel
            T = np.stack([t1, _cross(n, t1)])  # tangent basis (second row: across the tire)
            # Wheel angular acceleration from everything except friction.
            tau = min(max(cfg.motor_kv * (-d.ctrl[k] * RPM - w), -cfg.motor_torque), cfg.motor_torque)
            alpha_w = (tau - cfg.bearing_damping * w) / cfg.wheel_inertia
            cons.append((k, n, fn, r_b, r_w, u, slip, T, alpha_w, axis, squash))
        if not cons:
            return

        # Ball acceleration without friction: the normal forces and gravity.
        a_ball = m.opt.gravity + sum(c[2] * c[1] for c in cons) / BALL_MASS
        # Tangential slip each contact would have at the end of the step without friction.
        s = np.concatenate([c[7] @ (c[6] + dt * (a_ball - c[8] * c[5])) for c in cons])

        # Delassus matrix: how tangential impulses at the contacts change their slips.
        # Ball translation + ball rotation couple the contacts; each wheel only feels its own.
        nc = len(cons)
        Tb = np.zeros((2 * nc, 3 * nc))
        Mb = np.zeros((3 * nc, 3 * nc))
        for i, ci in enumerate(cons):
            Tb[2 * i : 2 * i + 2, 3 * i : 3 * i + 3] = ci[7]
            for j, cj in enumerate(cons):
                blk = np.eye(3) / BALL_MASS - _skew(ci[3]) @ _skew(cj[3]) / BALL_INERTIA
                if i == j:
                    blk += np.outer(ci[5], ci[5]) / cfg.wheel_inertia
                Mb[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = blk
        W = Tb @ Mb @ Tb.T
        limit = np.array([cfg.friction * c[2] * dt for c in cons])

        # Coulomb friction: stick if the friction cone allows it, else slide at mu N.
        P = -np.linalg.solve(W, s)
        mags = np.linalg.norm(P.reshape(nc, 2), axis=1)
        if np.any(mags > limit):
            # Projected Gauss-Seidel on the 2-D blocks; converges fast because each
            # contact's own wheel and ball resistance outweighs the coupling.
            P[:] = 0.0
            inv = [np.linalg.inv(W[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]) for i in range(nc)]
            for _ in range(60):
                worst = 0.0
                for i in range(nc):
                    sl = slice(2 * i, 2 * i + 2)
                    new = P[sl] - inv[i] @ (s[sl] + W[sl] @ P)
                    mag = np.sqrt(new @ new)
                    if mag > limit[i]:
                        new *= limit[i] / mag
                    worst = max(worst, np.abs(new - P[sl]).max())
                    P[sl] = new
                if worst < 1e-10:
                    break
        residual = s + W @ P

        force, torque = np.zeros(3), np.zeros(3)
        for i, (k, n, fn, r_b, r_w, u, slip, T, alpha_w, axis, squash) in enumerate(cons):
            ft = T.T @ P[2 * i : 2 * i + 2] / dt
            f = fn * n + ft
            force += f
            torque += _cross(r_b, ft)
            # Equal and opposite on the tire; only the part about the axle matters.
            d.qfrc_applied[self.dadr[k]] -= u @ f
            self.contacts[k] = (fn, np.sqrt(ft @ ft), squash, np.linalg.norm(residual[2 * i : 2 * i + 2]))
        d.xfrc_applied[self.ball_body, :3] = force
        d.xfrc_applied[self.ball_body, 3:] = torque

    # -- MuJoCo's own contacts, for logging ---------------------------------------------

    def _read_mujoco_contacts(self):
        m, d = self.model, self.data
        self.contacts = {}
        f6 = np.zeros(6)
        pos, vel, omega, _ = self.ball_state()
        for i in range(d.ncon):
            con = d.contact[i]
            g = {con.geom1, con.geom2}
            other = (g - {self.ball_geom}).pop() if self.ball_geom in g else None
            if other not in self.tire_geoms:
                continue
            k = self.tire_geoms[other]
            mujoco.mj_contactForce(m, d, i, f6)
            p = con.pos
            wb = self.wheel_bodies[k]
            axis = d.xmat[wb].reshape(3, 3)[:, 1]
            rel = vel + np.cross(omega, p - pos) - np.cross(axis * d.qvel[self.dadr[k]], p - d.xpos[wb])
            n = con.frame[:3]
            rel -= (rel @ n) * n
            self.contacts[k] = (f6[0], np.hypot(f6[1], f6[2]), -con.dist, np.linalg.norm(rel))

    # -- stepping -----------------------------------------------------------------------

    def step(self):
        if self.cfg.contact_model == "coulomb":
            self._coulomb_step()
            mujoco.mj_step(self.model, self.data)
        else:
            mujoco.mj_step(self.model, self.data)
            self._read_mujoco_contacts()

    def launch(self, record_every=5):
        """Feed the ball through the wheels and return its state when it comes out."""
        d, cfg = self.data, self.cfg
        self.reset()
        R_hT = self.R_head.T
        origin = np.array([0, 0, cfg.release_height])
        log = {k: [] for k in LOG_KEYS + tuple(f"{q}_{i}" for q in ("rim",) + CONTACT_KEYS for i in self.wheels)}
        w0 = [abs(d.qvel[self.dadr[i]]) for i in self.wheels]
        wmin = list(w0)
        axis = self.R_head @ cfg.spin_axis_head
        first_touch = last_touch = None
        steps = slipping = 0
        exited = False
        while d.time < cfg.max_time:
            self.step()
            steps += 1
            con = self.contacts
            pos, vel, omega, _ = self.ball_state()
            xh = (R_hT @ (pos - origin))[0]
            if con:
                first_touch = d.time if first_touch is None else first_touch
                last_touch = d.time
                slipping += any(v[3] > 0.05 for v in con.values())
            for i in self.wheels:
                wmin[i] = min(wmin[i], abs(d.qvel[self.dadr[i]]))
            left = first_touch is not None and not con
            if steps % record_every == 0 or left:
                log["t"].append(d.time)
                log["x"].append(xh)
                log["speed"].append(np.linalg.norm(vel))
                log["spin"].append(omega @ axis)
                # MuJoCo's own IMU sensors at the ball's center (body frame).
                log["gyro"].append(d.sensor("gyro").data.copy())
                log["accel"].append(d.sensor("accel").data.copy())
                for i in self.wheels:
                    log[f"rim_{i}"].append(abs(d.qvel[self.dadr[i]]) * cfg.wheel_radius)
                    for q, v in zip(CONTACT_KEYS, con.get(i, (0.0, 0.0, 0.0, np.nan))):
                        log[f"{q}_{i}"].append(v)
            if left and xh > 0:
                exited = True
                break
            if left and xh < entry_x(cfg):
                break  # spat back out of the nip
        log = {k: np.array(v) for k, v in log.items()}
        touched = first_touch is not None
        contact_steps = max(1, round((last_touch - first_touch) / cfg.timestep)) if touched else 1
        log["contact_time"] = (last_touch - first_touch) if touched else 0.0
        log["slip_fraction"] = slipping / contact_steps
        log["peak_squash"] = float(max(log[f"squash_{i}"].max(initial=0) for i in self.wheels))
        log["peak_normal"] = float(max(log[f"normal_{i}"].max(initial=0) for i in self.wheels))
        log["wheel_drop_pct"] = 100.0 * max((w0[i] - wmin[i]) / w0[i] if w0[i] > 0 else 0.0 for i in self.wheels)
        pos, vel, omega, quat = self.ball_state()
        return LaunchResult(cfg, exited, pos, vel, omega, quat, d.time, log)


def launch(cfg: MachineConfig, **kw) -> LaunchResult:
    return Machine(cfg).launch(**kw)


def calibrate(speed_mph, spin_rpm, axis_deg=0.0, iterations=6, tol=0.003, max_rpm=6000.0, **kw):
    """Find wheel rpms that actually deliver a target speed and spin.

    This is what an operator does with a radar gun and a spin tracker: dial in
    the textbook settings, measure what comes out, adjust, repeat. Here the
    adjustment is a Newton step with a finite-difference Jacobian, because once
    the ball is slipping, asking for more spin can give less. Returns the
    closest launch found, which is the best the machine can do if the target is
    out of reach.
    """
    target = np.array([speed_mph * MPH, spin_rpm * RPM])
    scale = np.array([target[0], max(abs(target[1]), 200 * RPM)])

    def measure(x):
        cfg = settings_for(x[0] / MPH, x[1] / RPM, axis_deg, **kw)
        if max(cfg.wheel_rpm) > max_rpm or min(cfg.wheel_rpm) < 0:
            return None, None
        r = launch(cfg)
        return (np.array([r.speed, r.signed_spin]), r) if r.exited else (None, None)

    x = target.copy()
    best, best_err = None, np.inf
    for _ in range(iterations):
        y, r = measure(x)
        if y is None:
            break
        err = np.abs((y - target) / scale).max()
        if err < best_err:
            best, best_err = r, err
        if err < tol:
            break
        h = 0.01 * scale
        cols = []
        for i in range(2):
            yi, _ = measure(x + np.eye(2)[i] * h)
            if yi is None:
                return best
            cols.append((yi - y) / h[i])
        # Damped least squares in scaled units: near the friction limit the
        # Jacobian goes singular (more rpm buys only more slip).
        J = np.stack(cols, axis=1) * (scale[None, :] / scale[:, None])
        r = (target - y) / scale
        step = np.linalg.solve(J.T @ J + 1e-3 * np.eye(2), J.T @ r) * scale
        x = x + np.clip(step, -0.25 * scale, 0.25 * scale)
    return best
