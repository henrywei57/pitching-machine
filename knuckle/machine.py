"""A two-wheel pitching machine in MuJoCo.

Nothing here hands the ball a spin rate. The wheels are driven by speed-
controlled motors, the ball is fed into the gap between them, and whatever
speed and spin it leaves with comes out of the contact: the tire and ball
squash into each other, friction can only push as hard as mu times the
squash force, and the wheels have real inertia, so they slow down a little as
they give up energy to the ball.

Contact models
--------------
"coulomb" (default): an explicit tire-ball contact law. The normal force is a
spring-damper on the overlap (the squash). Friction is solved every step as
Coulomb friction: the two contact impulses that would make both contacts
stick, projected back onto the friction cone |F_t| <= mu F_n when that asks
for too much. MuJoCo integrates the wheels, motors and ball.

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
the wheels). With tilt = 0 the "top" wheel sits above the ball and the
"bottom" wheel below it, both axles along head y. Tilting the head about the
barrel tilts the spin axis, exactly like rotating the head of a real machine
to throw a pitch that breaks sideways.
"""

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from .constants import BALL_INERTIA, BALL_MASS, BALL_RADIUS, INCH, MPH, PLATE_DISTANCE, RPM
from .seams import seam_points


@dataclass
class MachineConfig:
    top_rpm: float = 1800.0
    bottom_rpm: float = 1800.0
    tilt_deg: float = 0.0  # head rotation about the barrel; + rolls the top wheel toward 3B (-y)
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
    squeeze: float = 0.0127  # m (half an inch) of interference per wheel: (ball diameter - wheel gap) / 2
    friction: float = 0.7  # sliding friction, rubber tire on leather
    contact_stiffness: float = 1.3e5  # N/m per contact; tire and ball squashing together
    contact_damping: float = 60.0  # N s/m; hysteresis in the rubber and the ball

    feed_speed: float = 1.5  # m/s; how fast the chute rolls the ball into the wheels
    ball_quat: tuple = (1.0, 0.0, 0.0, 0.0)  # seam orientation in the chute (w, x, y, z), head frame

    timestep: float = 1e-5  # s; contact lasts ~5 ms, so it needs small steps
    max_time: float = 0.06  # s; give up if the ball never comes out

    def with_(self, **kw):
        return replace(self, **kw)

    # Ideal (rigid, no-slip) kinematics -------------------------------------------
    @property
    def top_surface_speed(self):
        return self.top_rpm * RPM * self.wheel_radius

    @property
    def bottom_surface_speed(self):
        return self.bottom_rpm * RPM * self.wheel_radius

    @property
    def ideal_speed(self):
        """Ball speed if it left the wheels rigidly, with no slip: the average surface speed."""
        return 0.5 * (self.top_surface_speed + self.bottom_surface_speed)

    @property
    def ideal_spin(self):
        """Spin (rad/s) if it left rigidly: surface-speed difference over the ball diameter.
        Positive is topspin for tilt = 0."""
        return (self.top_surface_speed - self.bottom_surface_speed) / (2.0 * BALL_RADIUS)


def settings_for(speed_mph, spin_rpm, wheel_radius=MachineConfig.wheel_radius, **kw):
    """Wheel rpms a naive operator would dial in for a target speed and spin.

    Positive spin_rpm is topspin (top wheel faster), negative is backspin.
    This inverts the ideal kinematics, so the real machine will come up short.
    """
    v = speed_mph * MPH
    dv = spin_rpm * RPM * BALL_RADIUS
    to_rpm = 1.0 / (RPM * wheel_radius)
    return MachineConfig(top_rpm=(v + dv) * to_rpm, bottom_rpm=(v - dv) * to_rpm, wheel_radius=wheel_radius, **kw)


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


def wheel_offset(cfg):
    """Distance from the barrel axis to each wheel's axle."""
    return cfg.wheel_radius + BALL_RADIUS - cfg.squeeze


def entry_x(cfg):
    """Head-frame x where the ball first touches both wheels."""
    rsum = cfg.wheel_radius + BALL_RADIUS
    return -np.sqrt(rsum**2 - wheel_offset(cfg) ** 2)


def nip_angle(cfg):
    """Angle between the contact normal and the wheel-to-wheel line at first touch.
    A rolling mill only grabs its stock if mu > tan(nip angle); so does this."""
    return np.arccos(wheel_offset(cfg) / (cfg.wheel_radius + BALL_RADIUS))


def build_mjcf(cfg, scene=False, plate_distance=PLATE_DISTANCE):
    """MJCF text for the machine and ball.

    scene=True adds the field dressing (mound, plate, strike zone) used for
    rendering; it has no effect on the launch.
    """
    R_head = head_rotation(cfg)
    c = wheel_offset(cfg)
    Rw, ww = cfg.wheel_radius, cfg.wheel_width
    # Inertia about a diameter: half the axle inertia (thin ring) plus the width term.
    # Nothing in the launch depends on it; it just has to be physically valid.
    Ip = 0.5 * cfg.wheel_inertia + cfg.wheel_mass * ww**2 / 12.0

    pairs = ""
    if cfg.contact_model == "mujoco":
        # With a constant impedance of 0.5 and solref = (-k, -b), a ball pinched
        # between two fixed axles feels 2 m (k r + b v) at each contact when
        # nothing slides (measured; tests/test_machine.py checks it), so divide by 2m.
        k, b = cfg.contact_stiffness / (2 * BALL_MASS), cfg.contact_damping / (2 * BALL_MASS)
        attrs = (
            f'condim="3" friction="{cfg.friction} {cfg.friction} 0.005 0.0001 0.0001" '
            f'solref="{-k:.9g} {-b:.9g}" solimp="0.5 0.5 0.001 0.5 2"'
        )
        pairs = f"""
  <contact>
    <pair name="top_pinch" geom1="ball" geom2="top_tire" {attrs}/>
    <pair name="bottom_pinch" geom1="ball" geom2="bottom_tire" {attrs}/>
  </contact>"""
    elif cfg.contact_model != "coulomb":
        raise ValueError(f"unknown contact_model {cfg.contact_model!r}")
    nocollide = 'contype="0" conaffinity="0"'

    # Red stitches so you can see the ball turn (visual only: no mass, no collisions).
    stitches = "\n      ".join(
        f'<geom type="sphere" size="0.0035" pos="{_fmt(p * BALL_RADIUS * 0.985)}" '
        f'rgba="0.8 0.1 0.1 1" {nocollide} group="1" mass="0"/>'
        for p in seam_points(72)[0]
    )

    frame = field = ""
    if scene:
        side = c + Rw + 0.03
        frame = f"""
      <geom type="box" pos="-0.02 0 {side}" size="0.20 {ww * 0.9} 0.012" rgba="0.15 0.25 0.55 1" {nocollide}/>
      <geom type="box" pos="-0.02 0 {-side}" size="0.20 {ww * 0.9} 0.012" rgba="0.15 0.25 0.55 1" {nocollide}/>
      <geom type="box" pos="-0.02 {ww * 0.9} 0" size="0.20 0.008 {side + 0.01}" rgba="0.15 0.25 0.55 0.35" {nocollide}/>
      <geom type="capsule" fromto="-0.35 0 {-BALL_RADIUS - 0.004} {entry_x(cfg) - 0.01} 0 {-BALL_RADIUS - 0.004}" size="0.004" rgba="0.6 0.6 0.6 1" {nocollide}/>"""
        xp = plate_distance
        zw, zt, zb = 8.5 * INCH, 3.5 * 0.3048, 1.5 * 0.3048  # zone half-width, top, bottom
        zone = [((xp, -zw, zb), (xp, -zw, zt)), ((xp, zw, zb), (xp, zw, zt)),
                ((xp, -zw, zt), (xp, zw, zt)), ((xp, -zw, zb), (xp, zw, zb))]
        zone = "\n    ".join(
            f'<geom type="capsule" fromto="{_fmt(a)} {_fmt(b)}" size="0.006" rgba="1 1 1 0.55" {nocollide}/>' for a, b in zone
        )
        field = f"""
    <geom name="mound" type="ellipsoid" pos="0 0 -0.05" size="2.7 2.7 0.30" rgba="0.62 0.42 0.27 1" {nocollide}/>
    <geom name="dirt" type="cylinder" pos="{xp + 0.2} 0 0.001" size="2.0 0.001" rgba="0.62 0.42 0.27 1" {nocollide}/>
    <geom name="plate" type="box" pos="{xp + 8.5 * INCH} 0 0.004" size="{8.5 * INCH} {8.5 * INCH} 0.004" rgba="1 1 1 1" {nocollide}/>
    <geom name="post" type="cylinder" fromto="-0.05 0 0 -0.05 0 {cfg.release_height - c - Rw - 0.04}" size="0.035" rgba="0.2 0.2 0.22 1" {nocollide}/>
    {zone}"""

    x0 = entry_x(cfg) - 0.002
    ball_pos = np.array([0, 0, cfg.release_height]) + R_head @ np.array([x0, 0, 0])
    ball_quat = mat_to_quat(R_head @ quat_to_mat(cfg.ball_quat))

    def wheel(name, z):
        return f"""
      <body name="{name}_wheel" pos="0 0 {z}">
        <joint name="{name}_axle" type="hinge" axis="0 1 0" damping="{cfg.bearing_damping}"/>
        <inertial pos="0 0 0" mass="{cfg.wheel_mass}" diaginertia="{Ip} {cfg.wheel_inertia} {Ip}"/>
        <geom name="{name}_tire" type="cylinder" size="{Rw} {ww / 2}" euler="1.5707963 0 0" material="tire" {nocollide}/>
        <geom type="box" size="{Rw * 0.75} {ww / 2 + 0.002} 0.012" material="hub" {nocollide} mass="0"/>
        <geom type="box" size="0.012 {ww / 2 + 0.002} {Rw * 0.75}" material="hub" {nocollide} mass="0"/>
      </body>"""

    return f"""
<mujoco model="two_wheel_pitching_machine">
  <compiler angle="radian" autolimits="true"/>
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
    <body name="head" pos="0 0 {cfg.release_height}" quat="{_fmt(mat_to_quat(R_head))}">{frame}{wheel("top", c)}{wheel("bottom", -c)}
    </body>
    <body name="ball" pos="{_fmt(ball_pos)}" quat="{_fmt(ball_quat)}">
      <freejoint name="ball"/>
      <inertial pos="0 0 0" mass="{BALL_MASS}" diaginertia="{BALL_INERTIA} {BALL_INERTIA} {BALL_INERTIA}"/>
      <geom name="ball" type="sphere" size="{BALL_RADIUS}" rgba="0.96 0.95 0.9 1" {nocollide}/>
      <site name="imu" pos="0 0 0" size="0.005"/>
      {stitches}
    </body>
  </worldbody>{pairs}
  <actuator>
    <velocity name="top_motor" joint="top_axle" kv="{cfg.motor_kv}" forcerange="{-cfg.motor_torque} {cfg.motor_torque}"/>
    <velocity name="bottom_motor" joint="bottom_axle" kv="{cfg.motor_kv}" forcerange="{-cfg.motor_torque} {cfg.motor_torque}"/>
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

WHEELS = ("top", "bottom")
LOG_KEYS = ("t", "x", "speed", "spin", "top_speed", "bottom_speed", "gyro", "accel")
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
        """Spin about the head's topspin axis (rad/s): + topspin, - backspin."""
        return float(self.omega @ (head_rotation(self.config) @ np.array([0.0, 1.0, 0.0])))

    @property
    def speed_efficiency(self):
        return self.speed / self.config.ideal_speed

    @property
    def spin_efficiency(self):
        s = self.config.ideal_spin
        return self.signed_spin / s if abs(s) > 1e-9 else np.nan

    def summary(self):
        c = self.config
        lines = [
            f"wheels       top {c.top_rpm:6.0f} rpm   bottom {c.bottom_rpm:6.0f} rpm   "
            f"(rim speed {c.top_surface_speed / MPH:5.1f} / {c.bottom_surface_speed / MPH:5.1f} mph)",
            f"ideal exit   {c.ideal_speed / MPH:6.1f} mph   {c.ideal_spin / RPM:+6.0f} rpm",
        ]
        if not self.exited:
            lines.append("ball never left the wheels (stalled, or spat back out of the nip)")
            return "\n".join(lines)
        eff = f"{100 * self.speed_efficiency:.1f}% of ideal speed"
        if np.isfinite(self.spin_efficiency) and abs(c.ideal_spin) > 100 * RPM:
            eff += f", {100 * self.spin_efficiency:.1f}% of ideal spin"
        L = self.log
        lines += [
            f"actual exit  {self.speed / MPH:6.1f} mph   {self.signed_spin / RPM:+6.0f} rpm   ({eff})",
            f"contact      {1e3 * L['contact_time']:.2f} ms   peak squash {1e3 * L['peak_squash']:.1f} mm   "
            f"peak normal force {L['peak_normal']:.0f} N   slipping {100 * L['slip_fraction']:.0f}% of the time   "
            f"wheels slowed {L['wheel_drop_pct']:.1f}%",
        ]
        return "\n".join(lines)


class Machine:
    """The compiled MuJoCo model for one configuration, plus the contact law."""

    def __init__(self, cfg: MachineConfig, scene=False):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(cfg, scene=scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.ball_body = m.body("ball").id
        self.ball_geom = m.geom("ball").id
        self.tire_geoms = {m.geom("top_tire").id: "top", m.geom("bottom_tire").id: "bottom"}
        self.wheel_bodies = {k: m.body(f"{k}_wheel").id for k in WHEELS}
        self.dadr = {k: m.jnt_dofadr[m.joint(f"{k}_axle").id] for k in WHEELS}
        self.qadr = {k: m.jnt_qposadr[m.joint(f"{k}_axle").id] for k in WHEELS}
        self.ball_qadr = m.jnt_qposadr[m.joint("ball").id]
        self.ball_dadr = m.jnt_dofadr[m.joint("ball").id]
        self.R_head = head_rotation(cfg)
        self.contacts = {}
        self.reset()

    # Wheel angular velocity about its axle (rad/s). The bottom wheel turns the
    # opposite way so both surfaces push the ball toward the plate.
    @property
    def wheel_targets(self):
        c = self.cfg
        return {"top": -c.top_rpm * RPM, "bottom": c.bottom_rpm * RPM}

    def reset(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        tgt = self.wheel_targets
        for k in WHEELS:
            d.qvel[self.dadr[k]] = tgt[k]  # wheels already up to speed, as on a real machine
        d.ctrl[:] = [tgt["top"], tgt["bottom"]]
        d.qvel[self.ball_dadr : self.ball_dadr + 3] = self.R_head @ np.array([self.cfg.feed_speed, 0, 0])
        mujoco.mj_forward(m, d)
        self.contacts = {}

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
        for i, k in enumerate(WHEELS):
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
            tau = min(max(cfg.motor_kv * (d.ctrl[i] - w), -cfg.motor_torque), cfg.motor_torque)
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
        log = {k: [] for k in LOG_KEYS + tuple(f"{q}_{w}" for q in CONTACT_KEYS for w in WHEELS)}
        w0 = {k: abs(d.qvel[self.dadr[k]]) for k in WHEELS}
        wmin = dict(w0)
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
            for k in WHEELS:
                wmin[k] = min(wmin[k], abs(d.qvel[self.dadr[k]]))
            left = first_touch is not None and not con
            if steps % record_every == 0 or left:
                log["t"].append(d.time)
                log["x"].append(xh)
                log["speed"].append(np.linalg.norm(vel))
                log["spin"].append(omega @ self.R_head[:, 1])
                # MuJoCo's own IMU sensors at the ball's center (body frame).
                log["gyro"].append(d.sensor("gyro").data.copy())
                log["accel"].append(d.sensor("accel").data.copy())
                for k in WHEELS:
                    log[f"{k}_speed"].append(abs(d.qvel[self.dadr[k]]) * cfg.wheel_radius)
                    for q, v in zip(CONTACT_KEYS, con.get(k, (0.0, 0.0, 0.0, np.nan))):
                        log[f"{q}_{k}"].append(v)
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
        log["peak_squash"] = float(max(log[f"squash_{k}"].max(initial=0) for k in WHEELS))
        log["peak_normal"] = float(max(log[f"normal_{k}"].max(initial=0) for k in WHEELS))
        log["wheel_drop_pct"] = 100.0 * max((w0[k] - wmin[k]) / w0[k] if w0[k] > 0 else 0.0 for k in WHEELS)
        pos, vel, omega, quat = self.ball_state()
        return LaunchResult(cfg, exited, pos, vel, omega, quat, d.time, log)


def launch(cfg: MachineConfig, **kw) -> LaunchResult:
    return Machine(cfg).launch(**kw)


def calibrate(speed_mph, spin_rpm, iterations=6, tol=0.003, max_rpm=6000.0, **kw):
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
        cfg = settings_for(x[0] / MPH, x[1] / RPM, **kw)
        if max(cfg.top_rpm, cfg.bottom_rpm) > max_rpm or min(cfg.top_rpm, cfg.bottom_rpm) < 0:
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
