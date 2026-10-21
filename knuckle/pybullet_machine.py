"""The same two-wheel machine in PyBullet, where this project started.

Same wheels, ball, squeeze, friction and contact spring as knuckle.machine,
built from PyBullet primitives with compliant contact (contactStiffness /
contactDamping). Used by experiments/compare_engines.py to see how much the
answer depends on the engine.

PyBullet is optional: pip install pybullet. (On macOS with Apple's command-line
tools, the source build may need SDKROOT=$(xcrun --show-sdk-path).)
"""

from dataclasses import dataclass

import numpy as np

from .constants import BALL_INERTIA, BALL_MASS, BALL_RADIUS
from .machine import MachineConfig, entry_x, wheel_offset


@dataclass
class BulletLaunch:
    exited: bool
    velocity: np.ndarray
    omega: np.ndarray
    contact_time: float

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    def signed_spin(self):
        return float(self.omega[1])  # topspin axis for tilt = 0


def launch(cfg: MachineConfig, timestep=None, solver_iterations=100) -> BulletLaunch:
    import pybullet as p

    if cfg.tilt_deg or cfg.elevation_deg or cfg.azimuth_deg:
        raise NotImplementedError("the PyBullet port only builds an untilted, level head")
    dt = timestep or cfg.timestep
    cid = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=solver_iterations, numSubSteps=0,
                                    physicsClientId=cid)
        h = cfg.release_height
        c = wheel_offset(cfg)
        # Bullet multiplies the two bodies' friction coefficients, and combines
        # contact springs in series, so give each side half the compliance.
        mu_each = np.sqrt(cfg.friction)
        k_each, d_each = 2 * cfg.contact_stiffness, 2 * cfg.contact_damping

        cyl = p.createCollisionShape(p.GEOM_CYLINDER, radius=cfg.wheel_radius, height=cfg.wheel_width, physicsClientId=cid)
        z_to_y = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])  # cylinder axis (z) -> world y
        Ip = 0.5 * cfg.wheel_inertia + cfg.wheel_mass * cfg.wheel_width**2 / 12.0
        wheels = {}
        for name, z, w in (("top", c, -cfg.top_rpm), ("bottom", -c, cfg.bottom_rpm)):
            body = p.createMultiBody(
                baseMass=0, basePosition=[0, 0, h + z],
                linkMasses=[cfg.wheel_mass], linkCollisionShapeIndices=[cyl], linkVisualShapeIndices=[-1],
                linkPositions=[[0, 0, 0]], linkOrientations=[z_to_y],
                linkInertialFramePositions=[[0, 0, 0]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
                linkParentIndices=[0], linkJointTypes=[p.JOINT_REVOLUTE], linkJointAxis=[[0, 0, 1]],
                physicsClientId=cid,
            )
            # Joint axis is link z = world y, so the signs match knuckle.machine.
            omega = w * 2 * np.pi / 60
            p.changeDynamics(body, 0, lateralFriction=mu_each, rollingFriction=0, spinningFriction=0, restitution=0,
                             contactStiffness=k_each, contactDamping=d_each, linearDamping=0, angularDamping=0,
                             jointDamping=cfg.bearing_damping, localInertiaDiagonal=[Ip, Ip, cfg.wheel_inertia],
                             # PyBullet silently caps joint speed at 100 rad/s (a 28 mph rim) by default.
                             maxJointVelocity=2000.0, physicsClientId=cid)
            p.resetJointState(body, 0, 0.0, omega, physicsClientId=cid)
            p.setJointMotorControl2(body, 0, p.VELOCITY_CONTROL, targetVelocity=omega, force=cfg.motor_torque,
                                    physicsClientId=cid)
            wheels[name] = body

        sph = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=cid)
        ball = p.createMultiBody(baseMass=BALL_MASS, baseCollisionShapeIndex=sph,
                                 basePosition=[entry_x(cfg) - 0.002, 0, h], physicsClientId=cid)
        p.changeDynamics(ball, -1, lateralFriction=mu_each, rollingFriction=0, spinningFriction=0, restitution=0,
                         contactStiffness=k_each, contactDamping=d_each, linearDamping=0, angularDamping=0,
                         localInertiaDiagonal=[BALL_INERTIA] * 3, maxJointVelocity=2000.0, physicsClientId=cid)
        p.resetBaseVelocity(ball, [cfg.feed_speed, 0, 0], [0, 0, 0], physicsClientId=cid)

        t, first, last, exited = 0.0, None, None, False
        while t < cfg.max_time:
            p.stepSimulation(physicsClientId=cid)
            t += dt
            # getContactPoints also returns near misses inside the collision margin.
            touching = any(cp[8] < 0 for wb in wheels.values() for cp in p.getContactPoints(ball, wb, physicsClientId=cid))
            if touching:
                first = t if first is None else first
                last = t
            x = p.getBasePositionAndOrientation(ball, physicsClientId=cid)[0][0]
            if first is not None and not touching:
                exited = x > 0
                break
        v, w = p.getBaseVelocity(ball, physicsClientId=cid)
        return BulletLaunch(exited, np.array(v), np.array(w), (last - first) if first else 0.0)
    finally:
        p.disconnect(cid)


def static_stiffness(cfg: MachineConfig, squeeze=0.008):
    """Per-contact normal force / squash with the ball parked in the pinch (for calibration checks)."""
    import pybullet as p

    cfg = cfg.with_(squeeze=squeeze, top_rpm=0.0, bottom_rpm=0.0, feed_speed=0.0)
    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, 0, physicsClientId=cid)
        p.setPhysicsEngineParameter(fixedTimeStep=cfg.timestep, numSolverIterations=100, physicsClientId=cid)
        k_each, d_each = 2 * cfg.contact_stiffness, 2 * cfg.contact_damping
        cyl = p.createCollisionShape(p.GEOM_CYLINDER, radius=cfg.wheel_radius, height=cfg.wheel_width, physicsClientId=cid)
        z_to_y = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])
        bodies = []
        for z in (wheel_offset(cfg), -wheel_offset(cfg)):
            b = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cyl, basePosition=[0, 0, 1 + z],
                                  baseOrientation=z_to_y, physicsClientId=cid)
            p.changeDynamics(b, -1, contactStiffness=k_each, contactDamping=d_each, physicsClientId=cid)
            bodies.append(b)
        sph = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS, physicsClientId=cid)
        ball = p.createMultiBody(baseMass=BALL_MASS, baseCollisionShapeIndex=sph, basePosition=[0, 0, 1], physicsClientId=cid)
        p.changeDynamics(ball, -1, contactStiffness=k_each, contactDamping=d_each, physicsClientId=cid)
        for _ in range(20000):
            p.stepSimulation(physicsClientId=cid)
        cp = p.getContactPoints(ball, bodies[0], physicsClientId=cid)[0]
        return cp[9] / -cp[8]  # normal force / penetration depth
    finally:
        p.disconnect(cid)
