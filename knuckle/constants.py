"""Regulation baseball, sea-level air, and the unit conversions baseball people use."""

import numpy as np

G = 9.81  # m/s^2

# Air at ~70 F, sea level (the default in Alan Nathan's trajectory calculator).
RHO_AIR = 1.20  # kg/m^3

# MLB rule 3.01: 5 to 5.25 oz, 9 to 9.25 in circumference.
BALL_MASS = 0.145  # kg
BALL_RADIUS = 0.0369  # m
BALL_AREA = np.pi * BALL_RADIUS**2
# Uniform-sphere approximation. A real ball has a dense cork/rubber pill, so the
# true value is a few percent lower; it only changes how spin couples to contact.
BALL_INERTIA = 0.4 * BALL_MASS * BALL_RADIUS**2

MPH = 0.44704  # m/s per mph
RPM = 2.0 * np.pi / 60.0  # rad/s per rpm
INCH = 0.0254  # m
FOOT = 0.3048  # m

# Machine exit point to the front edge of home plate. A pitcher releases about
# 6 ft in front of the rubber, so 60.5 ft - ~5.5 ft.
PLATE_DISTANCE = 55.0 * FOOT
