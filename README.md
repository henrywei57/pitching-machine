# pitching-machine

This is a simulation of a three-wheel pitching machine that I built in MuJoCo.

I wanted to see what actually happens to the ball inside a pitching machine,
and then what happens to it on the way to the plate. The big rule I set for
myself: the code never just *gives* the ball a spin rate. The wheels spin,
the ball gets squeezed between them, and whatever speed and spin it comes out
with is from the contact between the tires and the ball. Then the ball flies
through air that cares about where its seams are.

<p align="center">
  <img src="media/launch.gif" width="49%" alt="Slow-motion curveball going through three spinning wheels">
  <img src="media/knuckleballs.gif" width="49%" alt="Five knuckleballs from the catcher's view, each breaking a different way">
</p>
<p align="center"><sub>Left: a curveball going through the wheels in slow motion (about 250x). Right: five knuckleballs with the exact same machine settings. The only difference is how the ball was sitting in the chute.</sub></p>

## How to run it

You need Python 3.10 or newer.

```bash
git clone https://github.com/henrywei57/pitching-machine.git
cd pitching-machine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,render]"
```

Throw a pitch:

```bash
python -m pitching_machine.pitch curveball
```

You get something like this:

```
wheels       top 3251   lower 3B 2622   lower 1B 1993 rpm   (rims 96.7 / 78.0 / 59.3 mph)
ideal exit     78.0 mph    +2500 rpm   axis +30 deg
actual exit    77.5 mph    +2635 rpm   axis +30 deg   (99.4% of ideal speed, 105.4% of ideal spin)
contact      8.29 ms   peak squash 12.8 mm   peak normal force 1755 N   slipping 85% of the time   wheels slowed 3.5%
aim          +6.12 deg up   -2.27 deg toward 1B
at the plate  69.4 mph after 0.512 s   height 2.50 ft   side -0.3 in
break        +12.8 in horizontal (+ toward 1B)   -23.6 in vertical (vs. gravity + drag only)
```

The presets are `fastball`, `curveball`, `knuckle-curve` and `knuckleball`.
You can also make up your own pitch:

```bash
python -m pitching_machine.pitch --speed 85 --spin -2000 --axis 15 --plot pitch.png
python -m pitching_machine.pitch fastball --calibrate   # keeps adjusting the wheels until the "radar gun" says 90
python -m pitching_machine.pitch knuckleball --seed 3   # puts the ball in the chute a different way
python -m pitching_machine.pitch curveball --wheels 2   # the old two-wheel machine, for comparison
```

To watch it in MuJoCo's 3D viewer:

```bash
mjpython -m pitching_machine.view knuckleball   # on a Mac. Use plain python on Windows/Linux
```

To run the tests (37 of them, about 30 seconds) and remake every graph and GIF here:

```bash
pytest
for s in launch_anatomy contact_sweep fit_seam_model plate_scatter instrumented_ball compare_engines render; do
  python experiments/$s.py
done
```

`compare_engines.py` also needs PyBullet (`pip install -e ".[bullet]"`). On my
Mac it wouldn't build until I ran `export SDKROOT=$(xcrun --show-sdk-path)` first.

## The machine

There are three 10-inch tires spaced 120° apart around the path of the ball:
one on top, and two lower ones on the third-base side and the first-base side.
Each one has its own motor. The tires are squeezed half an inch into the ball.

The reason for three wheels instead of two is that three wheels can spin the
ball around **any** axis just by changing the wheel speeds. With two wheels
(top and bottom) you can only get topspin or backspin, and to throw anything
that breaks sideways you have to physically rotate the whole head. If the
ball didn't slip, the math would be:

```
rim speed of wheel i  =  ball speed  +  ball radius x spin x sin(wheel angle - spin axis angle)
```

Three wheels give three equations, which is enough to solve for speed, spin
rate *and* spin direction. `settings_for()` in
[`pitching_machine/machine.py`](pitching_machine/machine.py) runs this
backwards to get wheel speeds. In the curveball above, the three tires are at
96.7, 78.0 and 59.3 mph, and the spin axis comes out at 30° without tilting
anything.

That formula assumes the ball doesn't slip, though, and it does. So the
contact is where most of the actual work went:

- **The squash.** Each tire pushes on the ball like a spring
  (130,000 N/m) with some damping. So the harder they overlap, the harder
  they push.
- **Friction.** Every time step, the code figures out the friction forces
  that would make every tire grip the ball. If that would take more than
  μ × (the squash force), the tire slides instead. It's the same "static vs.
  kinetic friction" idea from physics class, but solved for three contacts
  at once, since they all push on the same ball.
- **MuJoCo** handles moving the wheels, motors and ball.

![Launch anatomy](figures/launch_anatomy.png)

Things I found out from this:

- **The ball doesn't start spinning right away.** For the first ~5 ms all
  three tires are sliding on the ball. Each one pushes with μ × N, and since
  the N's are about the same, the twisting forces cancel out. The ball speeds
  up but doesn't spin. Then the slowest tire (lower 1B side) catches up to
  the ball and grips it, and suddenly the other tires' friction isn't
  cancelled anymore. That's when the spin shows up, all in about 1 ms.
- **The ball comes out slower than the tires are spinning,** because the
  wheels slow down about 3.5% as they hand energy to the ball, and because
  the squashed contact point sits a little inside the tire.
- **The ball gets a little more spin than the formula says** (105%). The
  contact point is inside the ball's surface when it's squashed, so the same
  speed difference acts on a shorter lever.
- **A sensor inside the ball would feel about 1,750 g** while the tires grab
  it (this reads straight from MuJoCo's accelerometer).
- **Spinning pitches don't come out straight.** Topspin pitches leave 3–5°
  low and the backspin fastball leaves about 2° high. With topspin, the fast
  top tire is still sliding during the whole entry, and on that side its
  surface is moving forward *and down*. The two lower tires grab early and
  stop pushing, so nothing cancels that push. The aiming code fixes it,
  which is why the curveball above is aimed 6° up. A real operator would
  have to re-aim for each pitch type too.

![Contact sweep](figures/contact_sweep.png)

- **Too little squash means no spin at all.** The tires slip the whole way.
- **Too much squash with slippery tires and the ball gets spit back out.**
  This turns out to be the same rule as a rolling mill: the rollers only pull
  the metal in if μ > tan(nip angle). With μ = 0.4 it happens at 16 mm of
  squeeze.
- **The knuckle curve.** A knuckle curve is basically a curveball with a
  better grip (the pitcher digs a fingertip into the ball). On the machine,
  "grip" is the friction between tire and leather, so the `knuckle-curve`
  preset is the curveball with μ = 1.0 instead of 0.7. With the normal half
  inch of squeeze, the extra grip does almost nothing because the curveball
  already stops slipping. With only 6 mm of squeeze, the same extra grip adds
  about 1,100 rpm. So grip only helps while the ball is still slipping.

## The air

The aerodynamics are in [`pitching_machine/aero.py`](pitching_machine/aero.py).
There are three forces:

- **Drag.** Drag on a baseball isn't constant. Around 75–80 mph the air near
  the surface turns turbulent, stays attached longer, and the drag drops by
  more than half (this is called the drag crisis). I used the fit to Adair's
  data from Giordano & Nakanishi's textbook.
- **Magnus force** (why spin makes the ball curve), with the lift formula
  from Alan Nathan's trajectory calculator.
- **A seam force.** This was the hardest part, and it's what makes a
  knuckleball move. I drew the seam as a curve on the ball
  ([`seams.py`](pitching_machine/seams.py)). Every little piece of seam that's
  near the spot where the air leaves the ball pushes the ball a bit. I tuned
  how strong that push is so that a ball turned slowly in a wind tunnel gives
  the same sideways force Watts & Sawyer measured in 1975 (I used the formula
  Giordano & Nakanishi fit to their data). It matches that curve to 0.04%.

Seams in that zone also trip the airflow, which moves the drag crisis. A
knuckleball barely spins, so it keeps the same seam position for the whole
flight. That means its drag depends on how it happened to come out of the
machine. At 67 mph I got anywhere from C_D = 0.38 to 0.51. **There's no single
drag number for a knuckleball.**

![Seam model](figures/seam_model.png)

## Same settings, different pitches

Everything about the machine stays the same, and I aim at the middle of the
zone every time. The only thing that changes from pitch to pitch is how the
ball was sitting in the chute, which a real machine can't control.

![Plate scatter](figures/plate_scatter.png)

The fastball, curveball and knuckle curve all land within about 0.2 inches of
each other. The knuckleball is all over the place: about 11 inches side to
side and 10 inches up and down (one standard deviation).

At first I thought this might be the simulation being random or buggy, so I
checked. The same exact pitch thrown twice gives bit-for-bit the same answer.
Turning the ball in the chute moves the knuckleball in a smooth, repeatable
way. It's just really sensitive: up to a quarter inch at the plate for every
degree the ball is turned.

![Chute sensitivity](figures/chute_sensitivity.png)

### Does the physics engine matter?

I started this project in PyBullet and switched to MuJoCo later, so I built
the same machine in both and ran the same three pitches through each one
(the full table is in [`figures/engine_comparison.md`](figures/engine_comparison.md)):

| engine | fastball (90 mph, -2200 rpm dialed) | curveball (78 mph, +2500 rpm dialed) | knuckleball (67 mph, +0 rpm dialed) | repeatable |
|---|---|---|---|---|
| rigid, no slip (the textbook formula) | 90.0 mph, -2200 rpm | 78.0 mph, +2500 rpm | 67.0 mph, +0 rpm | n/a |
| MuJoCo + Coulomb contact law, dt 10 us (this repo's default) | 88.8 mph, -2327 rpm | 77.5 mph, +2635 rpm | 66.9 mph, -0 rpm | yes |
| MuJoCo + Coulomb contact law, dt 50 us | 88.6 mph, -2332 rpm | 77.4 mph, +2641 rpm | 66.8 mph, -0 rpm | yes |
| MuJoCo's own soft contact, dt 10 us | 73.8 mph, -2008 rpm | 66.1 mph, +2385 rpm | 58.4 mph, -2 rpm | yes |
| PyBullet compliant contact, dt 10 us | 87.7 mph, -2179 rpm | 76.3 mph, +2468 rpm | 64.7 mph, +5 rpm | yes |
| PyBullet compliant contact, dt 100 us | 85.9 mph, -2494 rpm | 75.9 mph, +2431 rpm | 64.2 mph, -11 rpm | yes |

Every engine gives the same answer every time, so none of them are random.
They disagree because they handle the contact differently. Some things I ran
into:

- **MuJoCo's built-in contact isn't good for this.** It lets friction "creep"
  for a few milliseconds even when the tire should be gripping. When the ball
  is being accelerated hard, the normal force also stops acting like a
  spring. That's why I wrote my own contact law. The built-in version is
  still there as `contact_model="mujoco"` if you want to compare.
- **PyBullet secretly caps everything at 100 rad/s.** That held the wheels to
  a 28 mph rim speed and the ball to 955 rpm. Every pitch came out at 27 mph
  until I figured that out and raised `maxJointVelocity`. PyBullet also
  multiplies the two objects' friction values together, and puts their
  springs in series, so I had to adjust for both. After that, it lands within
  about 1–3% of my contact law on speed.
- **Even getting MuJoCo's contact to be a spring of "k N/m" took
  measuring.** The ball squeezed between the tires felt exactly twice the
  force I expected, so there's a factor of 2 in the code and a test that
  checks it.
- **The flight part is converged.** It uses RK4 with 1 ms steps, and making
  the steps 10x smaller changes where the ball crosses the plate by less than
  0.01 inches.

## Using the ball as the sensor

Noel Perkins's lab at the University of Michigan built a baseball with a
wireless motion sensor (an IMU) inside it (McGinnis & Perkins, *Sensors*,
2012). That got me thinking about this. An accelerometer at the center of a
flying ball can't feel gravity, because everything is falling together. So
it only feels the air pushing on the ball. If you use the gyroscope to keep
track of which way the ball is facing, the ball can basically tell you the
drag, lift and seam forces on it at every moment, without needing any model
of the air.

[`experiments/instrumented_ball.py`](experiments/instrumented_ball.py) tries
this on a simulated knuckleball. I treat my full model as "reality" and put a
fake IMU inside the ball, with noise and bias, sampling at 1,000 times a
second. Then I compare what it measures against a simple model that uses
constant drag (C_D = 0.35) and no seams.

![Instrumented ball](figures/instrumented_ball.png)

- **The simple model misses the real plate location by about 13.5 inches.**
  Tracking the ball using only its own IMU readings gets within 0.7 inches.
- **The ball measures its own drag coefficient,** going from about 0.39 to
  0.47 as it slows down into the drag crisis. The simple model's 0.35 is
  wrong the entire time.
- **What the ball measures, minus what the simple model predicts, is the
  seam force.** It comes out of the data even though nothing in the
  measurement knows about seams. So the simulation makes a prediction, the
  ball measures what really happened, and the difference between them is the
  physics the simple model is missing.

## What's in the code

```
pitching_machine/
  machine.py           the MuJoCo machine: builds the model, my contact law, launching, calibrating
  aero.py              drag, Magnus force, seam force (does lots of balls at once)
  seams.py             the seam curve
  flight.py            flies the ball to the plate, measures break like Statcast does
  imu.py               the fake sensor inside the ball
  pitch.py             presets, aiming, and the command line tool
  view.py              the 3D viewer
  pybullet_machine.py  the same machine in PyBullet
experiments/           one script for each graph
tests/                 37 tests (contact springs, symmetry, same-answer-twice, step size, the IMU, ...)
```

## What's not realistic (yet)

- **I made up the machine's numbers.** The tire stiffness, damping, half
  inch of squeeze, μ = 0.7, wheel weight and motor strength are guesses at
  what a normal three-wheel machine might be like, not measurements from a
  real one. The sweeps above show how much each one matters.
- **The tire is treated like a single spring touching the ball at one
  point.** A real air-filled tire wraps around the ball.
- **The seam force reacts instantly.** In real air the wake takes a little
  time to change. Also, it was tuned using a ball turned around only one
  axis, so everything else is the model's guess.
- **Nobody fully knows how much seams matter on a ball that's spinning
  fast** (people call it the "seam-shifted wake"). I just turn the seam force
  down as spin goes up, so seams move fastballs and curveballs by inches, not
  feet. That's a placeholder. Knuckleballs barely spin, so they aren't
  affected.
- **The spin doesn't slow down in flight, and the ball is treated as solid
  all the way through.** A real ball has a cork center.
- **The PyBullet version only works with the machine level and not
  tilted.**

## Sources

- R. G. Watts and E. Sawyer, "Aerodynamics of a knuckleball," *American Journal of Physics* 43, 960 (1975).
- N. J. Giordano and H. Nakanishi, *Computational Physics*, 2nd ed. (2006), sections 2.2 and 2.3.
- R. K. Adair, *The Physics of Baseball*.
- A. M. Nathan, trajectory calculator and notes, baseball.physics.illinois.edu.
- R. S. McGinnis and N. C. Perkins, "A highly miniaturized, wireless inertial measurement unit for characterizing the dynamics of pitched baseballs and softballs," *Sensors* 12, 11933 (2012).
- E. Todorov, T. Erez and Y. Tassa, "MuJoCo: A physics engine for model-based control," IROS 2012.

My first version of this, in PyBullet, is in
[henrywei57/pitching-machine-sim](https://github.com/henrywei57/pitching-machine-sim).
