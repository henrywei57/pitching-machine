| engine | fastball (90 mph, -2200 rpm dialed) | curveball (78 mph, +2500 rpm dialed) | knuckleball (67 mph, +0 rpm dialed) | repeatable |
|---|---|---|---|---|
| rigid, no slip (the textbook formula) | 90.0 mph, -2200 rpm | 78.0 mph, +2500 rpm | 67.0 mph, +0 rpm | n/a |
| MuJoCo + Coulomb contact law, dt 10 us (this repo's default) | 88.8 mph, -2327 rpm | 77.5 mph, +2635 rpm | 66.9 mph, -0 rpm | yes |
| MuJoCo + Coulomb contact law, dt 50 us | 88.6 mph, -2332 rpm | 77.4 mph, +2641 rpm | 66.8 mph, -0 rpm | yes |
| MuJoCo's own soft contact, dt 10 us | 73.8 mph, -2008 rpm | 66.1 mph, +2385 rpm | 58.4 mph, -2 rpm | yes |
| PyBullet compliant contact, dt 10 us | 87.7 mph, -2179 rpm | 76.3 mph, +2468 rpm | 64.7 mph, +5 rpm | yes |
| PyBullet compliant contact, dt 100 us | 85.9 mph, -2494 rpm | 75.9 mph, +2431 rpm | 64.2 mph, -11 rpm | yes |
