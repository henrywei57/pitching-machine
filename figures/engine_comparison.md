| engine | fastball (90 mph, -2200 rpm dialed) | curveball (78 mph, +2500 rpm dialed) | knuckleball (67 mph, +0 rpm dialed) | repeatable |
|---|---|---|---|---|
| rigid, no slip (the textbook formula) | 90.0 mph, -2200 rpm | 78.0 mph, +2500 rpm | 67.0 mph, +0 rpm | n/a |
| MuJoCo + Coulomb contact law, dt 10 us (this repo's default) | 87.6 mph, -2329 rpm | 76.3 mph, +2614 rpm | 66.0 mph, -1 rpm | yes |
| MuJoCo + Coulomb contact law, dt 50 us | 87.4 mph, -2334 rpm | 76.2 mph, +2619 rpm | 65.9 mph, -0 rpm | yes |
| MuJoCo's own soft contact, dt 10 us | 72.1 mph, -2100 rpm | 64.6 mph, +2471 rpm | 57.3 mph, -5 rpm | yes |
| PyBullet compliant contact, dt 10 us | 85.4 mph, -2169 rpm | 73.8 mph, +2443 rpm | 63.3 mph, -1 rpm | yes |
| PyBullet compliant contact, dt 100 us | 85.7 mph, -2042 rpm | 74.0 mph, +2147 rpm | 63.0 mph, +104 rpm | yes |
