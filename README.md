# SO(3) state-estimation experiment

Reproduces the numbers and the figure of the subsection
*"A state-estimation illustration on SO(3)"* in

> *Resolution of the Borel–Kolmogorov Paradox via the Maximum Entropy Principle*,
> Trésor & Lukashchuk.

## What it shows

A latent orientation `R* ∈ SO(3)` is conditioned on an **exact heading (yaw)**
measurement, which pins it to a measure-zero 2-D submanifold of the group — a
genuine null-set conditioning event (the Borel–Kolmogorov situation). The pitch
angle `β` is then inferred. Two analysts process the *same* data and differ only
in the geometry they assume on the null set:

- **Geodesic / invariant** analyst — uses the rotation-invariant (Haar) volume.
  In the yaw–pitch–roll chart its surface density is `∝ cos β` (the SO(3) analogue
  of the `cos`-latitude weighting in the sphere example). This is the metric in
  which the isotropic measurement noise is isotropic (the §5 recipe).
- **Flat-Euler** analyst — treats the Euler-angle chart as a flat Euclidean box,
  i.e. drops the `cos β` Jacobian. This is the silent default of "linearizing in
  whatever chart is at hand".

**Ground truth (symmetric run).** The experiment runs under *both* generative
regimes. In the `isotropic` regime (`R* ~ Haar` + isotropic sensor noise) the
geodesic posterior is *exactly* the Bayes posterior and is calibrated, while the
flat-Euler prior (uniform in Euler angles) is not rotation-invariant and
over-covers. In the mirror `flat` regime (`R*` drawn uniform in the Euler chart)
the roles reverse: the flat-Euler posterior is exact Bayes and calibrated, and the
geodesic posterior under-covers. The calibrated analyst is always the one whose
metric matches the data-generating geometry; the script measures the cost of a
mismatch in either direction.

## Headline results (deterministic given the fixed seeds)

Exact null-set posterior of the pitch:

| quantity | geodesic (invariant) | flat-Euler |
|---|---|---|
| central 50% credible interval | `[-30°, 30°]` | `[-45°, 45°]` |
| `P(|pitch| < 30°)` | `0.500` | `0.333` |
| true coverage of its nominal 90% interval | `90.0%` | `98.8%` |

Calibration with an added isotropic attitude sensor of noise scale `σ`
(2000 Monte-Carlo trials/level), reported for both regimes:

- **isotropic-generated** (geodesic = exact Bayes): geodesic covers ≈90% at every
  `σ`; flat-Euler covers 89.5% at `σ = 3°` (sharp data, "nothing breaks") and
  over-covers up to 98.9% at `σ = 150°` (diffuse data).
- **flat-generated** (flat-Euler = exact Bayes): flat-Euler covers ≈90–92% at
  every `σ`; geodesic covers 88.3% at `σ = 3°` and under-covers down to 72.7% at
  `σ = 150°`.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python so3_experiment.py            # full run (~1 min); writes results.json
                                    # and ../../so3_state_estimation.pdf
python so3_experiment.py --quick    # fast smoke run
```

Outputs:
- `../../so3_state_estimation.pdf` — the figure included by the paper.
- `results.json` — every computed quantity.

All randomness is seeded inside `so3_experiment.py`; reruns reproduce the figure
and numbers exactly.
