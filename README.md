# SO(3) state-estimation experiment

Reproduces the numbers and the figure of the subsection
*"A state-estimation illustration on SO(3)"* in

> *Resolution of the Borel–Kolmogorov Paradox via the Maximum Entropy Principle*,
> Trésor & Lukashchuk.

## What it shows

A latent orientation `R* ∈ SO(3)` is drawn from **one prior**, the invariant
(Haar) measure `nu_bi`, in both experiments. Its heading is measured by a
**gate**: `R*` is recorded only when it lies within `delta` of the slice
`{yaw = yaw*}`, a measure-zero 2-D submanifold (the Borel–Kolmogorov
situation). The pitch `β` is then inferred with the help of the **same**
isotropic attitude sensor in both experiments. What changes between the two
experiments is only the metric in which the heading gate is measured:

- **(a) translation invariant** (`g_bi`): the angle between the body's pointing
  axis and the half great circle of pointing directions of heading `yaw*`;
- **(b) flat chart** (`g_flat`): `|yaw − yaw*|`, read off the Euler chart.

Two analysts share the prior and the likelihood, and apply Definition 5 of the
paper (MaxEnt posterior) with different metrics. From `nu_bi`, the
**translation invariant** analyst obtains a posterior on the slice that is
uniform in (β, γ) (the `g_bi` tube width ∝ 1/cos β cancels the Haar cos β),
and the **flat chart** analyst obtains one ∝ cos β. This is the SO(3) analogue
of the sphere example, where the invariant metric gives the uniform posterior
on a great circle.

**Ground truth (symmetric run).** As `delta → 0`, the exact Bayes posterior is
Definition 5 in the gate's metric. So each analyst is calibrated exactly when
the heading was measured in its own metric, and the calibrated curve swaps
between the two panels while prior and sensor stay fixed.

These two names, *translation invariant* and *flat chart*, are the ones used in
the paper's caption, on the figure, and throughout the code and `results.json`.

## Headline results (deterministic given the fixed seeds)

Exact null-set posterior of the pitch (heading known exactly, prior `nu_bi`):

| quantity | translation invariant | flat chart |
|---|---|---|
| central 50% credible interval | `[-45°, 45°]` | `[-30°, 30°]` |
| `P(|pitch| < 30°)` | `0.333` | `0.500` |
| coverage of its nominal 90% interval when the gate is in the other metric | `98.8%` | `71.3%` |

Calibration with the isotropic attitude sensor of noise scale `σ`
(2000 Monte-Carlo trials/level, `delta = 0.29°`):

- **heading measured in `g_bi`** (translation invariant = exact Bayes): the
  invariant analyst covers 88.5–90.5% at every `σ`; the flat chart one covers
  87.8% at `σ = 3°` and under-covers down to 70.2% at `σ = 90°`.
- **heading measured in `g_flat`** (flat chart = exact Bayes): the flat chart
  analyst covers 88.9–91.0% at every `σ`; the invariant one covers 89.5% at
  `σ = 3°` and over-covers up to 99.1% at `σ = 90°`.

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python so3_experiment.py            # full run (~2 min); writes results.json
                                    # and ../../so3_state_estimation.pdf
python so3_experiment.py --quick    # fast smoke run
```

Outputs:
- `../../so3_state_estimation.pdf` — the figure included by the paper.
- `results.json` — every computed quantity.

All randomness is seeded inside `so3_experiment.py`; reruns reproduce the figure
and numbers exactly.
