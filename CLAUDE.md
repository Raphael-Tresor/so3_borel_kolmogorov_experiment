# CLAUDE.md — SO(3) state-estimation experiment

> **Maintenance rule (read first).** This file documents the code in this
> directory. **Whenever you change the code, update this file in the same edit**
> so it never drifts from reality. **Keep it concise — under 300 lines.** If it
> grows past that, cut prose, not facts.

## Purpose

Reproducible numerical experiment for the paper subsection *"A state-estimation
illustration on SO(3)"* (`../../bj-template.tex`, `\label{fig:so3-estimation}`).
It shows that on `SO(3)` the **choice of metric changes the estimate an engineer
reports**, and (via the symmetric run over two generative regimes) that **the
calibrated analyst is always the one whose metric matches the data-generating
geometry** — neither metric is universally right. Every number quoted in the
paper and Figure 3 is produced here.

## Files

| File | Role |
|---|---|
| `so3_experiment.py` | The whole experiment. Run it to reproduce numbers + figure. |
| `requirements.txt` | Pinned deps (numpy 2.0.2, scipy 1.13.1, matplotlib 3.9.4; Python 3.9.6). |
| `README.md` | Human-facing summary + headline results table. |
| `results.json` | Generated: every computed quantity. Do not hand-edit. |
| `../../so3_state_estimation.pdf` | Generated: the figure the paper includes. |

The paper includes the PDF by relative path, so `so3_experiment.py` writes it to
the **repo root**, not this directory. Do not move it without editing the
`\includegraphics` line in `bj-template.tex`.

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python so3_experiment.py            # full run ~1 min: results.json + figure PDF
python so3_experiment.py --quick    # smoke run (coarse grid, few trials)
```

Repo convention: a venv named `.venv-so3/` at the repo root is git-ignored.
All randomness is seeded inside the script; reruns are bit-stable.

## Code map (`so3_experiment.py`)

- `Rz`, `Ry_grid`, `Rx_grid` — elementary rotation matrices, vectorised over
  angle grids.
- `verify_haar_cosine` — sanity check of the load-bearing fact (see Invariants).
- `exact_null_set_summary` — closed-form pitch posteriors for the exact-heading
  case; returns the credible intervals, the ±30° band probability, and the true
  coverage of the flat analyst's nominal 90% interval. **Headline numbers quoted
  in the text** (the posterior shapes are no longer plotted; they are the SO(3)
  analogue of the paper's Figure 2 and are given there by eq. so3-two-posteriors).
- `slice_rotations` / `geodesic_angle_to` / `posteriors_on_slice` — build the
  posterior over pitch on the constraint slice `{yaw = yaw*}` for both analysts,
  given one isotropic noisy attitude observation.
- `central_interval` — central credible interval of a 1-D density on the grid.
- `draw_truths(regime, ...)` — ground-truth orientations for one generative
  regime: `"isotropic"` = Haar (geodesic analyst is exact Bayes), `"flat"` =
  uniform in the Euler chart (flat-Euler analyst is exact Bayes). The symmetric
  run runs both.
- `run_calibration(..., regime=...)` — Monte-Carlo coverage vs noise scale
  `sigma` under one regime. **Panels (a) and (b).**
- `representative_trial` — one trial's posteriors (helper; not currently plotted).
- `make_figure` — writes the two-panel PDF: (a) calibration under
  isotropic-generated data, (b) calibration under flat-generated data.
- `main` — orchestrates both regimes, prints headline numbers, writes
  `results.json` + PDF.

## Invariants the code (and the paper) depend on

1. **Haar density ∝ cos(pitch)** in scipy's `ZYX` yaw–pitch–roll convention;
   yaw/roll uniform. This is the SO(3) analogue of the sphere's cos(latitude)
   and the source of the whole effect. `verify_haar_cosine` asserts it
   (< 5% MC error). If you change the Euler convention, re-derive this factor —
   `ZYX` gives `cos`, other conventions differ.
2. **Ground truth = whichever prior generated the data.** In the `"isotropic"`
   regime (Haar prior + isotropic noise) the geodesic posterior is exact Bayes,
   so geodesic coverage must land near nominal. In the `"flat"` regime (uniform
   in the Euler chart) the flat-Euler posterior is exact Bayes, so flat-Euler
   coverage must land near nominal. The calibrated analyst is always the one whose
   metric matches the generative prior; if the matching analyst drifts far from
   90%, that is the bug. Only the generative prior differs between regimes; the
   shared likelihood (isotropic geodesic-angle Gaussian) does not.
3. **The two analysts differ only in the `cos β` Jacobian.** `posteriors_on_slice`
   applies `cosb` to the geodesic branch and not to the flat branch; everything
   else (likelihood, γ-marginalisation, normalisation) is shared, and this holds
   in *both* regimes (the regime changes `draw_truths`, never the analysts). Keep
   it that way — that single factor *is* the experiment.

## Expected headline results (regression check)

Exact null-set posterior of pitch:

| quantity | geodesic | flat-Euler |
|---|---|---|
| central 50% interval | `[-30°, 30°]` | `[-45°, 45°]` |
| `P(|pitch| < 30°)` | `0.500` | `0.333` |
| true coverage of nominal 90% interval | `90.0%` | `98.8%` |

Calibration (2000 trials/level), the symmetric run:

- **Isotropic-generated regime** (panel a, geodesic is exact Bayes): geodesic
  ≈ 88.5–90% at every `sigma`; flat-Euler 89.5% at `sigma = 3°`, rising to 98.9%
  at `sigma = 150°` (over-covers = conservative).
- **Flat-generated regime** (panel b, flat-Euler is exact Bayes): flat-Euler
  ≈ 90–92% at every `sigma`; geodesic 88.3% at `sigma = 3°`, falling to 72.7% at
  `sigma = 150°` (under-covers = over-confident).

The mirror is the point: the calibrated curve swaps panels. Note the asymmetry is
real — the wrong flat analyst over-covers, the wrong geodesic analyst under-covers.

If your change moves these numbers, either it is a bug or the paper text and this
table must be updated together (and re-check `bj-template.tex`).

## When you change the code — checklist

- [ ] Update the **Code map** / **Invariants** / **results table** above.
- [ ] Re-run `python so3_experiment.py` (full, not `--quick`) to refresh
      `results.json` and the figure PDF.
- [ ] If numbers changed, update the prose and the `\caption` in
      `bj-template.tex` (subsection `subsec:methodology-robotics`) and recompile.
- [ ] Keep this file **under 300 lines**.
