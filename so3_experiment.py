#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
State estimation on SO(3): does the metric matter?
===================================================

Reproducible numerical experiment for the subsection
"A state-estimation illustration on SO(3)" of the paper

    Resolution of the Borel--Kolmogorov Paradox via the Maximum Entropy Principle
    (Tresor & Lukashchuk).

WHAT THIS SCRIPT DEMONSTRATES
-----------------------------
A latent orientation R* in SO(3) is observed by

  (i)  an *exact* heading (yaw) measurement  ->  this pins R* to a measure-zero
       2-D submanifold {yaw = yaw*} of the 3-D group: a genuine null-set
       conditioning event, the Borel--Kolmogorov situation; and
  (ii) an isotropic, noisy full-attitude measurement with concentration set by a
       tangent-space noise scale  sigma  (the standard "Gaussian on SO(3)" used in
       robotics; Barfoot 2017).

Two analysts process the *same* data and differ only in the geometry they assume
when they condition on the null set (i):

  * GEODESIC analyst -- uses the rotation-invariant (Haar) volume on the
    constraint submanifold.  In ZYX yaw-pitch-roll coordinates the invariant
    surface density is proportional to cos(pitch).  This is the metric in which
    the isotropic measurement noise is, in fact, isotropic -- the recipe of
    Section 5 ("use the metric in which the measurement noise is isotropic").

  * FLAT-EULER analyst -- treats the Euler-angle chart as if it were a flat
    Euclidean box, i.e. uses a uniform density in (pitch, roll).  This is the
    silent default a practitioner hits by "linearizing in whatever chart is at
    hand".  It drops the cos(pitch) Jacobian.

GROUND TRUTH AND THE SYMMETRIC RUN.  The experiment is run under BOTH generative
regimes, so the message is not "our metric is self-consistent" but "calibration
tracks whichever metric truly generated the data":

  * "isotropic" regime -- R* ~ Haar (rotation-invariant) with isotropic sensor
    noise.  Here the *geodesic* analyst is the exact Bayes posterior, so it is
    calibrated; the flat-Euler prior (uniform in Euler angles) is NOT rotation
    invariant, over-weights orientations near gimbal lock, and over-covers.

  * "flat" regime -- R* drawn uniform in the Euler (yaw,pitch,roll) chart, i.e.
    from the flat-Euler prior itself.  Now the *flat-Euler* analyst is the exact
    Bayes posterior and is calibrated, while the geodesic analyst is biased and
    under-covers.  This is the exact mirror of the isotropic regime: only the
    generative prior changes, the shared likelihood does not.

Nothing here is question-begging: each analyst is calibrated exactly when the
data are generated under its own geometry, and the experiment measures the cost
of assuming the wrong geometry, in either direction.

The script produces:
  * console output with all the headline numbers quoted in the paper,
  * results.json  with every computed quantity (for strict reproducibility),
  * ../../so3_state_estimation.pdf  -- the three-panel figure the paper includes.

Everything is deterministic given the seeds below.

Run:
    python so3_experiment.py
Dependencies: numpy, scipy, matplotlib (see requirements.txt).
"""

import json
import os
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# --------------------------------------------------------------------------- #
#  Elementary rotation matrices (vectorised over a grid of angles)            #
# --------------------------------------------------------------------------- #
def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Ry_grid(betas):
    """Stack of Ry(beta) for an array of angles -> (N,3,3)."""
    c, s = np.cos(betas), np.sin(betas)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, z, s], axis=-1),
                     np.stack([z, o, z], axis=-1),
                     np.stack([-s, z, c], axis=-1)], axis=-2)


def Rx_grid(gammas):
    """Stack of Rx(gamma) for an array of angles -> (N,3,3)."""
    c, s = np.cos(gammas), np.sin(gammas)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([o, z, z], axis=-1),
                     np.stack([z, c, -s], axis=-1),
                     np.stack([z, s, c], axis=-1)], axis=-2)


# --------------------------------------------------------------------------- #
#  Sanity check: Haar measure has density proportional to cos(pitch)          #
# --------------------------------------------------------------------------- #
def verify_haar_cosine(n=2_000_000, n_bins=30, seed=0):
    """Confirm the fact the whole experiment rests on: under the Haar (uniform)
    measure on SO(3), the ZYX pitch angle has marginal density proportional to
    cos(pitch), while yaw and roll are uniform.  Returns the max relative error
    of the cos fit."""
    rng = np.random.default_rng(seed)
    e = Rot.random(n, random_state=rng).as_euler("ZYX")  # [yaw, pitch, roll]
    pitch = e[:, 1]
    bins = np.linspace(-np.pi / 2, np.pi / 2, n_bins + 1)
    hist, _ = np.histogram(pitch, bins=bins, density=True)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    pred = np.cos(ctr) / 2.0  # cos normalised on (-pi/2, pi/2)
    rel_err = np.max(np.abs(hist - pred) / pred)
    return float(rel_err)


# --------------------------------------------------------------------------- #
#  Part A -- exact null-set posteriors (heading observed, no other sensor)    #
# --------------------------------------------------------------------------- #
def exact_null_set_summary():
    r"""Closed-form posteriors of the pitch angle on the slice {yaw = yaw*}.

    Prior R* ~ Haar  =>  pitch marginal density  p(b) = cos(b)/2  on (-pi/2,pi/2)
    (the GEODESIC / invariant answer, which is also the true Bayes posterior).
    The FLAT-EULER analyst instead reports the uniform density 1/pi on the same
    set.  We report the credible intervals, the level-band probability, and the
    MAP for each -- the SO(3) analogue of the sphere numbers in Section 4.
    """
    # Geodesic posterior  cos(b)/2 :  CDF F(b) = (sin b + 1)/2.
    def geo_quantile(p):           # F^{-1}(p)
        return np.arcsin(2.0 * p - 1.0)

    # Flat posterior  1/pi  on (-pi/2, pi/2):  CDF F(b) = (b + pi/2)/pi.
    def flat_quantile(p):
        return p * np.pi - np.pi / 2.0

    def band_prob_geo(b):          # P(|pitch| < b)
        return np.sin(b)
    def band_prob_flat(b):
        return 2.0 * b / np.pi

    deg = 180.0 / np.pi
    out = {}
    for level, tag in [(0.50, "ci50"), (0.90, "ci90")]:
        lo, hi = (1 - level) / 2, (1 + level) / 2
        out[tag] = {
            "level": level,
            "geodesic_deg": [geo_quantile(lo) * deg, geo_quantile(hi) * deg],
            "flat_deg":     [flat_quantile(lo) * deg, flat_quantile(hi) * deg],
        }
    band = 30.0 / deg  # +/- 30 degrees of level
    out["band_pm30deg"] = {
        "geodesic_prob": float(band_prob_geo(band)),
        "flat_prob":     float(band_prob_flat(band)),
    }
    out["map_pitch_deg"] = {"geodesic": 0.0, "flat": "undefined (uniform)"}
    # coverage of the FLAT analyst's 90% interval under the TRUE (cos) law:
    flat90 = out["ci90"]["flat_deg"]
    out["flat_ci90_true_coverage"] = float(band_prob_geo(flat90[1] / deg))
    return out


# --------------------------------------------------------------------------- #
#  Part B -- full estimation with an isotropic noisy sensor + calibration     #
# --------------------------------------------------------------------------- #
def geodesic_angle_to(Rgrid, R_obs):
    """Geodesic (rotation) angle between every R in Rgrid (...,3,3) and R_obs."""
    # trace(R^T R_obs) computed without forming the product explicitly:
    tr = np.einsum("...ij,ij->...", Rgrid, R_obs)  # since trace(R^T M)=sum(R*M)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(c)


def slice_rotations(yaw_star, betas, gammas):
    """Rotation matrices R(yaw*, beta, gamma) over the (beta,gamma) grid.

    Intrinsic ZYX:  R = Rz(yaw) Ry(beta) Rx(gamma).  Returns array (Nb, Ng, 3,3).
    """
    M = Rz(yaw_star)                              # (3,3)
    Rb = Ry_grid(betas)                           # (Nb,3,3)
    Rg = Rx_grid(gammas)                          # (Ng,3,3)
    MRb = np.einsum("ij,bjk->bik", M, Rb)         # Rz Ry  -> (Nb,3,3)
    R = np.einsum("bij,gjk->bgik", MRb, Rg)       # (Nb,Ng,3,3)
    return R


def posteriors_on_slice(yaw_star, R_obs, sigma, betas, gammas):
    """Return marginal pitch posteriors p(beta) for the geodesic and flat
    analysts, given the exact yaw constraint and one isotropic noisy attitude
    observation R_obs with tangent noise scale sigma."""
    R = slice_rotations(yaw_star, betas, gammas)          # (Nb,Ng,3,3)
    theta = geodesic_angle_to(R, R_obs)                   # (Nb,Ng)
    loglik = -(theta ** 2) / (2.0 * sigma ** 2)
    loglik -= loglik.max()                                # stabilise
    lik = np.exp(loglik)                                  # (Nb,Ng)

    dgamma = gammas[1] - gammas[0]
    cosb = np.cos(betas)                                  # invariant Jacobian

    # marginal over gamma (uniform measure dgamma for both analysts):
    m = lik.sum(axis=1) * dgamma                          # (Nb,)
    p_geo = cosb * m
    p_flat = m.copy()

    dbeta = betas[1] - betas[0]
    p_geo /= p_geo.sum() * dbeta
    p_flat /= p_flat.sum() * dbeta
    return p_geo, p_flat


def central_interval(betas, dens, level):
    """Central credible interval [lo,hi] of a 1-D density on the beta grid."""
    dbeta = betas[1] - betas[0]
    cdf = np.cumsum(dens) * dbeta
    cdf /= cdf[-1]
    lo = np.interp((1 - level) / 2, cdf, betas)
    hi = np.interp((1 + level) / 2, cdf, betas)
    return lo, hi


def draw_truths(regime, n_trials, rng):
    """Draw the ground-truth orientations for one generative regime.

    "isotropic" (regime A): R* ~ Haar (rotation-invariant prior). In the ZYX
    chart its pitch marginal is proportional to cos(pitch); the GEODESIC analyst,
    whose prior carries the cos(pitch) Jacobian, is then the exact Bayes analyst.

    "flat" (regime B): the truth is drawn *uniform in the Euler (yaw, pitch, roll)
    chart* -- the flat-Euler prior itself. The FLAT-EULER analyst, whose prior is
    uniform in the chart, is then the exact Bayes analyst, and the geodesic one is
    biased. This is the exact mirror of regime A: only the generative prior
    changes, the shared likelihood (isotropic geodesic-angle Gaussian) does not.
    """
    if regime == "isotropic":
        return Rot.random(n_trials, random_state=rng)
    if regime == "flat":
        yaw = rng.uniform(-np.pi, np.pi, n_trials)
        pitch = rng.uniform(-np.pi / 2, np.pi / 2, n_trials)
        roll = rng.uniform(-np.pi, np.pi, n_trials)
        return Rot.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=-1))
    raise ValueError(f"unknown regime {regime!r}")


def run_calibration(sigmas, n_trials, betas, gammas, seed=12345, level=0.90,
                    regime="isotropic"):
    """Monte-Carlo coverage of the nominal `level` pitch credible interval for
    each analyst, as a function of the noise scale sigma, under one generative
    `regime` ("isotropic" -> geodesic is exact Bayes; "flat" -> flat-Euler is)."""
    rng = np.random.default_rng(seed)
    cov_geo = np.zeros(len(sigmas))
    cov_flat = np.zeros(len(sigmas))
    width_geo = np.zeros(len(sigmas))
    width_flat = np.zeros(len(sigmas))

    # Pre-draw all truths/noise so the result is independent of sigma ordering.
    R_true = draw_truths(regime, n_trials, rng)
    eta = rng.standard_normal((len(sigmas), n_trials, 3))

    for si, sigma in enumerate(sigmas):
        hit_g = hit_f = 0
        wsum_g = wsum_f = 0.0
        for t in range(n_trials):
            Rt = R_true[t]
            yaw_star, beta_star, _ = Rt.as_euler("ZYX")
            # isotropic noisy attitude observation: R_obs = R_true * exp(eta^)
            R_obs = (Rt * Rot.from_rotvec(sigma * eta[si, t])).as_matrix()
            p_geo, p_flat = posteriors_on_slice(
                yaw_star, R_obs, sigma, betas, gammas)
            lo_g, hi_g = central_interval(betas, p_geo, level)
            lo_f, hi_f = central_interval(betas, p_flat, level)
            hit_g += (lo_g <= beta_star <= hi_g)
            hit_f += (lo_f <= beta_star <= hi_f)
            wsum_g += (hi_g - lo_g)
            wsum_f += (hi_f - lo_f)
        cov_geo[si] = hit_g / n_trials
        cov_flat[si] = hit_f / n_trials
        width_geo[si] = wsum_g / n_trials
        width_flat[si] = wsum_f / n_trials
    return cov_geo, cov_flat, width_geo, width_flat


def representative_trial(sigma, betas, gammas, seed):
    """One trial whose posterior makes a clear figure: returns the two pitch
    posteriors, the truth, and the noisy observation."""
    rng = np.random.default_rng(seed)
    Rt = Rot.random(random_state=rng)
    yaw_star, beta_star, _ = Rt.as_euler("ZYX")
    R_obs = (Rt * Rot.from_rotvec(sigma * rng.standard_normal(3))).as_matrix()
    p_geo, p_flat = posteriors_on_slice(yaw_star, R_obs, sigma, betas, gammas)
    return beta_star, p_geo, p_flat


# --------------------------------------------------------------------------- #
#  Figure                                                                     #
# --------------------------------------------------------------------------- #
def make_figure(betas, exact, sigmas, calib_iso, calib_flat, level, out_pdf):
    """Two panels, the symmetric calibration run: (a) data generated from the
    isotropic (geodesic-true) model; (b) data generated from the flat-chart
    (flat-Euler-true) model. The panels are mirror images: in each, the analyst
    matching the generative geometry sits on the nominal line and the other one
    drifts off it. (The exact null-set posterior shapes are the SO(3) analogue of
    Figure 2 and are given in the text by eq. (so3-two-posteriors); `exact` and
    `betas` are kept for the printed/JSON headline numbers, not plotted here.)

    calib_iso / calib_flat are (cov_geo, cov_flat) coverage arrays per regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.titlesize": 9, "figure.dpi": 200,
    })
    deg = 180.0 / np.pi
    C_GEO, C_FLAT = "#1f5fa6", "#c0392b"
    sig_deg = np.array(sigmas) * deg

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---- Panels (a), (b) : calibration under each generative regime ----
    def calibration_panel(ax, cov_geo, cov_flat, title):
        ax.axhline(level * 100, color="0.4", lw=1, ls=":")
        ax.text(sig_deg[-1], level * 100 + 0.6, f"nominal {int(level*100)}%",
                ha="right", va="bottom", color="0.4", fontsize=7.5)
        ax.plot(sig_deg, cov_geo * 100, "o-", color=C_GEO, lw=2, ms=4,
                label="geodesic (invariant)")
        ax.plot(sig_deg, cov_flat * 100, "s--", color=C_FLAT, lw=2, ms=4,
                label="flat-Euler chart")
        ax.set_xscale("log")
        ax.set_xlabel(r"sensor noise scale $\sigma$  (degrees)")
        ax.set_title(title)
        ax.legend(loc="center left", frameon=False)
        ax.grid(True, which="both", alpha=0.25)

    axA.set_ylabel(f"coverage of {int(level*100)}% interval (%)")
    calibration_panel(axA, calib_iso[0], calib_iso[1],
                      "(a) data generated: isotropic")
    calibration_panel(axB, calib_flat[0], calib_flat[1],
                      "(b) data generated: flat-chart")
    # share the y-range across the two panels for honest comparison
    ylo = min(axA.get_ylim()[0], axB.get_ylim()[0])
    yhi = max(axA.get_ylim()[1], axB.get_ylim()[1])
    axA.set_ylim(ylo, yhi)
    axB.set_ylim(ylo, yhi)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[figure] wrote {out_pdf}")


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000,
                    help="Monte-Carlo trials per noise level (default 2000).")
    ap.add_argument("--grid", type=int, default=181,
                    help="grid points per Euler angle (default 181).")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run (few trials / coarse grid).")
    args = ap.parse_args()
    if args.quick:
        args.trials, args.grid = 200, 91

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    out_pdf = os.path.join(repo_root, "so3_state_estimation.pdf")
    out_json = os.path.join(here, "results.json")

    # grids on the constraint slice (pitch in (-pi/2,pi/2), roll in (-pi,pi])
    eps = 1e-6
    betas = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, args.grid)
    gammas = np.linspace(-np.pi, np.pi, args.grid, endpoint=False)

    print("=" * 70)
    print("SO(3) state-estimation experiment -- Borel-Kolmogorov / MaxEnt paper")
    print("=" * 70)

    # 0. sanity
    rel = verify_haar_cosine()
    print(f"\n[0] Haar pitch-density vs cos(pitch): max rel. error = {rel:.4f}"
          f"  ({'OK' if rel < 0.05 else 'CHECK'})")

    # A. exact null-set posteriors
    exact = exact_null_set_summary()
    print("\n[A] Exact null-set posteriors of pitch (heading observed exactly)")
    print(f"    central 50% interval : geodesic {exact['ci50']['geodesic_deg'][0]:+.1f}"
          f" .. {exact['ci50']['geodesic_deg'][1]:+.1f} deg"
          f"   |  flat-Euler {exact['ci50']['flat_deg'][0]:+.1f}"
          f" .. {exact['ci50']['flat_deg'][1]:+.1f} deg")
    print(f"    P(|pitch| < 30 deg)  : geodesic {exact['band_pm30deg']['geodesic_prob']:.3f}"
          f"   |  flat-Euler {exact['band_pm30deg']['flat_prob']:.3f}")
    print(f"    flat-Euler's nominal 90% interval truly covers "
          f"{100*exact['flat_ci90_true_coverage']:.1f}%  (target 90%)")

    # B. calibration under BOTH generative regimes (the symmetric run):
    #    - "isotropic": data ~ Haar prior + isotropic noise  -> geodesic is exact Bayes;
    #    - "flat":      data ~ uniform-in-Euler-chart prior   -> flat-Euler is exact Bayes.
    # The two analysts are unchanged; only the law that generated the data changes.
    deg = 180.0 / np.pi
    sigmas = list(np.array([3, 6, 12, 25, 50, 90, 150]) / deg)
    print(f"\n[B] Calibration: {args.trials} trials/level, "
          f"{args.grid}x{args.grid} grid, sigma in "
          f"{[round(s*deg) for s in sigmas]} deg, both regimes ...")

    calib = {}
    for regime, calibrated in [("isotropic", "geodesic"), ("flat", "flat-Euler")]:
        cg, cf, wg, wf = run_calibration(
            sigmas, args.trials, betas, gammas, regime=regime)
        calib[regime] = (cg, cf, wg, wf)
        print(f"  regime={regime:9s} (exact Bayes = {calibrated}):")
        for s, g, f in zip(sigmas, cg, cf):
            print(f"    sigma={s*deg:5.0f} deg :  geodesic {100*g:5.1f}%   "
                  f"flat-Euler {100*f:5.1f}%   (nominal 90%)")

    # save everything
    def cal_block(regime):
        cg, cf, wg, wf = calib[regime]
        return {
            "sigma_deg": [s * deg for s in sigmas],
            "coverage_geodesic": cg.tolist(),
            "coverage_flat": cf.tolist(),
            "mean_width_geodesic_deg": (np.array(wg) * deg).tolist(),
            "mean_width_flat_deg": (np.array(wf) * deg).tolist(),
        }

    results = {
        "config": {"trials": args.trials, "grid": args.grid,
                   "sigmas_deg": [s * deg for s in sigmas], "level": 0.90},
        "haar_cos_rel_error": rel,
        "exact_null_set": exact,
        # data-generated-isotropic regime (geodesic analyst is exact Bayes):
        "calibration": cal_block("isotropic"),
        # symmetric run: data-generated-flat regime (flat-Euler is exact Bayes):
        "calibration_flat_generated": cal_block("flat"),
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[json] wrote {out_json}")

    # figure: (a) posteriors, (b) isotropic-generated calibration, (c) flat-generated
    make_figure(betas, exact, sigmas,
                calib["isotropic"][:2], calib["flat"][:2], 0.90, out_pdf)
    print("\nDone.")


if __name__ == "__main__":
    main()
