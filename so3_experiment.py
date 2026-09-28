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
ONE prior, ONE attitude sensor, TWO heading measurements.  In both experiments
the latent orientation R* is drawn from the same prior nu_bi (Haar) and is
observed by

  (i)  a heading GATE: R* is recorded only when it lies within `delta` of the
       2-D slice A = {yaw = yaw*}, the distance to A being measured in the
       metric of the experiment (see `slice_distance`).  As delta -> 0 this is a
       null-set conditioning event (the Borel--Kolmogorov situation) and, being
       the tube of Definition 5 of the paper, it is approached in that metric:
         - experiment (a), g_bi  : the angle between the pointing axis b1 = R* e1
           and the half great circle of pointing directions of heading yaw*;
         - experiment (b), g_flat: |yaw - yaw*|, read off the Euler chart;
  (ii) the SAME isotropic, noisy full-attitude sensor in both experiments, with
       tangent-space noise scale sigma (the standard "Gaussian on SO(3)" used in
       robotics; Barfoot 2017).

NAMING.  The two metrics are called, here and in the paper (caption of
fig:so3-estimation) and on the figure itself, by the same two names:

  * TRANSLATION INVARIANT, g_bi -- the unique metric (up to scale) invariant
    under left and right translation.
  * FLAT CHART, g_flat -- the Euler-angle chart declared Euclidean.

Keep this vocabulary when editing: "isotropic", "geodesic" and "flat-Euler" were
earlier names for these same two objects and are no longer used for them.

Two analysts share the prior nu_bi and the likelihood, and apply Definition 5
(MaxEnt posterior) on A with different metrics:

  * TRANSLATION INVARIANT analyst -- conditions nu_bi in g_bi.  The g_bi tube
    around A has width proportional to 1/cos(pitch), which cancels the
    cos(pitch) of the Haar density: the posterior on A is UNIFORM in
    (pitch, roll), the induced g_bi area (the SO(3) analogue of the uniform
    posterior on a great circle of the sphere, Prop. in Section 4).

  * FLAT CHART analyst -- conditions nu_bi in g_flat.  The tube has constant
    width in yaw, so the Haar density is simply restricted: the posterior on A
    is proportional to cos(pitch).

Each analyst is the exact Bayes posterior when the gate is measured in its own
metric, so the calibrated curve swaps between the two panels while the prior
and the sensor stay fixed: the experiment isolates the metric of the null-set
measurement, not a choice of prior.

The script produces:
  * console output with all the headline numbers quoted in the paper,
  * results.json  with every computed quantity (for strict reproducibility),
  * ../../so3_state_estimation.pdf  -- the two-panel figure the paper includes.

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


def verify_def5_slice_conditionals(n=4_000_000, delta=0.01, seed=1):
    """Confirm the fact the two analysts rest on: conditioning nu_bi on thin
    tubes around {yaw = 0} gives, for the pitch, a UNIFORM law when the tube is
    measured in g_bi and a cos(pitch)/2 law when it is measured in g_flat
    (Definition 5 as delta -> 0).  Returns the Kolmogorov--Smirnov distances
    {metric: (D to uniform, D to cos/2)}; the matching one should be small."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    R = Rot.random(n, random_state=rng)
    pitch = R.as_euler("ZYX")[:, 1]
    cdf_unif = lambda b: (b + np.pi / 2) / np.pi
    cdf_cos = lambda b: (np.sin(b) + 1.0) / 2.0
    out = {}
    for metric in ("invariant", "flat"):
        b = pitch[slice_distance(R, metric) < delta]
        out[metric] = (float(stats.kstest(b, cdf_unif).statistic),
                       float(stats.kstest(b, cdf_cos).statistic))
    return out


# --------------------------------------------------------------------------- #
#  Part A -- exact null-set posteriors (heading observed, no other sensor)    #
# --------------------------------------------------------------------------- #
def exact_null_set_summary():
    r"""Closed-form MaxEnt posteriors (Definition 5) of the pitch on the slice
    {yaw = yaw*}, from the SAME prior nu_bi (Haar), heading known exactly.

    TRANSLATION INVARIANT analyst (g_bi):  uniform density 1/pi on (-pi/2,pi/2).
    FLAT CHART analyst (g_flat):           p(b) = cos(b)/2.
    As on the sphere (Section 4: 1/3 vs 1/2, [-45,45] vs [-30,30]), we report the
    credible intervals, the level-band probability and the MAP for each, and the
    true coverage of each analyst's nominal 90% interval when the gate is
    measured in the OTHER metric.
    """
    # Invariant posterior  1/pi  on (-pi/2, pi/2):  CDF F(b) = (b + pi/2)/pi.
    def inv_quantile(p):           # F^{-1}(p)
        return p * np.pi - np.pi / 2.0

    # Flat posterior  cos(b)/2 :  CDF F(b) = (sin b + 1)/2.
    def flat_quantile(p):
        return np.arcsin(2.0 * p - 1.0)

    def band_prob_inv(b):          # P(|pitch| < b)
        return 2.0 * b / np.pi
    def band_prob_flat(b):
        return np.sin(b)

    deg = 180.0 / np.pi
    out = {}
    for level, tag in [(0.50, "ci50"), (0.90, "ci90")]:
        lo, hi = (1 - level) / 2, (1 + level) / 2
        out[tag] = {
            "level": level,
            "invariant_deg": [inv_quantile(lo) * deg, inv_quantile(hi) * deg],
            "flat_deg":      [flat_quantile(lo) * deg, flat_quantile(hi) * deg],
        }
    band = 30.0 / deg  # +/- 30 degrees of level
    out["band_pm30deg"] = {
        "invariant_prob": float(band_prob_inv(band)),
        "flat_prob":      float(band_prob_flat(band)),
    }
    out["map_pitch_deg"] = {"invariant": "undefined (uniform)", "flat": 0.0}
    # true coverage of each analyst's 90% interval when the other one is right:
    inv90, flat90 = out["ci90"]["invariant_deg"], out["ci90"]["flat_deg"]
    out["flat_ci90_coverage_under_invariant_gate"] = float(band_prob_inv(flat90[1] / deg))
    out["invariant_ci90_coverage_under_flat_gate"] = float(band_prob_flat(inv90[1] / deg))
    return out


# --------------------------------------------------------------------------- #
#  Part B -- full estimation with an isotropic noisy sensor + calibration     #
# --------------------------------------------------------------------------- #
def geodesic_angle_to(Rgrid, R_obs):
    """Geodesic (rotation) angle between every R in Rgrid (...,3,3) and R_obs.

    Note: "geodesic" here is literal (the rotation angle) and is shared by BOTH
    analysts through the likelihood; it is not the name of the invariant metric.
    """
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
    """Return marginal pitch posteriors p(beta) for the translation invariant
    and flat chart analysts, given the yaw constraint {yaw = yaw*} and one
    isotropic noisy attitude observation R_obs with tangent noise scale sigma.

    Both start from nu_bi and differ only in the metric of Definition 5, i.e. in
    the conditional of nu_bi on the slice (see `verify_def5_slice_conditionals`):
    g_bi -> uniform in (beta, gamma);  g_flat -> cos(beta)."""
    R = slice_rotations(yaw_star, betas, gammas)          # (Nb,Ng,3,3)
    theta = geodesic_angle_to(R, R_obs)                   # (Nb,Ng)
    loglik = -(theta ** 2) / (2.0 * sigma ** 2)
    loglik -= loglik.max()                                # stabilise
    lik = np.exp(loglik)                                  # (Nb,Ng)

    dgamma = gammas[1] - gammas[0]
    cosb = np.cos(betas)                                  # Haar density in the chart

    # marginal over gamma (uniform measure dgamma for both analysts):
    m = lik.sum(axis=1) * dgamma                          # (Nb,)
    p_inv = m.copy()          # nu_bi conditioned in g_bi: the 1/cos tube cancels cos
    p_flat = cosb * m         # nu_bi conditioned in g_flat: Haar density restricted

    dbeta = betas[1] - betas[0]
    p_inv /= p_inv.sum() * dbeta
    p_flat /= p_flat.sum() * dbeta
    return p_inv, p_flat


def central_interval(betas, dens, level):
    """Central credible interval [lo,hi] of a 1-D density on the beta grid."""
    dbeta = betas[1] - betas[0]
    cdf = np.cumsum(dens) * dbeta
    cdf /= cdf[-1]
    lo = np.interp((1 - level) / 2, cdf, betas)
    hi = np.interp((1 + level) / 2, cdf, betas)
    return lo, hi


def slice_distance(R, metric):
    """Distance from each rotation in R to the slice A = {yaw = 0}, in `metric`.

    "invariant" (g_bi): the minimal rotation angle to reach A.  A is the set of
    rotations whose pointing axis b1 = R e1 lies on the closed half great circle
    of heading 0 (roll is free), so the distance is the angle from b1 to that
    half circle: to the vertical plane when b1 points forward (x >= 0), to the
    nearest pole otherwise.
    "flat" (g_flat): |yaw|, read off the ZYX chart.

    Both are invariant under left rotation about the vertical, which is how the
    gate is moved to a general heading yaw* (see `draw_gated_truths`)."""
    if metric == "invariant":
        b1 = R.apply([1.0, 0.0, 0.0])
        to_plane = np.arcsin(np.clip(np.abs(b1[:, 1]), 0.0, 1.0))
        to_pole = np.arccos(np.clip(np.abs(b1[:, 2]), 0.0, 1.0))
        return np.where(b1[:, 0] >= 0.0, to_plane, to_pole)
    if metric == "flat":
        return np.abs(R.as_euler("ZYX")[:, 0])
    raise ValueError(f"unknown metric {metric!r}")


def draw_gated_truths(metric, n_trials, delta, rng, batch=400_000):
    """Ground truths of one experiment: R* ~ nu_bi (Haar), recorded only when it
    passes the heading gate {slice_distance < delta} measured in `metric`.

    The target headings yaw* are drawn uniformly; by left invariance of Haar and
    of both distances, gating at yaw = 0 and then rotating by Rz(yaw*) is exact.
    Returns (R_true, yaw_star)."""
    kept, n_kept = [], 0
    while n_kept < n_trials:
        R = Rot.random(batch, random_state=rng)
        keep = slice_distance(R, metric) < delta
        kept.append(R[keep])
        n_kept += int(keep.sum())
    R0 = Rot.concatenate(kept)[:n_trials]
    yaw_star = rng.uniform(-np.pi, np.pi, n_trials)
    return Rot.from_euler("Z", yaw_star) * R0, yaw_star


def run_calibration(sigmas, n_trials, betas, gammas, seed=12345, level=0.90,
                    metric="invariant", delta=0.005):
    """Monte-Carlo coverage of the nominal `level` pitch credible interval for
    each analyst, as a function of the noise scale sigma, when the heading gate
    is measured in `metric` ("invariant" -> the translation invariant analyst is
    exact Bayes; "flat" -> the flat chart analyst is).  Prior and sensor are the
    same for both metrics."""
    rng = np.random.default_rng(seed)
    cov_inv = np.zeros(len(sigmas))
    cov_flat = np.zeros(len(sigmas))
    width_inv = np.zeros(len(sigmas))
    width_flat = np.zeros(len(sigmas))

    # Pre-draw all truths/noise so the result is independent of sigma ordering.
    R_true, yaw_stars = draw_gated_truths(metric, n_trials, delta, rng)
    eta = rng.standard_normal((len(sigmas), n_trials, 3))

    for si, sigma in enumerate(sigmas):
        hit_i = hit_f = 0
        wsum_i = wsum_f = 0.0
        for t in range(n_trials):
            Rt = R_true[t]
            yaw_star = yaw_stars[t]                 # the gate's heading, not Rt's
            beta_star = Rt.as_euler("ZYX")[1]
            # isotropic noisy attitude observation: R_obs = R_true * exp(eta^)
            R_obs = (Rt * Rot.from_rotvec(sigma * eta[si, t])).as_matrix()
            p_inv, p_flat = posteriors_on_slice(
                yaw_star, R_obs, sigma, betas, gammas)
            lo_i, hi_i = central_interval(betas, p_inv, level)
            lo_f, hi_f = central_interval(betas, p_flat, level)
            hit_i += (lo_i <= beta_star <= hi_i)
            hit_f += (lo_f <= beta_star <= hi_f)
            wsum_i += (hi_i - lo_i)
            wsum_f += (hi_f - lo_f)
        cov_inv[si] = hit_i / n_trials
        cov_flat[si] = hit_f / n_trials
        width_inv[si] = wsum_i / n_trials
        width_flat[si] = wsum_f / n_trials
    return cov_inv, cov_flat, width_inv, width_flat


def representative_trial(sigma, betas, gammas, seed, metric="invariant",
                         delta=0.005):
    """One trial whose posterior makes a clear figure: returns the two pitch
    posteriors, the truth, and the noisy observation."""
    rng = np.random.default_rng(seed)
    R_true, yaw_stars = draw_gated_truths(metric, 1, delta, rng)
    Rt, yaw_star = R_true[0], yaw_stars[0]
    beta_star = Rt.as_euler("ZYX")[1]
    R_obs = (Rt * Rot.from_rotvec(sigma * rng.standard_normal(3))).as_matrix()
    p_inv, p_flat = posteriors_on_slice(yaw_star, R_obs, sigma, betas, gammas)
    return beta_star, p_inv, p_flat


# --------------------------------------------------------------------------- #
#  Figure                                                                     #
# --------------------------------------------------------------------------- #
def make_figure(betas, exact, sigmas, calib_inv, calib_flat, level, out_pdf):
    """Two panels, the symmetric calibration run: (a) heading measured in the
    translation invariant metric; (b) heading measured in the flat chart metric.
    Same prior nu_bi and same attitude sensor in both.
    The panels are mirror images: in each, the analyst matching the generative
    geometry sits on the nominal line and the other one drifts off it. (The
    exact null-set posterior shapes are the SO(3) analogue of Figure 2 and are
    given in the text by eq. (so3-two-posteriors); `exact` and `betas` are kept
    for the printed/JSON headline numbers, not plotted here.)

    calib_inv / calib_flat are (cov_inv, cov_flat) coverage arrays per regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.titlesize": 9, "figure.dpi": 200,
    })
    deg = 180.0 / np.pi
    C_INV, C_FLAT = "#1f5fa6", "#c0392b"
    sig_deg = np.array(sigmas) * deg

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---- Panels (a), (b) : calibration under each generative regime ----
    def calibration_panel(ax, cov_inv, cov_flat, title, legend_loc="lower left",
                          nominal_at="upper right"):
        ax.axhline(level * 100, color="0.4", lw=1, ls=":")
        # the annotation dodges the curves, which sit differently in each panel
        va, dy = ("bottom", +0.6) if nominal_at == "upper right" else ("top", -0.6)
        ax.text(sig_deg[-1], level * 100 + dy, f"nominal {int(level*100)}%",
                ha="right", va=va, color="0.4", fontsize=7.5)
        # the panel titles name the generating metric in the same words as these
        # legend entries and as the paper's caption, so a reader matches curve to
        # regime by reading rather than by translating between two vocabularies
        ax.plot(sig_deg, cov_inv * 100, "o-", color=C_INV, lw=2, ms=4,
                label=r"translation invariant ($g_{\mathrm{bi}}$)")
        ax.plot(sig_deg, cov_flat * 100, "s--", color=C_FLAT, lw=2, ms=4,
                label=r"flat chart ($g_{\mathrm{flat}}$)")
        ax.set_xscale("log")
        ax.set_xlabel(r"sensor noise scale $\sigma$  (degrees)")
        ax.set_title(title)
        ax.legend(loc=legend_loc, frameon=False)
        ax.grid(True, which="both", alpha=0.25)

    axA.set_ylabel(f"coverage of {int(level*100)}% interval (%)")
    calibration_panel(axA, calib_inv[0], calib_inv[1],
                      r"(a) measurement of $\alpha$: translation invariant")
    # in (b) the calibrated flat curve rides the nominal line and the invariant
    # one climbs above it, so the annotation drops below the line
    calibration_panel(axB, calib_flat[0], calib_flat[1],
                      r"(b) measurement of $\alpha$: flat chart",
                      nominal_at="lower right")
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
def run_delta_check(args, betas, gammas, out_json,
                    deltas=(0.1 / 57.29577951308232, 0.005, 1.0 / 57.29577951308232)):
    """Robustness of the calibration run to the gate tolerance delta (the
    resolution of the instrument measuring alpha).  Same seeds, sigmas and
    trials as the main run; only delta (in rad) changes.  delta = 0.005 rad
    (0.29 deg) reproduces the main run exactly."""
    deg = 180.0 / np.pi
    deltas_deg = [d * deg for d in deltas]
    sigmas = list(np.array([3, 6, 12, 25, 50, 90, 150]) / deg)
    out = {"config": {"trials": args.trials, "grid": args.grid,
                      "sigmas_deg": [s * deg for s in sigmas], "level": 0.90,
                      "deltas_deg": deltas_deg},
           "runs": {}}
    print(f"[delta-check] {args.trials} trials/level, deltas "
          f"{[round(d, 3) for d in deltas_deg]} deg")
    for d_rad, d in zip(deltas, deltas_deg):
        block = {}
        for metric in ("invariant", "flat"):
            ci, cf, _, _ = run_calibration(sigmas, args.trials, betas, gammas,
                                           metric=metric, delta=d_rad)
            block[metric] = {"coverage_invariant": ci.tolist(),
                             "coverage_flat": cf.tolist()}
            print(f"  delta={d:4.2f} deg  alpha measured in {metric:9s}: "
                  "inv " + " ".join(f"{100*x:5.1f}" for x in ci) +
                  "  | flat " + " ".join(f"{100*x:5.1f}" for x in cf))
        out["runs"][f"{d:.3f}"] = block
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[json] wrote {out_json}")


def make_delta_figure(in_json, out_pdf, level=0.90):
    """Supplement figure for the delta check, read from results_delta_check.json
    (no recomputation).  Same two panels and colours as the main figure; colour
    = analysis metric, line style = delta.  The solid delta = 0.29 deg line is
    the run of the main figure, drawn thicker and underneath so that the thin
    dotted / dashed lines stay visible where the three coincide."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.titlesize": 9, "figure.dpi": 200,
    })
    with open(in_json) as f:
        data = json.load(f)
    sig = np.array(data["config"]["sigmas_deg"])
    runs = data["runs"]                            # keys: delta in deg, "%.3f"
    C_INV, C_FLAT = "#1f5fa6", "#c0392b"
    # delta key -> (line style, width, z-order); main-run delta is the solid one
    style = {}
    for k in runs:
        d = float(k)
        if abs(d - 0.286) < 0.01:
            style[k] = ("-", 2.4, 2)
        elif d < 0.286:
            style[k] = (":", 1.4, 3)
        else:
            style[k] = ("--", 1.4, 3)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    panels = [(axA, "invariant",
               r"(a) measurement of $\alpha$: translation invariant"),
              (axB, "flat", r"(b) measurement of $\alpha$: flat chart")]
    for ax, metric, title in panels:
        # nominal line: thin solid light grey, so it is not read as a delta style
        ax.axhline(level * 100, color="0.65", lw=0.9, ls="-", zorder=1)
        for k, (ls, lw, z) in style.items():
            blk = runs[k][metric]
            ax.plot(sig, np.array(blk["coverage_invariant"]) * 100, ls,
                    color=C_INV, lw=lw, zorder=z)
            ax.plot(sig, np.array(blk["coverage_flat"]) * 100, ls,
                    color=C_FLAT, lw=lw, zorder=z)
        ax.set_xscale("log")
        ax.set_xlabel(r"sensor noise scale $\sigma$  (degrees)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axA.set_ylabel(f"coverage of {int(level*100)}% interval (%)")

    # two legend groups, each in the empty lower-left of a panel:
    # colour -> analysis metric (as in the main figure), line style -> delta
    metric_handles = [
        Line2D([], [], color=C_INV, lw=2, label=r"translation invariant ($g_{\mathrm{bi}}$)"),
        Line2D([], [], color=C_FLAT, lw=2, label=r"flat chart ($g_{\mathrm{flat}}$)")]
    delta_handles = []
    for k in sorted(runs, key=float):
        ls, lw, _ = style[k]
        lab = rf"$\delta = {float(k):.2g}^\circ$"
        if ls == "-":
            lab += " (main run)"
        delta_handles.append(Line2D([], [], color="0.25", ls=ls, lw=lw, label=lab))
    axA.legend(handles=metric_handles, loc="lower left", frameon=False)
    axB.legend(handles=delta_handles, loc="lower left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[figure] wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000,
                    help="Monte-Carlo trials per noise level (default 2000).")
    ap.add_argument("--grid", type=int, default=181,
                    help="grid points per Euler angle (default 181).")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run (few trials / coarse grid).")
    ap.add_argument("--delta-check", action="store_true",
                    help="robustness run over the gate tolerance delta only; "
                         "writes results_delta_check.json and the supplement "
                         "figure, not the main figure.")
    ap.add_argument("--delta-figure", action="store_true",
                    help="redraw the supplement delta figure from "
                         "results_delta_check.json, without recomputing.")
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

    delta_json = os.path.join(here, "results_delta_check.json")
    delta_pdf = os.path.join(repo_root, "so3_delta_robustness.pdf")
    if args.delta_check:
        run_delta_check(args, betas, gammas, delta_json)
        make_delta_figure(delta_json, delta_pdf)
        return
    if args.delta_figure:
        make_delta_figure(delta_json, delta_pdf)
        return

    print("=" * 70)
    print("SO(3) state-estimation experiment -- Borel-Kolmogorov / MaxEnt paper")
    print("=" * 70)

    # 0. sanity
    rel = verify_haar_cosine()
    print(f"\n[0] Haar pitch-density vs cos(pitch): max rel. error = {rel:.4f}"
          f"  ({'OK' if rel < 0.05 else 'CHECK'})")
    ks = verify_def5_slice_conditionals()
    for metric, (d_unif, d_cos) in ks.items():
        want_small = d_unif if metric == "invariant" else d_cos
        print(f"    Def.5 tube in {metric:9s}: KS to uniform {d_unif:.4f}, "
              f"to cos/2 {d_cos:.4f}  ({'OK' if want_small < 0.02 else 'CHECK'})")

    # A. exact null-set posteriors
    exact = exact_null_set_summary()
    print("\n[A] Exact null-set posteriors of pitch (heading observed exactly)")
    print(f"    central 50% interval : invariant {exact['ci50']['invariant_deg'][0]:+.1f}"
          f" .. {exact['ci50']['invariant_deg'][1]:+.1f} deg"
          f"   |  flat chart {exact['ci50']['flat_deg'][0]:+.1f}"
          f" .. {exact['ci50']['flat_deg'][1]:+.1f} deg")
    print(f"    P(|pitch| < 30 deg)  : invariant {exact['band_pm30deg']['invariant_prob']:.3f}"
          f"   |  flat chart {exact['band_pm30deg']['flat_prob']:.3f}")
    print(f"    flat chart's nominal 90% interval, heading gated in g_bi, covers "
          f"{100*exact['flat_ci90_coverage_under_invariant_gate']:.1f}%")
    print(f"    invariant's nominal 90% interval, heading gated in g_flat, covers "
          f"{100*exact['invariant_ci90_coverage_under_flat_gate']:.1f}%")

    # B. calibration under BOTH heading measurements (the symmetric run):
    #    - "invariant": heading gate measured in g_bi   -> invariant is exact Bayes;
    #    - "flat":      heading gate measured in g_flat -> flat chart is exact Bayes.
    # Prior (nu_bi), attitude sensor and analysts are unchanged; only the metric
    # in which the heading is measured changes.
    deg = 180.0 / np.pi
    delta = 0.005                               # gate half-width (rad), << sigma
    sigmas = list(np.array([3, 6, 12, 25, 50, 90, 150]) / deg)
    print(f"\n[B] Calibration: {args.trials} trials/level, "
          f"{args.grid}x{args.grid} grid, gate delta = {delta*deg:.2f} deg, sigma in "
          f"{[round(s*deg) for s in sigmas]} deg, both heading metrics ...")

    calib = {}
    for regime, calibrated in [("invariant", "translation invariant"),
                               ("flat", "flat chart")]:
        ci, cf, wi, wf = run_calibration(
            sigmas, args.trials, betas, gammas, metric=regime, delta=delta)
        calib[regime] = (ci, cf, wi, wf)
        print(f"  heading metric={regime:9s} (exact Bayes = {calibrated}):")
        for s, i, f in zip(sigmas, ci, cf):
            print(f"    sigma={s*deg:5.0f} deg :  invariant {100*i:5.1f}%   "
                  f"flat chart {100*f:5.1f}%   (nominal 90%)")

    # save everything
    def cal_block(regime):
        ci, cf, wi, wf = calib[regime]
        return {
            "sigma_deg": [s * deg for s in sigmas],
            "coverage_invariant": ci.tolist(),
            "coverage_flat": cf.tolist(),
            "mean_width_invariant_deg": (np.array(wi) * deg).tolist(),
            "mean_width_flat_deg": (np.array(wf) * deg).tolist(),
        }

    results = {
        "config": {"trials": args.trials, "grid": args.grid,
                   "sigmas_deg": [s * deg for s in sigmas], "level": 0.90,
                   "prior": "nu_bi (Haar) in both experiments",
                   "gate_delta_deg": delta * deg},
        "haar_cos_rel_error": rel,
        "def5_slice_ks": {m: {"to_uniform": u, "to_cos": c}
                          for m, (u, c) in ks.items()},
        "exact_null_set": exact,
        # the symmetric run, one block per metric of the heading measurement; in
        # each, the analyst of the same name is the exact Bayes posterior:
        "calibration_invariant_heading": cal_block("invariant"),
        "calibration_flat_heading": cal_block("flat"),
    }
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[json] wrote {out_json}")

    # figure: (a) invariant-generated calibration, (b) flat-chart-generated
    make_figure(betas, exact, sigmas,
                calib["invariant"][:2], calib["flat"][:2], 0.90, out_pdf)
    print("\nDone.")


if __name__ == "__main__":
    main()
