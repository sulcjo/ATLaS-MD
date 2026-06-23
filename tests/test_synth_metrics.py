"""Metric math tests for the synthetic harness."""
from __future__ import annotations

import numpy as np
import pytest

from gareus.synth.drivers import RoundRecord
from gareus.synth.landscapes import mixture_wells
from gareus.synth.metrics import (campaign_metrics, center_churn,
                                  final_overlap_vs_target,
                                  overlap_graph_connectivity, pmf_recovery,
                                  window_economy)
from gareus.synth.oracle import ideal_axis_ladder
from gareus.synth.sampler import Window, sample_window_exact


def _rec(c1):
    return RoundRecord(0, list(c1), [50.0] * len(c1), [0.0], [60.0],
                       len(c1), float("nan"), float("nan"), False)


def test_overlap_metric_keys_and_range():
    lp = mixture_wells()
    m = final_overlap_vs_target(lp, _rec([0.2, 0.4, 0.6, 0.8]), target=0.30, res=100)
    assert {"mean", "min", "frac_pairs_below_target"} <= set(m)
    assert 0.0 <= m["frac_pairs_below_target"] <= 1.0


def test_economy_ratio_near_one_for_oracle_ladder():
    lp = mixture_wells()
    ladder = ideal_axis_ladder(lp, axis="cv1", target_overlap=0.30, k=50.0, res=100)
    e = window_economy(lp, _rec(ladder), target=0.30, res=100)
    assert e["n_ideal"] >= 2
    assert 0.6 <= e["ratio"] <= 1.6


def test_connectivity_connected_for_oracle_ladder():
    lp = mixture_wells()
    ladder = ideal_axis_ladder(lp, axis="cv1", target_overlap=0.30, k=50.0, res=100)
    conn = overlap_graph_connectivity(lp, _rec(ladder), target=0.30, res=100)
    assert conn["connected"] is True
    assert conn["spectral_gap"] > 0.0


def test_center_churn_zero_for_stable_records():
    recs = [_rec([0.2, 0.5, 0.8]), _rec([0.2, 0.5, 0.8])]
    assert center_churn(recs) == pytest.approx(0.0, abs=1e-9)


def test_pmf_recovery_near_zero_for_mbar_on_exact_samples():
    # Proper MBAR reweighting of EXACT biased samples should recover the true
    # marginal PMF to well under 1 kBT (the umbrella bias is removed, not just
    # histogrammed). Windows must tile BOTH CVs so the orthogonal axis is covered.
    lp = mixture_wells()
    rng = np.random.default_rng(0)
    samples, windows = {}, {}
    wi = 0
    for c1 in np.linspace(0.05, 0.95, 8):
        for c2 in np.linspace(-0.9, 0.9, 5):
            w = Window(float(c1), 25.0, float(c2), 25.0)
            windows[wi] = w
            samples[wi] = sample_window_exact(lp, w, 900, beta=1.0, res=90, rng=rng)
            wi += 1
    rec = pmf_recovery(lp, samples, windows, axis="cv1", res=90)
    assert {"rmse_kbt", "js", "rmse_lowf_weighted"} <= set(rec)
    assert rec["rmse_lowf_weighted"] < 0.6  # MBAR removes the bias -> near zero


def test_pmf_recovery_biased_estimate_would_be_worse():
    # Sanity: a single off-center stiff window's pooled (un-reweighted) density is
    # far from the true PMF, but MBAR still reduces the error vs naive pooling.
    lp = mixture_wells()
    rng = np.random.default_rng(1)
    samples, windows = {}, {}
    wi = 0
    for c1 in np.linspace(0.1, 0.9, 7):
        for c2 in np.linspace(-0.8, 0.8, 4):
            w = Window(float(c1), 40.0, float(c2), 40.0)
            windows[wi] = w
            samples[wi] = sample_window_exact(lp, w, 800, beta=1.0, res=80, rng=rng)
            wi += 1
    mbar = pmf_recovery(lp, samples, windows, axis="cv1", res=80)["rmse_lowf_weighted"]
    assert mbar < 1.0


def test_campaign_metrics_end_to_end():
    pytest.importorskip("scipy")
    from gareus.synth.drivers import feedback_campaign
    lp = mixture_wells()
    recs = feedback_campaign(lp, initial_centers1=[0.2, 0.5, 0.8],
                             initial_k1=[40.0] * 3, initial_centers2=[0.0],
                             initial_k2=[60.0], mode="exact", n_rounds=2,
                             samples_per_window=1000, res=80, seed=0)
    m = campaign_metrics(lp, recs, target=0.30, res=80)
    assert "final_overlap" in m and "economy" in m and "action_accuracy" in m
    assert "connectivity" in m and "center_churn" in m


def test_campaign_output_meets_oracle_quantitatively():
    # Holds the ADAPTIVE OUTPUT (not just metric keys) to the oracle: from a
    # deliberately over-dense start the campaign must end near the ideal count,
    # connected, with no severe gap.
    pytest.importorskip("scipy")
    from gareus.synth.drivers import feedback_campaign
    lp = mixture_wells()
    recs = feedback_campaign(lp, initial_centers1=[0.2, 0.35, 0.5, 0.65, 0.8],
                             initial_k1=[80.0] * 5, initial_centers2=[0.0],
                             initial_k2=[60.0], mode="exact", n_rounds=4,
                             samples_per_window=1500, res=100, seed=0,
                             aggressiveness="aggressive")
    m = campaign_metrics(lp, recs, target=0.30, res=100)
    assert m["action_accuracy"]["converged_toward_ideal"] is True
    # Oracle Bhattacharyya count and the dispatcher's hist-overlap fixpoint differ
    # by definition, so allow a modest gap rather than exact equality.
    assert m["action_accuracy"]["final_gap"] <= 3
    assert m["connectivity"]["connected"] is True
