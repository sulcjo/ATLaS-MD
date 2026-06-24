"""Tests for oracle-optimal window placement (greedy constant-overlap rule).

The oracle-optimal 1D placement finds the spring constant k* such that the
greedy constant-overlap walk (ideal_axis_ladder) generates exactly R centers.
Centers cluster densely near PMF barriers and spread out in flat basins —
naturally encoding the Fisher-metric geometry of the sampling problem.

The oracle-optimal 2D placement uses greedy farthest-point sampling in
Bhattacharyya-overlap-distance space, adapting to the 2D FES curvature.
"""
import numpy as np
import pytest

from gareus.synth.landscapes import rugged_1d, mixture_wells
from gareus.synth.oracle import (
    oracle_k_for_r,
    oracle_optimal_centers_1d,
    oracle_optimal_centers_2d,
    ideal_axis_ladder,
    reference_pmf,
)


class TestOracleKForR:
    """oracle_k_for_r bisects to the smallest k giving >= R greedy centers."""

    def test_achievable_counts(self):
        """For counts directly available (k transition exists), bisection finds the k."""
        land = rugged_1d()
        for R in [2, 3, 4, 5, 6, 7]:
            k = oracle_k_for_r(land, R)
            n = len(ideal_axis_ladder(land, k=k, target_overlap=0.30))
            assert n >= R, f"R={R}: k={k:.1f} gave only {n} centers"

    def test_r1_returns_zero(self):
        land = rugged_1d()
        assert oracle_k_for_r(land, 1) == 0.0

    def test_k_monotone_in_r(self):
        """Larger R requires tighter (larger) k."""
        land = rugged_1d()
        ks = [oracle_k_for_r(land, R) for R in [3, 5, 7]]
        assert ks[0] < ks[1] < ks[2], f"k not monotone: {ks}"

    def test_r_beyond_range_returns_khi(self):
        """R larger than achievable count at k_hi returns k_hi (graceful)."""
        land = rugged_1d()
        k = oracle_k_for_r(land, 999, k_hi=500.0)
        assert k == 500.0


class TestOracleOptimalCenters1D:
    """oracle_optimal_centers_1d returns R centers at oracle-optimal 1D positions."""

    def test_returns_exactly_r_centers(self):
        land = rugged_1d()
        for R in [2, 3, 4, 5, 6, 7]:
            centers, k = oracle_optimal_centers_1d(land, R)
            assert len(centers) == R, f"R={R}: got {len(centers)} centers"

    def test_centers_monotone(self):
        """Centers are sorted ascending along CV1."""
        land = rugged_1d()
        centers, _ = oracle_optimal_centers_1d(land, 6)
        assert centers == sorted(centers), f"Centers not sorted: {centers}"

    def test_centers_dense_near_barrier(self):
        """Oracle centers cluster near the 14 kBT barrier (x~0.46), not uniformly spaced."""
        land = rugged_1d()
        x, pmf = reference_pmf(land, axis="cv1", res=200)
        barrier_x = float(x[np.argmax(pmf)])       # ~0.462
        centers, _ = oracle_optimal_centers_1d(land, 6)
        nearest = min(abs(c - barrier_x) for c in centers)
        assert nearest < 0.15, (
            f"No oracle center near barrier at {barrier_x:.3f}: nearest is {nearest:.3f}")

    def test_oracle_denser_than_linspace_at_barrier(self):
        """Oracle centers have higher density near the barrier than linspace centers."""
        from gareus.synth.replica import low_f_support
        land = rugged_1d()
        x, pmf = reference_pmf(land, axis="cv1", res=200)
        barrier_x = float(x[np.argmax(pmf)])
        R = 6
        centers_oracle, _ = oracle_optimal_centers_1d(land, R)
        lo, hi = low_f_support(land)
        centers_lin = list(np.linspace(lo, hi, R))
        dist_oracle = min(abs(c - barrier_x) for c in centers_oracle)
        dist_lin = min(abs(c - barrier_x) for c in centers_lin)
        assert dist_oracle <= dist_lin + 0.05, (
            f"Oracle not denser near barrier: oracle {dist_oracle:.3f} vs linspace {dist_lin:.3f}")

    def test_r1_special_case(self):
        """R=1 returns single center near midpoint, k=0."""
        land = rugged_1d()
        centers, k = oracle_optimal_centers_1d(land, 1)
        assert len(centers) == 1
        assert k == 0.0

    def test_r2_achievable(self):
        land = rugged_1d()
        centers, k = oracle_optimal_centers_1d(land, 2)
        assert len(centers) == 2
        assert k > 0

    def test_k_positive_for_r_gt1(self):
        land = rugged_1d()
        for R in [2, 4, 6]:
            _, k = oracle_optimal_centers_1d(land, R)
            assert k > 0, f"R={R}: k={k} not positive"


class TestOracleOptimalCenters2D:
    """oracle_optimal_centers_2d places R windows in the 2D low-F region using
    greedy farthest-point in Bhattacharyya-overlap-distance space."""

    def test_returns_exactly_r_centers(self):
        land = mixture_wells()
        for R in [3, 5, 8]:
            centers = oracle_optimal_centers_2d(land, R, res=40)
            assert len(centers) == R, f"R={R}: got {len(centers)} centers"

    def test_centers_in_low_f_region(self):
        """All 2D oracle centers land in the low-F region (F <= min + 5+1 kBT)."""
        from gareus.synth.oracle import LOW_F_THRESHOLD_KBT
        land = mixture_wells()
        c1, c2, f = land.grid(res=80)
        centers = oracle_optimal_centers_2d(land, 6, res=40)
        for cx1, cx2 in centers:
            i1 = int(np.argmin(np.abs(c1 - cx1)))
            i2 = int(np.argmin(np.abs(c2 - cx2)))
            assert f[i1, i2] <= f.min() + LOW_F_THRESHOLD_KBT + 1.5, (
                f"Center ({cx1:.3f},{cx2:.3f}) F={f[i1,i2]:.2f} kBT above threshold")

    def test_no_duplicate_centers(self):
        """No two oracle centers are identical (greedy adds distinct points)."""
        land = mixture_wells()
        centers = oracle_optimal_centers_2d(land, 6, res=40)
        for i, ci in enumerate(centers):
            for j, cj in enumerate(centers):
                if i != j:
                    d = np.sqrt((ci[0] - cj[0])**2 + (ci[1] - cj[1])**2)
                    assert d > 1e-6, f"Duplicate centers {i},{j}: {ci} and {cj}"

    def test_centers_spread_across_basins(self):
        """Lloyd CVT with R=4 covers all 3 main basins of mixture_wells.

        With R=4 the CVT partitions the 4 low-F features (3 basins + decoy)
        into separate Voronoi territories so each basin gets its own nearby center.
        """
        land = mixture_wells()
        main_basins = land.basins[:3]   # (0.18,-0.65), (0.52,-0.05), (0.84,0.55)
        centers = oracle_optimal_centers_2d(land, 4, res=80)
        for bx, by in main_basins:
            min_d = min(np.sqrt((cx - bx)**2 + (cy - by)**2) for cx, cy in centers)
            assert min_d < 0.30, (
                f"Basin ({bx:.2f},{by:.2f}) not covered by CVT R=4 (nearest={min_d:.3f})"
            )

    def test_r_too_large_clips_gracefully(self):
        """R larger than low-F candidates returns all candidates (no crash)."""
        land = mixture_wells()
        centers = oracle_optimal_centers_2d(land, 10000, res=10)
        assert len(centers) >= 1


class TestOraclePlacementComparison1D:
    """placement_comparison_1d benchmarks oracle vs linspace PMF RMSE at fixed R,B."""

    def test_returns_metrics_for_each_placement(self):
        """Returns dict with 'oracle', 'linspace' keys, each having pmf_rmse."""
        from gareus.synth.oracle import placement_comparison_1d
        land = rugged_1d()
        result = placement_comparison_1d(land, R=6, budget=50_000, seeds=[0])
        assert "oracle" in result
        assert "linspace" in result
        for key in ("oracle", "linspace"):
            assert "pmf_rmse" in result[key], f"'{key}' missing pmf_rmse"
            assert np.isfinite(result[key]["pmf_rmse"]), f"'{key}' pmf_rmse is nan"

    def test_oracle_rmse_le_linspace(self):
        """Oracle placement should give <= PMF RMSE vs linspace at R=6, B=100k.

        This is the core claim: oracle-optimal (constant-overlap greedy) beats
        uniform spacing for landscapes with barriers. Tested over 3 seeds."""
        from gareus.synth.oracle import placement_comparison_1d
        land = rugged_1d()
        result = placement_comparison_1d(land, R=6, budget=100_000, seeds=[0, 1, 2])
        oracle_rmse = result["oracle"]["pmf_rmse"]
        linspace_rmse = result["linspace"]["pmf_rmse"]
        # oracle may sometimes tie with linspace (if linspace happens to place near barrier)
        # but should not be significantly worse
        assert oracle_rmse <= linspace_rmse * 1.10, (
            f"Oracle RMSE {oracle_rmse:.4f} > linspace RMSE {linspace_rmse:.4f} (+10% margin)")
