# Analysis workflow

Run `gareus-analyze --help` to inspect installed analysis interface. Legacy/script workflows may use `analyze_gareus_mbar.py`; keep exact command, run directory, and output directory in analysis provenance.

Typical sequence:

1. Confirm production samples and canonical window table exist.
2. Reconstruct/use bias matrix with correct temperature and all active CV restraints.
3. Run MBAR/PMF analysis.
4. Read health report, coverage, overlap, weighted ESS, GaMD diagnostics.
5. Only then compare PMFs or quote free-energy differences.

Analysis has two layers. MBAR removes every configured umbrella restraint through reduced-bias matrix `u_nk`. GaMD runs then apply boost reweighting. Preserve estimator and input-population labels: a PMF is not defined solely by its plotted CV. Read [reweighting and estimators](reweighting.md) before selecting output.

Sample-query APIs can reconstruct arrays without a prewritten NPZ:

```python
from gareus.query import export_analysis_arrays_npz
export_analysis_arrays_npz("run_cln025", beta=1 / (8.314e-3 * 300))
```

See [PMF validity](pmf-validity.md) before interpretation.
