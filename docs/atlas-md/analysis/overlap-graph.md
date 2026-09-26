# Overlap graph

`gareus-analyze` draws the state-to-state overlap as a graph over the state layout, not as a K × K heatmap in state-index order. On a 2D λ-ladder grid, index order puts CV neighbours and rung partners far apart, so a heatmap cannot show where the gaps are.

## What is drawn

`overlap_matrix.png`:

- **Nodes:** each state (umbrella window) sits at its CV1 restraint centre, CV2 restraint centre and GaMD λ rung, labelled with its state index.
- **Edges:** each state is joined to its same-rung spatial neighbours (the layout neighbour rule used by the dashboard) and, at each centre, to the next rung up.
- **Gap edges:** the neighbour rule only joins nearby states, so a CV gap between two groups on one rung would leave no edge at all. Each rung's disconnected groups are joined by their closest pair (`kind = gap`). The break is then drawn with its measured, usually near-zero, overlap instead of disappearing.
- **Edge colour:** the pairwise symmetric MBAR state overlap $\sqrt{O_{ij} O_{ji}}$. It is computed on the pair's own samples with the global $f_k$ held fixed, and runs 0–0.5; two identical states give 0.5. Edges below `--min-ladder-state-overlap` (default 0.15) are drawn **red dashed**. An edge that could not be measured (a state without samples) is drawn **grey dotted** and is never counted as weak.
- **Density sheets:** the per-sample integrand of that overlap, $\sqrt{N_i N_j}\, W_{ni} W_{nj}$, is binned on the CV grid. Each sheet shows *where* in CV space a pair shares configurations, and integrates back to the edge's overlap. Same-rung pairs are drawn on their rung's λ layer; rung pairs on the layer halfway between the two rungs, so a λ gap shows as a pale or empty mid-layer.
- **Views:** two viewing angles side by side.

A run without a secondary CV gets a 2D (CV1 × λ) version with the density drawn as strips. A run without a ladder has a single λ layer.

## Why pairwise MBAR, not CV histograms

A CV histogram cannot see a rung gap: two rungs at one centre overlap ~1 in CV space by construction. The MBAR state overlap is measured in energy space, so it grades CV and λ edges on one scale. It is pairwise, not read off the full-union overlap matrix, because the full matrix dilutes each edge by the number of other states in the same region (chignolin_7: full-union median 0.089 vs pairwise 0.258 on the same 48 rung edges).

## Files

| File | Content |
| --- | --- |
| `overlap_matrix.png` | 3D overlap graph (2D for CV1-only runs) |
| `overlap_density_layers.png` | Each density sheet as a flat panel with state indices |
| `overlap_pairs_mbar.csv` | `i, j, kind (cv/rung/gap), layer_lambda, overlap, below_threshold` |
| `overlap_graph_3d.html` | Rotatable version with hover text (written only with a secondary CV and plotly installed) |
| `overlap_matrix_cv1_hist.png` | The previous heatmap: CV1-marginal histogram overlap in state-index order |

!!! note "Renamed output"
    Before 2026-09-26 `overlap_matrix.png` held the CV1-marginal heatmap. That figure is now `overlap_matrix_cv1_hist.png`. `overlap_matrix.csv` and the `neighbor_overlap` health numbers are unchanged.

`pmf_summary.json["overlap_graph"]` records the pair count, rung-pair and gap-pair counts, minimum and median overlap, the number of edges below threshold, and the number of edges that could not be measured (a state without samples).

## Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--no-overlap-graph` | off | Skip the graph, and delete any graph files left from an earlier analysis in the output directory |
| `--overlap-density-bins N` | 40 | Bins per CV axis for the density sheets |
| `--min-ladder-state-overlap X` | 0.15 | Threshold for the red dashed edges (shared with the ladder health rows) |

The graph is presentation only: it never changes the health verdict. The 0.15 threshold was calibrated on full-union values and has not been re-measured on the pairwise scale.
