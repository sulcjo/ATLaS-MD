# GENPEPT 7-run necessity benchmark suite

Sequence: `GYDPETGTWG`.

The suite is designed so run 01 is the largest/fullest reference run. Runs 02-06 are controls/ablations that test what is actually necessary. Run 07 is a reasonable integrated all-methods setting for practical use.

| Run | YAML | Purpose |
| --- | --- | --- |
| 01 | `genpept_run01_full_heavy_max.yaml` | Full heavy reference: broad generation + BH + NMA. |
| 02 | `genpept_run02_raw_generation_control.yaml` | Same raw generation and candidate count, but no BH/NMA. Tests pure generation. |
| 03 | `genpept_run03_bh_only_control.yaml` | Keeps BH, disables NMA. Tests BH marginal value. |
| 04 | `genpept_run04_nma_only_control.yaml` | Keeps NMA, disables BH. Candidate count is smaller because NMA expands all implicit minima when BH is absent. |
| 05 | `genpept_run05_minimal_full_necessary.yaml` | Cheaper full pipeline. Tests whether reduced BH/NMA settings retain most of the benefit. |
| 06 | `genpept_run06_pca_frontier_control.yaml` | PCA-frontier only control: BH/NMA disabled. Tests cheap archive-aware proposal/minimization. |
| 07 | `genpept_run07_reasonable_all_methods.yaml` | Practical all-methods run: broad generation + BH + NMA + PCA frontier, tuned to be reasonable rather than maximal. |

Suggested execution:

```bash
python GENPEPT.py --config genpept_run01_full_heavy_max.yaml
python GENPEPT.py --config genpept_run02_raw_generation_control.yaml
python GENPEPT.py --config genpept_run03_bh_only_control.yaml
python GENPEPT.py --config genpept_run04_nma_only_control.yaml
python GENPEPT.py --config genpept_run05_minimal_full_necessary.yaml
python GENPEPT.py --config genpept_run06_pca_frontier_control.yaml
python GENPEPT.py --config genpept_run07_reasonable_all_methods.yaml
```

Optional batch runner:

```bash
chmod +x run_all_genpept_suite.sh
./run_all_genpept_suite.sh
```

Optional comparison after outputs exist:

```bash
chmod +x compare_all_genpept_suite.sh
./compare_all_genpept_suite.sh
```

Interpretation:

- If run 02 approaches run 01, raw generation is sufficient.
- If run 03 approaches run 01, BH is doing most of the useful expansion.
- If run 04 approaches run 01, NMA can replace BH as the main expansion stage.
- If run 05 approaches run 01, the cheaper full settings are probably enough.
- If run 06 adds strong post_pca_explore gains, PCA frontier generation is a useful cheap exploration mode.
- If only run 01 is clearly best, the full heavy BH->NMA frontier-expansion pipeline is justified.
- If run 07 approaches run 01 while costing much less, use run 07 as the practical all-methods production profile.


## Run 06: adaptive PCA frontier control

`genpept_run06_pca_frontier_control.yaml` adds a cheap archive-aware exploration mode while keeping BH and NMA disabled. It tests whether the adaptive PCA frontier loop alone can recover useful new basins after initial implicit minimization.

Key settings:

```yaml
basin_hop: false
nma_expand: false
explore_loop: true
explore_rounds: 2
explore_proposals: 75000
explore_keep: 350
explore_bins: 45
explore_target_max_count: 0
explore_frontier_only: true
explore_allow_outside: true
explore_max_kept_per_bin: 4
```

Interpretation: compare `post_pca_explore` against `initial_implicit`. If it adds new coarse bins, archive score, and final PCA survivors, PCA-frontier generation is worth keeping as a cheap complement to BH/NMA.


## Run 07: reasonable all-methods combined run

`genpept_run07_reasonable_all_methods.yaml` combines all major exploration modes without using the maximal settings from run 01. It is intended as the practical integrated profile:

```yaml
basin_hop: true
bh_parent_seeds: 120
bh_steps: 3
nma_expand: true
nma_modes: 2
nma_amplitudes: "0.75,1.35"
explore_loop: true
explore_rounds: 1
explore_proposals: 50000
explore_keep: 250
explore_bh: false
```

Interpretation: compare run 07 against run 01 and run 05. If run 07 recovers most of run 01's archive score, effective basin count, final survivor diversity, and stage contribution while running much faster, it is the better default practical profile.
