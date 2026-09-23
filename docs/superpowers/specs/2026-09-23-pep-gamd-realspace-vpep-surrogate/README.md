# Pep-GaMD peptide-energy rework (real-space V_pep surrogate)

Everything about replacing the water-only auxiliary PME in the Pep-GaMD boost with a cheap
real-space surrogate for the peptide energy, collected in one place. Started 2026-09-23 on
ATLaS-MD / gareus v0.8.2.

| Path | Contents |
|---|---|
| `spec.md` | The design spec: problem, validity argument, force-group architecture (group 1 = measurement channel), RS energy and force algebra, surrogate candidates, offline and boosted-pilot gates, recording/reweighting semantics, implementation plan, test matrix, performance gate, risks, review decisions. Section 14 has the whole-codebase board findings the RS kernel must not inherit. |
| `bigboard-review/` | Seven-board adversarial review of `main` @ dfeb8f9 (2026-09-23). `README.md` has the consolidated verdicts, ranked findings with verification status, refuted claims and the method verdict. Each `bN_*/` folder has that board's `final_report.md`, `board_summary.json` and full `transcript.md`. |
| `benchmarks/` | Benchmark harnesses behind the cost numbers in the spec, run on aurum2 against the real chignolin_8 system: `c8_integ_bench.py` / `.sh` (integrator arms L, L2, P, B), `c8_integ_bench2.py` / `c8_biasgroup.sh` (real CV forces, bias groups split vs merged), `c8_integ_bench3.py` / `c8_mps59.sh` (stage-5 boost live, MPS at 59/48/16 contexts per GPU vs no MPS). |
| `benchmarks/results/` | Raw logs: jobs 2577764 (integrator arms), 2579032 (bias groups with real CVs), 2580889 (MPS at 59 contexts per GPU, stage 5, FSF < 1 confirmed). |

Related documents outside this folder:
- `ATLaS-MD_PepGaMD_integrator_v0.8.2_2026-09-22.docx` at the repo root: the full integrator
  report (method, cost anatomy, optimisations, chignolin_8 findings).
- `docs/atlas-md/developer/gpu-throughput-benchmark-todo.md`: the chignolin_9 checklist and the
  T1-T8 measurements.
