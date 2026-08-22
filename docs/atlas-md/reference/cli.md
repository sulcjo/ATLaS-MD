# Command-line programs

| Program | Exact help | Main use |
| --- | --- | --- |
| `gareus` | `gareus -h`, `gareus -hh [topic]` | Main workflow |
| `python -m gareus` | `python -m gareus -h` | Module equivalent |
| `python GENPEPT.py` | `python GENPEPT.py --help` | Seed conformers |
| `gareus-analyze` | `gareus-analyze --help` | MBAR package analysis |
| `gareus-suggest-cvs` | `gareus-suggest-cvs --help` | CV/window suggestions |
| `gareus-consolidate-traj` | `gareus-consolidate-traj --help` | Adaptive trajectory consolidation |
| `gareus-energy-decompose` | `gareus-energy-decompose -hh` | Energy decomposition |
| `gareus-test-run` | `gareus-test-run --help` | Dependency/tiny-workflow test |

`gareus -h` is operational help; `gareus -hh` is live encyclopedia and full option list. Use command help from installed checkout when exact option availability matters.

Core controls: `--seq`, `--out`, `--config`, `--profile`, `--run-mode`, `--window-mode`, `--cv1`, `--cv2`, `--seed`, `--resume`, `--platform`, `--device-index`, `--progress-mode`, and `--tui-mode`.
