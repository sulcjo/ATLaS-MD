# Troubleshooting

| Symptom | Check | Response |
| --- | --- | --- |
| Missing GaMD dependency | `gareus-test-run --check-deps --skip-if-missing` | Use CMD mode or install/configure external dependency |
| Resume fails | `segments.json`, checkpoint files, effective config | Resume same output dir; do not mix artifacts |
| Empty windows | final window table and sample chunks | Add/reposition bridges; collect new data |
| Low ESS | weighted diagnostics, all bias dimensions | Improve overlap/support, not raw frame count alone |
| PMF health `FAIL` | health report | Do not report PMF; repair coverage/reweighting |
| tICA state unreadable | adaptive metadata/state path | Refit or restore valid state; do not force CV2 switch |

For parser detail, use `gareus -hh`. For scientific diagnostics, read [PMF validity](../analysis/pmf-validity.md).
