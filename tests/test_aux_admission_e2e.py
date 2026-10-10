"""CPU end-to-end: adaptive aux-CV admission through the REAL adaptive-production epoch loop.

Real MD (GA dipeptide, OpenMM CPU, --run-mode cmd) through ``gareus.cli.main``: epochs 0-2 + final.
Only ``_discover`` is fixed (see conftest ``_fixed_discovery``: GA has no contact/H-bond pair, so the
U13 statistics cannot run on it); frame table, validation, alignment, freeze, backfill, applier,
window CSV, phase args, pull ramp, runtime aux force, sample writer and union pooling are the real code.
"""
import json

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _workers(states):
    return [s for s in states if (s.metadata or {}).get("aux")]


def _assert_integrator_kind(run, pep_gamd):
    """The campaign really ran Pep-GaMD (or plain cMD): read the recorded method settings of every phase."""
    manifests = sorted(run.out.rglob("run_manifest.json"))
    assert manifests
    for path in manifests:
        ms = json.loads(path.read_text()).get("method_settings") or {}
        if "gamd_boost_type" not in ms:
            continue
        if pep_gamd:
            assert ms["gamd_boost_type"] == "pep-gamd-lower-dual", path
            assert ms["pep_gamd_envelope_path"], path
        else:
            assert not str(ms["gamd_boost_type"]).startswith("pep-gamd"), path
            assert ms["pep_gamd_envelope_path"] is None, path
    if pep_gamd:
        assert list(run.out.rglob("shared_gamd_setup_globals.json")), "no Pep-GaMD envelope was calibrated"
        assert any(json.loads(p.read_text()).get("method_settings", {}).get("gamd_boost_type")
                   == "pep-gamd-lower-dual" for p in manifests), "no manifest records the Pep-GaMD boost type"


@pytest.mark.parametrize("pep_gamd", [False, True], ids=["cmd", "pep-gamd"])
def test_admission_through_epoch_loop_and_resume(tmp_path, small_adaptive_campaign, pep_gamd):
    run = small_adaptive_campaign(tmp_path, aux=True, pep_gamd=pep_gamd)
    ad = run.adaptive
    _assert_integrator_kind(run, pep_gamd)
    assert run.discover_calls == [1]                                      # admitted once, never re-discovered
    adm = json.loads((ad / "aux_admission.json").read_text())
    assert adm["epoch"] == 1 and len(adm["workers"]) == 1
    reg = run.registry()
    workers = _workers(reg.active_states())
    assert len(workers) == 1 and workers[0].gamd_lambda == 0.0
    w = workers[0]
    assert w.metadata["aux"]["burnin_phase_epoch"] == 2

    # epoch_002 ran with the worker: its window CSV has aux columns and every sample (all states) carries z
    header = (ad / "windows_epoch_002.csv").read_text().splitlines()[0]
    assert "aux_center" in header
    assert run.phase_has_column("epoch_002", "aux_z_00")
    s2 = run.samples("epoch_002")
    assert len(s2) > 0 and np.isfinite(s2["aux_z_00"].to_numpy(dtype=float)).all()
    # F09: the aux phase's gibbs-walk ledger records the log-space proposal algorithm with finite logs
    from gareus.query import load_exchanges
    ev = load_exchanges(run.phase_dir("epoch_002"))
    kinds = np.asarray(ev["kind"]).astype(str)
    moves = kinds == "swap"
    assert moves.any() and set(np.asarray(ev["proposal_algorithm"])[moves].tolist()) == {"gibbs_softmax_log_v2"}
    assert np.isfinite(np.asarray(ev["log_q_reverse"], dtype=float)[moves]).all()
    assert np.isfinite(np.asarray(ev["log_p_accept"], dtype=float)[moves]).all()

    # backfill for every pre-admission phase; recorded z matches z_from_positions on the post-admission phase
    from gareus.adaptive.aux_backfill import BACKFILL_FILENAME, check_backfill_against_recorded
    for label in ("epoch_000", "epoch_001"):
        assert (run.phase_dir(label) / BACKFILL_FILENAME).exists(), label
    chk = check_backfill_against_recorded(run.phase_dir("epoch_002"), run.model(), adaptive_dir=ad)
    print(f"recorded-vs-frame aux z: {chk}")
    assert chk["n_compared"] > 0 and np.isfinite(chk["max_abs_dev"])
    # the driver ran the kT-judged spot check once, on the first post-admission phase (never blocks)
    spot = json.loads((ad / "aux_admission.json").read_text())["spot_check"]
    print(f"driver spot check: {spot}")
    assert spot["epoch"] == 2 and spot["n_compared"] > 0
    assert np.isfinite(spot["max_abs_dev"]) and np.isfinite(spot["max_energy_err_kt"]) and len(spot["per_worker"]) == 1
    rep2 = json.loads((ad / "epoch_002" / "aux_discovery_report.json").read_text())
    assert rep2["spot_check"][-1]["max_energy_err_kt"] == spot["max_energy_err_kt"]

    # union pools every phase and includes the worker (manual build and the driver's own end-of-campaign union)
    from gareus.adaptive_production import build_union_state_mbar_inputs
    meta = build_union_state_mbar_inputs(ad, reg, output_prefix="e2e")
    assert int(w.state_id) in meta["state_ids"] and meta["aux_state_ids"] == [int(w.state_id)]
    assert meta["aux_model_sha256"] == run.model().model_sha256
    d = np.load(meta["arrays_npz"])
    assert np.isfinite(d["aux_z"]).all() and d["aux_z"].shape[0] == meta["n_samples"]
    col = meta["state_ids"].index(int(w.state_id))
    assert d["aux_k"][col] == pytest.approx(w.metadata["aux"]["aux_k_kcal_mol"]) and meta["N_k"][col] > 0
    import pandas as pd
    sources = {str(x).split("/")[0] for x in pd.read_csv(meta["samples_csv"])["source"]}
    assert {"epoch_000", "epoch_001", "epoch_002", "final"} <= sources, sources
    driver = json.loads((ad / "adaptive_union_mbar.json").read_text())
    assert driver["aux_state_ids"] == [int(w.state_id)]


def test_kill_after_registry_save_never_double_admits(tmp_path, small_adaptive_campaign):
    run = small_adaptive_campaign(tmp_path, aux=True, kill_after="registry_save:epoch_001")
    assert run.discover_calls == [1]
    assert (run.adaptive / "aux_admission.json").exists()
    assert len(_workers(run.registry().all_states())) == 1
    run.resume()
    assert run.discover_calls == [1]                                      # never re-discovered
    reg = run.registry()
    assert len(_workers(reg.all_states())) == 1
    assert run.phase_has_column("epoch_002", "aux_z_00")


def test_kill_after_freeze_before_save_readmits_from_the_record(tmp_path, small_adaptive_campaign):
    run = small_adaptive_campaign(tmp_path, aux=True, kill_after="action_report:epoch_001")
    assert run.discover_calls == [1]
    assert (run.adaptive / "aux_admission.json").exists()
    assert not _workers(run.registry().all_states())                    # the worker never reached the save
    run.resume()
    assert run.discover_calls == [1]                                      # re-admitted from the record
    reg = run.registry()
    assert len(_workers(reg.all_states())) == 1
    rep = json.loads((run.adaptive / "epoch_001" / "aux_discovery_report.json").read_text())
    assert rep["status"] == "ok" and rep["fixed_discovery"] is True        # the discovery record survives
    assert [e["status"] for e in rep["readmitted"]] == ["readmitted_from_record"]
    assert run.phase_has_column("epoch_002", "aux_z_00")
    from gareus.adaptive_production import build_union_state_mbar_inputs
    meta = build_union_state_mbar_inputs(run.adaptive, reg, output_prefix="e2e")
    assert meta["aux_state_ids"] == [int(_workers(reg.all_states())[0].state_id)]
