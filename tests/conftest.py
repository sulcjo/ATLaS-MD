"""Shared synthetic-swarm builders for the automatic CV2 selection tests.

`tests/test_swarm_analyze.py` keeps its own `_fake_round`/`_args` helpers untouched;
these fixtures extend the same on-disk shape with torsion features per member
(Task 1) and the configuration attributes the selector reads (Tasks 8-9). No MD,
no OpenMM: every member directory is written from literals.
"""
from __future__ import annotations

import csv
import json
import types
from pathlib import Path

import numpy as np
import pytest

from gareus.swarm.members import TRACE_COLUMNS

PHI_TORSIONS = [[0, 1, 2, 3], [4, 5, 6, 7]]
PSI_TORSIONS = [[8, 9, 10, 11], [12, 13, 14, 15]]


def make_fake_round(out: Path, *, n_members: int = 12, n_frames: int = 300, wide_anchor: bool = True,
                    with_features: bool = True, crash_one_member: bool = False, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rd = out / "swarm" / "round_000"
    rd.mkdir(parents=True)
    with (rd / "plan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["member_id", "cell_id", "cell_cv1", "cell_rg", "cell_e2e",
                                          "seed_id", "seed_pdb", "replicate", "velocity_seed"])
        w.writeheader()
        for m in range(n_members):
            w.writerow({"member_id": m, "cell_id": f"{m % 2}_0_0", "cell_cv1": m % 2, "cell_rg": 0,
                        "cell_e2e": 0, "seed_id": f"s{m // 2}", "seed_pdb": "/x", "replicate": m % 2,
                        "velocity_seed": m})
    json.dump({"n_cells": 2, "replicates_per_cell": n_members // 2, "n_members": n_members,
               "budget_ns": float(n_members), "seed_ns": 1.0,
               "edges": {"cv1": [0, 0.5, 1], "rg": [0, 2], "e2e": [0, 4]}},
              (rd / "plan_meta.json").open("w"))
    scale = 1.0 if wide_anchor else 0.069
    d = 2 * (len(PHI_TORSIONS) + len(PSI_TORSIONS))
    for m in range(n_members):
        md = rd / f"member_{m:04d}"
        (md / "frames").mkdir(parents=True)
        crashed = crash_one_member and m == n_members - 1
        rows = n_frames // 3 if crashed else n_frames
        cv1 = scale * rng.beta(2, 4, rows)
        hidden = rng.normal(size=(rows, d)) @ np.diag(np.linspace(2.0, 0.3, d))
        feats = np.tanh(hidden + 0.3 * ((cv1 - cv1.mean()) / max(cv1.std(), 1e-9))[:, None])
        with (md / "trace.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
            w.writeheader()
            for i in range(rows):
                w.writerow({"frame": i, "t_ps": 2.0 * (i + 1), "cv1": float(cv1[i]),
                            "rg_nm": float(0.6 + 0.3 * cv1[i] + 0.02 * rng.normal()),
                            "e2e_nm": float(1.0 + 0.2 * hidden[i, 1]),
                            "v_pep_kj": float(rng.normal(-50, 5)), "v_dih_kj": float(rng.normal(20, 2)),
                            "potential_kj": -1e4})
                if (i + 1) % 10 == 0:
                    (md / "frames" / f"frame_{i:05d}.pdb").write_text(
                        "ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        if with_features and not crashed:
            np.save(md / "torsion_features.npy", feats)
            json.dump({"phi_torsions": PHI_TORSIONS, "psi_torsions": PSI_TORSIONS},
                      (md / "torsion_index.json").open("w"))
        done = {"member_id": m, "n_frames": rows, "graft_fallback": False, "wall_s": 10.0,
                "ns_per_day": 400.0, "status": "md_failed" if crashed else "ok"}
        json.dump(done, (md / "done.json").open("w"))
    return rd


def make_swarm_args(**overrides) -> types.SimpleNamespace:
    base = dict(
        temperature_k=300.0, sigma0p_kcal_mol=6.0, sigma0d_kcal_mol=6.0, swarm_n_windows=8,
        swarm_overlap_sigma=1.5, swarm_target_beta_sigma=1.0, swarm_min_rungs=3, swarm_max_rungs=12,
        swarm_ess_floor=50, swarm_seeds_per_window=2, swarm_discard_block_frames=20,
        swarm_min_discard_ps=0.0, swarm_output_interval_ps=2.0, contact_adaptive_max_k_kcal=1200.0,
        contact_adaptive_min_k_kcal=5.0, swarm_round=0,
        # CV configuration the selector reads
        primary_cv="nonlocal-contacts", secondary_cv="none", contact_r0_a=12.0, contact_beta_a_inv=3.0,
        contact_min_sequence_separation=4, contact_atom_selection="heavy", contact_scheme="atom-pairs",
        contact_normalize=True, diversity_bank_preset="broad", max_replicas=128,
        swarm_n_windows_cv2=4, cv2_k_max=1000.0, cv2_k_min=0.0,
        cv_selection_residual_degree=1, cv_selection_max_nonlinear_r2=0.20,
        cv_selection_max_coupling_fraction=0.25, cv_selection_k2_reference_kcal=1.0,
        cv_selection_min_windows_cv1=4, cv_selection_min_gain_nats=0.02,
        cv_selection_fallback="cv1_only",
        # The fake swarm's CV2 is neither slow nor bimodal, so the default slowness ranking
        # (correctly) returns cv1_only; plumbing tests pin the legacy rule to get a pair.
        cv_selection_rank="gain",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


@pytest.fixture
def synthetic_swarm(tmp_path):
    """Factory: build a fake swarm round under a fresh directory; returns the run's out dir."""
    counter = {"n": 0}

    def _build(**kw):
        counter["n"] += 1
        out = tmp_path / f"run{counter['n']}"
        out.mkdir()
        make_fake_round(out, **kw)
        return out

    return _build


@pytest.fixture
def swarm_args():
    return make_swarm_args


# ---------------------------------------------------------------------------------------------------------
# Small adaptive-production campaign harness (adaptive-production e2e tests).
# ---------------------------------------------------------------------------------------------------------

class CampaignKilled(BaseException):
    """Raised by an injected kill point inside the adaptive-production driver. A BaseException, like the
    SIGTERM/SystemExit it stands for, so the driver's report-writing ``except Exception`` blocks cannot eat it."""


class CampaignReachedFinal(Exception):
    """Raised by a fake run_gareus when the driver reaches the final phase."""


def release_campaign_locks(root: Path) -> None:
    """A killed job leaves its .gareus_run.lock files behind with a dead PID, which the next job treats as
    stale; in one test process the PID is still alive, so remove them as the stale-lock path would."""
    for lock in Path(root).rglob(".gareus_run.lock"):
        lock.unlink(missing_ok=True)


@pytest.fixture
def campaign_locks():
    return release_campaign_locks


_SMALL_CAMPAIGN_WINDOWS = ("window,primary_cv_mode,primary_cv_center,primary_cv_k_kcal,gamd_lambda\n"
                           "0,distance,3.75,2.0,0.0\n1,distance,3.80,2.0,0.0\n2,distance,3.85,2.0,0.0\n")
SMALL_CAMPAIGN_AUX_K3 = 1.0


_SMALL_CAMPAIGN_PEP_GAMD = ["--run-mode", "gamd", "--gamd-boost-type", "pep-gamd-lower-dual",
                            "--gamd-cmd-steps", "100", "--equil-steps", "100", "--gamd-averaging-window", "50",
                            "--gamd-cmd-prep-steps", "50", "--gamd-equil-prep-steps", "50",
                            "--gamd-multiwindow-recon-prep-steps", "1", "--gamd-multiwindow-recon-cmd-steps", "2",
                            "--gamd-multiwindow-recon-steps", "2", "--gamd-recon-boosted-iters", "1",
                            "--gamd-recon-boosted-tol", "0.05", "--gamd-multiwindow-recon-report-interval", "1"]


def _small_campaign_argv(out: Path, windows_csv: Path, aux: bool, pep_gamd: bool = False):
    argv = ["--seq", "GA", "--out", str(out), "--seed", "2026", "--platform", "CPU", "--setup-platform", "CPU",
            "--box-shape", "dodecahedron", "--padding-nm", "0.55", "--ionic-strength-molar", "0.0",
            "--temperature-k", "300.0", "--run-mode", "cmd", "--timestep-fs", "1.0", "--friction-per-ps", "5.0",
            "--minimize-iterations", "5", "--nvt-warmup-steps", "2", "--nvt-warmup-timestep-fs", "0.25",
            "--npt-ramp-steps", "2", "--npt-ramp-timestep-fs", "0.25", "--npt-steps", "4",
            "--window-mode", "adaptive-production", "--windows-2d-csv", str(windows_csv),
            "--us-starting-structure-mode", "pull", "--us-pull-steps-per-window", "2",
            "--us-pull-timestep-fs", "0.25", "--us-pull-minimize-iterations", "1", "--us-allow-bad-windows",
            "--exchange-mode", "gibbs-walk", "--exchange-interval", "100", "--report-interval", "100",
            "--distance-output-interval", "100", "--traj-interval", "100", "--traj-format", "xtc",
            "--traj-solute-only", "--production-steps", "3000", "--ap-epochs", "3", "--ap-epoch-steps", "1000",
            "--ap-final-steps", "1000", "--checkpoint-interval", "0", "--progress-mode", "none",
            "--tui-mode", "none"]
    if pep_gamd:      # later flags win: --run-mode gamd replaces cmd
        argv = argv + _SMALL_CAMPAIGN_PEP_GAMD
    return argv + (["--ap-aux-discovery"] if aux else [])


class SmallCampaign:
    def __init__(self, out: Path, argv, discover_calls):
        self.out = Path(out)
        self.adaptive = self.out / "adaptive_production"
        self.argv = list(argv)
        self.discover_calls = discover_calls

    def registry(self):
        from gareus.adaptive_production import WindowStateRegistry
        return WindowStateRegistry.load(self.adaptive)

    def phases(self):
        """[(label, phase_dir)] in campaign order (epoch_000, epoch_001/baseline, ...)."""
        from gareus.adaptive import discovery_census as census
        return census.ordered_phases(self.adaptive)[0]

    def phase_dir(self, label_prefix: str) -> Path:
        hits = [p for lab, p in self.phases() if lab == label_prefix or lab.startswith(label_prefix + "/")]
        assert len(hits) == 1, (label_prefix, self.phases())
        return hits[0]

    def samples(self, label_prefix: str):
        import pandas as pd
        files = sorted((self.phase_dir(label_prefix) / "samples").glob("*/*.parquet"))
        assert files, label_prefix
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def phase_has_column(self, label_prefix: str, column: str) -> bool:
        return column in self.samples(label_prefix).columns

    def model(self):
        from gareus.auxiliary_cv.model import AuxModel
        return AuxModel.load(self.adaptive / "aux_model.json")

    def resume(self):
        from gareus.cli import main
        release_campaign_locks(self.out)
        main(self.argv + ["--resume"])


def _fixed_discovery(discover_calls):
    """``_discover`` stand-in: GA has no contact/H-bond descriptor pair, so the U13 partition (fitted on
    contacts + H-bonds only) cannot run on it; the statistics are Tasks 7-9's. The result is still real:
    a real emit_model z3 over the campaign's own phi/psi on its own 01_solvated_start.pdb ("direct"
    convention), a real FrozenPartition, and one worker on the lowest lambda = 0 state of the frame table,
    centred on that state's mean z."""
    import numpy as np
    from aux_discovery_fixture import frozen_partition
    from gareus.adaptive import aux_admission_io as H
    from gareus.adaptive.aux_discovery.pipeline import DiscoveryResult
    from gareus.adaptive.aux_discovery.z3_search import Z3Candidate, emit_model

    def _discover(*, ft, epoch, settings, out_dir, k3_max, eligible_parents=None):
        discover_calls.append(int(epoch))
        tors = np.asarray(ft.tors, dtype=np.float64)
        w = np.linspace(1.0, -0.5, tors.shape[1])
        # Unstandardised features (sd 1) and scale 1: z = w . (f - mu), sum|w| = 2, so XTC rounding (0.001 nm,
        # ~1e-2 rad) moves z by ~1e-2. A discovery-like standardisation (sd floor 0.05) on GA's barely moving
        # torsions amplified it to 0.035-0.051 against the 0.05 recorded-vs-frame tolerance (measured, Task 17).
        mu, sd = tors.mean(0), np.ones(tors.shape[1])
        zr = (tors - mu) @ w
        cand = Z3Candidate(groups=(0, 1), C=0.1, w_std=w, mu=mu, sd=sd, z_raw_train_sd=1.0,
                           info_gain=0.2, stability=0.9, corr_cv1=0.0, corr_cv2=0.0, basin_gain=0.1,
                           n_nonzero=int(w.size), passed=True, fail=[])
        model = emit_model(cand, ft.definition, H._full_topology(out_dir), label=f"e2e-epoch{int(epoch):03d}",
                           provenance={"source": "task17-e2e"})
        lam0 = np.asarray(ft.lam) == 0.0
        if eligible_parents is not None:              # the hook's admission limits (fix wave C1)
            lam0 &= np.isin(np.asarray(ft.state_id), np.asarray(list(eligible_parents), dtype=np.int64))
        parent = int(np.min(np.asarray(ft.state_id)[lam0]))
        c3 = float(np.mean(zr[np.asarray(ft.state_id) == parent]))
        chosen = [{"state_id": parent, "c3": c3, "k3": SMALL_CAMPAIGN_AUX_K3, "O": 0.3, "TV": 0.2,
                   "null": 0.05, "net": 0.15, "utility_q10": 0.04}]
        return DiscoveryResult("ok", {"fixed_discovery": True}, model, frozen_partition(), {"chosen": chosen})

    return _discover


@pytest.fixture
def small_adaptive_campaign():
    """Factory: a real (CPU, --run-mode cmd) GA-dipeptide adaptive-production campaign through
    ``gareus.cli.main``: epochs 0-2 + final, 3 windows, 1000 steps per phase, XTC every sample.

    ``aux=True`` adds --ap-aux-discovery, a passing aux_validation.json and the fixed discovery above.
    ``kill_after="registry_save:epoch_001"`` raises CampaignKilled right after the first
    WindowStateRegistry.save whose registry holds an aux worker (epoch 1's post-action save), before the
    post-save applied-actions ledger write; ``kill_after="action_report:epoch_001"`` raises in epoch 1's
    write_epoch_action_report, after the hook froze the admission but before apply/save. Weak-edge/R* proposals are off and the convergence gate always continues, so
    ``admit_aux`` is the only action and every numbered epoch runs."""
    import gareus.adaptive_production as ap
    from gareus.adaptive.aux_discovery.validation import main as validation_main
    from gareus.cli import main

    mp = pytest.MonkeyPatch()

    def _build(tmp_path: Path, *, aux: bool = True, kill_after: str = "", pep_gamd: bool = False):
        out = Path(tmp_path) / "run"
        out.mkdir(parents=True)
        windows = Path(tmp_path) / "windows.csv"
        windows.write_text(_SMALL_CAMPAIGN_WINDOWS)
        argv = _small_campaign_argv(out, windows, aux, pep_gamd)
        calls = []
        mp.setattr(ap, "propose_actions_from_diagnostics", lambda *a, **k: [])
        mp.setattr(ap, "evaluate_adaptive_convergence_gate",
                   lambda *a, **k: {"status": "continue", "stop_adaptive": False, "continue_reasons": []})
        if aux:
            from gareus.adaptive import aux_admission_io as H
            mp.setattr(H, "_discover", _fixed_discovery(calls))
            assert validation_main(["write", "--out", str(out / "adaptive_production" / "aux_validation.json"),
                                    "--commit", "e2e", "--timestep-fs", "1.0", "--k3-max", "10.0",
                                    "--finite-timestep", "pass", "--npt", "pass", "--cost", "pass"]) == 0
        run = SmallCampaign(out, argv, calls)
        if kill_after:
            armed = {"on": True}
            if kill_after == "registry_save:epoch_001":
                real_save = ap.WindowStateRegistry.save

                def _save(self, directory):
                    paths = real_save(self, directory)
                    if armed["on"] and any((s.metadata or {}).get("aux") for s in self.all_states()):
                        armed["on"] = False
                        raise CampaignKilled("killed right after registry.save (epoch 1)")
                    return paths

                mp.setattr(ap.WindowStateRegistry, "save", _save)
            elif kill_after == "action_report:epoch_001":
                # after the hook froze the admission, before the actions are applied and the registry saved
                real_report = ap.write_epoch_action_report

                def _report(epoch_dir, epoch, *a, **k):
                    if armed["on"] and int(epoch) == 1:
                        armed["on"] = False
                        raise CampaignKilled("killed after the admission freeze, before registry.save (epoch 1)")
                    return real_report(epoch_dir, epoch, *a, **k)

                mp.setattr(ap, "write_epoch_action_report", _report)
            else:
                raise AssertionError(f"unknown kill point {kill_after!r}")
            with pytest.raises(CampaignKilled):
                main(argv)
        else:
            main(argv)
        return run

    yield _build
    mp.undo()
