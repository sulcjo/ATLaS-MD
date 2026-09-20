"""Regenerate the tiny selector fixtures.

These are *generated* artifacts, deliberately not molecular run dumps: two
torsions, four features, two components. Their purpose is to pin the on-disk
encoding, so a later schema edit that silently invalidates stored artifacts
fails a test instead of a campaign.

    python tests/fixtures/cv_selection/generate.py
"""
from __future__ import annotations

from pathlib import Path

from gareus.cv_selection import contracts as C

HERE = Path(__file__).resolve().parent


def _features() -> list[dict]:
    rows: list[dict] = []
    for torsion, name in enumerate(("phi-1", "psi-1")):
        for trig in ("sin", "cos"):
            rows.append({
                "index": len(rows),
                "name": f"{name}-{trig}",
                "torsion_name": name,
                "residue_index": 1,
                "atom_indices": [4 * torsion, 4 * torsion + 1,
                                 4 * torsion + 2, 4 * torsion + 3],
                "trig": trig,
                "dihedral_sign_convention": "negated",
            })
    return rows


def _component(index: int) -> dict:
    sign = 1.0 if index == 1 else -1.0
    return {
        "component_index": index,
        "singular_value": 3.0 / index,
        "eigenvalue_tie_flagged": False,
        "right_singular_vector": [0.5, 0.5, 0.5, sign * 0.5],
        "residual_mean": [0.0, 0.0, 0.0, 0.0],
        "regression_coefficients": [[0.0] * 4, [0.10] * 4, [0.01] * 4],
        "primary_mean": 0.2,
        "primary_std": 0.05,
        "projection_mean": 0.0,
        "projection_std": 1.0,
    }


def build() -> dict[str, C._Artifact]:
    feature_schema = C.FeatureSchema.from_mapping({
        "schema": C.FEATURE_SCHEMA_VERSION,
        "topology_sha256": "0" * 63 + "1",
        "features": _features(),
    })
    candidate_set = C.CandidateSet.from_mapping({
        "schema": C.CANDIDATE_SET_VERSION,
        "kind": C.CANDIDATE_KIND_QUADRATIC_RESIDUAL,
        "feature_schema_sha256": feature_schema.sha256,
        "physical_system_sha256": "0" * 63 + "2",
        "training_rows_sha256": "0" * 63 + "3",
        "primary_definition": {
            "kind": "nonlocal-contact-fraction",
            "units": "dimensionless",
            "definition": {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0,
                           "min_sequence_separation": 4},
        },
        "components": [_component(1), _component(2)],
    })
    panel = C.ObservablePanel.from_mapping({
        "schema": C.OBSERVABLE_PANEL_VERSION,
        "primary": [{"observable_id": f"cdf::{source}::q{q:g}",
                     "kind": "cdf_probability",
                     "source_observable": source,
                     "discovery_quantile": q,
                     "halfwidth_tolerance": 0.02}
                    for source in ("rg_nm", "end_to_end_nm")
                    for q in (0.2, 0.4, 0.6, 0.8)],
        "cross_protocol_tolerance": 0.02,
        "diagnostic_partition_centers": 16,
        "diagnostic_histogram_bins": 20,
        "novelty_mass_tolerance": 0.02,
    })
    protocol = C.ProtocolSpec.from_mapping({
        "schema": C.PROTOCOL_VERSION,
        "study_id": "fixture-fixed-primary",
        "search_mode": C.SearchMode.FIXED_PRIMARY.value,
        "target": {"sequence": "GYDPETGTWG", "temperature_k": 300.0, "ensemble": "NPT",
                   "pressure_bar": 1.0, "physical_system_sha256": "0" * 63 + "2"},
        "blindness": {"generator_preset": "broad", "require_native_blind_allowlist": True,
                      "historical_data_role": "exploratory_only"},
        "discovery": {"candidate_set_sha256": candidate_set.sha256,
                      "feature_schema_sha256": feature_schema.sha256},
        # Unresolved on purpose: a generated example must never look launch-ready.
        "sampling": {"layout": "grid24-v1", "lambda_rungs": 4, "burn_in_ticks": None,
                     "measurement_ticks": None, "campaigns_per_arm": None},
        "resources": {"gpus": 4, "cpu_thread_cap": 192, "max_replicas": 128,
                      "gpu_hours_per_campaign": None, "total_gpu_hour_ceiling": None},
        "evaluation": {"observable_panel_sha256": panel.sha256},
        "uncertainty": {"policy": "joint-batch-v1", "calibration_certificate_sha256": None},
        "decision": {"provisional_choice_arm_id": None, "interval_family_size": None,
                     "advantage_claim_enabled": False},
    })
    arms = [{"arm_id": "contact-only", "secondary_component_index": None,
             "campaigns": 3, "gpu_hours_per_campaign": 8.0}]
    arms += [{"arm_id": f"contact+residual-pc{j:02d}", "secondary_component_index": j,
              "campaigns": 3, "gpu_hours_per_campaign": 8.0} for j in range(1, 7)]
    plan = C.TrialPlan.from_mapping({
        "schema": C.TRIAL_PLAN_VERSION,
        "plan_id": "fixture-screen",
        "study_id": protocol.study_id,
        "stage": C.Stage.SCREEN.value,
        "study_role": C.StudyRole.SCREEN.value,
        "measurement_phase_kind": C.PhaseKind.PRODUCTION.value,
        "protocol_sha256": protocol.sha256,
        "layout": "grid24-v1",
        "lambda_rungs": 4,
        "states_per_arm": 96,
        "max_replicas": 128,
        "arms": arms,
    })
    decision = C.Decision.from_mapping({
        "schema": C.DECISION_VERSION,
        "study_id": protocol.study_id,
        "integrity": C.Integrity.PASS.value,
        "precision": C.Precision.UNRESOLVED.value,
        "dependence": C.Dependence.UNRESOLVED_CORRELATION.value,
        "reproducibility": C.Reproducibility.UNRESOLVED.value,
        "cross_protocol": C.CrossProtocol.NOT_COMPARED.value,
        "support": C.Support.UNRESOLVED.value,
        "advantage": C.Advantage.NOT_TESTED.value,
        "decision": C.DecisionOutcome.INSUFFICIENT_EVIDENCE.value,
        "reasons": [C.ReasonCode.UNRESOLVED_CORRELATION.value],
        "unresolved_regions": [],
        "tested_protocol_ids": [arm["arm_id"] for arm in arms],
        "input_sha256": {"candidate_set": candidate_set.sha256,
                         "observable_panel": panel.sha256,
                         "protocol": protocol.sha256},
        "selected_arm_id": None,
    })
    return {"feature_schema": feature_schema, "candidate_set": candidate_set,
            "observable_panel": panel, "protocol": protocol, "trial_plan": plan,
            "decision": decision}


def main() -> None:
    for name, artifact in build().items():
        (HERE / f"{name}.json").write_bytes(artifact.to_json_bytes())
        print(f"wrote {name}.json  sha256={artifact.sha256}")


if __name__ == "__main__":
    main()
