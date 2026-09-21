import json
import types
import pytest


def _residual_meta(sha=None):
    meta = {
        "enabled": True,
        "mode": "residual-torsion-pc",
        "scalar_reconstruction": {"schema_version": "residual_full_expression_v1"},
    }
    if sha is not None:
        meta["pair_model_sha256"] = sha
    return meta


def _write_model(tmp_path, sha):
    p = tmp_path / "cv_pair_model.json"
    p.write_text(json.dumps({"schema": "atlas-cv-selection-pair-model-v1", "sha256": sha}))
    return p


def test_resume_refuses_a_changed_pair_model():
    from gareus.production import reconcile_resume_secondary_cv_metadata
    meta = _residual_meta("a" * 64)
    with pytest.raises(RuntimeError, match="frozen CV2 pair model changed"):
        reconcile_resume_secondary_cv_metadata(meta, [0.0, 1.0], current_pair_sha256="b" * 64)


def test_resume_accepts_the_same_pair_model_and_records_a_first_digest():
    from gareus.production import reconcile_resume_secondary_cv_metadata
    meta = _residual_meta("a" * 64)
    out = reconcile_resume_secondary_cv_metadata(meta, [0.0, 1.0], current_pair_sha256="a" * 64)
    assert out["pair_model_sha256"] == "a" * 64
    first = reconcile_resume_secondary_cv_metadata(_residual_meta(), [0.0], current_pair_sha256="c" * 64)
    assert first["pair_model_sha256"] == "c" * 64


def test_resume_refuses_a_legacy_residual_checkpoint_without_evaluator_identity():
    from gareus.production import reconcile_resume_secondary_cv_metadata
    meta = {"enabled": True, "mode": "residual-torsion-pc", "pair_model_sha256": "a" * 64}
    with pytest.raises(RuntimeError, match="lacks the verified residual_full_expression_v1"):
        reconcile_resume_secondary_cv_metadata(meta, [0.0, 1.0], current_pair_sha256="a" * 64)


def test_resume_without_a_pair_digest_is_unchanged_for_legacy_runs():
    from gareus.production import reconcile_resume_secondary_cv_metadata
    out = reconcile_resume_secondary_cv_metadata({"enabled": True, "mode": "rama-map"}, [0.0])
    assert "pair_model_sha256" not in out


def test_manifest_records_the_pair_model_digest(tmp_path):
    from gareus.provenance import _method_settings, pair_model_sha256
    model = _write_model(tmp_path, "d" * 64)
    args = types.SimpleNamespace(seq="GYDPETGTWG", secondary_cv="residual-torsion-pc",
                                 secondary_cv_model=str(model), temperature_k=300.0)
    s = _method_settings(args)
    assert s["cv_pair_model_sha256"] == "d" * 64 and s["secondary_cv_model"] == str(model)
    assert pair_model_sha256(None) is None
    assert _method_settings(types.SimpleNamespace(seq="G"))["cv_pair_model_sha256"] is None


def test_a_configured_but_missing_pair_model_is_an_error_not_a_silent_none(tmp_path):
    from gareus.provenance import pair_model_sha256
    with pytest.raises(FileNotFoundError, match="missing file"):
        pair_model_sha256(tmp_path / "gone.json")
