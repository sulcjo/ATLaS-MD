# tests/test_aux_resume_guards.py
import ast
import inspect
import json
import types

import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.kernel_identity import EXCHANGE_ENERGY_VERSION, EXCHANGE_ENERGY_VERSION_AUX


def _model(tmp_path, name="aux.json", c=1.0):
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [c, 0.0]))
    path = tmp_path / name
    m.write(path)
    return m, str(path)


def _snapshot_dir(tmp_path, identity):
    run = tmp_path / "run"
    (run / "windows").mkdir(parents=True)
    (run / "windows" / "seg_001.json").write_text(json.dumps({"segment_id": "seg_001", "windows": [],
                                                              "kernel_identity": identity}))
    return run


def _args(model_path=None):
    return types.SimpleNamespace(aux_cv_model=model_path)


def test_resume_refusal_is_two_sided_and_model_bound(tmp_path):
    from gareus.production import refuse_resume_of_aux_campaign
    m, path = _model(tmp_path)
    _other, other_path = _model(tmp_path, "other.json", c=2.0)
    aux_run = _snapshot_dir(tmp_path / "a", {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX,
                                             "aux_model_sha256": m.model_sha256})
    legacy_run = _snapshot_dir(tmp_path / "l", {"exchange_energy_version": EXCHANGE_ENERGY_VERSION})
    refuse_resume_of_aux_campaign(aux_run, _args(path))                        # same model: resumes
    refuse_resume_of_aux_campaign(legacy_run, _args(None))                     # legacy: unchanged
    refuse_resume_of_aux_campaign(tmp_path / "empty", _args(path))             # nothing recorded yet
    with pytest.raises(RuntimeError, match="same --aux-cv-model"):
        refuse_resume_of_aux_campaign(aux_run, _args(None))
    with pytest.raises(RuntimeError, match="is not this campaign's auxiliary model"):
        refuse_resume_of_aux_campaign(aux_run, _args(other_path))
    with pytest.raises(RuntimeError, match="ran without auxiliary-CV states"):
        refuse_resume_of_aux_campaign(legacy_run, _args(path))


def test_kernel_identity_resume_check_is_args_aware(tmp_path):
    from gareus.production import verify_kernel_identity_on_resume
    _m, path = _model(tmp_path)
    meta = {"enabled": True, "mode": "alpha"}
    aux = types.SimpleNamespace(secondary_cv="alpha", aux_cv_model=path)
    plain = types.SimpleNamespace(secondary_cv="alpha")

    def _record(version):
        (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {"exchange_energy_version": version}}))

    _record(EXCHANGE_ENERGY_VERSION_AUX)
    verify_kernel_identity_on_resume(aux, tmp_path, meta)
    with pytest.raises(RuntimeError, match="resume refused"):
        verify_kernel_identity_on_resume(plain, tmp_path, meta)
    _record(EXCHANGE_ENERGY_VERSION)
    verify_kernel_identity_on_resume(plain, tmp_path, meta)
    with pytest.raises(RuntimeError, match="resume refused"):
        verify_kernel_identity_on_resume(aux, tmp_path, meta)


def test_window_order_key_is_nan_safe():
    from gareus.production import _aux_window_order_key
    nan = float("nan")
    a = _aux_window_order_key([0.1, 0.2], [1.0, 1.0], [nan, 0.5], [0.0, 2.0], 2)
    assert a == _aux_window_order_key([0.1, 0.2], [1.0, 1.0], [nan, 0.5], [0.0, 2.0], 2)
    assert a[0][2] is None


def _run_gareus_tree():
    import gareus.production as production
    return ast.parse(inspect.getsource(production.run_gareus))


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", getattr(n.func, "id", None)) == name]


def _fast_resume_branch(tree):
    """The `if fast_resume:` whose first statement is the refusal (run_gareus has other fast_resume ifs)."""
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
            and n.test.id == "fast_resume" and n.body and isinstance(n.body[0], ast.Expr)
            and _calls(n.body[0], "refuse_resume_of_aux_campaign")]
    assert len(hits) == 1
    return hits[0]


def test_refusal_runs_first_on_every_resume_branch():
    branch = _fast_resume_branch(_run_gareus_tree())
    fast, other = branch.body[0], branch.orelse[0]
    assert [ast.unparse(a) for a in _calls(fast, "refuse_resume_of_aux_campaign")[0].args] == ["out_dir", "args"]
    assert isinstance(other, ast.If) and "resume" in ast.unparse(other.test)
    assert _calls(other, "refuse_resume_of_aux_campaign")


def test_fast_resume_takes_the_aux_table_from_the_checkpoint_and_sets_the_window_key():
    body = ast.Module(body=_fast_resume_branch(_run_gareus_tree()).body, type_ignores=[])
    assert _calls(body, "aux_table_from_checkpoint")
    causes = {k.value.value for c in _calls(body, "refuse_aux_population_change") for k in c.keywords
              if k.arg == "cause" and isinstance(k.value, ast.Constant)}
    assert causes == {"checkpoint resume"}
    assert any(isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_aux_window_key" for t in n.targets)
               for n in ast.walk(body))


def test_stage_b_engineering_warning_is_gone():
    import gareus.production as production
    assert "aux-cv-allow-unpersisted" not in inspect.getsource(production.run_gareus)


def test_window_order_key_maps_none_placeholders_to_none():
    from gareus.production import _aux_window_order_key
    a = _aux_window_order_key([0.1], [1.0], [None], [None], 1)
    assert a == ((0.1, 1.0, None, None),)


# --- refused-resume manifest stamping (controller ruling, c10 safety) -----------------------------

_CLI_BASE = ["--seq", "GA", "--cv1", "contacts"]
_CLI_AUX = ["--aux-cv-model", "m.json", "--windows-2d-csv", "w.csv", "--run-mode", "cmd",
            "--exchange-mode", "gibbs-walk"]


def _legacy_campaign_dir(tmp_path):
    run = _snapshot_dir(tmp_path, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION})
    manifest = run / "run_manifest.json"
    manifest.write_text(json.dumps({"method_settings": {"exchange_energy_version": EXCHANGE_ENERGY_VERSION},
                                    "resolved_args": {"seq": "GA"}}, indent=2))
    return run, manifest


@pytest.mark.parametrize("flag", ["--resume", "--extend"])
def test_refused_aux_resume_never_stamps_a_non_aux_manifest(tmp_path, flag):
    from gareus.cli import main
    run, manifest = _legacy_campaign_dir(tmp_path)
    before = manifest.read_bytes()
    listing = sorted(p.name for p in run.iterdir())
    with pytest.raises(RuntimeError, match="ran without auxiliary-CV states"):
        main(_CLI_BASE + _CLI_AUX + [flag, "--out", str(run)])
    assert manifest.read_bytes() == before
    assert sorted(p.name for p in run.iterdir()) == listing      # nothing written at all


def test_refused_aux_resume_checks_the_final_production_subdirectory(tmp_path):
    from gareus.cli import main
    root = tmp_path / "root"
    root.mkdir()
    run, _manifest = _legacy_campaign_dir(tmp_path)
    run.rename(root / "final_production")
    manifest = root / "final_production" / "run_manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(RuntimeError, match="ran without auxiliary-CV states"):
        main(_CLI_BASE + _CLI_AUX + ["--resume", "--out", str(root)])
    assert manifest.read_bytes() == before


def test_legacy_resume_passes_the_main_guard(tmp_path):
    """c10 pin: a non-aux --resume of a non-aux campaign gets past the guard (it then stops for want of a
    production checkpoint, as before)."""
    from gareus.cli import main
    run, _manifest = _legacy_campaign_dir(tmp_path)
    assert main(_CLI_BASE + ["--resume", "--out", str(run)]) is None
