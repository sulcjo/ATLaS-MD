"""--ap-aux-validation {required,off}: the aux_validation.json gate on worker admission can be switched off."""
import json
from types import SimpleNamespace

import pytest

from gareus.adaptive import aux_admission_io as H
from gareus.adaptive.aux_discovery.settings import (AUX_CAMPAIGN_OPTIONS_FILENAME, AuxDiscoverySettings,
                                                    resolve_aux_campaign_options)
from test_aux_admission_hook import _args, _run


def _spy(monkeypatch):
    seen = {}
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())

    def disc(**kw):
        seen["k3_max"] = kw["k3_max"]
        return H.DiscoveryResultStub.ok_with_workers([(1, 1.2, 2.0)])
    monkeypatch.setattr(H, "_discover", disc)
    monkeypatch.setattr(H, "check_validation_record",
                        lambda *a, **k: pytest.fail("aux_validation.json must not be read with validation off"))
    return seen


def test_off_admits_without_a_record_and_takes_k3_cap_from_settings(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    seen = _spy(monkeypatch)
    args = _args(tmp_path); args.ap_aux_validation = "off"
    out = _run(ad, tmp_path, args=args)
    assert len(out) == 1 and out[0][0] == "admit_aux" and not (ad / "aux_validation.json").exists()
    assert seen["k3_max"] == AuxDiscoverySettings().k3_max_unvalidated
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert rep["validation"]["validation"] == "off" and rep["validation"]["k3_max_source"].startswith("aux_settings")
    adm = json.loads((ad / "aux_admission.json").read_text())
    assert adm["validation"]["validation"] == "off" and adm["validation"]["k3_max"] == seen["k3_max"]


def test_k3_cap_follows_the_frozen_settings(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    (ad / "aux_settings.json").write_text(json.dumps({
        "schema": "atlas-aux-discovery-settings-v1", "settings": {"k3_max_unvalidated": 2.5}}))
    seen = _spy(monkeypatch)
    args = _args(tmp_path); args.ap_aux_validation = "off"
    _run(ad, tmp_path, args=args)
    assert seen["k3_max"] == 2.5


def test_required_is_the_default_and_still_blocks_without_a_record(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    monkeypatch.setattr(H, "_discover", lambda **kw: H.DiscoveryResultStub.ok_with_workers([(1, 1.2, 2.0)]))
    assert _run(ad, tmp_path) == []
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert rep["status"] == "validation_missing" and "validation" not in rep["validation"]


def test_freeze_mismatch_refused_and_legacy_record_is_required(tmp_path):
    off = SimpleNamespace(ap_continue_states=True, ap_aux_validation="off")
    req = SimpleNamespace(ap_continue_states=True)
    assert resolve_aux_campaign_options(tmp_path, off)["options"]["aux_validation"] == "off"
    with pytest.raises(RuntimeError, match="aux_validation=off.*frozen per campaign"):
        resolve_aux_campaign_options(tmp_path, req)
    other = tmp_path / "req"; other.mkdir()
    rec = resolve_aux_campaign_options(other, req)                           # required: record bytes unchanged
    assert rec["options"] == {"continue_states": True}                       # (= legacy record without the key)
    with pytest.raises(RuntimeError, match="aux_validation=required"):
        resolve_aux_campaign_options(other, off)


def test_yaml_key_and_default(tmp_path):
    from gareus.cli import parse_args
    cfg = tmp_path / "c.yaml"; cfg.write_text("ap_aux_validation: 'off'\n")
    a = parse_args(["--config", str(cfg), "--out", str(tmp_path / "o"), "--seq", "GYDPETGTWG"])
    assert a.ap_aux_validation == "off" and a.adaptive_production_aux_validation == "off"
    b = parse_args(["--out", str(tmp_path / "o"), "--seq", "GYDPETGTWG"])
    assert b.ap_aux_validation == "required"
