import numpy as np


def test_v2_samples_pool_into_union_mbar(tmp_path, monkeypatch):
    import aux_union_fixture as fixture
    from test_aux_sidechain_dictionary import residue_fixture
    from gareus.adaptive_production import build_union_state_mbar_inputs
    from gareus.auxiliary_cv.sample_schema import build_sample_schema
    from gareus.auxiliary_cv.sidechain_dictionary import build_sidechain_dictionary
    from gareus.auxiliary_cv.sidechain_model import SidechainModel

    topology, system = residue_fixture("PHE")
    dictionary = build_sidechain_dictionary(topology, system)
    coefficients = np.array([0.7, -0.4], dtype=float)
    model = SidechainModel.from_dictionary(dictionary, coefficients, topology=topology)
    schema = build_sample_schema(topology, (), [model])
    assert schema.schema == "atlas-aux-samples-v2"
    assert len(schema.torsion_quads) == 1

    monkeypatch.setattr(fixture, "MODEL", model)
    monkeypatch.setattr(fixture, "SCHEMA", schema)
    monkeypatch.setattr(fixture, "KI", {**fixture.KI, "aux_model_sha256": model.model_sha256})
    campaign = fixture.build_campaign(tmp_path)
    meta = build_union_state_mbar_inputs(campaign.ad, campaign.registry, output_prefix="v2")

    arrays = np.load(meta["arrays_npz"])
    assert meta["aux_model_sha256"] == model.model_sha256
    assert arrays["aux_z"].shape == (meta["n_samples"],)
    assert np.isfinite(arrays["aux_z"]).all()
    assert meta["aux_state_ids"] == [campaign.worker_id]
    assert arrays["aux_k"][meta["state_ids"].index(campaign.worker_id)] == 2.0
    assert meta["aux_burnin_exclusions"]["phases"][0]["burnin_workers"] == [campaign.worker_id]
