from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from gareus.adaptive.aux_discovery.pipeline import local_split_ids, search_local_candidates
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings
from gareus.adaptive.aux_discovery.frames import FrameTable
from gareus.adaptive.aux_discovery import pipeline


def frame_fixture(*, cv_only=False, constant_chi=False):
    n = 720
    region = np.tile([0, 0, 1, 1], 180)
    x = np.tile([-1., 1., -1., 1.], 180)
    labels = ((2 * region - 1) * x > 0).astype(np.int64)
    chi = np.c_[np.sin(x * np.pi / 3), np.cos(x * np.pi / 3)]
    if constant_chi:
        chi[:] = 1.
    if cv_only:
        chi[:, 0] = labels * 2. - 1.
    split = np.repeat([0, 1, 2], n // 3)
    phase = np.repeat(['epoch_000/baseline', 'epoch_001/baseline', 'epoch_002/baseline'], n // 3)
    replica = np.tile(np.repeat(np.arange(12), 20), 3)
    step = np.tile(np.arange(20), 36)
    cv1 = region.astype(float)
    if cv_only:
        cv1 = labels.astype(float)
    ft = FrameTable(phase=phase, epoch=np.repeat([0, 1, 2], n // 3), replica=replica, step=step,
                    state_id=np.zeros(n, dtype=np.int64), lam=np.zeros(n), cv1=cv1, cv2=None,
                    tors=np.c_[np.sin(x), np.cos(x)], tors_theta_iupac=np.empty((n, 0)),
                    hc=np.empty((n, 0)), hb=np.empty((n, 0)), basin=np.empty((n, 0)),
                    definition=SimpleNamespace(hc_labels=(), hb_labels=(), schema_sha256='schema'),
                    sources=[], chi=chi,
                    sidechain_dictionary=SimpleNamespace(primitives=('sin', 'cos')))
    return ft, labels, region, split


def run(ft, labels, regions, split, feature_space='sidechain'):
    n = len(labels)
    settings = AuxDiscoverySettings(feature_space=feature_space, l1_c_grid=(.1, 1.), n_null_z3=3,
                                    max_cv_corr=.7)
    source = ('partition-identity-k2', 2, labels)
    return search_local_candidates(ft, [source], regions, split != 2, split == 2, settings,
                                   split_ids=split)


def test_local_adapter_preserves_opposite_region_relationships_and_global_projection():
    ft, labels, regions, split = frame_fixture()
    sampled = ft.take(np.array([0, 3, 12]))
    np.testing.assert_array_equal(sampled.chi, ft.chi[[0, 3, 12]])
    assert sampled.sidechain_dictionary is ft.sidechain_dictionary
    result = run(ft, labels, regions, split)
    assert result['null_gate']['passed']
    candidates = [c for c in result['candidates'] if c.family == 'sidechain']
    assert {c.region for c in candidates} == {0, 1}
    assert all(c.partition_id == 'partition-identity-k2' for c in candidates)
    by_region = sorted(candidates, key=lambda c: c.region)
    active = np.argmax(np.abs(by_region[0].coefficients) + np.abs(by_region[1].coefficients))
    assert by_region[0].coefficients[active] * by_region[1].coefficients[active] < 0
    values = [candidate.values(np.c_[ft.tors, ft.chi]) for candidate in candidates]
    assert all(np.isfinite(z).all() for z in values)


def test_automatic_split_keeps_each_phase_replica_carrier_whole():
    ft, _labels, _regions, holdout = frame_fixture()
    ids = local_split_ids(ft, holdout != 2, holdout == 2, seed=31)
    assert np.all(ids[holdout == 2] == 2)
    for carrier in np.unique(ft.lineage[holdout != 2]):
        assert np.unique(ids[(ft.lineage == carrier) & (holdout != 2)]).size == 1


def test_auto_search_includes_backbone_sidechain_mixed_and_all_source_partitions():
    ft, labels, regions, split = frame_fixture()
    settings = AuxDiscoverySettings(feature_space='auto', l1_c_grid=(.1, 1.), n_null_z3=2)
    result = search_local_candidates(ft, [('partition-a', 2, labels), ('partition-b', 3, 1-labels)],
                                     regions, split != 2, split == 2, settings, split_ids=split)
    assert result['families'] == ['backbone', 'sidechain', 'mixed']
    assert result['source_partition_ids'] == ['partition-a', 'partition-b']
    assert {c.partition_id for c in result['candidates']} <= {'partition-a', 'partition-b'}


def test_holdout_only_region_does_not_create_candidate():
    ft, labels, regions, split = frame_fixture()
    regions = regions.copy()
    regions[split == 2] = 8
    result = run(ft, labels, regions, split)
    assert not any(c.region == 8 for c in result['candidates'])


def test_cv_only_sidechain_signal_is_rejected_by_same_search_guard():
    ft, labels, regions, split = frame_fixture(cv_only=True)
    result = run(ft, labels, regions, split)
    assert result['null_gate']['status'] == 'no_real_candidate'
    assert result['candidates'] == []


def test_constant_chi_has_no_candidate_and_trapped_null_is_unavailable():
    ft, labels, regions, split = frame_fixture(constant_chi=True)
    result = run(ft, labels, regions, split)
    assert result['null_gate']['status'] == 'no_real_candidate'
    assert result['candidates'] == []

    ft, labels, regions, split = frame_fixture()
    ft = replace(ft, phase=np.arange(len(labels)).astype(str), replica=np.zeros(len(labels), dtype=np.int64))
    result = run(ft, labels, regions, split)
    assert result['null_gate']['status'] == 'null_uninformative_trapped_lineages'
    assert result['null_gate']['passed'] is False
    assert result['candidates']


def test_invalid_active_features_are_not_dropped_silently():
    ft, labels, regions, split = frame_fixture()
    ft.chi[0, 0] = np.nan
    result = run(ft, labels, regions, split)
    assert result['status'] == 'invalid_local_features'


def test_run_discovery_uses_winning_k2_partition_for_placement_even_when_k4_passes(monkeypatch):
    ft, labels, regions, split = frame_fixture()
    frozen = object()
    labels_k4 = 1 - labels
    part = SimpleNamespace(status='ok', choice=SimpleNamespace(k=4, table=[]), frozen=frozen,
                           bins=regions, hidden_fraction=.6, co_occurrence=[], lineage_info=.2,
                           per_k=[{'k': 2, 'labels': labels, 'triggered': True, 'hidden_fraction': .6,
                                   'co_occurrence': [], 'lineage_info': .2},
                                  {'k': 4, 'labels': labels_k4, 'triggered': True, 'hidden_fraction': .6,
                                   'co_occurrence': [], 'lineage_info': .2}],
                           conditioning={'selected': ['cv1']}, n_cond_bins=2)
    partition_id = pipeline._partition_identity(2, labels, regions, split != 2, part.conditioning)
    candidate = SimpleNamespace(partition_id=partition_id, region=0, pair=(0, 1), family='sidechain',
                                coefficients=np.ones(ft.tors.shape[1] + ft.chi.shape[1]),
                                weights=np.ones(ft.tors.shape[1] + ft.chi.shape[1]), c=.1,
                                local_gain=.2, prevalence=.5, score=.1, offset=.2, scale=1.3,
                                values=lambda X: X[:, 0])
    observed = {}
    monkeypatch.setattr(pipeline, 'search_local_candidates', lambda *a, **k: {
        'status': 'ok', 'candidates': [candidate], 'chosen': candidate,
        'candidate_summaries': [{'partition_id': partition_id}], 'null_gate': {'passed': True}})
    monkeypatch.setattr(pipeline, 'emit_local_model', lambda *a, **k: SimpleNamespace(model_sha256='model'))
    def place(z, lab, *a, **kwargs):
        observed.update(labels=np.asarray(lab).copy(), kwargs=kwargs, z=np.asarray(z).copy())
        return {'n_states': 1, 'n_candidates': 1, 'n_eligible': 1, 'skipped_states': [],
                'selection_log': [], 'chosen': [{'state_id': 0, 'c3': 0.1, 'k3': 2.0}]}
    monkeypatch.setattr(pipeline, 'place_workers', place)
    monkeypatch.setattr(pipeline, 'fit_partition', lambda *a, **k: part)
    result = pipeline.run_discovery(ft, train=split != 2, holdout=split == 2,
                                    settings=AuxDiscoverySettings(feature_space='sidechain',
                                                                  l1_c_grid=(.1, 1.), n_null_z3=2),
                                    full_topology=None, k3_max=3, epoch=2)
    assert result.status == 'ok'
    assert result.local_candidate.partition_id == result.report['local_candidate']['partition_id']
    assert result.source_partition_k == 2
    np.testing.assert_array_equal(result.source_partition_labels, labels)
    np.testing.assert_array_equal(observed['labels'], labels)
    assert observed['kwargs']['k_labels'] == 2
    assert observed['kwargs']['local_support']['pair'] == (0, 1)
    assert observed['kwargs']['local_support']['region'] == 0
    np.testing.assert_array_equal(observed['z'], candidate.values(np.column_stack((ft.tors, ft.chi))))


def test_local_discovery_returns_no_worker_without_placement(monkeypatch):
    ft, labels, regions, split = frame_fixture()
    part = SimpleNamespace(status='ok', choice=SimpleNamespace(k=2, table=[]), frozen=object(), bins=regions,
                           hidden_fraction=.6, co_occurrence=[], lineage_info=.2,
                           per_k=[{'k': 2, 'labels': labels, 'triggered': True, 'hidden_fraction': .6,
                                   'co_occurrence': [], 'lineage_info': .2}],
                           conditioning={'selected': ['cv1']}, n_cond_bins=2)
    partition_id = pipeline._partition_identity(2, labels, regions, split != 2, part.conditioning)
    candidate = SimpleNamespace(partition_id=partition_id, region=0, pair=(0, 1), family='sidechain',
                                coefficients=np.ones(ft.tors.shape[1] + ft.chi.shape[1]),
                                weights=np.ones(ft.tors.shape[1] + ft.chi.shape[1]), c=.1,
                                local_gain=.2, prevalence=.5, score=.1, offset=0., scale=1.,
                                values=lambda X: X[:, 0])
    monkeypatch.setattr(pipeline, 'search_local_candidates', lambda *a, **k: {
        'status': 'ok', 'candidates': [candidate], 'chosen': candidate,
        'candidate_summaries': [], 'null_gate': {'passed': True}})
    monkeypatch.setattr(pipeline, 'emit_local_model', lambda *a, **k: SimpleNamespace(model_sha256='model'))
    monkeypatch.setattr(pipeline, 'place_workers', lambda *a, **k: {
        'n_states': 1, 'n_candidates': 0, 'n_eligible': 0, 'skipped_states': [],
        'selection_log': [], 'chosen': []})
    monkeypatch.setattr(pipeline, 'fit_partition', lambda *a, **k: part)
    result = pipeline.run_discovery(ft, train=split != 2, holdout=split == 2,
                                    settings=AuxDiscoverySettings(feature_space='sidechain'),
                                    full_topology=None, k3_max=3, epoch=2)
    assert result.status == 'no_worker' and result.placement['chosen'] == []


def test_local_model_emits_exact_projection_coefficients_offset_and_scale(monkeypatch):
    from gareus.auxiliary_cv.sidechain_core import Projection

    ft, _labels, _regions, _split = frame_fixture()
    ft = replace(ft, tors=ft.tors[:, :2], tors_theta_iupac=np.ones((ft.n, 1)), chi=np.empty((ft.n, 0)),
                 sidechain_dictionary=SimpleNamespace(version='dictionary-v1', system_digest='system',
                                                     topology_digest='topology', atom_keys=(), exclusions=(),
                                                     torsions=()),
                 definition=SimpleNamespace(phi_quads=np.array([[0, 1, 2, 3]]), psi_quads=np.empty((0, 4), int),
                                            phi_labels=('phi_A1',), psi_labels=(), tors_labels=('tors_A',)))
    residue0 = SimpleNamespace(index=0, id='0', name='ALA', insertionCode='', chain=SimpleNamespace(id='A'))
    residue1 = SimpleNamespace(index=1, id='1', name='ALA', insertionCode='', chain=SimpleNamespace(id='A'))
    topology = SimpleNamespace(atoms=lambda: [SimpleNamespace(index=i, residue=(residue0 if i < 2 else residue1))
                                               for i in range(4)])
    import mdtraj as md
    from gareus.adaptive.aux_discovery import descriptors
    production_definition = SimpleNamespace(tors_labels=('tors_A',), phi_quads=np.array([[0, 1, 2, 3]]),
                                            psi_quads=np.empty((0, 4), int), phi_labels=('phi_A1',),
                                            psi_labels=())
    monkeypatch.setattr(md.Topology, 'from_openmm', staticmethod(lambda _top: object()))
    monkeypatch.setattr(descriptors, 'descriptor_definition', lambda _top: production_definition)
    monkeypatch.setattr('gareus.auxiliary_cv.atom_mapping.topology_metadata',
                        lambda _top: ((('A', '0', '', 'ALA', 'N', 'N'),), ((0, 1),)))
    captured = {}
    class Model:
        @classmethod
        def from_mapping(cls, raw):
            captured['raw'] = raw
            captured['projection'] = Projection(tuple(Primitive(tuple(tuple(q) for q in f['orbit']), f['trig'],
                                                                  f['harmonic'], f['sign'])
                                                       for f in raw['features']), raw['coefficients'],
                                                raw['offset'], raw['scale'])
            return SimpleNamespace(projection=captured['projection'], model_sha256='model')
    from gareus.auxiliary_cv.sidechain_core import Primitive
    monkeypatch.setattr('gareus.auxiliary_cv.sidechain_model.SidechainModel', Model)
    candidate = SimpleNamespace(partition_id='partition', region=1, pair=(0, 1), family='backbone',
                                coefficients=np.array([.7, -.2]), offset=.4, scale=1.3,
                                values=lambda X: (X @ np.array([.7, -.2]) + .4) / 1.3)
    model = pipeline.emit_local_model(candidate, ft, topology, epoch=3)
    assert model.model_sha256 == 'model'
    assert captured['projection'].coefficients == (.7, -.2)
    assert captured['projection'].offset == .4 and captured['projection'].scale == 1.3
    assert captured['raw']['units'] == 'dimensionless'
    assert captured['raw']['provenance']['partition_id'] == 'partition'


def test_auto_without_chi_uses_legacy_backbone_search(monkeypatch):
    ft, labels, regions, split = frame_fixture()
    ft = replace(ft, chi=None, sidechain_dictionary=None)
    choice = SimpleNamespace(k=2, table=[], labels=labels)
    part = SimpleNamespace(status='ok', choice=choice, frozen=object(), bins=regions,
                           hidden_fraction=.6, co_occurrence=[], lineage_info=.2,
                           per_k=[{'k': 2, 'labels': labels, 'triggered': True, 'hidden_fraction': .6,
                                   'co_occurrence': [], 'lineage_info': .2}],
                           conditioning={'selected': ['cv1']}, n_cond_bins=2)
    calls = []
    candidate = SimpleNamespace(groups=(0, 1), C=.1, summary=lambda: {'passed': True})
    monkeypatch.setattr(pipeline, 'fit_partition', lambda *a, **k: part)
    monkeypatch.setattr(pipeline, '_cooccurring_pairs', lambda *a, **k: [(0, 1)])
    monkeypatch.setattr(pipeline, 'search_z3_sources', lambda *a, **k: (calls.append('backbone') or candidate,
                                                                         [candidate]))
    monkeypatch.setattr(pipeline, 'z3_null_gate', lambda *a, **k: {'passed': True})
    monkeypatch.setattr(pipeline, 'emit_model', lambda *a, **k: SimpleNamespace(model_sha256='model'))
    monkeypatch.setattr(pipeline, 'z3_values', lambda *a, **k: np.zeros(ft.n))
    monkeypatch.setattr(pipeline, 'place_workers', lambda *a, **k: {
        'n_states': 1, 'n_candidates': 1, 'n_eligible': 1, 'skipped_states': [], 'selection_log': [],
        'chosen': [{'state_id': 0}]})
    monkeypatch.setattr(pipeline, 'search_local_candidates', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('auto without chi must delegate to legacy backbone search')))
    result = pipeline.run_discovery(ft, train=split != 2, holdout=split == 2,
                                    settings=AuxDiscoverySettings(feature_space='auto'),
                                    full_topology=None, k3_max=3, epoch=2)
    assert calls == ['backbone'] and result.status == 'ok'
