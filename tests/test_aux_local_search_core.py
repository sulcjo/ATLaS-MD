import numpy as np
import pytest

from gareus.adaptive.aux_discovery.local_search_core import fit_local_candidates, compare_null


def data():
    # Balanced in each fit/tune/confirmation block, with opposite regional effects.
    region = np.tile([0, 0, 1, 1], 180)
    x = np.tile([-1., 1., -1., 1.], 180)
    y = ((2*region-1)*x > 0).astype(int)
    features = np.c_[np.sin(x*np.pi/3), np.cos(x*np.pi/3)]
    split = np.repeat([0, 1, 2], 240)
    return features, y, region, split


def fit(X, y, region, split, families=None):
    return fit_local_candidates(X, y, region, split, partition_id='partition-two',
                                families=families or {'sidechain': (0, 1)}, c_grid=(.1, 1.),
                                min_train=20, min_holdout=20)


def test_opposite_regions_and_exported_projection():
    X, y, region, split = data()
    candidates = fit(X, y, region, split)
    assert len(candidates) == 2
    for c in candidates:
        assert c.partition_id == 'partition-two'
        assert c.local_gain > .6
        assert c.score > .29
        np.testing.assert_allclose(c.values(X), ((X-c.mean)/c.sd) @ c.weights/c.scale)
    assert candidates[0].coefficients[0] * candidates[1].coefficients[0] < 0


def test_holdout_does_not_change_projection():
    X, y, region, split = data()
    a = fit(X, y, region, split)
    changed = y.copy()
    changed[split == 2] = 1-changed[split == 2]
    b = fit(X, changed, region, split)
    for c, d in zip(sorted(a, key=lambda z:z.region), sorted(b, key=lambda z:z.region)):
        np.testing.assert_array_equal(c.coefficients, d.coefficients)
        assert c.scale == d.scale


def test_no_holdout_region_invents_training_candidate():
    X, y, region, split = data()
    region[split == 2] = 8
    assert fit(X, y, region, split) == []


def test_null_shared_across_families_and_trapped_status():
    X, y, region, split = data()
    seen = []
    def search(labels):
        seen.append(labels.copy())
        return fit(X, labels, region, split, {'sidechain': (0,1), 'mixed': (0,1)})
    # Phase-local identities cannot span training/holdout.
    lineage = np.array([f'{s}:{i//24}' for i,s in enumerate(split)])
    result = compare_null(y, lineage, np.arange(len(y)), search, n_null=3, seed=4)
    assert len(seen) == 4
    assert result['n_null_run'] == 3
    assert result['real_best'] > 0
    trapped = compare_null(y, y.astype(str), np.arange(len(y)), search, n_null=3)
    assert trapped['status'] == 'null_uninformative_trapped_lineages'
    assert trapped['passed'] is False


def test_degenerate_features_and_bad_input():
    X, y, region, split = data()
    assert fit(np.ones_like(X), y, region, split) == []
    X[0,0] = np.nan
    with pytest.raises(ValueError):
        fit(X,y,region,split)
    with pytest.raises(ValueError):
        compare_null(y, y, np.arange(len(y)), lambda _: [], n_null=0)


def test_partition_matrix_uses_the_same_shifts():
    X, y, region, split = data()
    labels = np.c_[y, 1-y]
    seen = []
    def search(lab):
        np.testing.assert_array_equal(lab[:,1], 1-lab[:,0])
        seen.append(lab.copy())
        return fit(X, lab[:,0], region, split)
    lineage = np.array([f'{s}:{i//24}' for i,s in enumerate(split)])
    compare_null(labels, lineage, np.arange(len(y)), search, n_null=2)
    assert len(seen) == 3


def test_fitted_candidate_compiles_into_projection_without_rescaling_twice():
    from gareus.auxiliary_cv.sidechain_core import Primitive, Projection
    X, y, region, split = data()
    theta = np.tile([-1., 1., -1., 1.], 180)*np.pi/3
    for c in fit(X, y, region, split):
        p = Projection((Primitive(((0,1,2,3),),'sin'), Primitive(((0,1,2,3),),'cos')),
                       tuple(c.coefficients), c.offset, c.scale)
        np.testing.assert_allclose(p.from_angles(theta[:,None]), c.values(X), atol=1e-14)


def test_null_refuses_nan_scores_instead_of_flooring_to_zero():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='nonfinite candidate score'):
        compare_null(np.array([0,1]), np.array(['a','a']), np.array([0,1]),
                     lambda _: [SimpleNamespace(score=np.nan)], n_null=1)
