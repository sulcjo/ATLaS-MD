from __future__ import annotations

import math

import numpy as np


def test_gibbs_mh_acceptance_can_reject_heatbath_choice() -> None:
    from gareus.production import _gibbs_mh_acceptance_probability

    alpha = _gibbs_mh_acceptance_probability(
        delta_kj=4.0,
        beta=1.0,
        q_forward=0.80,
        q_reverse=0.10,
    )

    assert 0.0 < alpha < 1.0


def test_gibbs_window_proposal_distribution_normalizes_stay_candidate() -> None:
    from gareus.production import _gibbs_window_proposal_distribution

    bias_matrix_kj = np.asarray(
        [
            [0.0, 5.0, 7.0],
            [1.0, 0.0, 4.0],
            [6.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    holders = np.asarray([0, 1, 2], dtype=np.int64)

    proposal = _gibbs_window_proposal_distribution(
        beta=0.5,
        bias_matrix_kj=bias_matrix_kj,
        replica_index=0,
        current_window=0,
        replica_of_window=holders,
    )

    assert proposal["windows"].tolist() == [0, 1, 2]
    assert math.isclose(float(proposal["probabilities"].sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    stay_idx = int(np.where(proposal["windows"] == 0)[0][0])
    assert float(proposal["deltas_kj"][stay_idx]) == 0.0
    assert float(proposal["probabilities"][stay_idx]) > 0.0
