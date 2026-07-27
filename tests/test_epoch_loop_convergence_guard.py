import pytest

from gareus.adaptive_production import _epoch_loop_missing_convergence


@pytest.mark.parametrize(
    "epoch,max_epochs,gate_converged,require_convergence_before_final,expected",
    [
        # Mid-budget: never raises regardless of convergence or policy.
        (0, 3, True, False, False),
        (0, 3, False, False, False),
        (0, 3, False, True, False),
        (1, 3, False, True, False),
        # Last epoch (epoch == max_epochs - 1): only raises when NOT
        # converged and the policy requires convergence before final.
        (2, 3, True, True, False),   # converged on the last epoch -> no raise
        (2, 3, False, True, True),   # never converged, policy demands it -> raise
        (2, 3, False, False, False),  # never converged but policy allows it -> no raise
        (2, 3, True, False, False),
        # max_epochs == 1: the only epoch is also the last epoch.
        (0, 1, True, True, False),
        (0, 1, False, True, True),
        (0, 1, False, False, False),
        # start_epoch on a resumed run doesn't change which epoch is "last" --
        # the check is an absolute epoch+1 >= max_epochs comparison.
        (4, 5, False, True, True),
        (4, 5, True, True, False),
    ],
)
def test_epoch_loop_missing_convergence(epoch, max_epochs, gate_converged, require_convergence_before_final, expected):
    assert _epoch_loop_missing_convergence(epoch, max_epochs, gate_converged, require_convergence_before_final) is expected
