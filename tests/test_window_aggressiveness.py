"""
Tests for N3: remove_block_radius must be >= 1 for all aggressiveness modes.
"""
import types
import pytest

# Import the aggressiveness settings function from gareus
try:
    from gareus.windows import _adaptive_window_aggressiveness_settings
    HAS_GAREUS = True
except ImportError:
    HAS_GAREUS = False


def _make_args(mode: str):
    """Return a minimal args namespace with adaptive_window_aggressiveness set."""
    return types.SimpleNamespace(adaptive_window_aggressiveness=mode)


@pytest.mark.skipif(not HAS_GAREUS, reason="gareus not importable")
@pytest.mark.parametrize("mode", ["conservative", "balanced", "aggressive", "very-aggressive"])
def test_remove_block_radius_is_at_least_1(mode):
    """Every aggressiveness mode must have remove_block_radius >= 1."""
    settings = _adaptive_window_aggressiveness_settings(_make_args(mode))
    radius = settings.get("remove_block_radius", 1)
    assert radius >= 1, (
        f"Mode '{mode}' has remove_block_radius={radius}; "
        "must be >= 1 to prevent adjacent window removal creating unvalidated gaps"
    )


@pytest.mark.skipif(not HAS_GAREUS, reason="gareus not importable")
def test_very_aggressive_block_radius_changed():
    """Regression: very-aggressive previously had remove_block_radius=0 (the bug)."""
    settings = _adaptive_window_aggressiveness_settings(_make_args("very-aggressive"))
    radius = settings.get("remove_block_radius", 1)
    assert radius >= 1, (
        f"very-aggressive remove_block_radius={radius}; "
        "the old value of 0 allowed adjacent removal — must be >= 1"
    )


# If the gareus import fails, fall back to a standalone test that
# verifies the logic directly
def test_block_radius_0_allows_adjacent_removal_bug():
    """
    Demonstrates why radius=0 is wrong: with radius=0, windows 2 and 3
    can both appear in removal candidates for a 5-window chain [1,2,3,4,5].
    """
    def _select_with_block_radius(candidates, radius):
        selected = []
        blocked = set()
        for c in candidates:
            if c in blocked:
                continue
            selected.append(c)
            for b in range(c - radius, c + radius + 1):
                blocked.add(b)
        return selected

    candidates = [2, 3]  # adjacent windows, both want to be removed
    selected_r0 = _select_with_block_radius(candidates, radius=0)
    selected_r1 = _select_with_block_radius(candidates, radius=1)

    assert len(selected_r0) == 2, "radius=0 allows both adjacent windows (the bug)"
    assert len(selected_r1) == 1, "radius=1 blocks the second adjacent window (the fix)"
