"""Guard against a stale `epoch_window_map.csv` silently mis-attributing every
sample of an adaptive-production phase to the wrong umbrella state.

Root cause (real run `RUNS/chignolin_6`, full writeup in
`docs/chignolin_6_low_ess_root_cause.md`): `--us-auto-drop-bad-windows` prunes
umbrella windows *post-pull* and `gareus/production.py` renumbers every
window-indexed array around the survivors (0..N-1), but the phase's own
`epoch_window_map.csv` -- written earlier by the state registry as an IDENTITY
map over all *active* states -- is never rewritten. The MBAR union loader
trusts that map's row count blindly, so every local window index at or after
the first dropped index is attributed to the wrong state: on chignolin_6 that
was ~1.82M of 12.1M samples (15%), including 856,767 samples scored against a
sign-flipped CV2 centre (2,012 kT of fabricated bias) and one state (20) that
was never actually sampled at all.

Ground truth these fixtures are modelled on (chignolin_6):

| phase                  | map rows | real windows | dropped (pre-drop local) | correct repair          |
|------------------------|----------|--------------|--------------------------|-------------------------|
| final/baseline         | 27       | 24           | [20, 22, 23]             | 20->21, 21->24, 22->25, 23->26 |
| epoch_001/baseline     | 24       | 23           | [20]                     | 20->21, 21->22, 22->23  |
| epoch_001/topup_004    | 3        | 2            | (none of its own)        | 0->21, 1->22            |
| final/topup_003        | 2        | 1            | (none of its own)        | 0->21                   |

The two topup phases carry no drop record of their own -- theirs is inherited
from the sibling `baseline` sub-run of the same parent phase, which is where
the auto-drop actually ran.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gareus.mbar_analysis.loaders_adaptive import (
    _PHANTOM_EPOCH_WINDOW_BASE,
    _dropped_state_ids_for_phase,
    _phase_dropped_post_pull_windows,
    _phase_recorded_window_count,
    _validate_and_repair_epoch_window_map,
)


# --- fixture builders -------------------------------------------------------

def _write_map(phase_dir: Path, rows: list) -> list:
    """Write an epoch_window_map.csv from (epoch_window, state_id, c1, c2) tuples.

    Column set deliberately mirrors the real chignolin_6 maps, which carry
    *no* primary_k/secondary_k columns (those fall back to the registry
    per-field in `_epoch_bias_param_vectors`).
    """
    phase_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with (phase_dir / 'epoch_window_map.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch_window', 'state_id', 'primary_center', 'secondary_center'])
        w.writeheader()
        for ew, sid, c1, c2 in rows:
            row = {'epoch_window': str(ew), 'state_id': str(sid),
                   'primary_center': repr(c1), 'secondary_center': repr(c2)}
            w.writerow(row)
            out.append(row)
    return out


def _identity_map(phase_dir: Path, n: int, sec_centers: list) -> list:
    return _write_map(phase_dir, [(i, i, 0.05 * (i // 10), sec_centers[i]) for i in range(n)])


def _write_meta(phase_dir: Path, dropped=None, dropped_section: str = 'window_metadata',
                n_windows=None, n_windows_A=None) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    if dropped is not None:
        meta[dropped_section] = {'dropped_post_pull_bad_windows': list(dropped)}
    if n_windows_A is not None:
        meta['windows_A'] = [0.0] * int(n_windows_A)
    if meta:
        (phase_dir / 'gareus_metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    if n_windows is not None:
        (phase_dir / 'analysis_metadata_validation.json').write_text(
            json.dumps({'n_windows': int(n_windows), 'status': 'ok'}), encoding='utf-8')


def _mapping(rows: list) -> dict:
    """Exactly what `_load_epoch_task` builds from the returned rows.

    Faithful mirror of the real caller (gareus/mbar_analysis/
    loaders_union_parquet.py), phantom keys included -- see `_real_mapping`.
    """
    return {int(r['epoch_window']): int(r['state_id']) for r in rows}


def _real_mapping(rows: list) -> dict:
    """The part of `_mapping` a sample can actually reach: local windows only.

    A repaired map keeps the *dropped* states' rows (so the state-keyed
    native-window-param parser downstream still sees this phase's own params
    for them, instead of silently falling back to the possibly-recentered
    final registry) and parks them at `_PHANTOM_EPOCH_WINDOW_BASE`+ -- far
    beyond any real local window index, so no sample can resolve to one.
    """
    return {ew: sid for ew, sid in _mapping(rows).items()
            if ew < _PHANTOM_EPOCH_WINDOW_BASE}


def _phantom_state_ids(rows: list) -> set:
    """The state_ids of the retained-but-unreachable (dropped) rows."""
    return {sid for ew, sid in _mapping(rows).items()
            if ew >= _PHANTOM_EPOCH_WINDOW_BASE}


# 27 secondary centres in the shape of the real run: the CV2 ladder, with the
# three late bridge states (24-26) out at the extremes.
_SEC27 = [-0.906, 0.657, 0.350, 0.374, -0.258, -0.370, 0.144, 0.341, -0.298, -0.494,
          0.307, 0.900, 1.121, -0.700, 0.120, -0.310, -0.171, 0.402, -1.153, 0.170,
          -1.628, -1.438, 1.407, 1.759, -0.980, 1.220, 0.640]


# --- metadata readers -------------------------------------------------------

def test_phase_dropped_post_pull_windows_reads_window_metadata(tmp_path):
    """final/baseline's real layout: the record sits under window_metadata."""
    _write_meta(tmp_path, dropped=[20, 22, 23])
    assert _phase_dropped_post_pull_windows(tmp_path) == [20, 22, 23]


def test_phase_dropped_post_pull_windows_reads_secondary_cv_section(tmp_path):
    """epoch_001/baseline's real layout: window_metadata has no record at all;
    the only copy lives under gareus_metadata.json's `secondary_cv` block
    (production.py writes it into both dicts, but only one survives per phase)."""
    _write_meta(tmp_path, dropped=[20], dropped_section='secondary_cv')
    assert _phase_dropped_post_pull_windows(tmp_path) == [20]


def test_phase_dropped_post_pull_windows_none_without_record(tmp_path):
    _write_meta(tmp_path, n_windows=20)
    assert _phase_dropped_post_pull_windows(tmp_path) is None
    assert _phase_dropped_post_pull_windows(tmp_path / 'nope') is None


def test_phase_recorded_window_count_prefers_validation_json(tmp_path):
    _write_meta(tmp_path, n_windows=24, n_windows_A=27)
    assert _phase_recorded_window_count(tmp_path) == 24


def test_phase_recorded_window_count_falls_back_to_windows_A(tmp_path):
    _write_meta(tmp_path, n_windows_A=23)
    assert _phase_recorded_window_count(tmp_path) == 23


def test_phase_recorded_window_count_none_without_metadata(tmp_path):
    assert _phase_recorded_window_count(tmp_path) is None


# --- consistent maps are left completely alone ------------------------------

def test_consistent_map_returned_unchanged(tmp_path):
    rows = _identity_map(tmp_path, 20, _SEC27)
    _write_meta(tmp_path, n_windows=20)
    out, notes = _validate_and_repair_epoch_window_map(
        tmp_path, rows, window_ids=np.arange(20), cv2=np.array(_SEC27[:20]))
    assert _mapping(out) == _mapping(rows)
    assert notes == []


def test_consistent_map_without_any_metadata_is_untouched(tmp_path):
    """The configuration every pre-existing loader fixture in this suite uses:
    no gareus_metadata.json, no analysis_metadata_validation.json -- the real
    window count can only come from the samples themselves. Detection must not
    fire here."""
    rows = _write_map(tmp_path, [(0, 0, 0.0, -0.9), (1, 1, 0.0, 0.65), (2, 2, 0.0, 0.35)])
    out, notes = _validate_and_repair_epoch_window_map(
        tmp_path, rows, window_ids=np.array([0, 0, 1, 1, 2, 2]),
        cv2=np.array([-0.9, -0.9, 0.65, 0.65, 0.35, 0.35]))
    assert _mapping(out) == {0: 0, 1: 1, 2: 2}
    assert notes == []


def test_unsampled_trailing_window_is_not_a_mismatch(tmp_path):
    """A phase whose last window produced no samples is NOT corruption: the
    authoritative recorded window count still matches the map, so the map must
    be accepted as-is rather than "repaired" into a shifted mapping."""
    rows = _identity_map(tmp_path, 5, _SEC27)
    _write_meta(tmp_path, n_windows=5)
    out, notes = _validate_and_repair_epoch_window_map(
        tmp_path, rows, window_ids=np.array([0, 1, 2, 3]), cv2=np.array(_SEC27[:4]))
    assert _mapping(out) == _mapping(rows)
    assert notes == []


# --- repair from the phase's own drop record --------------------------------

def test_final_baseline_shape_repaired_from_own_drop_record(tmp_path):
    """chignolin_6 final/baseline: 27 identity map rows, 24 windows really run,
    dropped local [20, 22, 23]."""
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, dropped=[20, 22, 23], n_windows=24)
    out, notes = _validate_and_repair_epoch_window_map(phase, rows)
    m = _real_mapping(out)
    assert len(m) == 24
    assert sorted(m) == list(range(24))                     # renumbered 0..N-1
    assert _phantom_state_ids(out) == {20, 22, 23}          # dropped rows kept, unreachable
    assert {i: m[i] for i in range(20)} == {i: i for i in range(20)}
    assert {20: m[20], 21: m[21], 22: m[22], 23: m[23]} == {20: 21, 21: 24, 22: 25, 23: 26}
    # centres travel with the state they belong to
    assert float(out[20]['secondary_center']) == pytest.approx(_SEC27[21])
    assert float(out[23]['secondary_center']) == pytest.approx(_SEC27[26])
    assert len(notes) == 1
    assert 'final/baseline' in notes[0]
    assert '27' in notes[0] and '24' in notes[0]
    assert '20->21' in notes[0] and '23->26' in notes[0]
    assert 'chignolin_6_low_ess_root_cause.md' in notes[0]


def test_epoch001_baseline_shape_repaired_and_cv2_fingerprint_confirms(tmp_path):
    """chignolin_6 epoch_001/baseline: 24 map rows, 23 windows, state 20 dropped
    (record under `secondary_cv`). The sampled CV2 means are the independent
    evidence the repair is right -- they match the *true* state's centre, which
    is exactly how the original investigation identified the shift."""
    phase = tmp_path / 'epoch_001' / 'baseline'
    rows = _identity_map(phase, 24, _SEC27)
    _write_meta(phase, dropped=[20], dropped_section='secondary_cv', n_windows=23)
    true_states = list(range(20)) + [21, 22, 23]
    window_ids = np.repeat(np.arange(23), 4)
    cv2 = np.repeat(np.array([_SEC27[s] for s in true_states]), 4)
    out, notes = _validate_and_repair_epoch_window_map(phase, rows, window_ids=window_ids, cv2=cv2)
    m = _real_mapping(out)
    assert len(m) == 23
    assert _phantom_state_ids(out) == {20}
    assert {20: m[20], 21: m[21], 22: m[22]} == {20: 21, 21: 22, 22: 23}
    assert 'epoch_001/baseline' in notes[0]
    assert 'cv2' in notes[0].lower()


# --- repair of a topup, from the sibling baseline's drop record -------------

def test_topup_repaired_from_sibling_baseline_drop_record(tmp_path):
    """chignolin_6 epoch_001/topup_004: its own map is a *compacted* subset
    (local 0,1,2 -> states 20,21,22) with 2 windows really run, and it carries
    no drop record of its own. The record lives on the sibling baseline of the
    same parent epoch, whose own map translates it to dropped state_id 20."""
    epoch = tmp_path / 'epoch_001'
    _identity_map(epoch / 'baseline', 24, _SEC27)
    _write_meta(epoch / 'baseline', dropped=[20], dropped_section='secondary_cv', n_windows=23)
    topup = epoch / 'topup_004_75786000'
    rows = _write_map(topup, [(0, 20, 0.114, _SEC27[20]),
                              (1, 21, 0.152, _SEC27[21]),
                              (2, 22, 0.153, _SEC27[22])])
    _write_meta(topup, n_windows=2)
    out, notes = _validate_and_repair_epoch_window_map(topup, rows)
    assert _real_mapping(out) == {0: 21, 1: 22}
    assert _phantom_state_ids(out) == {20}
    assert 'topup_004' in notes[0]
    assert 'baseline' in notes[0]          # names where the drop record came from


def test_final_topup_repaired_from_sibling_baseline_drop_record(tmp_path):
    """chignolin_6 final/topup_003: 2 map rows, 1 window; only state 20 of the
    sibling's dropped set {20, 22, 23} appears in this map, so exactly one row
    is removed."""
    final = tmp_path / 'final'
    _identity_map(final / 'baseline', 27, _SEC27)
    _write_meta(final / 'baseline', dropped=[20, 22, 23], n_windows=24)
    topup = final / 'topup_003_478000'
    rows = _write_map(topup, [(0, 20, 0.114, _SEC27[20]), (1, 21, 0.152, _SEC27[21])])
    _write_meta(topup, n_windows=1)
    out, notes = _validate_and_repair_epoch_window_map(topup, rows)
    assert _real_mapping(out) == {0: 21}
    assert _phantom_state_ids(out) == {20}
    assert notes


def test_sibling_drop_record_is_translated_through_a_non_identity_map(tmp_path):
    """The drop record holds *pre-drop local indices*, not state_ids -- they
    only coincide when the map is an identity map. A registry with a retired
    state hands out a non-identity map (chignolin_6's real topup maps look like
    11->12, 12->15), so the record must be read as a position into the map it
    was recorded against."""
    epoch = tmp_path / 'epoch_002'
    _write_map(epoch / 'baseline', [(0, 3, 0.0, _SEC27[3]), (1, 7, 0.0, _SEC27[7]),
                                    (2, 9, 0.0, _SEC27[9])])
    _write_meta(epoch / 'baseline', dropped=[1], n_windows=2)
    got = _dropped_state_ids_for_phase(epoch / 'baseline')
    assert got is not None
    dropped_ids, src = got
    assert dropped_ids == {7}                      # position 1 -> state_id 7, not state 1
    assert Path(src).name == 'baseline'


# --- fail-closed branches ---------------------------------------------------

def test_mismatch_without_any_drop_record_fails_closed(tmp_path):
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, n_windows=24)
    with pytest.raises(ValueError) as exc:
        _validate_and_repair_epoch_window_map(phase, rows)
    msg = str(exc.value)
    assert 'final/baseline' in msg
    assert '27' in msg and '24' in msg
    assert 'chignolin_6_low_ess_root_cause.md' in msg


def test_mismatch_detected_from_samples_alone_without_metadata(tmp_path):
    """The check that needs no metadata at all: the map claims more windows
    than the phase's own Parquet samples ever reference."""
    phase = tmp_path / 'epoch_001' / 'baseline'
    rows = _identity_map(phase, 24, _SEC27)
    with pytest.raises(ValueError) as exc:
        _validate_and_repair_epoch_window_map(
            phase, rows, window_ids=np.repeat(np.arange(23), 2))
    assert '24' in str(exc.value) and '23' in str(exc.value)


def test_drop_record_that_does_not_reconcile_the_counts_fails_closed(tmp_path):
    """A record that removes the wrong number of rows must never be applied --
    the surviving count is the only thing proving the record belongs to this
    map."""
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, dropped=[20], n_windows=24)      # removes 1, needs 3
    with pytest.raises(ValueError) as exc:
        _validate_and_repair_epoch_window_map(phase, rows)
    assert '26' in str(exc.value) or 'reconcile' in str(exc.value).lower()


def test_repair_rejected_when_cv2_fingerprint_gets_worse(tmp_path):
    """Guards the one genuinely unsound corner of inheriting a sibling's drop
    record: a dropped window is only phantom *for the sub-run that dropped it*
    (chignolin_6's final/topup_001 legitimately re-pulled state 23, which
    final/baseline had dropped). If the deletion the record implies moves the
    sampled CV2 means *away* from their assigned restraint centres, the
    inference was wrong -- refuse rather than repair."""
    final = tmp_path / 'final'
    _write_map(final / 'baseline', [(0, 20, 0.114, _SEC27[20]), (1, 21, 0.152, _SEC27[21])])
    _write_meta(final / 'baseline', dropped=[0], n_windows=1)
    topup = final / 'topup_009_1000'
    rows = _write_map(topup, [(0, 20, 0.114, _SEC27[20]), (1, 21, 0.152, _SEC27[21])])
    _write_meta(topup, n_windows=1)
    # The single sampled window really *is* state 20 (its cv2 sits on state
    # 20's centre), so deleting state 20's row would mis-attribute it to 21.
    with pytest.raises(ValueError) as exc:
        _validate_and_repair_epoch_window_map(
            topup, rows, window_ids=np.zeros(8, dtype=int),
            cv2=np.full(8, _SEC27[20]))
    assert 'cv2' in str(exc.value).lower()


def test_samples_beyond_the_repaired_map_fail_closed(tmp_path):
    """If the samples reference a local window index the repaired map no longer
    covers, those samples would be silently dropped (remapped to -1) -- so the
    recorded window count must be wrong too, and we must not proceed."""
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, dropped=[20, 22, 23], n_windows=24)
    with pytest.raises(ValueError) as exc:
        _validate_and_repair_epoch_window_map(
            phase, rows, window_ids=np.arange(26))     # 26 > 24 surviving windows
    assert '26' in str(exc.value) or 'beyond' in str(exc.value).lower()


def test_env_override_downgrades_unrepairable_mismatch_to_a_warning(monkeypatch, tmp_path):
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, n_windows=24)
    monkeypatch.setenv('GAREUS_ALLOW_STALE_WINDOW_MAP', '1')
    out, notes = _validate_and_repair_epoch_window_map(phase, rows)
    assert _mapping(out) == _mapping(rows)             # loaded with the stale map, as asked
    assert notes and 'GAREUS_ALLOW_STALE_WINDOW_MAP' in notes[0]


def test_env_override_does_not_preempt_a_real_repair(monkeypatch, tmp_path):
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, dropped=[20, 22, 23], n_windows=24)
    monkeypatch.setenv('GAREUS_ALLOW_STALE_WINDOW_MAP', '1')
    out, _notes = _validate_and_repair_epoch_window_map(phase, rows)
    assert _real_mapping(out)[20] == 21 and len(_real_mapping(out)) == 24


# --- A10: mid-campaign secondary-CV redefinition ----------------------------

def _write_manifest(phase_dir: Path, secondary_cv: str) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / 'run_manifest.json').write_text(
        json.dumps({'resolved_args': {'secondary_cv': secondary_cv, 'temperature_k': 300.0}}),
        encoding='utf-8')


def test_secondary_cv_regime_change_note_is_none_for_one_regime(tmp_path):
    from gareus.mbar_analysis.loaders_union_parquet import _secondary_cv_regime_change_note
    for name in ('epoch_000', 'epoch_001'):
        _write_manifest(tmp_path / name, 'tica-linear')
    assert _secondary_cv_regime_change_note(
        [tmp_path / 'epoch_000', tmp_path / 'epoch_001'], [10, 20]) is None


def test_secondary_cv_regime_change_note_is_none_without_manifests(tmp_path):
    from gareus.mbar_analysis.loaders_union_parquet import _secondary_cv_regime_change_note
    assert _secondary_cv_regime_change_note([tmp_path / 'epoch_000'], [10]) is None


def test_secondary_cv_regime_change_note_names_both_regimes_and_phases(tmp_path):
    """chignolin_6: epoch_000 ran torsion-pca, epoch_001+/final ran tica-linear.
    Per-epoch native window params keep u_nk internally consistent within each
    epoch, but state k is then not one Hamiltonian across the switch."""
    from gareus.mbar_analysis.loaders_union_parquet import _secondary_cv_regime_change_note
    _write_manifest(tmp_path / 'epoch_000', 'torsion-pca')
    _write_manifest(tmp_path / 'epoch_001' / 'baseline', 'tica-linear')
    _write_manifest(tmp_path / 'final' / 'baseline', 'tica-linear')
    note = _secondary_cv_regime_change_note(
        [tmp_path / 'epoch_000', tmp_path / 'epoch_001' / 'baseline', tmp_path / 'final' / 'baseline'],
        [3130000, 4050000, 1000000])
    assert note is not None
    assert 'torsion-pca' in note and 'tica-linear' in note
    assert 'epoch_000' in note and 'epoch_001/baseline' in note
    assert '3,130,000' in note                      # per-regime sample counts
    assert 'Hamiltonian' in note


# --- end-to-end through the real union loader -------------------------------

def _write_multiwindow_epoch(epoch_dir: Path, windows: list, samples: list) -> None:
    """Real Parquet samples + window snapshot for a multi-window phase.

    `windows` is a list of (center1, k1, center2, k2) in local window order;
    `samples` a list of (step, replica, window_id, cv1, cv2). Same low-level
    writers (ParquetSampleWriter/SegmentRegistry/WindowSnapshot) as the other
    loader fixtures in this suite, generalized past their single window.
    """
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    epoch_dir.mkdir(parents=True, exist_ok=True)
    reg = SegmentRegistry(epoch_dir)
    seg_id = reg.open_segment('run_001', None, len(windows))
    WindowSnapshot(epoch_dir).snapshot(
        seg_id,
        [{'window_id': i, 'center1': c1, 'k1': k1, 'center2': c2, 'k2': k2}
         for i, (c1, k1, c2, k2) in enumerate(windows)],
        cv1_type='contacts', cv2_type='tica-linear',
    )
    writer = ParquetSampleWriter(epoch_dir / 'samples' / seg_id, flush_rows=1000)
    max_step = 0
    for step, replica, wid, cv1, cv2 in samples:
        writer.write_sample(step, replica, wid, cv1, cv2, -100.0, 5.0, 2.0, 0.4)
        max_step = max(max_step, step)
    writer.close()
    reg.close_segment(seg_id, end_step=max_step)


def _write_union_registry(adaptive_dir: Path, rows: list) -> None:
    adaptive_dir.mkdir(parents=True, exist_ok=True)
    with (adaptive_dir / 'final_registry_used_for_mbar.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['state_id', 'usable_for_mbar', 'primary_center',
                                          'primary_k', 'secondary_center', 'secondary_k',
                                          'burnin_steps'])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _reg_row(sid: int, c1: float, c2: float) -> dict:
    return {'state_id': str(sid), 'usable_for_mbar': 'True', 'primary_center': repr(c1),
            'primary_k': '44.3', 'secondary_center': repr(c2), 'secondary_k': '148.5',
            'burnin_steps': '0'}


def _write_temperature(*dirs) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / 'run_args.json').write_text(json.dumps({'temperature_k': 300.0}), encoding='utf-8')


def test_load_parquet_adaptive_union_repairs_a_stale_map_end_to_end(tmp_path):
    """The payoff case: an already-finished run with a stale map re-analyses
    correctly, with the samples landing in the states that really generated
    them and a warning in meta['load_notes'] (which analyze() folds into
    pmf_summary.json's warnings)."""
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union

    adaptive = tmp_path / 'adaptive_production'
    _write_temperature(tmp_path, adaptive)
    centers2 = [-0.906, -1.628, -1.438, 1.407]           # states 0..3
    _write_union_registry(adaptive, [_reg_row(s, 0.05 * s, centers2[s]) for s in range(4)])

    epoch = adaptive / 'epoch_000'
    # Three windows really ran, holding what were states 0, 2 and 3: state 1
    # was dropped post-pull, and the surviving windows renumbered 0..2.
    true_states = [0, 2, 3]
    _write_multiwindow_epoch(
        epoch,
        windows=[(0.05 * s, 44.3, centers2[s], 148.5) for s in true_states],
        samples=[(i * 50, 0, w, 0.05 * true_states[w] + 0.001 * i, centers2[true_states[w]])
                 for w in range(3) for i in range(6)],
    )
    _identity_map(epoch, 4, centers2)                    # the stale 4-row map
    _write_meta(epoch, dropped=[1], n_windows=3)

    d = load_parquet_adaptive_union(adaptive)

    # state_id_to_k is identity here (registry states 0..3), so the k indices
    # are the state_ids: nothing may land in state 1, which never ran.
    assert sorted(set(np.asarray(d.window, dtype=np.int64).tolist())) == [0, 2, 3]
    notes = d.meta.get('load_notes') or []
    assert any('[stale window map]' in n for n in notes), notes
    assert any('1->2' in n and '2->3' in n for n in notes), notes
    # Each sample now sits on its own restraint centre, so its own state's
    # reduced bias is ~0 rather than the hundreds of kT the shift fabricated.
    own = d.u_nk[np.arange(d.cv.size), np.asarray(d.window, dtype=np.int64)]
    assert float(np.max(own)) < 1.0


def test_load_parquet_adaptive_union_fails_closed_on_unrepairable_stale_map(tmp_path):
    """Same layout, but with no drop record anywhere: refuse rather than
    silently mis-attribute."""
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union

    adaptive = tmp_path / 'adaptive_production'
    _write_temperature(tmp_path, adaptive)
    centers2 = [-0.906, -1.628, -1.438, 1.407]
    _write_union_registry(adaptive, [_reg_row(s, 0.05 * s, centers2[s]) for s in range(4)])
    epoch = adaptive / 'epoch_000'
    true_states = [0, 2, 3]
    _write_multiwindow_epoch(
        epoch,
        windows=[(0.05 * s, 44.3, centers2[s], 148.5) for s in true_states],
        samples=[(i * 50, 0, w, 0.05 * true_states[w], centers2[true_states[w]])
                 for w in range(3) for i in range(6)],
    )
    _identity_map(epoch, 4, centers2)
    _write_meta(epoch, n_windows=3)
    with pytest.raises(ValueError) as exc:
        load_parquet_adaptive_union(adaptive)
    assert 'chignolin_6_low_ess_root_cause.md' in str(exc.value)


def test_load_parquet_adaptive_union_warns_on_mid_campaign_cv2_redefinition(tmp_path):
    """A10 end-to-end: the regime-change warning must reach meta['load_notes'],
    the channel analyze() turns into pmf_summary.json warnings."""
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union

    adaptive = tmp_path / 'adaptive_production'
    _write_temperature(tmp_path, adaptive)
    _write_union_registry(adaptive, [_reg_row(0, 0.0, -0.906), _reg_row(1, 0.05, 0.657)])
    for i, (name, sid, regime) in enumerate([('epoch_000', 0, 'torsion-pca'),
                                             ('epoch_001', 1, 'tica-linear')]):
        ep = adaptive / name
        c2 = -0.906 if sid == 0 else 0.657
        _write_multiwindow_epoch(ep, windows=[(0.05 * sid, 44.3, c2, 148.5)],
                                 samples=[(j * 50, 0, 0, 0.05 * sid, c2) for j in range(5)])
        _write_map(ep, [(0, sid, 0.05 * sid, c2)])
        _write_meta(ep, n_windows=1)
        _write_manifest(ep, regime)

    d = load_parquet_adaptive_union(adaptive)
    notes = d.meta.get('load_notes') or []
    assert any('[cv2 regime change]' in n for n in notes), notes
    joined = ' '.join(notes)
    assert 'torsion-pca' in joined and 'tica-linear' in joined
    assert 'epoch_000' in joined and 'epoch_001' in joined


# --- repair from the phase's own surviving window table ---------------------
#
# `umbrella_explicit_windows.csv` is the window table the sampler actually ran
# (post-drop, renumbered 0..N-1) and it records each local window's own
# primary/secondary centre. Matching those centres back onto the map -- order
# preserving, since the survivors keep their relative order -- identifies the
# spurious rows directly, with no drop record needed at all. This is the repair
# source the root-cause writeup recommends, and it is the only one that covers
# the phases whose drop record is missing or does not explain the whole gap
# (real examples on disk: chignolin_5/final/baseline 30 map rows vs 29 windows
# with no record at all, chignolin_sigma3_2d/final/baseline 28 vs 23 with a
# 2-entry record).

def _write_explicit_windows(phase_dir: Path, centers: list) -> None:
    """Write an umbrella_explicit_windows.csv from (primary, secondary) pairs."""
    phase_dir.mkdir(parents=True, exist_ok=True)
    with (phase_dir / 'umbrella_explicit_windows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['window', 'primary_center', 'primary_k',
                                          'secondary_cv_center', 'secondary_cv_k_kcal_mol'])
        w.writeheader()
        for i, (c1, c2) in enumerate(centers):
            w.writerow({'window': str(i), 'primary_center': repr(c1), 'primary_k': '44.3',
                        'secondary_cv_center': repr(c2), 'secondary_cv_k_kcal_mol': '148.5'})


def test_phase_real_window_centers_reads_the_explicit_window_table(tmp_path):
    from gareus.mbar_analysis.loaders_adaptive import _phase_real_window_centers
    _write_explicit_windows(tmp_path, [(0.0, -0.9), (0.0654, 1.12)])
    got = _phase_real_window_centers(tmp_path)
    assert got == [(0.0, -0.9), (0.0654, 1.12)]
    assert _phase_real_window_centers(tmp_path / 'nope') is None


def test_repaired_from_explicit_window_table_without_any_drop_record(tmp_path):
    """The chignolin_sigma3_2d/final/baseline shape: a big shift, and a drop
    record that (if it exists at all) does not account for it. The surviving
    window table alone pins every local window to its real state."""
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 6, _SEC27)
    _write_meta(phase, n_windows=4)
    # states 0, 2, 4, 5 really ran (1 and 3 are phantom rows)
    real = [(0.0, _SEC27[0]), (0.0, _SEC27[2]), (0.0, _SEC27[4]), (0.0, _SEC27[5])]
    _write_explicit_windows(phase, real)
    out, notes = _validate_and_repair_epoch_window_map(phase, rows)
    assert _real_mapping(out) == {0: 0, 1: 2, 2: 4, 3: 5}
    assert _phantom_state_ids(out) == {1, 3}
    assert notes and 'umbrella_explicit_windows.csv' in notes[0]


def test_explicit_window_table_wins_over_a_disagreeing_drop_record(tmp_path):
    """Direct evidence of what ran beats an inferred drop record: the record
    here would delete the wrong row."""
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 4, _SEC27)
    _write_meta(phase, dropped=[0], n_windows=3)
    _write_explicit_windows(phase, [(0.0, _SEC27[0]), (0.0, _SEC27[1]), (0.0, _SEC27[3])])
    out, _notes = _validate_and_repair_epoch_window_map(phase, rows)
    assert _real_mapping(out) == {0: 0, 1: 1, 2: 3}
    assert _phantom_state_ids(out) == {2}      # NOT state 0, as the record claimed


def test_trailing_phantom_row_is_repaired_without_any_shift(tmp_path):
    """The chignolin_5/final/baseline shape: the one phantom row sits at the
    tail, so no sampled window was ever mis-attributed -- the map is still
    trimmed, and the note must say plainly that nothing shifted."""
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 4, _SEC27)
    _write_meta(phase, n_windows=3)
    _write_explicit_windows(phase, [(0.0, _SEC27[0]), (0.0, _SEC27[1]), (0.0, _SEC27[2])])
    out, notes = _validate_and_repair_epoch_window_map(phase, rows)
    assert _real_mapping(out) == {0: 0, 1: 1, 2: 2}
    assert _phantom_state_ids(out) == {3}
    assert notes and 'no local window changed state' in notes[0]
    # The note must NOT tell the reader an earlier run-level analysis is fine:
    # an unshifted phase says nothing about the joint MBAR solve it took part in.
    assert 'does NOT validate an earlier analysis of the run' in notes[0]


def test_explicit_window_table_of_the_wrong_length_is_ignored(tmp_path):
    """A table that disagrees with the phase's own recorded window count is not
    trusted -- fall through to the drop record instead of matching against it."""
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, dropped=[20, 22, 23], n_windows=24)
    _write_explicit_windows(phase, [(0.0, _SEC27[i]) for i in range(10)])   # stale/partial
    out, notes = _validate_and_repair_epoch_window_map(phase, rows)
    assert _real_mapping(out)[20] == 21 and len(_real_mapping(out)) == 24
    assert 'drop record' in notes[0] or 'dropped state_id' in notes[0]


def test_unmatchable_explicit_window_table_falls_through_to_fail_closed(tmp_path):
    """Centres that appear nowhere in the map (a table from a different phase)
    must not be force-fitted."""
    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 4, _SEC27)
    _write_meta(phase, n_windows=3)
    _write_explicit_windows(phase, [(9.0, 9.0), (9.1, 9.1), (9.2, 9.2)])
    with pytest.raises(ValueError):
        _validate_and_repair_epoch_window_map(phase, rows)


# --- dropped states' rows are RETAINED, only made unreachable ---------------
#
# The first fix round deleted the dropped rows outright. That compacted the
# local-index lookup correctly but also deleted those states' *per-epoch native
# window params*, which `_parse_epoch_window_map_native_params` reads out of the
# very same row list (keyed by state_id; it never looks at `epoch_window`). MBAR
# still needs a bias column for a dropped state -- it is cross-evaluated against
# every other epoch's samples -- so losing the row silently falls that column
# back to the final-registry snapshot, i.e. re-creates for those states exactly
# the stale-snapshot bug the 2026-08-04 per-epoch-native-params fix exists to
# prevent. The flat `epoch_NNN/epoch_window_map.csv` maps carry primary_k and
# secondary_k and differ from the registry in every row on real runs
# (chignolin_6/epoch_000: secondary_center -2.1337 in the map vs -0.9058 in the
# registry), so this is a live hazard, not a theoretical one. The end-to-end
# proof that the params really do survive into u_nk lives in
# tests/test_union_mbar_per_epoch_bias.py.

def test_dropped_rows_are_kept_verbatim_out_of_band(tmp_path):
    """Every original row survives the repair, dropped ones included, with all
    their columns untouched -- only `epoch_window` moves out of reach."""
    phase = tmp_path / 'epoch_000'
    rows = _write_map(phase, [(0, 0, 0.0, -2.1337), (1, 1, 0.0, -1.1031),
                              (2, 2, 0.05, 0.657)])
    _write_meta(phase, dropped=[1], n_windows=2)
    out, _notes = _validate_and_repair_epoch_window_map(phase, rows)

    assert _real_mapping(out) == {0: 0, 1: 2}
    assert _phantom_state_ids(out) == {1}
    # Same set of state_ids as the original map: nothing was discarded.
    assert {int(r['state_id']) for r in out} == {0, 1, 2}
    # ... and state 1's own recorded params came through byte-identically.
    (phantom,) = [r for r in out if int(r['state_id']) == 1]
    original = [r for r in rows if int(r['state_id']) == 1][0]
    assert {k: v for k, v in phantom.items() if k != 'epoch_window'} == \
           {k: v for k, v in original.items() if k != 'epoch_window'}


def test_phantom_indices_are_unreachable_and_keep_the_lookup_fast_path(tmp_path):
    """The parked indices must be (a) far past any real local window index, so
    no sample can ever resolve to one, and (b) non-negative -- a negative key
    would drop `_vectorized_map_lookup` off its dense-LUT fast path onto a
    per-element Python loop over millions of samples.

    Non-negativity is only one of the two fast-path preconditions the phantom
    base has to satisfy; the other (the LUT the widened key range implies still
    fitting under `_VECTORIZED_LOOKUP_MAX_TABLE_SIZE`) is checked, by measuring
    which path actually ran, in
    `test_repaired_map_keeps_the_vectorized_lookup_fast_path` at the end of this
    file. Nothing here would notice that one being violated.
    """
    from gareus.mbar_analysis.loaders_adaptive import _vectorized_map_lookup

    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, dropped=[20, 22, 23], n_windows=24)
    out, _notes = _validate_and_repair_epoch_window_map(phase, rows)

    wmap = _mapping(out)                       # what the real caller builds
    assert min(wmap) >= 0
    assert min(ew for ew in wmap if ew >= _PHANTOM_EPOCH_WINDOW_BASE) > 24 * 1000
    # Every local index a sample of this phase can hold resolves to a survivor.
    got = _vectorized_map_lookup(np.arange(24), wmap, default=-1)
    assert got.tolist() == [wmap[i] for i in range(24)]
    assert 20 not in got.tolist() and 22 not in got.tolist() and 23 not in got.tolist()


# --- the phantom base must not cost the vectorized lookup its fast path -----
#
# `_PHANTOM_EPOCH_WINDOW_BASE` and `_VECTORIZED_LOOKUP_MAX_TABLE_SIZE` are two
# constants picked independently of one another, in different sections of
# `loaders_adaptive.py`, that nonetheless meet: parking a dropped state's row at
# the phantom base widens the `wmap` dict's key range from ~24 to ~1,000,003,
# and the real caller feeds that whole dict into `_vectorized_map_lookup`
# against the phase's entire window_id column (millions of rows per phase). The
# LUT is sized by the largest key, so the current values are affordable only
# because the cap sits an order of magnitude above the base.
#
# Nothing observable changes when that stops being true -- the fallback path is
# the exact original comprehension, so the *output* is identical either way and
# no result-checking test can see the difference. The only observable is WHICH
# path ran, hence the counting dict below. Without it, raising the phantom base
# past the cap (or lowering the cap under the base) would quietly turn a
# multi-million-row vectorized remap into a per-element Python loop with the whole
# suite still green. NOTE the module also asserts the inequality directly at import
# time; that assertion is a fast, loud signal but is not itself
# revert-checkable, so the teeth live here.

class _GetCountingDict(dict):
    """A `wmap` that records whether anything did a per-element ``dict.get``.

    `_vectorized_map_lookup`'s dense-LUT path only ever touches ``keys()``,
    ``values()`` and ``len()``; its ``_naive`` fallback is a comprehension of
    ``mapping.get(int(x), default)`` calls, one per sample. So a non-zero
    `get_calls` after a lookup means the fast path was abandoned -- and it is
    the only difference between the two paths that is visible from outside,
    since their return values are equal by construction.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.get_calls = 0

    def get(self, *a, **kw):
        self.get_calls += 1
        return super().get(*a, **kw)


def test_repaired_map_keeps_the_vectorized_lookup_fast_path(tmp_path):
    """A repaired phase's `wmap` must still remap through the dense LUT.

    Real shape: chignolin_6's `final/baseline` (27 map rows, 24 real windows,
    3 dropped), so the dict the caller builds holds 24 reachable keys in 0..23
    plus 3 parked phantoms, and the LUT it needs spans
    `_PHANTOM_EPOCH_WINDOW_BASE + 3` entries -- ~8 MB of transient int64 to
    remap a 24-entry mapping. That is the standing price of keeping the dropped
    states' rows, and it is only payable while it stays under
    `_VECTORIZED_LOOKUP_MAX_TABLE_SIZE`.
    """
    from gareus.mbar_analysis import loaders_adaptive as _la

    phase = tmp_path / 'final' / 'baseline'
    rows = _identity_map(phase, 27, _SEC27)
    _write_meta(phase, dropped=[20, 22, 23], n_windows=24)
    out, _notes = _validate_and_repair_epoch_window_map(phase, rows)

    expected_map = _mapping(out)
    # It really is the phantoms, not the two dozen real windows, that size the
    # table -- otherwise this test would prove nothing about the coupling.
    assert max(expected_map) == _PHANTOM_EPOCH_WINDOW_BASE + 2
    assert max(_real_mapping(out)) == 23

    wmap = _GetCountingDict(expected_map)
    window_ids = np.repeat(np.arange(24, dtype=np.int32), 7)   # a window_id column
    got = _la._vectorized_map_lookup(window_ids, wmap, default=-1, dtype=np.int32)

    assert got.tolist() == [expected_map[int(w)] for w in window_ids]
    assert wmap.get_calls == 0, (
        f'_vectorized_map_lookup fell back to its per-element Python path: the LUT it '
        f'needs ({max(expected_map) + 1} entries, set by _PHANTOM_EPOCH_WINDOW_BASE='
        f'{_PHANTOM_EPOCH_WINDOW_BASE}) no longer fits under '
        f'_VECTORIZED_LOOKUP_MAX_TABLE_SIZE={_la._VECTORIZED_LOOKUP_MAX_TABLE_SIZE}.')


def test_get_counting_wmap_actually_detects_the_slow_path(monkeypatch):
    """Positive control for the detector the test above relies on.

    Without this, `get_calls == 0` could just mean the counting dict never sees
    a `get` under any circumstances, and the guard would be vacuous. The cap is
    read at call time, so shrinking it below the phantom base forces the
    fallback with no source edit -- and the returned values must stay identical,
    which is the reason the fallback is safe (and the reason nothing else can
    detect it).
    """
    from gareus.mbar_analysis import loaders_adaptive as _la

    wmap = _GetCountingDict({i: i for i in range(24)})
    wmap[_PHANTOM_EPOCH_WINDOW_BASE] = 24
    monkeypatch.setattr(_la, '_VECTORIZED_LOOKUP_MAX_TABLE_SIZE', 1000)

    window_ids = np.arange(24, dtype=np.int32)
    got = _la._vectorized_map_lookup(window_ids, wmap, default=-1, dtype=np.int32)

    assert got.tolist() == list(range(24))
    assert wmap.get_calls == 24
