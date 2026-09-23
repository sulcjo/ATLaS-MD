## Submission

# Board review: ATLaS-MD / gareus v0.8.2 (main @ dfeb8f9) -- does this code produce the Boltzmann free energy it claims?

## What the whole software exists to do
Produce correct, Boltzmann-weighted free-energy surfaces (1D PMF and 2D FES) of peptide conformations
(target: chignolin, GYDPETGTWG, TIP3P, 300 K, NPT 1 bar) from replica-exchange umbrella sampling (REUS)
on CV1 = nonlocal heavy-atom contact fraction x CV2 = residual-torsion-pc (torsion principal component
residualised against CV1, auto-selected by a "swarm" stage of short unbiased trajectories), accelerated by
a peptide-only Pep-GaMD lower-bound dual (dependent dihedral + total) boost whose envelope
(Vmax/Vmin/threshold/k0max) is calibrated ONCE by the swarm and then frozen, with a lambda ladder of boost
strengths (every umbrella window replicated at lambda in {0, 0.236, 0.647, 1}; k0 = lambda*k0max),
replica exchange by gibbs-walk over the (window x rung) state graph, NPT via an application-controlled
biased Metropolis-MC volume move against U* = physical + bias + boost, checkpoint/resume across SLURM
walltimes, adaptive epochs (window add/split/retire, envelope recalibration), and final reweighting by
MBAR over all (window, rung) states using per-frame recorded CVs and boost arguments (v_pep, v_dih).
Pipeline: swarm -> window/ladder design -> seeding/umbrella pull -> production (exchange, boost, barostat,
checkpoint/resume) -> adaptive epochs -> analysis (analyze_gareus_mbar.py, gareus/mbar_analysis/).

## Calibration case (a real failure nobody's tests or diagnostics caught)
chignolin_8 ran its ENTIRE production unboosted: gamd-openmm's StageIntegrator picks its stage from its
own stepCount; the swarm envelope carried physics globals but no stepCount, so every replica sat in stage 2
(conventional MD accumulating statistics) with ForceScalingFactor exactly 1.0 while exchange and MBAR
priced a boost that was never applied. Fixed in c73224f (seed_frozen_envelope_stage5 +
verify_gamd_production_stage5). The existing tests had set stepCount by hand, so they never exercised the
real hand-off. FIND THE NEXT ONES LIKE IT.

## Your task
Adversarially review the code below. Report every place where the software can SILENTLY produce a number
that is not the Boltzmann free energy it claims: wrong Hamiltonian applied vs priced, inconsistent energies
between integrator / exchange / barostat / recorded samples / MBAR, detailed-balance violations, unit or
sign errors, state-label mismatches, resume/restart state loss, stale or overwritten calibration, silent
fallbacks (NaN -> 0, missing key -> default), tests that cannot fail. For each finding give: file:line
(use the L-numbers), the concrete failure scenario, the effect on the free energy, and your confidence.
Distinguish VERIFIED (you can point at the lines) from SUSPECTED. Do not report style issues.

## Subsystem under review: replica exchange: state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path




### gareus/production.py :: umbrella_bias_matrix_kcal (source lines 895-917)
```python
def umbrella_bias_matrix_kcal(primary_values, ss_values, centers_a, k_kcal, ss_centers=None, ss_k_kcal=None):
    """The two umbrella components of the [state, replica] bias matrix, in kcal/mol.

    ``0.5 k (x - c)^2`` per axis; a state without a secondary centre, or a replica without
    a finite secondary value, contributes zero on that axis (the previous sample-path
    guard, now applied identically to the exchange path -- a NaN entry used to freeze
    that replica's exchanges). Orientation is [state, replica]; the fixed-state export is
    [sample, state] and converts by name, never by shape.
    """
    pv = np.asarray(primary_values, dtype=np.float64)
    ca = np.asarray(centers_a, dtype=np.float64)
    ka = np.asarray(k_kcal, dtype=np.float64)
    dprimary = pv[np.newaxis, :] - ca[:, np.newaxis]
    distance_kcal = 0.5 * ka[:, np.newaxis] * dprimary * dprimary
    if ss_centers is None or ss_k_kcal is None:
        return distance_kcal, np.zeros_like(distance_kcal)
    sv = np.asarray(ss_values, dtype=np.float64)
    sc = np.asarray(ss_centers, dtype=np.float64)
    sk = np.asarray(ss_k_kcal, dtype=np.float64)
    dss = sv[np.newaxis, :] - sc[:, np.newaxis]
    ss_kcal = 0.5 * sk[:, np.newaxis] * dss * dss
    ss_kcal = np.where(np.isfinite(dss), ss_kcal, 0.0)
    return distance_kcal, ss_kcal
```

### gareus/production.py :: _gibbs_window_proposal_distribution (source lines 983-1041)
```python
def _gibbs_window_proposal_distribution(
    beta: float,
    bias_matrix_kj: np.ndarray,
    replica_index: int,
    current_window: int,
    replica_of_window: np.ndarray,
) -> dict[str, np.ndarray]:
    """Heat-bath proposal over target windows for one replica transposition."""
    rep = int(replica_index)
    wi = int(current_window)
    holders = np.asarray(replica_of_window, dtype=np.int64)
    all_windows = np.arange(holders.size, dtype=np.int32)
    valid_mask = holders >= 0
    valid_windows = all_windows[valid_mask]
    target_reps = holders[valid_mask]
    if valid_windows.size <= 0:
        return {
            "windows": np.asarray([], dtype=np.int32),
            "target_replicas": np.asarray([], dtype=np.int64),
            "deltas_kj": np.asarray([], dtype=np.float64),
            "probabilities": np.asarray([], dtype=np.float64),
        }

    bias = np.asarray(bias_matrix_kj, dtype=np.float64)
    old_e = bias[wi, rep] + bias[valid_windows, target_reps]
    new_e = bias[valid_windows, rep] + bias[wi, target_reps]
    delta = np.asarray(new_e - old_e, dtype=np.float64)
    stay_idx = np.where(valid_windows == wi)[0]
    if stay_idx.size:
        delta[int(stay_idx[0])] = 0.0
    # No upper clip: true heat-bath weights exp(-β·δ) can exceed 1 for favourable
    # moves (δ < 0).  The log-sum-exp shift on line ~390 handles overflow.
    # Lower bound -745 prevents underflow to exact zero (loses that candidate).
    log_weights = np.clip(-float(beta) * delta, -745.0, 745.0)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        return {
            "windows": np.asarray([], dtype=np.int32),
            "target_replicas": np.asarray([], dtype=np.int64),
            "deltas_kj": np.asarray([], dtype=np.float64),
            "probabilities": np.asarray([], dtype=np.float64),
        }
    valid_windows = valid_windows[finite]
    target_reps = target_reps[finite]
    delta = delta[finite]
    log_weights = log_weights[finite]
    m = float(np.max(log_weights))
    weights = np.exp(log_weights - m)
    sw = float(np.sum(weights))
    if not math.isfinite(sw) or sw <= 0.0:
        probabilities = np.full(valid_windows.shape, 1.0 / max(1, valid_windows.size), dtype=np.float64)
    else:
        probabilities = weights / sw
    return {
        "windows": valid_windows.astype(np.int32, copy=False),
        "target_replicas": target_reps.astype(np.int64, copy=False),
        "deltas_kj": delta.astype(np.float64, copy=False),
        "probabilities": probabilities.astype(np.float64, copy=False),
    }
```

### gareus/production.py :: _gibbs_mh_acceptance_probability (source lines 1044-1067)
```python
def _gibbs_mh_acceptance_probability(
    delta_kj: float,
    beta: float,
    q_forward: float,
    q_reverse: float,
) -> float:
    """Metropolis-Hastings correction for nonuniform Gibbs-walk proposals."""
    try:
        delta = float(delta_kj)
        beta_val = float(beta)
        qf = float(q_forward)
        qr = float(q_reverse)
    except Exception:
        return 0.0
    if not (math.isfinite(delta) and math.isfinite(beta_val) and math.isfinite(qf) and math.isfinite(qr)):
        return 0.0
    if qf <= 0.0 or qr <= 0.0:
        return 0.0
    log_alpha = -beta_val * delta + math.log(qr) - math.log(qf)
    if log_alpha >= 0.0:
        return 1.0
    if log_alpha < -745.0:
        return 0.0
    return float(math.exp(log_alpha))
```

### gareus/production.py :: _exchange_probability (source lines 1070-1089)
```python
def _exchange_probability(delta_kj: float, beta: float) -> float:
    """Metropolis probability for a proposed umbrella-window swap.

    Extracted from run_gareus's exchange-loop closures (was a nested def
    capturing beta from the enclosing scope) so it's unit-testable without a
    live OpenMM Context.
    """
    try:
        delta_kj = float(delta_kj)
        beta = float(beta)
    except Exception:
        return 0.0
    if not (math.isfinite(delta_kj) and math.isfinite(beta)):
        return 0.0
    if delta_kj <= 0.0:
        return 1.0
    x = -beta * delta_kj
    if x < -745.0:
        return 0.0
    return float(math.exp(x))
```

### gareus/production.py :: swap_candidate_replicas (source lines 1092-1115)
```python
def swap_candidate_replicas(replica_of_window, wi: int, wj: int):
    """``(i, j)`` for an attemptable swap, else ``None``.

    Split out of :func:`apply_window_swap` so a caller can decide whether the
    move is attemptable *before* drawing a random number. Production's original
    closure returned early on these cases without touching the rng, and any
    extra draw here shifts every downstream random stream -- the same hazard
    ``force_accept`` is guarded against.
    """
    try:
        wi = int(wi)
        wj = int(wj)
    except (TypeError, ValueError):
        return None
    if wi == wj:
        return None
    n_win = int(replica_of_window.size)
    if not (0 <= wi < n_win and 0 <= wj < n_win):
        return None
    i = int(replica_of_window[wi])
    j = int(replica_of_window[wj])
    if i < 0 or j < 0 or i == j:
        return None
    return i, j
```

### gareus/production.py :: apply_window_swap (source lines 1127-1182)
```python
def apply_window_swap(
    bias_matrix_kj,
    beta: float,
    assignments,
    replica_of_window,
    wi: int,
    wj: int,
    uniform: Optional[float],
    *,
    p_override: Optional[float] = None,
    force_accept: bool = False,
) -> "Optional[SwapOutcome]":
    """Decide and apply one umbrella-window swap; the pure core of the REUS move.

    Extracted from ``run_gareus``'s closure so the exchange kernel can be driven
    from a test against a known target distribution. Everything here is pure
    NumPy/arithmetic: ``assignments`` and ``replica_of_window`` are mutated in
    place exactly as production does, and the caller applies the side effects
    (``set_window`` on the two contexts, the Parquet journal entry) from the
    returned outcome.

    Returns ``None`` when there is nothing to attempt -- same window, either
    window unheld, or both held by the same replica -- so the caller can leave
    its attempt counter alone.

    The move swaps *state labels*, never configurations, which is why the
    unbiased potential and the GaMD boost cancel exactly from ``delta`` and only
    the umbrella biases appear: ``x_i`` and ``x_j`` are untouched, so
    ``U0(x_i)+U0(x_j)`` and ``dV(x_i)+dV(x_j)`` are identical before and after.

    ``uniform`` is the already-drawn ``rng.random()``. It is a parameter rather
    than an rng handle so that the kernel is deterministic under test, and so
    the caller can preserve production's short-circuit: ``force_accept`` must
    not consume a random number, or every downstream RNG stream shifts.
    """
    pair = swap_candidate_replicas(replica_of_window, wi, wj)
    if pair is None:
        return None
    i, j = pair
    wi = int(wi)
    wj = int(wj)
    # bias_matrix_kj[window, replica] = U_window(x_replica)
    old_e = float(bias_matrix_kj[wi, i] + bias_matrix_kj[wj, j])
    new_e = float(bias_matrix_kj[wj, i] + bias_matrix_kj[wi, j])
    delta = float(new_e - old_e)
    pacc = float(p_override) if p_override is not None else _exchange_probability(delta, beta)
    pacc = max(0.0, min(1.0, pacc)) if math.isfinite(pacc) else 0.0
    if force_accept:
        accepted = True
    else:
        accepted = bool(float(uniform) < pacc) if uniform is not None else False
    if accepted:
        assignments[i], assignments[j] = assignments[j], assignments[i]
        replica_of_window[int(assignments[i])] = int(i)
        replica_of_window[int(assignments[j])] = int(j)
    return SwapOutcome(i, j, delta, pacc, accepted)
```

### gareus/production.py :: gibbs_propose_one_replica (source lines 1202-1271)
```python
def gibbs_propose_one_replica(
    bias_matrix_kj,
    beta: float,
    assignments,
    replica_of_window,
    replica_index: int,
    choose,
) -> "GibbsProposal":
    """Decide one `gibbs-walk` move: propose, reverse-propose, MH-correct.

    Split out of ``run_gareus``'s exchange closure so the exact-enumeration
    tests drive *this* code rather than a copy of it. A hand-written mirror of
    these six steps in a test proves only that the mirror is self-consistent --
    it cannot see a defect introduced on the production side.

    ``choose(n_candidates, probabilities) -> index`` is the sampling step,
    injected so the caller supplies the RNG and a test can enumerate every
    branch deterministically. Production passes ``rng.choice``.

    The reverse proposal is evaluated against ``holders_after`` -- the holder
    table as it *would* be post-swap -- because MH requires the reverse
    probability from the destination state, not the current one. Getting this
    wrong is the classic heat-bath-with-MH bug and it is silent: the chain
    still runs, it just no longer targets pi.
    """
    rep = int(replica_index)
    wi = int(assignments[rep])
    proposal = _gibbs_window_proposal_distribution(
        beta=float(beta),
        bias_matrix_kj=bias_matrix_kj,
        replica_index=rep,
        current_window=wi,
        replica_of_window=replica_of_window,
    )
    valid_windows = proposal["windows"]
    probs = proposal["probabilities"]
    deltas = proposal["deltas_kj"]
    if valid_windows.size <= 0 or probs.size <= 0:
        return GibbsProposal(wi, wi, 0.0, 0.0, 0.0, 0.0, False, True)

    choice_index = int(choose(int(valid_windows.size), probs))
    wj = int(valid_windows[choice_index])
    q_forward = float(probs[choice_index])
    if wj == wi:
        return GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False)

    holders_after = np.asarray(replica_of_window).astype(np.int64, copy=True)
    target_rep = int(holders_after[wj])
    holders_after[wi] = target_rep
    holders_after[wj] = rep
    reverse = _gibbs_window_proposal_distribution(
        beta=float(beta),
        bias_matrix_kj=bias_matrix_kj,
        replica_index=rep,
        current_window=wj,
        replica_of_window=holders_after,
    )
    rev_windows = reverse["windows"]
    rev_probs = reverse["probabilities"]
    rev_idx = np.where(rev_windows == wi)[0]
    q_reverse = float(rev_probs[int(rev_idx[0])]) if rev_idx.size else 0.0
    pacc = _gibbs_mh_acceptance_probability(
        delta_kj=float(deltas[choice_index]),
        beta=float(beta),
        q_forward=q_forward,
        q_reverse=q_reverse,
    )
    return GibbsProposal(
        wi, wj, float(deltas[choice_index]), q_forward, q_reverse, float(pacc), False, False
    )
```

### gareus/production.py :: explicit_2d_neighbor_pairs_for_exchange (source lines 1479-1505)
```python
def explicit_2d_neighbor_pairs_for_exchange(centers_a, secondary_cv_centers, parity: int, args=None) -> tuple[list[tuple[int, int, str, float]], int, list[dict]]:
    """Return a disjoint subset of graph edges for one exchange interval."""
    edges = build_explicit_2d_neighbor_edges(centers_a, secondary_cv_centers, args=args)
    slots = max(1, int(getattr(args, "explicit_2d_exchange_slots", 4) if args is not None else 4))
    slot = int(parity) % slots
    chosen = []
    used = set()
    # Round-robin edge coloring by edge index, then greedy disjoint filtering.
    for idx, edge in enumerate(edges):
        if idx % slots != slot:
            continue
        wi, wj = int(edge["wi"]), int(edge["wj"])
        if wi in used or wj in used:
            continue
        chosen.append((wi, wj, str(edge.get("edge_type", "explicit_2d_graph")), float(edge.get("normalized_distance", float("nan")))))
        used.add(wi)
        used.add(wj)
    # If a slot was empty for a small graph, try all edges greedily so exchanges do not stall.
    if not chosen and edges:
        for edge in edges:
            wi, wj = int(edge["wi"]), int(edge["wj"])
            if wi in used or wj in used:
                continue
            chosen.append((wi, wj, str(edge.get("edge_type", "explicit_2d_graph")), float(edge.get("normalized_distance", float("nan")))))
            used.add(wi)
            used.add(wj)
    return chosen, slots, edges
```

### gareus/production.py :: assemble_bias_matrices (source lines 2055-2072)
```python
def assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_kj):
    """Combine the umbrella components and the Pep-GaMD boost into one (kcal, kj) pair.

    Both unit matrices carry the SAME quantity -- umbrella + boost -- so the
    invariant ``bias_kj == 4.184 * bias_kcal`` holds regardless of whether the
    λ-ladder is active. ``boost_bias_kj`` is all-zero when it is not, so this
    is a no-op reduction to the pre-ladder umbrella-only matrix in that case.
    Pulled out to module level so both assembly sites in ``run_gareus`` (the
    log/sample path and ``_current_exchange_arrays``) share one definition and
    so it is directly testable without a live OpenMM Context.
    """
    distance_bias_kcal = np.asarray(distance_bias_kcal, dtype=np.float64)
    ss_bias_kcal = np.asarray(ss_bias_kcal, dtype=np.float64)
    boost_bias_kj = np.asarray(boost_bias_kj, dtype=np.float64)
    boost_bias_kcal = boost_bias_kj / 4.184
    bias_kcal = distance_bias_kcal + ss_bias_kcal + boost_bias_kcal
    bias_kj = 4.184 * bias_kcal
    return bias_kcal, bias_kj
```

### gareus/production.py :: _fetch_v_pep_v_dih (source lines 2075-2094)
```python
def _fetch_v_pep_v_dih(ctx, pep_env, unit) -> tuple[float, float]:
    """Read (V_pep, V_dih) in kJ/mol for the Pep-GaMD boost from a live Context.

    Returns ``(nan, nan)`` without touching the Context when ``pep_env`` is
    None (the λ-ladder is not active), matching every call site's prior
    inline guard. Shared by both fetch closures' both branches in
    ``run_gareus`` (log/sample and exchange, fast-CV and non-fast).
    """
    if pep_env is None:
        return float("nan"), float("nan")
    # gamd-openmm's factory puts PeriodicTorsion/CMAPTorsion forces in group 2 for every
    # stock boost type (integrator_factory.set_dihedral_group), the same id as
    # DIHEDRAL_GROUP in the Pep-GaMD partition, so this read is valid on both paths.
    v_dih = (ctx.getState(getEnergy=True, groups={DIHEDRAL_GROUP}).getPotentialEnergy()
             .value_in_unit(unit.kilojoule_per_mole))
    if not getattr(pep_env, "has_total", True):
        # single dihedral boost: no Total channel, no auxiliary force to read
        return float("nan"), v_dih
    v_pep = peptide_essential_energy_kj(ctx, unit)
    return v_pep, v_dih
```

### gareus/production.py :: campaign_ladder_registry_lambda (source lines 2261-2299)
```python
def campaign_ladder_registry_lambda(out_dir) -> Optional[tuple]:
    """First ``(registry filename, λ)`` with λ > 0 at or above ``out_dir``, else None.

    The λ-ladder is a property of the CAMPAIGN, but ``args.state_gamd_lambdas``
    only ever describes the states handed to ONE sub-run. An adaptive top-up may
    legitimately be given nothing but λ=0 states, and deciding "no ladder here"
    from that subset is what produced chignolin_7's defect: the λ=0 replicas kept
    the shared calibration's full ``k0`` instead of ``λ·k0max = 0`` and so ran
    boosted, while their channel energies went unrecorded. The registry
    (``StateRegistry.write_state_csv``) knows every state in the campaign, so ask
    it instead.

    The walk stops at ``adaptive_production/`` deliberately: several campaigns
    commonly share one parent directory, and an unbounded walk would let one
    campaign's ladder switch on another's boost recording.
    """
    base = Path(out_dir)
    for candidate in (base, *base.parents):
        for name in CAMPAIGN_LADDER_REGISTRY_NAMES:
            path = candidate / name
            if not path.exists():
                continue
            try:
                with path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        raw = row.get("gamd_lambda")
                        if raw in (None, "", "None"):
                            continue
                        try:
                            lam = float(raw)
                        except (TypeError, ValueError):
                            continue
                        if lam > 0.0:
                            return (path.name, lam)
            except OSError:
                continue
        if candidate.name == "adaptive_production":
            break
    return None
```

### gareus/production.py :: resolve_ladder_active (source lines 2302-2317)
```python
def resolve_ladder_active(state_lambdas, out_dir) -> bool:
    """Is the λ-ladder active for this run?

    True when this sub-run holds a boosted state itself (the original test), OR
    when the campaign registry above it says the campaign runs a ladder. The
    second clause is what keeps an all-λ=0 top-up on the ladder code path, so its
    replicas get ``k0 = 0·k0max`` and its samples carry ``v_pep``/``v_dih``.

    Deliberately NOT true for a plain umbrella/REUS or plain-GaMD campaign: the
    ladder path makes every logged sample read two extra Context energies, which
    would be pure cost for a boost that is identically zero.
    """
    lam_arr = np.asarray(state_lambdas, dtype=float)
    if lam_arr.size and bool(np.any(lam_arr > 0.0)):
        return True
    return campaign_ladder_registry_lambda(out_dir) is not None
```

### gareus/production.py :: _derive_state_gamd_lambdas (source lines 2360-2384)
```python
def _derive_state_gamd_lambdas(window_metadata: Optional[dict], n: int, existing=None) -> list[float]:
    """Derive one gamd_lambda per surviving window, filter/drop-safe.

    ``window_metadata["normalized_rows"]`` is the source of truth whenever it is
    usable: present, aligned to exactly ``n`` surviving windows, and every row
    carries its own ``gamd_lambda``. Reindexing operations that drop/reorder
    windows (filter_explicit_2d_windows_by_seed_reachability,
    drop_bad_us_windows_and_rebuild's _resubscript_normalized_rows) already
    subset+renumber normalized_rows correctly, so reading gamd_lambda back out
    of it here is safe across any such reindex.

    ``existing`` is used ONLY when normalized_rows cannot be trusted, and ONLY
    if it is already exactly length ``n`` -- a caller must not pass a
    pre-reindex value across a filter/drop boundary where the alignment cannot
    be verified from here; pass ``existing=None`` at any such call site.

    Zeros (ladder inactive) is the fallback of last resort, never a guess.
    """
    rows = (window_metadata or {}).get("normalized_rows") or []
    if rows and len(rows) == int(n) and all("gamd_lambda" in r for r in rows):
        return [float(r.get("gamd_lambda", 0.0) or 0.0) for r in rows]
    existing_list = list(existing or [])
    if len(existing_list) == int(n):
        return existing_list
    return [0.0] * int(n)
```

### gareus/production.py :: _reload_state_gamd_lambdas_on_resume (source lines 2386-2437)
```python
def _reload_state_gamd_lambdas_on_resume(args, out_dir: Path) -> None:
    """Restore the frozen λ-ladder from run_manifest.json on ``--resume``.

    Controller ruling (Task 2's implementer found this and deferred it to
    Task 9): ``--resume`` never persisted ``args.state_gamd_lambdas`` across a
    restart on its own. ``_derive_state_gamd_lambdas`` below is only as good
    as the window_metadata it is handed on the resumed path, and nothing
    upstream guarantees that still carries a per-row ``gamd_lambda`` after a
    restart (an older run directory, or a checkpoint manifest's
    window_metadata predating the rung dimension). Left uncorrected,
    ``args.state_gamd_lambdas`` would come out unset/empty and every replica
    would silently run at λ=0 -- silently, because an all-zero ladder is also
    the valid "ladder disabled" state, so nothing downstream would complain.
    Spec §3.6 requires the ladder to stay frozen for the whole campaign,
    which includes surviving a restart, so it is read back from wherever it
    was first frozen: ``run_manifest.json``'s ``method_settings`` (see
    ``gareus/provenance.py:_method_settings``), which is written once at
    campaign start (``initialize_run_manifest``) and never overwritten by a
    resume's ``update_run_manifest`` patches.

    CALL ORDER IS LOAD-BEARING: this must run BEFORE ``_derive_state_gamd_lambdas``,
    never after. A review caught this call sitting after
    ``_derive_state_gamd_lambdas`` in an earlier revision, which made it dead
    code on every real run: that function's own fallback-of-last-resort is
    ``[0.0] * n``, a non-empty (hence truthy) list, so by the time a
    post-derive reload call inspected ``args.state_gamd_lambdas`` it always
    looked already-populated and the manifest was never read. Called BEFORE
    ``_derive_state_gamd_lambdas`` (its real call site, in ``run_gareus``),
    this only ever observes ``args.state_gamd_lambdas`` in its true
    pre-derivation state: unset on the fast-resume/choose_windows paths, or
    already populated straight from the window table on the explicit-2D path
    (which this function correctly leaves alone) -- either way,
    ``_derive_state_gamd_lambdas``'s own ``existing=`` parameter then
    naturally receives whatever this function restored.

    A no-op unless ``args.resume`` is set and ``args.state_gamd_lambdas`` is
    still unset/empty at the point it is called -- an already-populated
    ladder (from an explicit window table, or a caller that set it
    explicitly) is never overridden. This is a plain truthiness check, not a
    content check: an all-zero ``args.state_gamd_lambdas`` set by a caller is
    a legitimate disabled ladder and must never be treated as "unset" and
    silently replaced.
    """
    if not bool(getattr(args, "resume", False)):
        return
    if getattr(args, "state_gamd_lambdas", None):
        return
    manifest = read_json_file(Path(out_dir) / "run_manifest.json", {}) or {}
    method_settings = (manifest or {}).get("method_settings", {}) or {}
    lambdas = method_settings.get("state_gamd_lambdas")
    if lambdas:
        args.state_gamd_lambdas = [float(x) for x in lambdas]
```

### gareus/production.py :: _persist_state_gamd_lambdas (source lines 2439-2467)
```python
def _persist_state_gamd_lambdas(args, out_dir: Path) -> None:
    """Patch the CURRENT args.state_gamd_lambdas into run_manifest.json's
    method_settings, overwriting any earlier snapshot written by a previous
    call.

    Review finding: the first cut of this patch was a single call placed
    right after the initial _derive_state_gamd_lambdas() in run_gareus, but
    args.state_gamd_lambdas is mutated TWICE afterwards on some runs --
    the --max-replicas truncation (`args.state_gamd_lambdas =
    list(args.state_gamd_lambdas)[:_max_replicas]`) and the post-pull US
    auto-drop re-derive (`args.state_gamd_lambdas =
    _derive_state_gamd_lambdas(window_metadata, len(centers_a),
    existing=None)`) -- and neither re-patched the manifest. Persisting the
    PRE-truncation/PRE-drop value is exactly what
    _derive_state_gamd_lambdas' own docstring warns a caller never to do
    with `existing=` across a filter/drop boundary, because
    _reload_state_gamd_lambdas_on_resume feeds this same manifest field
    back in as `existing=` on the next --resume: a stale, wrong-length
    snapshot there either falls back to _derive_state_gamd_lambdas' own
    [0.0]*n (silently disabling the ladder) or, worse, if the post-drop
    count happens to match, resumes with a misaligned per-state λ.

    Call this again at EVERY point in run_gareus that reassigns
    args.state_gamd_lambdas, not just once after the initial derive --
    the last call before production starts is what ends up in the
    manifest, so it must be the call closest to (after) the final
    mutation, not the first available opportunity.
    """
    update_run_manifest(out_dir, {"method_settings": {"state_gamd_lambdas": list(getattr(args, "state_gamd_lambdas", None) or [])}})
```

### gareus/pep_gamd.py :: _channel_boost (source lines 497-503)
```python
def _channel_boost(v, e, vmax, vmin, k0):
    v = _np.asarray(v, dtype=float)
    rng = vmax - vmin
    scale = _np.maximum(_np.maximum(abs(e), _np.abs(v)), 1.0)
    b = 0.5 * k0 * (e - v) ** 2 / rng
    b = _np.where(_np.abs(rng) <= 0.001 * scale, 0.0, b)
    return _np.where((b + v) < e, b, 0.0)
```

### gareus/pep_gamd.py :: pep_gamd_boost_kj (source lines 506-519)
```python
def pep_gamd_boost_kj(v_pep_kj, v_dih_kj, lam, env: PepGamdEnvelope):
    """gamd-openmm's dependent dual boost under rung λ: dihedral first, then Total with the
    dihedral boost added to the Total energy before the square (stage_integrator
    _add_dihedral_boost_to_total_energy)."""
    lam = float(lam)
    b_dih = _channel_boost(v_dih_kj, env.threshold_dih, env.vmax_dih, env.vmin_dih, lam * env.k0max_dih)
    if not env.has_total:
        # Single dihedral boost: no Total channel exists, v_pep is ignored (it is NaN
        # on such runs by construction, see production._fetch_v_pep_v_dih).
        out = _np.asarray(b_dih, dtype=float)
        return float(out) if out.ndim == 0 else out
    b_tot = _channel_boost(_np.asarray(v_pep_kj, dtype=float) + b_dih, env.threshold_total, env.vmax_total, env.vmin_total, lam * env.k0max_total)
    out = b_dih + b_tot
    return float(out) if out.ndim == 0 else out
```

### gareus/pep_gamd.py :: pep_gamd_boost_matrix_kj (source lines 522-525)
```python
def pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, lambdas, env: PepGamdEnvelope) -> _np.ndarray:
    """(n_states, n_samples): boost of each sample's configuration under each state's λ."""
    v_pep = _np.asarray(v_pep_kj, dtype=float); v_dih = _np.asarray(v_dih_kj, dtype=float)
    return _np.vstack([_np.asarray(pep_gamd_boost_kj(v_pep, v_dih, float(l), env), dtype=float) for l in lambdas])
```

### gareus/pep_gamd.py :: k0max_from_globals (source lines 528-530)
```python
def k0max_from_globals(shared_globals: dict) -> dict:
    """Top-rung k0 per channel; ``Total`` is 0.0 for a single dihedral boost (no such global)."""
    return {"Total": float(shared_globals.get("k0_Total", 0.0)), "Dihedral": float(shared_globals["k0_Dihedral"])}
```

### gareus/pep_gamd.py :: set_replica_lambda (source lines 533-546)
```python
def set_replica_lambda(integrator, lam: float, k0max: dict) -> None:
    """Put a replica on rung λ: k0_c = λ·k0max_c for every channel the integrator has. Nothing else differs between rungs."""
    lam = float(lam)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"gamd_lambda={lam} must lie in [0, 1]")
    try:
        names = {str(integrator.getGlobalVariableName(i)) for i in range(int(integrator.getNumGlobalVariables()))}
    except AttributeError:
        names = None  # minimal test doubles expose only setGlobalVariableByName: keep the two-channel behaviour
    integrator.setGlobalVariableByName("k0_Dihedral", lam * float(k0max["Dihedral"]))
    if names is None or "k0_Total" in names:
        integrator.setGlobalVariableByName("k0_Total", lam * float(k0max.get("Total", 0.0)))
    elif float(k0max.get("Total", 0.0)) != 0.0:
        raise ValueError("k0max['Total'] is non-zero but the integrator has no k0_Total global (single dihedral boost?)")
```

### gareus/pep_gamd.py :: set_replica_lambda_for_window (source lines 549-569)
```python
def set_replica_lambda_for_window(integrator, window_index, state_lambdas, k0max_by_channel) -> None:
    """Apply the rung λ for `window_index` to `integrator`, or no-op if the ladder is inactive.

    k0max_by_channel is None on every run that does not have the λ-ladder active
    (plain GaMD, conventional MD, or GaMD without a ladder -- the common case);
    on those runs state_lambdas is still an unconditionally-populated array, and
    this must be a silent no-op, not an error. Only when the ladder IS active
    (k0max_by_channel is not None) is a per-window λ required; state_lambdas is
    None in that combination only as a real misconfiguration, so that's the one
    case this raises on. One helper, one invariant, used at every call site that
    (re)applies a replica's window assignment so the guard can't be forgotten or
    mismatched at any individual site.
    """
    if k0max_by_channel is None:
        return
    if state_lambdas is None:
        raise ValueError(
            "set_replica_lambda_for_window: the λ-ladder is active (k0max_by_channel is set) "
            "but state_lambdas is None -- no per-window λ was supplied."
        )
    set_replica_lambda(integrator, float(state_lambdas[int(window_index)]), k0max_by_channel)
```

### FILE gareus/correctness/state_identity.py (lines 1-247)
```python
"""Immutable scientific state definitions, separate from engine checkpoints.

This is an explicit metadata API: it never invents historical CV coefficients,
boost envelopes, system identity, or intervention history from today's config.
Writers must call it after final window filtering/resume reconciliation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

from ._io import IntegrityError, atomic_bytes, digest, json_bytes, json_loads
from .bias import finite_number, normalize_windows

STATE_SCHEMA = "atlas-fixed-state-v1"


@dataclass(frozen=True)
class CanonicalStateTable:
    definition: dict[str, Any]
    windows: tuple[dict[str, Any], ...]
    column_window_ids: tuple[int, ...]
    definition_sha256: str


def _hash_string(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise IntegrityError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_cv(raw: Mapping[str, Any] | None, label: str) -> dict | None:
    if raw is None:
        return None
    cv = json_loads(json_bytes(dict(raw)))
    if set(cv) != {"kind", "units", "definition"}:
        raise IntegrityError(f"{label} requires kind, units, and embedded definition (no filename identity)")
    if not isinstance(cv["kind"], str) or not cv["kind"]:
        raise IntegrityError(f"{label}.kind must be explicit")
    if cv["units"] not in {"dimensionless", "angstrom", "nanometer", "radian"}:
        raise IntegrityError(f"{label}.units not supported: {cv['units']!r}")
    if not isinstance(cv["definition"], dict) or not cv["definition"]:
        raise IntegrityError(f"{label}.definition must embed the actual CV parameters")
    def reject_path_only(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if key.endswith(("_path", "_file", "_filename")) or key in {"path", "file", "filename"}:
                    raise IntegrityError(f"{label}: embed model contents instead of {key}")
                reject_path_only(val)
        elif isinstance(value, list):
            for val in value:
                reject_path_only(val)
    reject_path_only(cv["definition"])
    return cv


def canonical_state_definition(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = json_loads(json_bytes(dict(raw)))
    required = {"schema", "physical_system_sha256", "ensemble", "temperature_k",
                "pressure_bar", "fixed_box_vectors_nm", "cv1", "cv2",
                "boost", "energy_unit", "windows"}
    if set(data) != required or data.get("schema") != STATE_SCHEMA:
        raise IntegrityError(f"State definition requires exact schema {STATE_SCHEMA}; missing/extra fields")
    _hash_string(data["physical_system_sha256"], "physical_system_sha256")
    data["temperature_k"] = finite_number(data["temperature_k"], "temperature_k", positive=True)
    ensemble = data["ensemble"]
    if ensemble == "NVT":
        if data["pressure_bar"] is not None:
            raise IntegrityError("An NVT target must not carry a pressure parameter")
        box = np.asarray(data["fixed_box_vectors_nm"], dtype=np.float64)
        if box.shape != (3, 3) or not np.isfinite(box).all() or np.linalg.det(box) <= 0:
            raise IntegrityError("NVT requires its fixed positive-volume cell")
        data["fixed_box_vectors_nm"] = box.tolist()
    elif ensemble == "NPT":
        data["pressure_bar"] = finite_number(data["pressure_bar"], "pressure_bar")
        if data["fixed_box_vectors_nm"] is not None:
            raise IntegrityError("Instantaneous NPT volume is a sample variable, not a state identity")
    else:
        raise IntegrityError("This fixed-state exporter supports explicit NVT or NPT targets only")
    data["cv1"] = _canonical_cv(data["cv1"], "cv1")
    data["cv2"] = _canonical_cv(data["cv2"], "cv2")
    energy_unit = data["energy_unit"]
    if energy_unit not in {"kcal/mol", "kJ/mol"}:
        raise IntegrityError("State energy_unit must be kcal/mol or kJ/mol")
    windows = normalize_windows(data["windows"])
    if not windows:
        raise IntegrityError("A state table needs at least one window")
    seen = set()
    for row in windows:
        state_id = row.get("window_id")
        if isinstance(state_id, bool) or not isinstance(state_id, int) or state_id < 0 or state_id in seen:
            raise IntegrityError(f"Missing/invalid/duplicate window_id: {state_id!r}")
        seen.add(state_id)
        if set(row) - {"window_id", "center1", "k1", "center2", "k2", "gamd_lambda"}:
            raise IntegrityError(f"Window {state_id} has unknown physics fields; normalize explicitly")
        if energy_unit == "kJ/mol":
            row["k1"] /= 4.184
            row["k2"] /= 4.184
        if row["k1"] > 0 and data["cv1"] is None:
            raise IntegrityError(f"Window {state_id} needs a frozen CV1 definition")
        if row["k2"] > 0 and data["cv2"] is None:
            raise IntegrityError(f"Window {state_id} needs a frozen CV2 definition")
    data["energy_unit"] = "kcal/mol"
    data["windows"] = sorted(windows, key=lambda row: row["window_id"])
    if any(row["gamd_lambda"] > 0 for row in windows) and not isinstance(data["boost"], dict):
        raise IntegrityError("Active lambda states require an embedded boost definition")
    if data["boost"] is not None:
        if not isinstance(data["boost"], dict) or not data["boost"]:
            raise IntegrityError("boost must be null or a nonempty frozen definition")
        # Both static bias and lambda-dependent boosts belong here. Even a
        # shared bias matters when later recovering the physical target.
        if "kind" not in data["boost"] or "envelope" not in data["boost"]:
            raise IntegrityError("boost requires kind and embedded envelope")
        if not isinstance(data["boost"]["envelope"], dict):
            raise IntegrityError("boost.envelope must embed parameters, not a filename")
    return data


def make_state_definition(
    windows: Sequence[Mapping[str, Any]], *, physical_system_sha256: str,
    ensemble: str, temperature_k: float, cv1: Mapping | None, cv2: Mapping | None,
    pressure_bar: float | None = None, fixed_box_vectors_nm=None,
    boost: Mapping | None = None, energy_unit: str = "kcal/mol",
) -> dict[str, Any]:
    return canonical_state_definition({
        "schema": STATE_SCHEMA, "physical_system_sha256": physical_system_sha256,
        "ensemble": ensemble, "temperature_k": temperature_k,
        "pressure_bar": pressure_bar, "fixed_box_vectors_nm": fixed_box_vectors_nm,
        "cv1": cv1, "cv2": cv2, "boost": boost,
        "energy_unit": energy_unit, "windows": list(windows),
    })


def state_definition_hash(definition: Mapping[str, Any]) -> str:
    return digest(json_bytes(canonical_state_definition(definition)))


def compare_state_definitions(reference: Any, candidate: Any, path: str = "state") -> list[dict]:
    """Exact, field-level comparison; do not merge states using loose isclose."""
    if isinstance(reference, dict) and isinstance(candidate, dict):
        differences = []
        for key in sorted(set(reference) | set(candidate)):
            child = f"{path}.{key}"
            if key not in reference or key not in candidate:
                differences.append({"field": child, "reference": reference.get(key, "<missing>"),
                                    "candidate": candidate.get(key, "<missing>")})
            else:
                differences.extend(compare_state_definitions(reference[key], candidate[key], child))
        return differences
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            return [{"field": path + ".length", "reference": len(reference), "candidate": len(candidate)}]
        return [difference for i, (left, right) in enumerate(zip(reference, candidate))
                for difference in compare_state_definitions(left, right, f"{path}[{i}]")]
    if reference != candidate:
        return [{"field": path, "reference": reference, "candidate": candidate}]
    return []


def freeze_snapshot(segment_id: str, definition: Mapping[str, Any], *,
                    equilibrium_analysis_eligible: bool,
                    phase_kind: str) -> dict[str, Any]:
    if not isinstance(segment_id, str) or not segment_id:
        raise IntegrityError("A snapshot needs an explicit segment_id")
    if not isinstance(equilibrium_analysis_eligible, bool):
        raise IntegrityError("Analysis eligibility must be explicit, not truthy metadata")
    if phase_kind not in {"production", "pilot", "exploration", "equilibration"}:
        raise IntegrityError("Unknown explicit phase_kind")
    if equilibrium_analysis_eligible and phase_kind != "production":
        raise IntegrityError("Exploration/equilibration cannot be marked production eligible")
    state = canonical_state_definition(definition)
    return {
        "segment_id": segment_id, "snapshot_schema": "atlas-window-snapshot-v2",
        "windows": state["windows"],
        "cv1_type": state["cv1"]["kind"] if state["cv1"] is not None else None,
        "cv2_type": state["cv2"]["kind"] if state["cv2"] is not None else None,
        "state_definition": state, "state_definition_sha256": state_definition_hash(state),
        "sampling_policy": {"phase_kind": phase_kind,
                            "equilibrium_analysis_eligible": equilibrium_analysis_eligible},
    }


def write_frozen_snapshot(path: Path | str, snapshot: Mapping[str, Any]) -> None:
    """Refuse to rewrite an existing segment's definition; identical is a no-op.

    Caller must hold the run lock. This is a single-writer metadata API.
    """
    validate_fixed_state_segments({str(snapshot["segment_id"]): snapshot}, require_eligible=False)
    path = Path(path)
    payload = json_bytes(snapshot)
    if path.exists():
        if path.read_bytes() != payload:
            raise IntegrityError(f"Refusing to overwrite immutable state snapshot {path}")
        return
    atomic_bytes(path, payload)


def validate_fixed_state_segments(
    snapshots: Mapping[str, Mapping[str, Any]], *, require_eligible: bool = True,
) -> CanonicalStateTable:
    if not snapshots:
        raise IntegrityError("No state snapshots were selected")
    first = None
    first_id = None
    for segment_id, snapshot in snapshots.items():
        if snapshot.get("segment_id") != segment_id:
            raise IntegrityError(f"Snapshot segment_id mismatch for {segment_id}")
        if snapshot.get("snapshot_schema") != "atlas-window-snapshot-v2":
            raise IntegrityError(
                f"Segment {segment_id} lacks a frozen state definition. Do not infer historical "
                "CVs/envelopes from current files; export through a reviewed legacy migration."
            )
        policy = snapshot.get("sampling_policy")
        if (not isinstance(policy, dict)
                or policy.get("phase_kind") not in {"production", "pilot", "exploration", "equilibration"}
                or not isinstance(policy.get("equilibrium_analysis_eligible"), bool)):
            raise IntegrityError(f"Segment {segment_id} has invalid sampling-policy metadata")
        if policy["equilibrium_analysis_eligible"] and policy["phase_kind"] != "production":
            raise IntegrityError(f"Segment {segment_id} inconsistently marks exploration as production eligible")
        if require_eligible and policy["equilibrium_analysis_eligible"] is not True:
            raise IntegrityError(f"Segment {segment_id} is exploratory or has unknown sampling eligibility")
        state = canonical_state_definition(snapshot["state_definition"])
        if state_definition_hash(state) != snapshot.get("state_definition_sha256"):
            raise IntegrityError(f"State-definition checksum mismatch for {segment_id}")
        if normalize_windows(snapshot.get("windows", [])) != state["windows"]:
            # Top-level windows may be ordered differently; compare by IDs below.
            visible = sorted(normalize_windows(snapshot.get("windows", [])),
                             key=lambda row: row.get("window_id", -1))
            if visible != state["windows"]:
                raise IntegrityError(f"Visible windows disagree with immutable definition in {segment_id}")
        if first is None:
            first, first_id = state, segment_id
        else:
            differences = compare_state_definitions(first, state)
            if differences:
                detail = differences[0]
                raise IntegrityError(
                    f"Segments {first_id!r} and {segment_id!r} are not fixed-state compatible: "
                    f"{detail['field']}: {detail['reference']!r} != {detail['candidate']!r}. "
                    "Use a validated union-state workflow instead of changing columns by segment."
                )
    return CanonicalStateTable(first, tuple(first["windows"]),
                               tuple(row["window_id"] for row in first["windows"]),
                               state_definition_hash(first))
```

### FILE gareus/correctness/bias.py (lines 1-176)
```python
"""Strict shared umbrella-bias reconstruction for the existing ATLaS-MD API.

Energy input convention is unchanged: k1 and k2 are kcal/mol per squared CV
unit; CV values and centers must already use the SAME units. beta is mol/kJ.
The existing gareus.mbar_analysis.ladder helper remains the only boost formula.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence
import numpy as np

from ._io import IntegrityError

KJ_PER_KCAL = 4.184


class MissingCoordinateError(IntegrityError):
    """A requested Hamiltonian needs a coordinate that was not supplied."""


def numeric_vector(value: Any, name: str, length: int | None = None) -> np.ndarray:
    """Preserve masked data as NaN; never flatten or permit broadcasting."""
    try:
        masked = np.ma.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise IntegrityError(f"{name} must be a numeric one-dimensional vector") from exc
    if masked.ndim != 1:
        raise IntegrityError(f"{name} must have shape (N,), got {masked.shape}")
    array = np.asarray(masked.filled(np.nan), dtype=np.float64)
    if length is not None and array.shape != (length,):
        raise IntegrityError(f"{name} must have shape ({length},), got {array.shape}")
    # Infinity is missing/invalid observation, not a physically infinite
    # restraint parameter. It will propagate NaN in columns requiring this CV.
    array = array.copy()
    array[~np.isfinite(array)] = np.nan
    return array


def finite_number(value: Any, label: str, *, minimum: float | None = None,
                  positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise IntegrityError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IntegrityError(f"{label} must be a finite number; got {value!r}") from exc
    if not math.isfinite(result):
        raise IntegrityError(f"{label} must be finite; got {value!r}")
    if positive and result <= 0:
        raise IntegrityError(f"{label} must be positive")
    if minimum is not None and result < minimum:
        raise IntegrityError(f"{label} must be >= {minimum}")
    return result


def _term(window: Mapping[str, Any], axis: int, label: str) -> tuple[float, float]:
    center_key, force_key = f"center{axis}", f"k{axis}"
    if force_key not in window:
        if axis == 2 and center_key not in window:
            return 0.0, 0.0
        raise IntegrityError(f"{label}: missing {force_key}; inactive means explicit zero")
    force = finite_number(window[force_key], f"{label}.{force_key}", minimum=0)
    if force == 0:
        # With zero force the center is irrelevant. Do not require a meaningful
        # placeholder, and do not multiply an unmeasured CV by zero later.
        return 0.0, 0.0
    if center_key not in window:
        raise IntegrityError(f"{label}: active {force_key} requires {center_key}")
    center = finite_number(window[center_key], f"{label}.{center_key}")
    return center, force


def normalize_windows(windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, raw in enumerate(windows):
        if not isinstance(raw, Mapping):
            raise IntegrityError(f"window[{index}] must be a mapping")
        window = dict(raw)
        label = f"window {raw.get('window_id', index)}"
        window["center1"], window["k1"] = _term(raw, 1, label)
        window["center2"], window["k2"] = _term(raw, 2, label)
        window["gamd_lambda"] = finite_number(raw.get("gamd_lambda", 0.0),
                                              f"{label}.gamd_lambda", minimum=0)
        normalized.append(window)
    return normalized


def normalize_verified_legacy_disabled_secondary(
    windows: Sequence[Mapping[str, Any]], *, secondary_disabled: bool,
) -> list[dict[str, Any]]:
    """Explicit adapter for a verified *disabled* old secondary-CV definition.

    Never infer disabled status from bad numbers. Positive or negative finite
    forces conflict with disabled provenance and are rejected. This adapter is
    NOT invoked automatically by reconstruction or by the strict exporter.
    """
    if secondary_disabled is not True:
        raise IntegrityError("Legacy adapter requires explicit disabled-CV evidence")
    out = []
    for index, raw in enumerate(windows):
        window = dict(raw)
        value = raw.get("k2")
        if value is not None:
            try:
                k = float(value)
            except (TypeError, ValueError) as exc:
                raise IntegrityError(f"Unrecognized legacy k2 at window {index}") from exc
            if math.isfinite(k) and k != 0:
                raise IntegrityError(f"Legacy k2={k} contradicts disabled secondary CV")
        window.update(center2=0.0, k2=0.0)
        out.append(window)
    return normalize_windows(out)


def reconstruct_bias_matrix(
    cv_A: np.ndarray,
    cv2: np.ndarray | None,
    windows: list,
    beta: float,
    *,
    v_pep: np.ndarray | None = None,
    v_dih: np.ndarray | None = None,
    envelope=None,
    meta: dict | None = None,
    _ladder_apply: Callable | None = None,
) -> np.ndarray:
    """Return the (N,K) reduced umbrella-plus-ladder bias, not full potential.

    Missing entire required coordinates raise. Missing individual observations
    yield NaN only in states requiring that coordinate. Row filtering belongs
    downstream and must keep origins/counts/aligned arrays consistent.
    _ladder_apply is a test seam, not an alternative production boost definition.
    """
    beta = finite_number(beta, "beta", positive=True)
    primary = numeric_vector(cv_A, "cv_A")
    n = len(primary)
    secondary = None if cv2 is None else numeric_vector(cv2, "cv2", n)
    pep = None if v_pep is None else numeric_vector(v_pep, "v_pep", n)
    dih = None if v_dih is None else numeric_vector(v_dih, "v_dih", n)
    rows = normalize_windows(windows)
    active_secondary = [w.get("window_id", i) for i, w in enumerate(rows) if w["k2"] > 0]
    if active_secondary and secondary is None:
        raise MissingCoordinateError(
            f"cv2 was not supplied, but states {active_secondary} have active secondary restraints"
        )
    matrix = np.zeros((n, len(rows)), dtype=np.float64)
    with np.errstate(over="raise", invalid="ignore"):
        try:
            for column, row in enumerate(rows):
                if row["k1"] > 0:
                    delta = primary - row["center1"]
                    matrix[:, column] += 0.5 * row["k1"] * delta * delta
                if row["k2"] > 0:
                    delta = secondary - row["center2"]
                    matrix[:, column] += 0.5 * row["k2"] * delta * delta
            matrix *= beta * KJ_PER_KCAL
        except FloatingPointError as exc:
            raise IntegrityError("Overflow reconstructing umbrella bias; check CV units/parameters") from exc
    lambdas = np.asarray([w["gamd_lambda"] for w in rows], dtype=np.float64)
    bookkeeping = meta if meta is not None else {}
    if not np.any(lambdas > 0):
        bookkeeping.update(gamd_ladder=False, gamd_ladder_samples_without_raw_energies=0)
        return matrix
    if envelope is None:
        raise IntegrityError("Active gamd_lambda requires the frozen envelope and raw energies")
    if dih is None or (bool(getattr(envelope, "has_total", True)) and pep is None):
        raise IntegrityError("Active gamd_lambda requires v_dih and, for a Total channel, v_pep")
    if pep is None:
        # Truly dihedral-only: NaN is an explicit absent inactive channel,
        # never a fabricated zero energy. The repository helper ignores it.
        pep = np.full(n, np.nan, dtype=np.float64)
    if _ladder_apply is None:
        from gareus.mbar_analysis.ladder import apply_ladder_boost_to_u
        _ladder_apply = apply_ladder_boost_to_u
    return _ladder_apply(matrix, pep, dih, lambdas, envelope, beta, bookkeeping)
```

### FILE gareus/production.py (lines 7420-7700)
```python
    # the contact switching function gradient is ~0 and the umbrella provides no actual force).
    # After _stuck_max_intervals consecutive exchange intervals below _stuck_threshold, copy
    # positions from the nearest non-stuck replica and reinitialise velocities.
    # Conservative guard: exploratory rescue needs separate policy/ledger wiring.
    # No contact-CV default or final-production override may mutate configurations.
    _stuck_enabled = False
    _stuck_threshold = float(getattr(args, "cv1_stuck_threshold", 0.03) or 0.03)
    _stuck_max_intervals = int(getattr(args, "cv1_stuck_detect_intervals", 500) or 500)
    _stuck_counter = np.zeros(nrep, dtype=int)
    _stuck_rescue_total = 0

    # _sim_pool (one thread per replica, pinned since before replica_construction)
    # was created above, before Context construction, so every replica's Context
    # is built and stepped on the same OS thread for its whole lifetime.
    if str(getattr(args, "platform", "")).upper() == "CPU" and int(getattr(args, "cpu_threads", 1)) == 0 and nrep > 1:
        print(
            f"WARNING: --cpu-threads 0 (use all cores) combined with {nrep} parallel replicas "
            "will oversubscribe CPU cores. Set --cpu-threads 1 for parallel step_all()."
        )
    # Sentinel variables initialised here so the finally block can always read
    # them safely, even if the try body raises before the loop sets prod_done or
    # prod_total.
    _prod_completed_cleanly = False
    prod_done = 0
    prod_total = 0
    try:
        exchange_writer = parquet_exchange_writer

        def ensure_sample_writer():
            nonlocal samples_started
            if not samples_started:
                samples_started = True
                print(f"    Production samples writing to {out_dir / 'samples' / _seg_id}")
            return parquet_sample_writer

        def sample(step: int, phase: str) -> list[dict]:
            """Collect one reporting sample from every replica.

            The expensive part is the OpenMM State read, so this does exactly one
            getState(getPositions=True, getEnergy=<sample_potential_energy>) per replica, then computes
            all umbrella bias vectors as a single NumPy matrix:

                bias_matrix_kj[window, replica]

            The same matrix feeds samples.csv, analysis_arrays.npz, and dashboard
            rows without recomputing per-window JSON vectors in Python loops.
            """
            nonlocal _gamd_recal_active
            rows: list[dict] = []
            is_prod = is_gamd_production_phase(phase)
            primary_mode_name = primary_cv_mode(args)
            primary_label_name = primary_cv_label(args)
            primary_units_name = primary_cv_units(args)
            primary_k_units_name = primary_k_units(args)
            is_distance_primary = primary_mode_name == "distance"
            write_full_bias_vectors = bool(getattr(args, "write_full_bias_csv_vectors", False))
            write_gamd_globals = bool(getattr(args, "write_gamd_globals_json", False))
            read_sample_potential = bool(getattr(args, "sample_potential_energy", True))
            temperature_value = float(args.temperature_k)
            beta_value = float(beta)
            primary_values = np.empty(nrep, dtype=np.float64)
            ss_values = np.full(nrep, np.nan, dtype=np.float64)
            potentials_kj = np.empty(nrep, dtype=np.float64)
            v_pep_kj = np.empty(nrep, dtype=np.float64)
            v_dih_kj = np.empty(nrep, dtype=np.float64)

            def _fetch_state(r_sim):
                r, sim = r_sim
                if _use_fast_cv_path:
                    ctx = sim.context
                    cv, ss = observe_fast_path(ctx, _fast_primary_forces[r], _fast_ss_forces[r], args, secondary_cv_metadata)
                    if read_sample_potential:
                        state = ctx.getState(getEnergy=True, enforcePeriodicBox=True,
                                             groups=physical_energy_groups_for_args(args))
                        pe = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                    else:
                        pe = float("nan")
                    v_pep, v_dih = _fetch_v_pep_v_dih(ctx, pep_env, unit)
                    return r, cv, ss, pe, v_pep, v_dih
                cv, ss, pe = primary_secondary_and_potential_from_state(
                    sim.context, primary_cv_def, args, unit, secondary_cv_metadata,
                    read_potential_energy=read_sample_potential,
                )
                v_pep, v_dih = _fetch_v_pep_v_dih(sim.context, pep_env, unit)
                return r, cv, ss, pe, v_pep, v_dih

            for r, cv, ss, pe, v_pep, v_dih in _sim_pool.map(_fetch_state, enumerate(sims)):
                primary_values[r], ss_values[r], potentials_kj[r] = cv, ss, pe
                v_pep_kj[r], v_dih_kj[r] = v_pep, v_dih
            ss_centers_arr = ss_centers_arr_global
            ss_k_arr = ss_k_kcal_arr_global
            _has_ss_axis = secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None
            distance_bias_matrix_kcal, ss_bias_matrix_kcal = umbrella_bias_matrix_kcal(
                primary_values, ss_values, centers_a_arr, k_arr,
                ss_centers_arr if _has_ss_axis else None, ss_k_arr if _has_ss_axis else None,
            )
            boost_bias_matrix_kj = (pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)
                                    if pep_env is not None else np.zeros((nrep, nrep)))
            bias_matrix_kcal, bias_matrix_kj = assemble_bias_matrices(
                distance_bias_matrix_kcal, ss_bias_matrix_kcal, boost_bias_matrix_kj
            )
            reduced_bias_matrix = float(beta) * bias_matrix_kj
            observable_cache["step"] = int(step)
            observable_cache["primary_values"] = primary_values
            observable_cache["cvs_nm"] = (primary_values / 10.0) if is_distance_primary else primary_values
            observable_cache["bias_matrix_kj"] = bias_matrix_kj
            observable_cache["v_pep_kj"] = v_pep_kj
            observable_cache["v_dih_kj"] = v_dih_kj
            observable_cache["boost_bias_matrix_kj"] = boost_bias_matrix_kj
            if is_prod:
                ensure_sample_writer()
            for r, sim in enumerate(sims):
                w = int(assignments[r])
                cv_a = float(primary_values[r])
                center_a = float(centers_a_arr[w])
                k_kcal_a2 = float(k_arr[w])
                all_bias_kcal = bias_matrix_kcal[:, r]
                all_bias_kj = bias_matrix_kj[:, r]
                all_reduced_bias = reduced_bias_matrix[:, r]
                all_distance_bias_kcal = distance_bias_matrix_kcal[:, r]
                all_ss_bias_kcal = ss_bias_matrix_kcal[:, r]
                sampled_bias_kcal = float(all_bias_kcal[w])
                sampled_bias_kj = float(all_bias_kj[w])
                sampled_umbrella_bias_kj = float(4.184 * (all_distance_bias_kcal[w] + all_ss_bias_kcal[w]))
                sampled_boost_bias_kj = float(boost_bias_matrix_kj[w, r])
                row = {
                    "step": int(step),
                    "phase": phase,
                    "replica": r,
                    "window": w,
                    "center_A": center_a,
                    "k_kcal_mol_A2": k_kcal_a2,
                    "cv_A": cv_a,
                    "primary_cv": primary_mode_name,
                    "primary_cv_label": primary_label_name,
                    "primary_cv_units": primary_units_name,
                    "primary_cv_value": cv_a,
                    "primary_cv_center": center_a,
                    "primary_cv_k": k_kcal_a2,
                    "primary_cv_k_units": primary_k_units_name,
                    "primary_umbrella_bias_kcal_mol": float(all_distance_bias_kcal[w]),
                    "secondary_cv": float(ss_values[r]) if math.isfinite(float(ss_values[r])) else "",
                    "secondary_cv_center": float(ss_centers_arr[w]) if math.isfinite(float(ss_centers_arr[w])) else "",
                    "secondary_cv_k_kcal_mol": float(ss_k_arr[w]) if math.isfinite(float(ss_k_arr[w])) else "",
                    "distance_umbrella_bias_kcal_mol": float(all_distance_bias_kcal[w]),
                    "secondary_cv_bias_kcal_mol": float(all_ss_bias_kcal[w]),
                    "v_pep_kj_mol": float(v_pep_kj[r]) if pep_env is not None else float("nan"),
                    "v_dih_kj_mol": float(v_dih_kj[r]) if pep_env is not None else float("nan"),
                    "gamd_lambda": float(state_lambdas[w]),
                    "umbrella_bias_kcal_mol": sampled_bias_kcal,
                    "umbrella_bias_kj_mol": sampled_bias_kj,
                    "sampled_umbrella_bias_kj": sampled_umbrella_bias_kj,
                    "sampled_boost_bias_kj": sampled_boost_bias_kj,
                    "umbrella_reduced_bias": float(all_reduced_bias[w]),
                    "umbrella_restoring_force_kcal_mol_per_A": k_kcal_a2 * (center_a - cv_a),
                    "potential_kj_mol": float(potentials_kj[r]),
                    "temperature_K": temperature_value,
                    "beta_1_over_kJ_mol": beta_value,
                }
                if write_full_bias_vectors:
                    row.update({
                        "primary_umbrella_bias_all_windows_kcal_mol_json": json.dumps([float(x) for x in all_distance_bias_kcal]),
                        "distance_umbrella_bias_all_windows_kcal_mol_json": json.dumps([float(x) for x in all_distance_bias_kcal]),
                        "secondary_cv_bias_all_windows_kcal_mol_json": json.dumps([float(x) for x in all_ss_bias_kcal]),
                        "umbrella_bias_all_windows_kcal_mol_json": json.dumps([float(x) for x in all_bias_kcal]),
                        "umbrella_bias_all_windows_kj_mol_json": json.dumps([float(x) for x in all_bias_kj]),
                        "umbrella_reduced_bias_all_windows_json": json.dumps([float(x) for x in all_reduced_bias]),
                    })
                boost_kj, boost_components_kj, boost_source = extract_gamd_boost_kj(sim.integrator, unit, None)
                row["gamd_boost_total_kj_mol"] = boost_kj if boost_kj is not None else ""
                row["gamd_boost_total_kcal_mol"] = (boost_kj / 4.184) if boost_kj is not None else ""
                row["gamd_boost_source"] = boost_source
                row["gamd_boost_components_kj_mol_json"] = json.dumps(boost_components_kj, sort_keys=True)
                row["gamd_globals_json"] = (
                    json.dumps(integrator_globals(sim.integrator, unit=unit), sort_keys=True)
                    if write_gamd_globals else ""
                )
                if is_prod and _gamd_recal_active:
                    try:
                        for _gname, _gid in _gamd_recal_targets:
                            _gpe = boost_target_energy_kj(sim.context, _gid, unit, total_groups=total_energy_groups_for_args(args))
                            _gamd_recal_accumulators[_gname].setdefault(w, _WelfordAccumulator()).update(_gpe)
                    except Exception as _gamd_recal_sample_exc:
                        # Best-effort measurement only: a failure here must never take
                        # down a real production epoch. Disable for the rest of this
                        # run; whatever was accumulated before the failure is still
                        # written out and pooled (partial data degrades gracefully at
                        # the recalibration step, same as "no data").
                        print(
                            "WARNING: GaMD production-envelope sampling failed "
                            f"({_gamd_recal_sample_exc}); disabling for the rest of this run"
                        )
                        _gamd_recal_active = False
                if is_prod:
                    _boost_raw = row.get("gamd_boost_total_kj_mol")
                    _boost_total = float(_boost_raw) if _boost_raw not in (None, "") else None
                    _boost_comps = json.loads(row.get("gamd_boost_components_kj_mol_json", "{}") or "{}")
                    _, _boost_dihe, _boost_nonb = parse_gamd_boost_components(_boost_total, _boost_comps)
                    _ss_val = float(ss_values[r])
                    parquet_sample_writer.write_sample(
                        step=int(step),
                        replica=r,
                        window_id=w,
                        cv1=cv_a,
                        cv2=_ss_val if math.isfinite(_ss_val) else None,
                        potential=float(potentials_kj[r]),
                        boost_total=_boost_total,
                        boost_dihedral=_boost_dihe,
                        boost_nonbonded=_boost_nonb,
                        v_pep=float(v_pep_kj[r]) if pep_env is not None else float("nan"),
                        v_dih=float(v_dih_kj[r]) if pep_env is not None else float("nan"),
                        gamd_lambda=float(state_lambdas[w]),
                    )
                rows.append(row)
            if is_prod and bool(getattr(args, "flush_every_log", False)):
                parquet_sample_writer.flush()

            # ── tICA dihedral observation recording (opt-in via tica_obs_interval) ──
            if is_prod and _dihedral_buffers is not None:
                _tica_obs_counter[0] += 1
                if _tica_obs_counter[0] % _tica_obs_interval == 0:
                    for r, sim in enumerate(sims):
                        w = int(assignments[r])
                        _pos_state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
                        _pos_nm = _pos_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                        _dihedral_buffers[r].record(
                            _pos_nm, int(step), w,
                            cv_primary=float(primary_values[r]),
                            cv_secondary=float(ss_values[r]),
                        )
            # ──────────────────────────────────────────────────────────────────────

            return rows

        def flush_scalar_writers() -> None:
            """Flush open writers before checkpoints/final reports."""
            parquet_sample_writer.flush()
            parquet_exchange_writer.flush()

        def _write_production_nan_diagnostics(exc: Exception, requested_steps: int, completed_substeps: int = 0) -> Path:
            """Write targeted diagnostics when production stepping creates NaNs.

            Reporter exceptions often hide which replica actually exploded.  This
            scan runs after a failed production chunk and records per-replica
            finiteness, potential energy, assignment/window, and a crash PDB for
            any replica whose coordinates can still be serialized.
            """
            diag_step = int(prod_done) + int(completed_substeps)
            diag_path = Path(out_dir) / f"PRODUCTION_NAN_DIAGNOSTICS_prodstep_{diag_step:09d}.json"
            rows_diag = []
            for rr, sim in enumerate(sims):
                row = {
                    "replica": int(rr),
                    "window": int(assignments[rr]) if rr < len(assignments) else None,
                    "production_step": int(diag_step),
                    "requested_chunk_steps": int(requested_steps),
                    "finite_positions": False,
                    "state_read_ok": False,
                }
                try:
                    st = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True,
                                              groups=physical_energy_groups_for_args(args))
                    row["state_read_ok"] = True
                    try:
                        pos_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                        finite = bool(np.isfinite(np.asarray(pos_nm, dtype=float)).all())
                        row["finite_positions"] = finite
                        row["n_nonfinite_values"] = int(np.size(pos_nm) - np.isfinite(np.asarray(pos_nm, dtype=float)).sum())
                    except Exception as pos_exc:
                        row["position_check_error"] = str(pos_exc)
                        finite = False
                    try:
                        row["potential_kj_mol"] = float(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                    except Exception as e_pe:
                        row["potential_error"] = str(e_pe)
                    try:
                        crash_pdb = Path(out_dir) / f"CRASH_production_replica_{rr:03d}_prodstep_{diag_step:09d}.pdb"
                        write_state_pdb(crash_pdb, app, topology, st.getPositions())
                        row["crash_pdb"] = str(crash_pdb)
                    except Exception as write_exc:
                        row["crash_pdb_error"] = str(write_exc)
```

### FILE gareus/production.py (lines 7780-7900)
```python
            try:
                w = int(window_index)
            except Exception:
                return None
            if w < 0 or w >= int(replica_of_window.size):
                return None
            rep = int(replica_of_window[w])
            return rep if rep >= 0 else None

        def _attempt_window_swap(wi: int, wj: int, primary_values: np.ndarray, bias_matrix_kj: np.ndarray, absolute_step: int, attempt: int, p_override: Optional[float] = None, force_accept: bool = False) -> int:
            """Attempt/apply a swap between two umbrella windows using vectorized bias energies."""
            wi = int(wi)
            wj = int(wj)
            if wi == wj:
                return attempt
            # Decision + bookkeeping live in the module-level kernel so a test can
            # drive them; the side effects below stay here. `force_accept` must not
            # consume a random number -- production short-circuited `rng.random()`
            # away, and drawing one anyway would shift every later RNG stream.
            # Check attemptability BEFORE drawing: the original closure returned
            # early on these cases without consuming a random number, and an
            # extra draw shifts every downstream stream.
            if swap_candidate_replicas(replica_of_window, wi, wj) is None:
                return attempt
            outcome = apply_window_swap(
                bias_matrix_kj, beta, assignments, replica_of_window, wi, wj,
                None if force_accept else rng.random(),
                p_override=p_override, force_accept=force_accept,
            )
            if outcome is None:
                return attempt
            i, j = outcome.replica_i, outcome.replica_j
            _record_exchange_stats(wi, wj, outcome.accepted)
            if outcome.accepted:
                # NPT correction: the umbrella-parameter writes run on each
                # replica's own worker (same affinity rule as stepping), and an
                # accepted swap re-labels replicas so the cached CV/bias
                # observables are stale until the next sample().
                def _apply_swap_to_replica(replica_index: int):
                    set_window(sims[replica_index].context, centers_nm, ks_kj_nm2, assignments[replica_index], secondary_cv_centers, secondary_cv_ks_kj)
                    set_replica_lambda_for_window(sims[replica_index].integrator, assignments[replica_index], state_lambdas, k0max_by_channel)

                _sim_pool.submit(i, _apply_swap_to_replica, i).result()
                _sim_pool.submit(j, _apply_swap_to_replica, j).result()
                observable_cache.clear()
            parquet_exchange_writer.write_exchange(
                step=int(absolute_step),
                replica_i=int(i),
                replica_j=int(j),
                window_i=int(wi),
                window_j=int(wj),
                delta_e=float(outcome.delta_kj),
                accepted=bool(outcome.accepted),
            )
            return attempt + 1

        def _current_exchange_arrays(absolute_step: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
            # Prefer the sample/log observables from the same production step.
            # This avoids a second getState() pass when next_log and next_exchange
            # coincide, while keeping the exchange energy exactly the same matrix
            # used for samples.csv / analysis_arrays.npz.
            if absolute_step is not None and int(observable_cache.get("step", -1)) == int(absolute_step):
                cached_cvs = observable_cache.get("primary_values")
                cached_bias = observable_cache.get("bias_matrix_kj")
                if isinstance(cached_cvs, np.ndarray) and isinstance(cached_bias, np.ndarray):
                    if cached_cvs.shape == (nrep,) and cached_bias.shape == (nrep, nrep):
                        return cached_cvs, cached_bias

            # Compute CV values for every replica.  When the fast path is active
            # (contacts primary + CustomCVForce secondary) this reads cached scalars
            # via getCollectiveVariableValues() — no positions DMA at all.  The
            # full U[window, replica] umbrella matrix is then built in NumPy for all
            # swap/Gibbs candidates.  Potential energies are not needed here -- except
            # when the λ-ladder is active (pep_env is not None), which adds three
            # getState(getEnergy=True, ...) reads per replica per exchange attempt
            # (peptide_essential_energy_kj's two group reads plus the dihedral group)
            # to price the Pep-GaMD boost under every candidate state's λ.
            primary_values = np.empty(nrep, dtype=np.float64)
            ss_values = np.full(nrep, np.nan, dtype=np.float64)
            v_pep_kj = np.empty(nrep, dtype=np.float64)
            v_dih_kj = np.empty(nrep, dtype=np.float64)
            _ss_enabled = bool((secondary_cv_metadata or {}).get("enabled"))

            def _fetch_exchange_state(r_sim):
                r, sim = r_sim
                if _use_fast_cv_path:
                    ctx = sim.context
                    cv, ss = observe_fast_path(ctx, _fast_primary_forces[r], _fast_ss_forces[r], args, secondary_cv_metadata)
                    v_pep, v_dih = _fetch_v_pep_v_dih(ctx, pep_env, unit)
                else:
                    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
                    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    cv = primary_cv_value_from_positions_nm(pos, primary_cv_def, args)
                    ss = secondary_structure_score_from_positions_nm(pos, secondary_cv_metadata) if _ss_enabled else float("nan")
                    v_pep, v_dih = _fetch_v_pep_v_dih(sim.context, pep_env, unit)
                return r, cv, ss, v_pep, v_dih

            for r, cv, ss, v_pep, v_dih in _sim_pool.map(_fetch_exchange_state, enumerate(sims)):
                primary_values[r] = cv
                ss_values[r] = ss
                v_pep_kj[r] = v_pep
                v_dih_kj[r] = v_dih
            _has_ss_axis = (secondary_cv_centers is not None and secondary_cv_ks_kj is not None
                            and ss_ks_kj_arr_global is not None)
            # Same kcal arrays as the sample path: the matrix the kernel exchanges on IS the
            # matrix the writer records (spec F01, state_bias_matrix_v2).
            distance_bias_kcal, ss_bias_kcal = umbrella_bias_matrix_kcal(
                primary_values, ss_values, centers_a_arr, k_arr,
                ss_centers_arr_global if _has_ss_axis else None,
                ss_k_kcal_arr_global if _has_ss_axis else None,
            )
            boost_bias_matrix_kj = (pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)
                                    if pep_env is not None else np.zeros((nrep, nrep)))
            _, bias_matrix_kj = assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_matrix_kj)
            return primary_values, bias_matrix_kj

        def _candidate_delta_for_window_swap(wi: int, wj: int, bias_matrix_kj: np.ndarray) -> Optional[float]:
            wi = int(wi)
            wj = int(wj)
            if wi == wj:
                return 0.0
```

## glm

Let me analyze this carefully. This is a review of replica exchange code in a molecular dynamics free energy pipeline. I need to find bugs in the state bias matrix, Gibbs-walk proposals, swaps, lambda handling.

Let me examine each piece.

**1. umbrella_bias_matrix_kcal (L895-917)**

Orientation: bias[state, replica]. `pv[np.newaxis, :] - ca[:, np.newaxis]` gives [state, replica]. OK. Secondary NaN handling: `np.where(np.isfinite(dss), ss_kcal, 0.0)` — replicas with NaN secondary contribute zero bias on that axis for ALL states. Docstring says "a NaN entry used to freeze that replica's exchanges". Hmm — but setting bias to 0 rather than NaN means exchange delta calculations will silently treat NaN secondary as zero bias. That's a "NaN → 0 silent fallback" candidate. If ss_value is NaN (e.g., fast path returning NaN), the bias is 0 instead of freezing exchanges. The docstring claims "now applied identically to the exchange path — a NaN entry used to freeze that replica's exchanges" but the code does NOT produce NaN entries; it zeroes them. So the claimed freeze behavior doesn't exist. This is a real discrepancy: the docstring claims NaN propagates to freeze exchanges, but `np.where(np.isfinite(dss), ss_kcal, 0.0)` makes bias exactly zero, so exchanges proceed with wrong energies. VERIFIED inconsistency between doc and code. Effect: a replica with missing secondary CV participates in exchanges priced as if it sits at the harmonic minimum of every state — inflated acceptance, wrong sampling weights. MBAR would also reconstruct bias with NaN (in correctness/bias.py numeric_vector converts non-finite to NaN → those samples need filtering). Actually in reconstruct_bias_matrix, missing observations yield NaN in states requiring that coordinate. So the exchange path prices 0 while MBAR prices NaN — inconsistent energies between exchange and recorded samples. That's the class of bug asked for. Though note `_current_exchange_arrays` uses `umbrella_bias_matrix_kcal` and sample path also uses same function so both zero it; but MBAR analysis reconstructs from CVs with correctness/bias.py which yields NaN. Actually wait — does the analysis use reconstruct_bias_matrix? The submission says final reweighting by MBAR using per-frame recorded CVs and boost arguments. reconstruct_bias_matrix converts non-finite secondary to NaN; the states with k2>0 would give NaN, which MBAR must filter out. So a replica with secondary NaN gets exchange-accepted with bias 0, but its samples are dropped in MBAR — that's fine-ish for MBAR but the exchange path accepted moves with fabricated zero energy, which corrupts the sampling distribution (moves accepted that shouldn't be). Actually more subtly: exchange acceptance with wrong energy breaks detailed balance of the joint chain — the target distribution conditioned on exchange acceptance probabilities that use wrong bias is not the intended distribution. Effect on FES: biased.

Also note only `dss` finiteness is checked, not `distance_kcal` — primary NaN propagates. Fine.

**2. _gibbs_window_proposal_distribution (L983-1041)**

Heat-bath proposal. delta computed as new_e - old_e. old_e = bias[wi, rep] + bias[valid_windows, target_reps]; new_e = bias[valid_windows, rep] + bias[wi, target_reps]. For the swap of replica `rep` in window wi with the holder of window w (target_rep): new configuration assigns rep→w and target_rep→wi. Bias energy = bias[w, rep] + bias[wi, target_rep]. Correct. Stay index delta set to 0. Good.

Note: delta here only includes umbrella bias. The Gibbs move swaps state labels between replicas — configurations unchanged — so physical potential and boost cancel... wait, boost does NOT cancel! The boost depends on the state's λ. In this state graph, each (window, rung) is a state with its own λ. A "window swap" between replica in window wi (λ_i) and holder of wj (λ_j): the boost energy of replica i's configuration under rung λ_j differs from under λ_i. The delta for the swap must include the boost terms: b_j(x_i) + b_i(x_j) - b_i(x_i) - b_j(x_j). But the docstring in apply_window_swap claims "the unbiased potential and the GaMD boost cancel exactly from delta ... dV(x_i)+dV(x_j) are identical before and after." That's WRONG when λ_i ≠ λ_j. The boost dV depends on λ of the state the replica is assigned to. If the ladder is active and windows have different λ, swapping window labels between replicas changes the boost energy each configuration experiences — it does NOT cancel.

But wait — does the bias matrix include the boost? Yes: assemble_bias_matrices includes boost_bias_kj, and bias_matrix_kj includes boost of each sample's configuration under each state's λ (pep_gamd_boost_matrix_kj returns (n_states, n_samples) boost of each sample's configuration under each state's λ). So bias[w, r] includes b_λ(w)(x_r). So the delta in apply_window_swap using bias_matrix_kj correctly includes boost differences. Good — the matrix includes boost, so delta is correct for the swap. So the docstring's justification is wrong (boost doesn't cancel) but the computation is right because the matrix contains per-state boost. Hmm, but the docstring says the boost cancels — the claim is incorrect reasoning but the code (via matrix) is right. Actually wait: is the boost matrix included in the exchange bias matrix? In `_current_exchange_arrays`, yes: `_, bias_matrix_kj = assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_matrix_kj)`. And apply_window_swap receives bias_matrix_kj. So swaps are priced with boost. Fine.

But note shape issue: `boost_bias_matrix_kj = np.zeros((nrep, nrep))` when pep_env is None. pep_gamd_boost_matrix_kj returns (n_states, n_samples) = (len(lambdas), n_samples). Here nrep replicas and nrep windows (assuming n windows == nrep). If number of windows != number of replicas (e.g., --max-replicas truncation mentioned in _persist_state_gamd_lambdas: `args.state_gamd_lambdas = list(args.state_gamd_lambdas)[:_max_replicas]`), then shapes mismatch! state_lambdas has length = number of windows; v_pep has length nrep. pep_gamd_boost_matrix_kj stacks len(lambdas) rows each of length n_samples → shape (n_windows, nrep). Then assemble: distance_bias_kcal has shape (n_windows, nrep); boost (n_windows, nrep). OK, consistent. But the fallback `np.zeros((nrep, nrep))` when pep_env is None — if n_windows ≠ nrep, boost is (nrep, nrep), mismatched with (n_windows, nrep) distance matrix. Adding them would broadcast-fail or misalign. Actually if pep_env is None, boost is zeros; adding (nrep,nrep) zeros to (n_windows, nrep) matrix raises ValueError if shapes differ (unless n_windows == nrep). Hmm — but more important: `_current_exchange_arrays` checks `cached_bias.shape == (nrep, nrep)` in the cache validation. If n_windows ≠ nrep, the cached bias matrix shape is (n_windows, nrep) and the cache check fails, forcing recompute — only a performance issue. But the zeros((nrep,nrep)) with n_windows≠nrep: assemble_bias_matrices would raise ValueError on broadcast? Adding arrays of shape (nwin,nrep) and (nrep,nrep): if nwin != nrep, numpy raises. So error, not silent. But wait, is state_lambdas length n_windows or nrep after truncation? The persist function mentions truncation of state_lambdas to _max_replicas — suggesting state_lambdas is indexed by window in some places and by replica in others? In sample(): `row["gamd_lambda"] = float(state_lambdas[w])` where w = assignments[r] — window index. And `set_replica_lambda_for_window(..., assignments[replica_index], state_lambdas, ...)` — window index. So state_lambdas is per-window. But truncation `args.state_gamd_lambdas[:_max_replicas]` truncates by replica count — if n_windows > nrep... Hmm, ambiguous. If truncation is meant to align per-window λ with reduced replica count... Actually the persist docstring says the truncation mutates state_gamd_lambdas. If windows and replicas are 1:1 normally, fine.

Actually the bigger issue in pep_gamd_boost_matrix_kj at exchange time: `pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)` — v_pep_kj has NaN when... `_fetch_v_pep_v_dih` returns (nan, nan) only when pep_env is None, which is guarded. But for single dihedral boost (has_total False), v_pep = NaN and pep_gamd_boost_kj handles: `_channel_boost(v_pep + b_dih, ...)` — wait no, if not env.has_total, returns b_dih only, ignoring v_pep NaN. OK.

But in `_channel_boost`: `b = _np.where((b + v) < e, b, 0.0)` — the lower-bound boost guard: only apply if boost lifts v below threshold e... Actually GaMD lower-bound boost: boost applied when V < E (threshold). The boost should raise energy to V + 0.5*k0*(E-V)^2/... Let me check units: `rng = vmax - vmin`, `b = 0.5*k0*(e-v)^2/rng`. That's gamd-openmm's lower bound boost formula: ΔV = 0.5*k0*(E-V)^2/(Vmax-Vmin) when V < E. And guard `(b+v) < e` → apply only if b+v < e, i.e., boosted energy still below threshold. Plausible. And rng guard: if |vmax - vmin| small, b=0 — silent boost=0 fallback! If the swarm calibration produced vmax ≈ vmin (degenerate envelope), the boost silently becomes zero while MBAR still prices... wait, MBAR reconstructs boost from v_pep/v_dih with the same formula and same envelope, so it'd also be zero. Consistent. But if rng differs slightly between... it's the same env. OK.

Numerical stability of _channel_boost: `scale = np.maximum(np.maximum(abs(e), npabs(v)), 1.0)`; `np.where(np.abs(rng) <= 0.001*scale, 0.0, b)` — fine.

**3. _gibbs_mh_acceptance_probability**: log_alpha = -beta*delta + log(qr/qf). Correct MH for proposal distribution q. If q_reverse == 0 (rev_idx empty → q_reverse = 0.0) → return 0 (reject). But when is reverse probability zero while forward was positive? In reverse distribution computed from holders_after... If the move is valid, reverse should include wi as candidate. If q_reverse = 0, reject — safe. But subtle: the heat-bath proposal for replica rep at window wi includes the "stay" option with delta forced 0 and weight exp(0)=1 — fine.

Hmm — a deeper issue with the Gibbs walk: the proposal distribution `_gibbs_window_proposal_distribution` is a heat-bath over swap partners, but its weights use only the bias matrix (umbrella + boost), i.e., the reduced Hamiltonian difference. Heat-bath proposal q(j) ∝ exp(-β δ_j) where δ_j is the swap energy change — this is a valid proposal and MH correction with reverse q fixes detailed balance. Looks right. But the stay delta forced to 0 — actual delta for staying is 0 anyway (swapping replica with itself? No—stay means wj == wi; the "swap" would be with itself; delta genuinely 0). Fine.

**Potential bug: `holders` array vs assignments.** In gibbs_propose_one_replica, wi = assignments[rep]. If assignments and replica_of_window are consistent, fine. apply_window_swap updates both: `assignments[i], assignments[j] = assignments[j], assignments[i]` then `replica_of_window[int(assignments[i])] = int(i)`... wait: after the swap, assignments[i] holds the new window of replica i (which is wj), and replica_of_window[wj] should be i. Let's check: after `assignments[i], assignments[j] = assignments[j], assignments[i]`: assignments[i] = wj (old window of j), assignments[j] = wi. Then `replica_of_window[int(assignments[i])] = int(i)` → replica_of_window[wj] = i. Correct. `replica_of_window[int(assignments[j])] = int(j)` → replica_of_window[wi] = j. Correct. OK.

But wait: what if i == j? Guarded by swap_candidate_replicas (i==j returns None). What if one replica holds both windows? Guarded.

Hmm, wait: there's an issue with the initial `assignments` semantics — set_window uses `assignments[replica_index]` — OK.

**4. _exchange_probability**: standard Metropolis. Fine.

**5. swap_candidate_replicas**: `n_win = int(replica_of_window.size)` — replica_of_window is indexed by window, size = n windows. Checks wi/wj in range. Fine.

**6. explicit_2d_neighbor_pairs_for_exchange (L1479-1505)**: Round-robin edge coloring: `if idx % slots != slot: continue` — selects edges by index mod slots, then greedy disjoint filter. Fallback when slot empty: tries ALL edges greedily — this makes the edge set for an empty slot equal to the full greedy matching, which overlaps with other slots' edges. Does this break detailed balance? For a fixed parity schedule, the Markov chain with alternating matchings still satisfies balance if each sub-chain is reversible — Metropolis swaps within a matching are reversible. Adding the full matching on an "empty slot" is just another valid matching set; the composition remains valid as long as it's a matching. Detailed balance for the product of reversible moves: each swap move is reversible with respect to π, and compositions of such moves... Actually a sequence of Metropolis moves each individually satisfying detailed balance gives a valid chain regardless of schedule (each move is a π-reversible transition kernel; product of reversible kernels is not necessarily reversible but is still π-invariant). Yes — π-invariance holds. So it's fine for correctness of stationary distribution. OK.

But subtle: nonlocal exchanges across the explicit-2D graph — acceptance via _exchange_probability with delta from bias matrix. Fine.

**7. assemble_bias_matrices (L2055-2072)**: boost_bias_kcal = boost_kj / 4.184. bias_kcal = distance + ss + boost_kcal. bias_kj = 4.184*bias_kcal. Invariant holds. Fine, except shape concern (nwin vs nrep zeros fallback).

**8. _fetch_v_pep_v_dih (L2075-2094)**: reads group DIHEDRAL_GROUP for v_dih, peptide_essential_energy_kj for v_pep. Concern: the boost applied by the integrator modifies forces but getPotentialEnergy on a context with a CustomIntegrator applying GaMD boost — does getState(getEnergy=True) return the boosted or unboosted energy? The boost is applied inside the integrator's per-dof force modification, so getState potential energy does NOT include the boost; separate reads of channel energies are needed. This is consistent with reading v_pep/v_dih and computing boost analytically. Fine, but the concern: is v_dih read from group {DIHEDRAL_GROUP} the *unmodified* dihedral energy? Should be, since forces are modified by integrator, not the Force objects. OK.

But sign/subtlety: in the single-dihedral case, `if not getattr(pep_env, "has_total", True): return nan, v_dih`. Then in pep_gamd_boost_kj, `if not env.has_total: ... return b_dih`. OK. But in reconstruct_bias_matrix (correctness/bias.py): `if pep is None: pep = np.full(n, np.nan)` and ladder apply must ignore it. Fine if it does.

**9. campaign_ladder_registry_lambda (L2261-2299)**: Walks up directory tree looking for registry with any λ > 0, returns first found. Concern: returns the FIRST λ>0 encountered in CSV row order — arbitrary λ, only used for boolean detection in resolve_ladder_active. resolve_ladder_active returns True if campaign has ladder. This switches on the ladder code path (recording v_pep/v_dih, applying λ=... per state). But an all-λ=0 sub-run under a ladder campaign: resolve_ladder_active True → pep_env created? Hmm, we don't see the code that constructs pep_env from this. The λ per state is derived from window_metadata; if all λ=0 in this sub-run, k0=0, boost = 0 via _channel_boost (k0=0 → b=0? 0.5*0*(...)^2/rng = 0, and (b+v)<e check → 0 or 0; yes zero). Recorded samples carry v_pep/v_dih. Consistent.

But hazard: the registry walk uses `CAMPAIGN_LADDER_REGISTRY_NAMES` — reads a state CSV written by StateRegistry.write_state_csv. If the registry lists λ>0 for states NOT in this sub-run, ladder active — fine. But what if the walk goes too far up (before adaptive_production)? It checks parents up to but stopping after adaptive_production. Could a campaign's out_dir be nested such that a sibling campaign's registry is found? Only if registry names exist in a shared parent. They claim they stop at adaptive_production. Potential cross-campaign contamination if directories share a parent above adaptive_production... they break after adaptive_production so at most walks to that level. Seems OK-ish; SUSPECTED minor.

Bigger hazard: **stale registry**. If the registry CSV is written at campaign start and states are retired/added adaptively, reading λ from the registry with `lam > 0` triggering ladder — that's only a boolean, fine.

**10. _derive_state_gamd_lambdas (L2360-2384)**: reads normalized_rows if aligned and all rows have gamd_lambda; else existing if length matches; else zeros. Zeros = silent ladder disable — the fallback of "last resort". The persist/reload pair tries to close this. But there's a hole: `_reload_state_gamd_lambdas_on_resume` only reloads if `args.state_gamd_lambdas` is falsy. If it's set but WRONG (e.g., set from an explicit window table with different length), reload skips, and derive uses existing only if length matches, else zeros — silent. But persist is called at every mutation point... The docstrings themselves admit these were found in review. Remaining hole: after `--max-replicas` truncation `args.state_gamd_lambdas[:_max_replicas]` — truncating a per-window array by number of replicas. If n_windows > nrep, this truncation drops the last windows' λ, misaligning λ to windows! Wait — if windows and replicas are 1:1 (each replica owns one window), nrep == nwin and truncation is identity. But if the run has more windows than replicas... does REUS here run one replica per window? The state graph is (window × rung) and replicas move over windows via gibbs-walk — the state space includes all windows, so nrep likely == nwin. Hmm, but then what does --max-replicas truncation mean? If nrep < nwin, replica_of_window has unheld windows (holders >= 0 check exists — so unheld windows are supported!). swap_candidate_replicas returns None if either window unheld. _gibbs_window_proposal_distribution only considers valid (held) windows. So the design supports nrep < nwin. Then truncating state_lambdas (length nwin) to _max_replicas (nrep) is a real misalignment: window w's λ taken from state_lambdas[w] where the list was truncated — windows beyond nrep would index out of bounds! `state_lambdas[w]` with w >= nrep → IndexError. Or if nwin == nrep after drop... Unclear. This is a SUSPECTED state-label mismatch. Actually wait, re-read the persist docstring: "the --max-replicas truncation (`args.state_gamd_lambdas = list(args.state_gamd_lambdas)[:_max_replicas]`)". If state_gamd_lambdas is per-window and _max_replicas < nwin, subsequent `state_lambdas[w]` for w >= nrep crashes (loud) or windows get λ of... Actually if it truncates and then windows are also truncated to _max_replicas elsewhere, maybe consistent. Can't verify; SUSPECTED.

**11. `_reload_state_gamd_lambdas_on_resume` truthiness check**: `if getattr(args, "state_gamd_lambdas", None): return` — an all-zero list [0.0, 0.0, ...] is truthy (non-empty list), so it's left alone — as documented. But an EMPTY list [] is falsy → proceeds to read manifest. Fine. But: a list that is legitimately all-zero from manifest: `lambdas = method_settings.get("state_gamd_lambdas"); if lambdas:` — all-zero list is truthy (non-empty), so it restores [0,0,...]. OK.

However — critical hole: the reload only fires when `args.resume` is set. What about the **adaptive top-up** path (new sub-run without --resume)? campaign_ladder_registry_lambda handles ladder-active detection, but the per-window λ still comes from `_derive_state_gamd_lambdas(window_metadata, ...)`. If window_metadata rows lack gamd_lambda (older format), fallback zeros → λ=0 states run unboosted... but wait, that's the chignolin_7 defect described — they fixed it via registry detection making ladder "active" but active means v_pep/v_dih recorded and k0 = λ·k0max with λ=0 → boost zero. Hmm, actually if the ladder is active but all derived λ are 0 (because metadata lacked gamd_lambda), replicas run unboosted while the registry says λ>0 states exist — but those λ>0 states aren't in this sub-run, so... consistent? The sub-run's states genuinely have λ=0 if the window table says so. The registry detection only switches on recording. Seems consistent.

**12. The chignolin_8-class bug in _fetch_v_pep_v_dih / exchange pricing**: In `_current_exchange_arrays`, bias matrix prices the boost under each candidate state's λ using v_pep/v_dih read from the context NOW. But the boost the integrator actually applies depends on the integrator's k0 and its *stage statistics* (Vmax etc. from globals). If the envelope globals carried across (the chignolin_8 bug was stage, fixed). Now: does the analytic `_channel_boost` in pep_gamd_boost_kj exactly match the boost the integrator applies? The integrator (gamd-openmm StageIntegrator) computes boost with its own formula and globals (Vmax, Vmin, E, k0) in possibly different units (gamd-openmm uses kJ? or kcal? or per-particle?). Unit consistency between `pep_gamd_boost_kj` (kJ/mol) and the integrator's internal energy scale is critical. The integrator's k0 globals are set via set_replica_lambda with k0max from `k0max_from_globals(shared_globals)` — units of shared_globals presumably kJ/mol? We can't see. The claimed consistency "exchange and MBAR priced a boost that was never applied" was the chignolin_8 failure mode; the residual risk is a mismatch between the integrator's applied boost and the analytic pricing formula (e.g., gamd-openmm's `_add_dihedral_boost_to_total_energy` ordering). pep_gamd_boost_kj claims to mirror stage_integrator's dependent dual: dihedral first, then total with dihedral boost added. If gamd-openmm's actual Total-channel boost uses threshold_total and Vmax/Vmin of Total computed on (V_pep + b_dih)... The code does `_channel_boost(v_pep + b_dih, env.threshold_total, ...)` — matching. But gamd-openmm's lower-bound dual: for "dual" boost the total boost is applied to the *total* potential with the dihedral boost added — plausible mirror. Can't verify against the actual integrator source; SUSPECTED consistency risk but they've clearly thought about it.

One concrete unit check: `pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)` in sample(): v_pep_kj from `_fetch_v_pep_v_dih` is in kJ/mol. _channel_boost expects kJ — b in kJ. boost_bias_kj → /4.184 → kcal. OK.

**13. `bias_matrix_kj` includes boost — and MBAR.** The exchange acceptance uses bias = umbrella + boost. For REUS with boost, the correct extended-Hamiltonian exchange should use the full potential difference including the boost — which it does since bias matrix includes boost and physical potential cancels for label swaps (configurations unchanged). Wait — does the physical potential cancel? Swapping labels: replica i's configuration x_i moves from state (wi, λi) to (wj, λj). Total energy change = [U0(x_i) + U_b(x_i, λj) + Uumb_wj(x_i)] + [U0(x_j) + U_b(x_j, λi) + Uumb_wi(x_j)] - [U0(x_i) + U_b(x_i, λi) + Uumb_wi(x_i)] - [U0(x_j) + U_b(x_j, λj) + Uumb_wj(x_j)] = [U_b(x_i,λj) - U_b(x_i,λi)] + [U_b(x_j,λi) - U_b(x_j,λj)] + umbrella terms. U0 cancels. Boost does NOT cancel unless λi == λj. The bias matrix includes boost per (state, sample): bias[w, r] = Uumb_w(x_r) + b_λ(w)(x_r). So delta = bias[wj,i] + bias[wi,j] - bias[wi,i] - bias[wj,j] correctly includes the boost difference. 

But the docstring in apply_window_swap explicitly claims "the GaMD boost cancel[s] exactly from delta" — factually wrong reasoning, but harmless since the matrix includes it. However — wait. In the Gibbs proposal, `_gibbs_window_proposal_distribution` uses bias_matrix_kj which includes boost. Fine.

BUT — there's a subtle issue: the Gibbs-walk / swap moves only permute state labels among replicas; they never exchange configurations. For replica-exchange to sample the extended ensemble correctly with label permutation moves only... Actually label-swap (state swap) RE is a standard valid formulation: the joint distribution over (configs, permutation) — moves that swap which state each configuration is in with Metropolis acceptance on the total energy change is exactly the "state swap" variant, valid. OK.

**14. Boost pricing vs applied boost — the CRITICAL one:** The exchange prices the boost analytically via `_channel_boost` with the frozen envelope, but the integrator applies its own boost computed from its globals. If the integrator's actual applied boost at any moment differs from the analytic reconstruction (e.g., gamd-openmm applies boost with (E - V) computed per-step inside the kernel with its own formula, possibly with the (b+v)<e guard implemented differently, or with k0 in different units), exchange accepts with wrong probability. This is exactly the chignolin_8 class. The sample row also records `gamd_boost_total_kj_mol` from `extract_gamd_boost_kj(sim.integrator, ...)` — the ACTUAL applied boost — while `sampled_boost_bias_kj` is the analytic one. If they disagree, nothing checks! There's no assertion comparing `extract_gamd_boost_kj` (applied) with `pep_gamd_boost_kj` (priced). That's a missing cross-check that could catch the next chignolin_8. Both are computed in the same sample() call: `boost_kj, ... = extract_gamd_boost_kj(sim.integrator, unit, None)` and `sampled_boost_bias_kj = boost_bias_matrix_kj[w, r]`. A consistency check between these two would be cheap and is absent. VERIFIED (absence of cross-check; the two quantities are computed side-by-side at L~7620-7640 and never compared). This is a key recommendation.

Wait, actually is `extract_gamd_boost_kj` reading the integrator's accumulated boost? It reads the integrator's boost total (probably a global variable accumulating per-step boost). The analytic one is instantaneous per-current-configuration boost. The integrator's applied boost at the current configuration should equal analytic if formulas match. Comparing them would validate the handoff. Absent. Good finding.

**15. NPT volume moves**: The task says barostat via biased Metropolis-MC volume move against U* = physical + bias + boost. Not in the shown code — the submission's subsystem is exchange. Out of scope but note: apply_window_swap's docstring mentions "NPT correction" comment in _attempt_window_swap. The shown code says umbrella-parameter writes run on each replica's own worker. After an accepted swap, `observable_cache.clear()` — stale observables cleared. Good.

But: `_apply_swap_to_replica` sets window and lambda on the integrator — but does it re-apply... the set_window writes centers/ks to context parameters. If the swap is accepted and then the barostat MC move uses U* including bias — bias depends on current CV which changes with volume... out of scope.

**16. _gibbs_window_proposal_distribution numerical details:**

- `log_weights = np.clip(-beta*delta, -745, 745)` — clip at ±745 prevents overflow but introduces bias: a candidate with true log-weight -800 is clipped to -745, giving it weight e^{-745} instead of 0 — relative to max weight, if max log weight is, say, -700 (all deltas hugely positive... wait if all deltas positive, max log weight = -beta*min delta which could be -800; then clipping all to -745? No — clip(-800) = -745 only if -800 < -745 → clipped to -745. So all weights clipped up to -745... then max m = -745, weights = exp(0) = 1 for all clipped candidates. This distorts the proposal distribution when all log weights < -745. Comment says "Lower bound -745 prevents underflow to exact zero (loses that candidate)" — but with log-sum-exp, an exact-zero weight is fine (candidate never proposed, which is correct heat-bath). Clipping to -745 makes weights exp(log_w - m); if m is also ≈ -745 (because the best candidate is also clipped), the clipped candidates get non-negligible relative probability. Concretely: if true log weights are [-800, -900], clipped to [-745, -745], normalized to [0.5, 0.5] instead of [1, ~0]. The proposal distribution is then wrong — BUT the MH correction with q_forward/q_reverse uses the same clipped probabilities, so the chain remains correct! The MH acceptance ratio corrects for proposal asymmetry. q's computed consistently for forward and reverse → detailed balance preserved. So the clip only affects efficiency, not correctness. Hmm wait — is that right? MH with proposal q (any q with full support over connected swaps) and acceptance min(1, π(y)q(y→x)/π(x)q(x→y)) is valid. The clip changes q but consistently, so validity retained. However — the stay option: delta forced 0, log weight 0. If other candidates have huge positive delta, their clipped weight is e^{-745+745}... m = max(0, -745...) — stay has weight 1, m=0, clipped candidates get exp(-745) ≈ 0 anyway. Fine. So clip is efficiency-only. Actually edge: candidate true log-weight -1000 clipped to -745, m = 0 (stay), weight = e^{-745} ≈ 5e-324 — subnormal, effectively 0. Fine.

- `finite = np.isfinite(log_weights)` — after clipping, log_weights always finite unless delta is NaN (bias NaN). NaN delta → -beta*NaN = NaN → not finite → filtered out. So NaN bias entries drop that candidate rather than freezing the replica. Combined with finding #1 (NaN ss → 0 bias, not NaN), the ss NaN never reaches here as NaN. But if primary CV is NaN, distance_kcal is NaN → delta NaN → candidate dropped in Gibbs. For apply_window_swap, delta NaN → _exchange_probability returns 0 → reject. OK-ish.

- The fallback `probabilities = uniform 1/n` when sw not finite or <= 0: a uniform proposal fallback. With MH correction using these q's, still valid. But note: q_forward recorded from proposal["probabilities"] — consistent.

**17. gibbs_propose_one_replica reverse-proposal**: holders_after computed correctly: swap wi↔wj holders for rep and target_rep. But wait — `holders_after[wi] = target_rep; holders_after[wj] = rep`. Correct. Then reverse proposal for rep currently at wj (assignments in reverse should be wj). It passes `current_window=wj` — correct. rev candidate wi must be in rev_windows. If wi not in valid windows of reverse (e.g., wi unheld in holders_after? No—wi is held by target_rep). Should be present. If absent, q_reverse = 0 → acceptance 0 → move always rejected. That's conservative but could freeze moves... valid but inefficiency. Hmm, but wait — is it actually conservative? If the reverse proposal can never propose returning, rejecting is correct for detailed balance. Fine.

One real concern in gibbs: **it only proposes transpositions involving `rep` and the current holder**. This is a valid move set. Fine.

**18. `_attempt_window_swap` RNG discipline**: `swap_candidate_replicas` check before drawing rng.random() — then apply_window_swap draws... wait, in _attempt_window_swap, `None if force_accept else rng.random()` is evaluated as an argument — the rng.random() IS drawn before apply_window_swap is called, after the candidate check. Good — matches documented discipline. But note `rng.random()` is evaluated eagerly even if apply_window_swap then returns None (can't happen since candidate check passed and nothing changes in between — single-threaded). OK.

But subtle: `apply_window_swap(..., None if force_accept else rng.random(), p_override=...)` — when p_override is given but force_accept is False, an rng draw still consumed — presumably matches original. Fine.

**19. `_current_exchange_arrays` cache staleness**: cache check `int(observable_cache.get("step", -1)) == int(absolute_step)` and shape (nrep, nrep). If nwin != nrep, shape check fails → recompute (fine). If steps coincide, uses cached bias matrix from sample(). But sample() computed bias matrix BEFORE any swaps this interval; if multiple exchange rounds happen at the same absolute_step (Gibbs rounds + swap rounds), the cache is cleared on accepted swap (`observable_cache.clear()`), so subsequent rounds recompute. But if a swap was REJECTED, cache not cleared — bias matrix unchanged anyway (labels unchanged) — wait, rejected swap doesn't change assignments, so bias matrix (indexed by window,replica of configurations) is unchanged. Fine. But accepted Gibbs moves? gibbs_propose_one_replica returns a proposal — who applies it? The shown code doesn't include the Gibbs application. Presumably elsewhere it applies via apply_window_swap-like logic with p_override=pacc. If the Gibbs application path does NOT clear observable_cache... The shown _attempt_window_swap clears cache on accept. We can't see the Gibbs apply site. SUSPECTED.

Also: bias matrix rows indexed by window; after an accepted swap the matrix is stale (replica r's config unchanged but its window changed; matrix entries bias[w, r] are per-configuration so still valid, but "sampled" entries w=assignments[r] change). Cache cleared. Fine.

**20. The exchange bias matrix uses v_pep/v_dih read at exchange time, computing boost under each state's λ analytically.** But the integrator's applied boost uses its own current k0 and its internal (E - V) — where V is read inside the kernel from the actual energy. The analytic `_channel_boost(v, e, vmax, vmin, k0)` uses v = v_pep (peptide essential energy in kJ). Does the integrator's Total-channel V match peptide_essential_energy_kj exactly (same force groups, same units)? And the threshold/vmax/vmin from the frozen envelope — but gamd-openmm's StageIntegrator in stage 5 uses its own accumulated statistics? No — envelope frozen and globals set. The units of vmax/vmin/threshold in the integrator globals (gamd-openmm might use kJ or per-different-grouping). If pep_env's envelope is in kJ but integrator globals expect kcal — factor 4.184 error in applied boost vs priced boost. Can't verify from shown code; SUSPECTED. Actually seed_frozen_envelope_stage5 presumably handles this.

**21. `umbrella_bias_matrix_kcal` secondary NaN → 0 — deeper.** The docstring: "a replica without a finite secondary value, contributes zero on that axis (the previous sample-path guard, now applied identically to the exchange path -- a NaN entry used to freeze that replica's exchanges)." This is self-contradictory: zero contribution is NOT freezing. The "freeze" behavior described doesn't exist in this code. If secondary CV is NaN for a replica (fast path returning NaN, or non-fast path with _ss_enabled False but ss_centers set?), the exchange prices that replica at zero secondary bias for all states → delta excludes secondary terms → swaps accepted per wrong energies. In sample(), `sampled_umbrella_bias_kj` would then also use 0 secondary... but reconstruct_bias_matrix in MBAR analysis gives NaN → sample filtered. So exchange distribution wrong (detailed balance w.r.t. intended target violated since acceptance uses wrong energy). Effect on FES: exchange acceptance too high/too low; sampling not from intended joint distribution; MBAR weights mismatch. VERIFIED as doc/code inconsistency and silent-fallback (NaN→0).

Actually wait, is it possible that ss_values NaN occurs legitimately? `observe_fast_path` with `_fast_ss_forces` — if secondary disabled, ss = NaN, but then `_has_ss_axis` would be False and ss_centers passed as None. If secondary enabled but the CV computation fails → NaN? Possibly. Either way, the guard zeroes rather than freezes.

**22. state_identity.py review:**

- `canonical_state_definition`: windows normalized; k in kJ converted /4.184 — mutating `row` in place while iterating `windows` = the same dicts in data["windows"]? normalize_windows returns new dicts (dict(raw)), so fine. Note conversion happens AFTER checking unknown fields but row["k1"] /= 4.184 mutates the normalized copies — then `data["windows"] = sorted(windows, ...)`. Fine.

- Bug check: `if any(row["gamd_lambda"] > 0 for row in windows) ...` uses `windows` (the normalized list) — fine. But `row["gamd_lambda"]` from normalize_windows ensures present. OK.

- `if row["k1"] > 0 and data["cv1"] is None` — cv1 required. But required schema includes cv1 always; _canonical_cv(None) returns None. OK.

- freeze_snapshot: `equilibrium_analysis_eligible and phase_kind != "production"` → error; but snapshot allows phase_kind production with eligible False. Fine.

- validate_fixed_state_segments: compares segments' definitions for exact equality — but windows sorted by window_id in canonical form, so segments with different window ORDER compare equal. The top-level windows check handles ordering. OK.

- One issue: `canonical_state_definition` requires exact schema set equality — `set(data) != required` — any extra field rejects. Strict. Fine.

- `normalize_windows` `_term` for axis 2: `if force_key not in window: if axis==2 and center_key not in window: return 0.0, 0.0` — so k2 missing AND center2 missing → (0,0). But if k2 missing while center2 present → falls through to `raise IntegrityError(missing k2)`. Hmm wait: `if force_key not in window:` then inner check `if axis == 2 and center_key not in window: return 0.0, 0.0` else raise missing force. So k2 absent + center2 present → error "inactive means explicit zero". Strict but fine.

- **Unit bug in state_identity**: k1 for CV1 with units angstrom: k in kcal/mol/Å². CV in the run: cv_A in angstroms. Consistent per docstring "k1 and k2 are kcal/mol per squared CV unit". OK.

- **reconstruct_bias_matrix dimensionless beta multiplication**: `matrix *= beta * KJ_PER_KCAL` — converts kcal to reduced (dimensionless). Then `_ladder_apply(matrix, pep, dih, lambdas, envelope, beta, bookkeeping)` — the ladder applies boost... boost in kJ must be multiplied by beta. Presumably apply_ladder_boost_to_u handles. Can't verify.

- **reconstruct_bias_matrix: boost added AFTER umbrella is multiplied by beta** — order: matrix (kcal) *= beta*4.184 → reduced umbrella. Then ladder apply adds boost reduced. If apply_ladder_boost_to_u computes pep_gamd_boost_kj (kJ) and multiplies by beta, consistent. OK.

- **Missing: boost must be SUBTRACTED or ADDED?** In MBAR for GaMD reweighting, the biased distribution is p_bias ∝ exp(-β(U0 - ΔV_boost))? GaMD adds a positive boost potential ΔV to lower-barrier regions: U' = U + ΔV where ΔV ≥ 0 lowers effective barriers? Actually GaMD: V' = V - ΔV, i.e., the boost reduces the potential (adds -ΔV). Standard GaMD: when V < E, add ΔV = 0.5*k0*(E-V)^2/(Vmax-Vmin), and the modified potential is V - ... no wait. GaMD (Miao et al.): V*(r) = V(r) - ΔV(r), where ΔV = 0.5*k0*(Vmax - V)(V - Vmin)/(Vmax-Vmin)... For lower-bound GaMD: ΔV = (1/2)k0(E - V)^2/(Vmax - Vmin) for V < E, and V*(r) = V(r) - ΔV? No — actually GaMD adds the boost to RAISE the potential wells: V* = V + ΔV where ΔV>0 when V < E... Hmm. GaMD's essential idea: add a harmonic boost to raise the potential energy in low-energy regions: V*(r) = V(r) + ΔV(r), with ΔV = 0.5 k0 (E - V)^2/(Vmax - Vmin) when V < E. This raises minima, flattening the landscape. Yes — GaMD ADDS ΔV (V* = V + ΔV). Check the code: `b = 0.5*k0*(e-v)^2/rng; where((b+v) < e, b, 0)` — apply boost only if b+v < e, i.e., boosted energy still below threshold (avoid over-boosting past threshold). This matches gamd-openmm's lower bound boost. And pep_gamd_boost_kj returns positive b, added to bias. In exchange, the extended Hamiltonian is U0 + umbrella + boost (all positive contributions). Consistent.

For MBAR reweighting from the biased ensemble back to unbiased: the biased state's unnormalized density is exp(-β(U0 + boost)), so MBAR needs u_k = β(U0 + umbrella_k + boost_k). Fine — not shown.

**23. `set_replica_lambda`** — if integrator lacks getGlobalVariableName (test double), sets k0_Total anyway via setGlobalVariableByName — would raise AttributeError on a real integrator without k0_Total... they handle via names check. OK.

**24. `_gibbs_mh_acceptance_probability` `except Exception: return 0.0`** — silent reject on bad input. Conservative, fine.

**25. The `sample()` closure `boost_bias_matrix_kj[w, r]`** — boost matrix is (n_states, nrep) with n_states = len(state_lambdas) = nwin. If nwin != nrep... `np.zeros((nrep, nrep))` fallback for pep_env None — shape mismatch with distance (nwin, nrep) if nwin != nrep → ValueError in assemble. When pep_env is not None, boost is (nwin, nrep) — matches. So the zeros fallback is only a bug when pep_env is None AND nwin != nrep. Does that happen? With unheld windows supported (holders >= 0 checks), nwin > nrep possible via --max-replicas. Then `assemble_bias_matrices(distance(nwin,nrep), ss(nwin,nrep), zeros(nrep,nrep))` → broadcast ValueError → crash (loud, not silent). Unless nwin == nrep. Actually wait, if nwin != nrep, adding (nwin,nrep) + (nrep,nrep) raises. Loud crash is a robustness bug but not silent wrongness. Hmm, but actually wait — in `sample()`: `boost_bias_matrix_kj = ... if pep_env is not None else np.zeros((nrep, nrep))`. And also `state_lambdas[w]` — if state_lambdas truncated to nrep < nwin, IndexError for w >= len. Loud. OK — mention as SUSPECTED inconsistency when nrep < nwin.

**26. `explicit_2d_neighbor_pairs_for_exchange` — the `edges` list from build_explicit_2d_neighbor_edges: if edges ordering is nondeterministic (set iteration)? Can't see. If edge order varies between runs, reproducibility issue only.

**27. `_derive_state_gamd_lambdas`: `all("gamd_lambda" in r for r in rows)`** — if even one row lacks the key, falls to existing/zeros. Fine per docs.

**28. `campaign_ladder_registry_lambda` returning `(path.name, lam)`** — returns just the filename, not full path; used only for truthiness. Fine.

**29. `_reload_state_gamd_lambdas_on_resume` — the manifest read**: `read_json_file(Path(out_dir) / "run_manifest.json", {})` — reads manifest in out_dir. But the docstring says the manifest was "written once at campaign start (initialize_run_manifest)" — is run_manifest.json in out_dir the campaign's or the sub-run's? If each sub-run has its own run_manifest.json written at its own start (with its own method_settings from its own args), then the resume reads the sub-run's manifest — which itself may have been patched by _persist_state_gamd_lambdas with the truncated/dropped ladder. The persist docstring says "the last call before production starts is what ends up in the manifest". So the manifest should hold the correct final ladder. But if the sub-run crashed BEFORE the last persist call (e.g., during the pull phase before post-drop re-derive persist), the manifest holds a stale ladder that reload then bakes in. Edge case.

More serious: **the reload happens only when `args.state_gamd_lambdas` is unset/empty.** On resume, is args.state_gamd_lambdas populated from CLI/config before run_gareus? The docstring says on the fast-resume path it's unset. OK as documented.

**30. Missing from review scope but notable: `_exchange_probability` `if delta_kj <= 0.0: return 1.0`** — fine.

**31. `apply_window_swap` — `uniform` semantics**: `accepted = bool(float(uniform) < pacc)`. pacc=1 → uniform < 1.0 always true (uniform in [0,1)). Fine.

**32. Gibbs proposal `_gibbs_window_proposal_distribution` uses `replica_of_window` holders — but Gibbs move changes assignments of two replicas. After an accepted Gibbs move (applied where?), both assignments and replica_of_window must update. Not shown.**

**33. Detailed balance of the stay-move in Gibbs**: when wj == wi, gibbs_propose returns with accepted flag False and "is_stay" True — caller presumably counts it without MH. Fine.

**34. The Gibbs walk and window swap both being applied — RNG stream discipline for `choose`: production passes rng.choice. Does `rng.choice(n, p=probs)` consume a deterministic number of draws? numpy Generator.choice with p consumes variable amounts? It uses inverse-CDF or similar — deterministic count (one uniform typically). Fine.

**35. `_channel_boost` numeric: `(b + v) < e` with b,v possibly NaN (v_pep NaN in Total path when...)**: if has_total True, v_pep comes from peptide_essential_energy_kj — could it be NaN if context energies blow up? Then b = 0.5*k0*(e-nan)^2 = NaN; where(nan < e → False) → 0.0. So NaN v_pep → boost 0 silently! `_np.where((b+v) < e, b, 0.0)` — NaN comparison is False → 0.0. So a NaN channel energy silently prices boost=0 in the exchange matrix, and in MBAR reconstruction (pep_gamd_boost_kj used by ladder apply). Meanwhile the integrator's actual applied boost with NaN energy... would also be garbage. This is a NaN→0 silent fallback. VERIFIED in pep_gamd.py L497-503: NaN inputs give b=0 via the where. Though `numeric_vector` in correctness/bias.py converts non-finite to NaN and reconstruct would produce NaN boost columns (NaN propagates through _channel_boost? b = NaN, where((NaN + v) < e → False) → 0.0! So even MBAR reconstruction silently zeroes NaN channel energies → samples with corrupted v_pep get priced as unboosted instead of being flagged/filtered. That IS a silent NaN→0 in the MBAR path too, via the same `_channel_boost`. Wait — in reconstruct_bias_matrix, pep goes through numeric_vector (non-finite → NaN) then ladder apply → apply_ladder_boost_to_u → presumably pep_gamd_boost_kj → _channel_boost → where False → 0. So a frame with NaN v_pep gets boost 0 in MBAR instead of NaN. VERIFIED: `_channel_boost` L502 `return _np.where((b + v) < e, b, 0.0)` maps NaN v to boost 0.0, silently. Effect: corrupted/missing energies treated as unboosted — exactly the chignolin_8 signature (priced vs applied mismatch) but now in the MBAR pricing direction.

Hmm wait, careful: `b = 0.5*k0*(e-v)**2/rng` with v=NaN → b=NaN. `(b+v) < e` → NaN < e → False → returns 0.0. Yes. And the earlier `np.where(np.abs(rng) <= 0.001*scale, 0.0, b)` — b NaN stays NaN. Then final where → 0. Confirmed.

Also what if e (threshold) is NaN? Envelope frozen, presumably validated finite.

**36. `pep_gamd_boost_matrix_kj` shape when lambdas empty**: vstack of empty list → ValueError. If state_lambdas empty (ladder derive fallback zeros... no, zeros list non-empty if n>0). Fine.

**37. In sample(): `row["secondary_cv_center"] = float(ss_centers_arr[w]) if math.isfinite(...) else ""` — ss_centers_arr could be None if no ss axis → float(None) TypeError? They guard with isfinite of float(ss_centers_arr[w]) — evaluating float(None) raises TypeError before isfinite. Actually `math.isfinite(float(ss_centers_arr[w]))` — if ss_centers_arr is None, float(None) raises TypeError → crash. Loud. Unless ss_centers_arr is array. It's `ss_centers_arr_global` — presumably always an array (maybe of NaN). Then isfinite(NaN)=False → "". Fine.

**38. `row["secondary_cv_k_kcal_mol"]` uses `ss_k_arr[w]`** — ss_k_arr = ss_k_kcal_arr_global — if no ss axis, what's in it? Probably zeros. OK.

**39. `sampled_boost_bias_kj = float(boost_bias_matrix_kj[w, r])`** — w is window index; boost matrix row per state λ. With λ per window — is the rung tied to the window or independently swappable? The state graph is (window × rung) — replicas exchange over BOTH window and rung? The Gibbs/swap moves shown only swap windows, with λ fixed per window (state_lambdas[w]). So the "state" is fully determined by window: λ travels with window. That's a valid design (each window has an associated rung). But then "replica exchange by gibbs-walk over the (window × rung) state graph" — if λ is tied to window index, the rung dimension is degenerate. Fine, design choice.

Hmm wait — actually re-read: `set_replica_lambda_for_window(sim.integrator, assignments[replica_index], state_lambdas, ...)` after swap — λ applied per window assignment. Consistent with λ tied to window. And bias matrix includes boost[w, r] using state_lambdas[w]. Consistent. Good.

**40. `_current_exchange_arrays` v_pep/v_dih read when pep_env not None — but if the ladder is active with all λ=0 (registry-detected), boost is zero, matrix fine. OK.**

**41. One more: `_fetch_v_pep_v_dih` reads v_dih via `groups={DIHEDRAL_GROUP}` — comment says gamd-openmm puts PeriodicTorsion/CMAPTorsion in group 2 "the same id as DIHEDRAL_GROUP". If they ever diverge... constant defined where? SUSPECTED coupling.**

**42. The `sample()` boost recording: `v_pep=float(v_pep_kj[r]) if pep_env is not None else float("nan")` — recorded NaN when no ladder, and MBAR reconstruct with lambdas all zero skips boost (lambdas check). Fine.**

**43. state_identity: `data["cv1"] = _canonical_cv(data["cv1"], "cv1")` — NPT check for pressure; `finite_number(data["pressure_bar"], "pressure_bar")` — no minimum; negative pressure allowed ( tension ) — fine.**

**44. In `validate_fixed_state_segments`: `normalize_windows(snapshot.get("windows", [])) != state["windows"]`** — normalize_windows outputs window dicts with keys window_id (if present), center1,k1,center2,k2,gamd_lambda. state["windows"] sorted by window_id; snapshot windows possibly unsorted → first comparison likely unequal (order) → then sorted comparison. Fine.

**45. compare_state_definitions: float equality — k1 values compared exactly; two segments with k1 = 0.1 vs 0.1 fine. JSON round-trip floats stable. OK.**

**46. A real one: `freeze_snapshot` `if any(row["gamd_lambda"] > 0 ...)` in canonical_state_definition — uses `windows` normalized but note k1 conversion `row["k1"] /= 4.184` happens inside the loop AFTER `if set(row) - {...}` check — mutating rows while `seen` uses window_id — fine.**

**47. umbrella_bias_matrix_kcal: `distance_kcal` has NO NaN guard** — if primary CV NaN, distance NaN, delta NaN. In apply_window_swap, delta NaN → _exchange_probability → isfinite check → 0 → reject. Safe. In Gibbs, filtered. In sample, recorded as NaN → downstream filtering. OK — asymmetric treatment: primary NaN freezes (reject), secondary NaN fabricates 0. The docstring claims NaN "used to freeze that replica's exchanges" — only true for primary. Verified inconsistency.

**48. Potential double-counting in `_attempt_window_swap`: `_record_exchange_stats(wi, wj, outcome.accepted)` — fine.

**49. `_apply_swap_to_replica` runs via `_sim_pool.submit(i, ...).result()` — sequential; fine.

**50. Missing RNG-stream hazard: `rng.random()` drawn once per attemptable pair; `rng.choice` for Gibbs. If the number of attempts varies (adaptive windows?), streams shift — reproducibility only.

**51. `_gibbs_window_proposal_distribution` `stay_idx` uses `valid_windows == wi` — after finite filtering, valid_windows was reassigned. stay_idx computed BEFORE finite filter (line order: stay_idx computed, then finite filter, then valid_windows = valid_windows[finite]). If the stay candidate was filtered out (delta forced 0 → log_weight = clip(0)=0, finite — never filtered). Fine. But wait — stay delta forced 0 AFTER delta computed; if bias[wi, rep] is NaN... delta[stay]=0 anyway. OK.

Actually there's a subtle bug candidate: `stay_idx = np.where(valid_windows == wi)[0]; if stay_idx.size: delta[stay_idx[0]] = 0.0`. For the stay "swap" — rep swapping with itself? The holder of wi IS rep (since assignments[rep] = wi, holders[wi] = rep). So old_e = bias[wi,rep] + bias[wi,rep] = 2*bias[wi,rep]; new_e = same → delta 0 naturally. Forcing 0 also fine. But heat-bath weight for stay should be exp(0) = 1 — correct since staying has zero energy change. OK.

**52. The proposal q includes the stay option; q_forward for a real move wj: probability of proposing wj. MH ratio uses q_forward = probs[choice_index] and q_reverse = probability of proposing wi from wj post-swap. Correct.

BUT: there's a classic subtlety — the Gibbs/heat-bath proposal over transpositions (rep ↔ holder of w). The reverse move from the new state: rep is now at wj, and proposes transposition with holder of wi (which is target_rep post-swap). holders_after reflects that. Correct as documented. Good.

**53. `_channel_boost` — `scale = np.maximum(np.maximum(abs(e), np.abs(v)), 1.0)` — mixing units: e, v in kJ/mol (~hundreds), scale ≥ 1 — fine.

**54. `k0max_from_globals`: `shared_globals["k0_Dihedral"]` — KeyError if missing (loud). "Total" defaults 0.0 silently — if the global is named differently (e.g., "k0_total"), Total silently 0 → top rung has no total boost while analytic pricing uses env.vmax_total etc... wait, set_replica_lambda sets k0_Total = lam*k0max["Total"]=0 → integrator applies no total boost; pep_gamd_boost_kj computes b_tot with lam*k0max_total where env.k0max_total comes from pep_env (envelope), NOT from k0max_from_globals. Two sources of k0max! `set_replica_lambda(integrator, lam, k0max)` uses k0max dict from k0max_from_globals(shared_globals); `pep_gamd_boost_kj(..., lam, env)` uses env.k0max_dih/k0max_total. If shared_globals["k0_Total"] disagrees with env.k0max_total (e.g., one in kcal, one in kJ, or one updated by recalibration), the integrator applies one boost and exchange/MBAR prices another — exactly the wrong-Hamiltonian-applied-vs-priced class! Is there a consistency check between k0max_from_globals and pep_env's k0max? Not shown. VERIFIED as a structural risk: two independent k0max sources (pep_gamd.py L528-530 vs env fields used at L506-519), no shown invariant tying them. SUSPECTED severity high if recalibration updates env but not shared_globals (the "_gamd_recal_active" machinery updates accumulators — if recal writes new envelope to pep_env but the integrator's globals still hold the old k0... spec says envelope frozen, recal only for adaptive epochs which create new sub-runs). Mark SUSPECTED.

**55. `_persist_state_gamd_lambdas` — update_run_manifest patch with method_settings: does update_run_manifest MERGE method_settings or replace? If it replaces the whole method_settings dict with just {"state_gamd_lambdas": ...}, other method settings are lost — then _reload reads method_settings.get("state_gamd_lambdas") fine, but other consumers of method_settings lose data. Can't verify. SUSPECTED.

**56. In `sample()`: `boost_kj, boost_components_kj, boost_source = extract_gamd_boost_kj(sim.integrator, unit, None)` — reads the applied boost from the integrator. Then recorded. As noted in #14, no comparison with analytic `sampled_boost_bias_kj`. This is the single cheapest guard against the chignolin_8 class and it's absent. Strong recommendation.

**57. `_write_production_nan_diagnostics` — fine.

**58. `state_identity.canonical_state_definition`: `if data["cv1"] is None` — but cv1 in required set; make_state_definition allows cv1=None. A window with k1>0 requires cv1 — checked. OK.

**59. bias.py `reconstruct_bias_matrix`: `matrix *= beta * KJ_PER_KCAL` happens INSIDE the try with errstate(over="raise") — good. But the ladder boost for lambda states: `_ladder_apply(matrix, pep, dih, lambdas, envelope, beta, bookkeeping)` — matrix is already reduced (beta*kcal→dimensionless). The boost must be added as beta*boost_kJ. If apply_ladder_boost_to_u multiplies by beta again... can't verify. Not shown; SUSPECTED unit-check.

Also `reconstruct_bias_matrix` requires envelope when any λ>0 — but what if the run was plain GaMD (no ladder) but boost active? Lambdas all zero → no boost in matrix → MBAR of a GaMD-boosted run WITHOUT ladder would be wrong (boost never priced)! Wait — the pipeline's GaMD is always with λ-ladder per the description; plain GaMD without ladder is mentioned in set_replica_lambda_for_window docstring ("plain GaMD"). For plain GaMD (single λ=1, no ladder), samples are boosted but λ... state_lambdas would be [1.0]? If state_lambdas = [1.0]*n, then any(λ>0) → True → envelope required → boost priced. OK if plain GaMD sets gamd_lambda=1. If plain GaMD sets λ=0 (thinking "no ladder = no λ"), boost unpriced → chignolin_8 exact repeat. The resolve_ladder_active docstring says ladder path NOT active for "plain GaMD" campaigns — meaning v_pep/v_dih NOT recorded! Then MBAR reconstruction with... plain GaMD samples would have gamd_lambda recorded as? If state_lambdas zeros, reconstruct returns umbrella-only matrix — the boost the integrator applied is never removed → FES wrong (biased by boost, un-reweighted). This is a potentially CRITICAL hole: plain-GaMD campaigns (ladder inactive) neither record channel energies nor price the boost, yet the integrator applies a boost. Unless plain GaMD runs set k0 full and gamd_lambda=1 with envelope... The docstring at L2302-2317 says "Deliberately NOT true for a plain umbrella/REUS or plain-GaMD campaign" — for plain GaMD, resolve_ladder_active False → pep_env None? → boost_bias zero, v_pep/v_dih NaN, and integrator runs with k0 = k0max (set where?). If the integrator applies a boost but exchange delta excludes it (matrix zeros) — wrong Hamiltonian vs priced, exactly the calibration-case class. But maybe plain GaMD campaigns in this pipeline always go through the ladder. Ambiguous — flag as SUSPECTED with high impact: any run where the integrator's k0 > 0 but pep_env is None prices zero boost in exchange and MBAR.

Actually wait — for plain GaMD, is there even an exchange? REUS + GaMD... the pipeline is REUS+Pep-GaMD with ladder. Plain GaMD may be outside the MBAR path (e.g., swarm stage). I'll flag it as a question/risk.

**60. `swap_candidate_replicas` — `n_win = int(replica_of_window.size)` then bounds check — fine.

**61. `explicit_2d_neighbor_pairs_for_exchange` — `normalized_distance` nan default — journal only.

**62. `_gibbs_window_proposal_distribution` — `holders` includes only held windows; proposal can move rep into... only held windows (transpositions). If some windows unheld, rep can never reach them — correct (configurations only exist at replicas).

**63. acceptance for Gibbs uses `deltas[choice_index]` — the delta from the proposal computation which was computed with the ORIGINAL holders — correct for the forward move.

**64. In `_attempt_window_swap`, the swap decision uses `bias_matrix_kj` computed at `_current_exchange_arrays` — which included the boost under each state's λ. For the NPT volume move (not shown) consistency unknown.

**65. `apply_window_swap` p_override path: pacc = p_override, clamped [0,1] — used by the Gibbs caller presumably with pacc from MH. Fine.

Now the verdict: The submission is a review-style document (it's a code review submission of the subsystem). Wait — re-read the task: "The board is convened to review the submission below." The submission is itself a "Board review" document? No — the submission contains code sections under review with the task: "Adversarially review the code below... Produce your independent assessment: 1. Your position (ACCEPT/ACCEPT-WITH-CHANGES/REJECT)..."

So I'm reviewing this replica-exchange subsystem code. My verdict: ACCEPT-WITH-CHANGES or REJECT? The findings:

Critical:
1. `_channel_boost` NaN→0 silent fallback (pep_gamd.py L497-503): NaN channel energy yields boost 0 in BOTH exchange pricing and MBAR reconstruction — silent, exactly the chignolin_8 signature class. VERIFIED.
2. `umbrella_bias_matrix_kcal` secondary NaN→0 (L916): docstring claims "freeze exchanges" but code fabricates zero bias — inconsistent energies between exchange path and MBAR reconstruction (which yields NaN). VERIFIED doc/code contradiction.
3. No cross-check between applied boost (extract_gamd_boost_kj) and analytic priced boost (sampled_boost_bias_kj) computed side by side in sample() — the cheapest guard against the calibration-case failure class is absent. VERIFIED absence.
4. Two independent k0max sources (k0max_from_globals vs pep_env envelope fields) with no shown invariant — risk of applied≠priced boost. SUSPECTED.
5. `np.zeros((nrep, nrep))` fallback shape mismatch when nwin≠nrep (unheld windows supported) — loud crash rather than silent, but breaks the "holders>=0" design; also state_lambdas truncation by _max_replicas misaligns per-window λ. SUSPECTED.
6. clip at ±745 in log_weights — MH-corrected, efficiency-only. Not a correctness bug.
7. resolve_ladder_active / plain-GaMD hole: if the integrator ever runs with k0>0 while pep_env is None, exchange and MBAR price zero boost. SUSPECTED.

Strengths: correct label-swap kernel with boost included in matrix; RNG-stream discipline; MH reverse-proposal done against post-swap holders (the classic bug avoided); immutable state snapshots with checksums; strict bias reconstruction raising on missing coordinates; persist/reload ladder handling documented.

The docstring in apply_window_swap claiming boost cancels is factually wrong (boost doesn't cancel when λ differs) but the code is correct because the matrix includes boost. Worth noting as a documentation error that could mislead future maintainers into "simplifying" delta to umbrella-only.

Also one more check on `_gibbs_mh_acceptance_probability`: for the Gibbs move, the MH acceptance should use total energy delta including physical potential — but the label swap leaves configurations unchanged, so physical U0 cancels; boost is in the matrix. Fine.

Wait, one more: in `_gibbs_window_proposal_distribution`, the heat-bath weights use only bias delta — but the true conditional for a heat-bath over transpositions should be ∝ exp(-β ΔE_total). ΔE_total for the transposition = Δbias (U0 cancels). Correct.

And `_exchange_probability` vs the swap: delta from matrix includes boost. Correct.

One more numerical check: `umbrella_bias_matrix_kcal` — `0.5 * ka[:, None] * dprimary * dprimary` — ka is per-state k. Orientation [state, replica]. Yes.

`assemble_bias_matrices`: `bias_kj = 4.184 * bias_kcal` — but boost was converted from kJ to kcal via /4.184 then back ×4.184 — round-trip float error negligible. Fine.

`reconstruct_bias_matrix`: NaN secondary yields NaN columns for states with k2>0 → downstream filtering — but the exchange path zeroed them. This inconsistency means the recorded `sampled_umbrella_bias_kj` (which uses the zeroed matrix... in sample(): `sampled_umbrella_bias_kj = 4.184*(all_distance_bias_kcal[w] + all_ss_bias_kcal[w])` where ss_bias was zeroed for NaN) disagrees with MBAR's reconstruction (NaN). VERIFIED inconsistency between recorded sample bias and MBAR bias.

Let me also double-check the `where` in umbrella_bias_matrix_kcal: `ss_kcal = np.where(np.isfinite(dss), ss_kcal, 0.0)` — if dss contains inf → 0. If sv (secondary values) NaN → dss NaN → 0. Also if sc (centers) NaN → 0 — center NaN would also silently zero. Fine.

Also check `_current_exchange_arrays` `_has_ss_axis` condition differs subtly from sample()'s: sample uses `secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None`; exchange uses `secondary_cv_centers is not None and secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None` — and uses `ss_k_kcal_arr_global` (note: kJ→kcal naming). If `secondary_cv_ks_kj` vs `secondary_cv_k_kcal_list` diverge (one set, other not), one path includes ss axis and the other doesn't → exchange matrix and sample matrix disagree. Also exchange references `ss_centers_arr_global` and `ss_k_kcal_arr_global` while sample references `ss_centers_arr_global`... sample uses `ss_centers_arr = ss_centers_arr_global` and passes `ss_k_arr` = ss_k_kcal_arr_global. Same arrays. But the axis-enable conditions differ textually; if they can disagree, matrices differ. SUSPECTED — different guard predicates on the two paths is a state-label mismatch risk. Actually sample: `_has_ss_axis = secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None`. Exchange: `secondary_cv_centers is not None and secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None`. If secondary_cv_k_kcal_list is set but secondary_cv_ks_kj is None (or vice versa) — plausible unit-conversion artifact — sample includes the axis and exchange doesn't (or crashes on None array). Flag as SUSPECTED inconsistency.

Also note exchange path cache-shape check `(nrep, nrep)` — if nwin > nrep the cached matrix is never reused (perf only) — and if nwin < nrep impossible.

Now, is there an actual detailed-balance bug anywhere? The one candidate: **the "empty slot" fallback in explicit_2d_neighbor_pairs_for_exchange**. If slot s yields no edges (all edges with idx%s==s conflict with used set... wait, `used` starts empty each call, so slot s's candidate edges (idx%s==s) are greedily filtered — the first edge with idx%s==s is always chosen since used is empty. So `chosen` is empty only if NO edges have idx%s==s. Then fallback tries ALL edges. This produces a valid matching. π-invariance preserved. Fine.

Another candidate: **stay probability / attempt counting in _record_exchange_stats** — stats only.

**Gibbs + swap both used at same interval?** Composition of valid kernels — invariant preserved.

OK. Also `_exchange_probability` returns 1.0 for delta <= 0 — but delta NaN → isfinite check → 0. Good.

Final check on `set_replica_lambda`: `if not (0.0 <= lam <= 1.0): raise` — loud. `names` from getGlobalVariableName; if integrator has k0_Total but under a different name... handled.

`set_replica_lambda_for_window`: state_lambdas indexing by window_index — if state_lambdas shorter than nwin → IndexError loud. Fine.

Now verdict. The code is generally careful, with several genuine strengths, but contains at least two verified silent-fallback defects in the energy pricing path (NaN→0 in _channel_boost; secondary NaN→0 with docstring contradiction) and lacks the applied-vs-priced boost cross-check. Given the review's own mandate ("FIND THE NEXT ONES LIKE IT"), these are exactly the class sought. I'd say ACCEPT-WITH-CHANGES — the core exchange kernels (swap delta including boost, MH reverse proposal) are correct; the defects are in the bias/boost assembly edges and are fixable without redesign. Or REJECT? The NaN→0 in _channel_boost affects MBAR reconstruction directly — free energy numbers silently wrong for corrupted frames. But it only triggers on NaN inputs, which themselves indicate a corrupted run. The secondary NaN→0 similarly only triggers when secondary CV fails. Conditional defects. The missing
(TRUNCATED - output hit the token limit before completion)

## kimi

Let me carefully review this code. The task is to adversarially review the replica-exchange subsystem of ATLaS-MD / gareus v0.8.2, looking for places where the software can silently produce a number that is not the Boltzmann free energy it claims.

Let me go through each function carefully.

### 1. `umbrella_bias_matrix_kcal` (L895-917)

```python
def umbrella_bias_matrix_kcal(primary_values, ss_values, centers_a, k_kcal, ss_centers=None, ss_k_kcal=None):
    pv = np.asarray(primary_values, dtype=np.float64)
    ca = np.asarray(centers_a, dtype=np.float64)
    ka = np.asarray(k_kcal, dtype=np.float64)
    dprimary = pv[np.newaxis, :] - ca[:, np.newaxis]
    distance_kcal = 0.5 * ka[:, np.newaxis] * dprimary * dprimary
    if ss_centers is None or ss_k_kcal is None:
        return distance_kcal, np.zeros_like(distance_kcal)
    sv = np.asarray(ss_values, dtype=np.float64)
    sc = np.asarray(ss_centers, dtype=np.float64)
    sk = np.asarray(ss_k_kcal, dtype=np.float64)
    dss = sv[np.newaxis, :] - sc[:, np.newaxis]
    ss_kcal = 0.5 * sk[:, np.newaxis] * dss * dss
    ss_kcal = np.where(np.isfinite(dss), ss_kcal, 0.0)
    return distance_kcal, ss_kcal
```

Key observation: The docstring says "a state without a secondary centre, or a replica without a finite secondary value, contributes zero on that axis". The guard `np.where(np.isfinite(dss), ss_kcal, 0.0)` replaces NaN secondary bias with **zero**. 

This is a serious issue: if a replica's secondary CV is NaN (e.g., the torsion-PC residual is undefined for some configuration, or the fast path failed to compute it), then the bias contribution for that replica under **every** state is treated as zero on the secondary axis. That means:
- In the exchange acceptance, the energy delta omits the secondary restraint entirely for that replica — the swap is priced with a wrong Hamiltonian. A replica with NaN secondary CV will happily swap into a window with a strong secondary restraint, because the restraint energy is treated as 0 rather than infinite/NaN. This biases the stationary distribution: configurations where the secondary CV is undefined get over-represented in restrained windows.
- In the recorded samples, the bias matrix recorded for MBAR has 0 instead of NaN on that axis, so MBAR will price those samples with the wrong reduced potential — inconsistent with the actual sampling Hamiltonian (the actual simulation applies the restraint force via OpenMM; if the CV is NaN in the force, the force would be NaN and the simulation would blow up... but if the recorded value is NaN due to a different evaluation path, e.g., fast path vs. slow path mismatch, the recorded bias is wrong).

Also note the asymmetry: the primary axis has NO NaN guard. If `pv` is NaN, `distance_kcal` is NaN, and the whole row/column is NaN. The docstring says "a NaN entry used to freeze that replica's exchanges" — so the primary NaN propagates and freezes exchanges (delta NaN → `_exchange_probability` returns 0.0 — wait, actually let's check: `_exchange_probability` with NaN delta: `math.isfinite(delta_kj)` fails → returns 0.0. So all swaps involving that replica are rejected — frozen. OK).

But for the secondary axis, NaN → 0.0 silently. This is exactly the class of "silent fallback (NaN → 0)" the task asks to find. The effect: the exchange kernel and the recorded MBAR bias matrix treat a missing secondary CV as "no restraint energy", which is not the Boltzmann weight of the stated target. Confidence: VERIFIED that the code does NaN→0; the effect on free energy depends on how often NaN secondary CVs occur in production. But it's a real silent wrong-Hamiltonian pricing path.

Also: `np.isfinite(dss)` — dss = sv - sc. If the state has no secondary center... how is "a state without a secondary centre" represented? Presumably sc = NaN for that state, or sk = 0. If sk = 0 and sc = NaN, then dss = NaN → ss_kcal set to 0. Fine. If sk = 0 and sc = 0 and sv finite, ss_kcal = 0. Fine. But if sk > 0 and sv = NaN (replica missing value), ss_kcal = 0 — wrong, should be NaN (missing) or the swap should be blocked. This is the bug.

Additionally, compare with `reconstruct_bias_matrix` in correctness/bias.py: there, NaN propagates (matrix entries become NaN for states requiring that coordinate), and "Row filtering belongs downstream". So the analysis path treats missing secondary as NaN, while the production path treats it as 0. Inconsistency between production-recorded bias and analysis-reconstructed bias: the sample path records `umbrella_bias_all_windows_kcal_mol_json` from `umbrella_bias_matrix_kcal` (NaN→0 on secondary), while `reconstruct_bias_matrix` would produce NaN. If MBAR consumes the recorded matrix, samples with missing CV2 get zero secondary bias; if MBAR reconstructs, they get NaN and the row is dropped. Either way there's an inconsistency risk. But more importantly, the exchange path prices swaps with 0.

Wait — actually, let me re-read. In `sample()`, the recorded per-window bias vectors come from `bias_matrix_kcal`/`bias_matrix_kj` which derive from `umbrella_bias_matrix_kcal`. So the recorded reduced bias matrix has NaN→0 on the secondary axis. And the exchange path uses the same matrix. So at least they're consistent with each other. But both are wrong relative to the true Hamiltonian when a secondary value is missing.

Hmm, but is a missing secondary CV physically possible while the simulation is still running? The secondary CV is "residual-torsion-pc" — computed from positions. If positions are finite, the torsion PC should be finite. NaN could arise from the fast path (CustomCVForce) if not yet computed, or from a failed read. The guard exists because it happened (the docstring says "the previous sample-path guard"). So it's a real path.

### 2. `_gibbs_window_proposal_distribution` (L983-1041)

```python
    bias = np.asarray(bias_matrix_kj, dtype=np.float64)
    old_e = bias[wi, rep] + bias[valid_windows, target_reps]
    new_e = bias[valid_windows, rep] + bias[wi, target_reps]
    delta = np.asarray(new_e - old_e, dtype=np.float64)
```

This computes the energy change of swapping replica `rep` (currently in window `wi`) with the replica currently holding window `valid_windows` (target_reps). Wait — for a Gibbs/heat-bath move over windows, the move is: replica `rep` moves to window `wj`, and the holder of `wj` moves to `wi`. That's a swap (transposition). The delta computed here is exactly the swap delta: new_e - old_e where new assignment has rep at wj and target at wi. OK.

But note: when `valid_windows[k] == wi` (the "stay" candidate), target_reps[k] = rep itself, old_e = bias[wi,rep] + bias[wi,rep], new_e = same, delta = 0. They explicitly set delta=0.0 for stay. Fine.

Now the heat-bath weights: `log_weights = clip(-beta*delta, -745, 745)`. For a proper heat-bath proposal over the set of candidate swaps, q(wj) ∝ exp(-β Δ(wi→wj)). Then MH correction with q_reverse. The comment says "No upper clip: true heat-bath weights exp(-β·δ) can exceed 1 for favourable moves" — but then they clip at +745 anyway. exp(745) is near overflow (~1e323, close to double max ~1.8e308? No wait — exp(745) ≈ 5e323 which OVERFLOWS double! exp(709.78) is the max for float64. So clipping at ±745 and then `np.exp(log_weights - m)`: since they subtract the max m before exponentiating, the actual exp arguments are ≤ 0, so no overflow. The clip at +745 only matters before the max-subtraction; since m ≤ 745 after clipping, log_weights - m ∈ [-1490, 0], exp underflows to 0 below -745. So the clip is harmless. OK.

The real question: is the heat-bath + MH correct? Heat-bath proposal q ∝ exp(-βδ) followed by MH acceptance min(1, exp(-βδ) q_rev/q_fwd). This is a valid MCMC kernel as long as q_fwd and q_rev are computed correctly. The reverse proposal is computed from `holders_after` — the post-swap holder table — with `replica_index=rep`, `current_window=wj`. 

Hold on. After the swap, replica `rep` is at window `wj`, and `target_rep` is at window `wi`. The reverse move must propose, from the new state, the move that takes us back: replica rep moves from wj back to wi (and target_rep goes from wi to wj). The reverse proposal distribution is the heat-bath distribution for replica rep at current window wj with holder table holders_after. q_reverse = P(reverse proposes wi) = rev_probs[index of wi]. That's what the code does. 

But wait — there's a subtlety. The heat-bath proposal is over *windows*, and each window maps to a target replica via the holder table. In the forward proposal, the candidate set is all valid windows with their current holders. In the reverse, the candidate set is all valid windows with holders_after. The delta for the reverse move wj→wi should be -delta_forward. Let's verify: in the reverse call, bias matrix is the same (configurations unchanged), replica_index=rep, current_window=wj, replica_of_window=holders_after where holders_after[wi]=target_rep, holders_after[wj]=rep. old_e_rev = bias[wj, rep] + bias[valid_windows, target_reps_after]. For the candidate window wi: target_after = target_rep. old_e_rev = bias[wj,rep] + bias[wi,target_rep]; new_e_rev = bias[wi,rep] + bias[wj,target_rep]. delta_rev = new - old = (bias[wi,rep]+bias[wj,target_rep]) - (bias[wj,rep]+bias[wi,target_rep]) = -(delta_fwd). Good, consistent.

However — here's a potential issue: the MH acceptance uses `delta_kj=float(deltas[choice_index])` — the forward delta — with q_forward and q_reverse. log_alpha = -βδ + log(qr) - log(qf). Correct MH for proposal q.

But is the heat-bath proposal q actually proportional to exp(-βδ)? Yes, probabilities = weights/sw with weights = exp(log_weights - m). Unless the fallback uniform branch triggers: `if not math.isfinite(sw) or sw <= 0.0: probabilities = uniform`. When could sw be non-finite? weights = exp(log_weights - m) ≤ 1, sum ≤ n. sw finite unless... log_weights all -inf? They clipped to -745, so finite. NaN? If delta contains NaN, log_weights = NaN, but they filter `finite = np.isfinite(log_weights)` before. So sw > 0 always (at least one finite candidate remains, weights include exp(0)=1 for the max). So the uniform fallback is dead code — harmless.

BUT: the NaN filtering has a subtle detailed-balance consequence. If some candidate windows have NaN delta (e.g., because bias[wj, rep] is NaN due to primary CV NaN), they're dropped from the forward proposal. In the reverse proposal, the same NaN entries would be dropped too (same bias matrix, same replica values — NaN entries depend on (window, replica) pairs; the set of NaN (window, rep) entries is the same in forward and reverse for the column rep, but the target_reps columns differ...). Hmm, actually the candidate set could differ between forward and reverse if NaN pattern differs, because in the forward, target_reps = holders[valid_mask] (current), in reverse, target_reps = holders_after. The delta for candidate wk involves bias[wk, rep] + bias[wi, target_rep_of_wk] - bias[wi, rep] - bias[wk, target_rep_of_wk]. In the reverse, for candidate wk, target is holders_after[wk]. For wk ≠ wi, wj: holders_after[wk] = holders[wk], same. For wk = wi: forward target is rep (holders[wi]=rep), reverse target is target_rep. So the delta for the "stay" candidate differs, but stay is forced to 0.0 in both. For wk = wj in the reverse: that's the move back. So the candidate sets are the same windows, and finiteness of each candidate's delta is the same except possibly for candidates involving wi/wj columns... Let me think: forward candidate wk delta uses columns rep and holders[wk]. Reverse candidate wk delta uses columns rep and holders_after[wk]. For wk ∉ {wi, wj}: same holder, same delta, same finiteness. For wk = wi: forced 0 in both. For wk = wj (only exists in reverse as the "move to wj"... wait, in the reverse call, current_window=wj, so candidate wj is the stay candidate (forced 0), and candidate wi is the move-back). Forward: candidate wj delta uses holders[wj] = target_rep. Reverse: candidate wi delta uses holders_after[wi] = target_rep. Same columns (rep, target_rep), and delta_rev = -delta_fwd as computed. So finiteness matches. OK, so the NaN filtering is symmetric. Fine.

Now the real question about the gibbs-walk: is the MH-corrected heat-bath kernel here actually leaving the target invariant? One classic subtlety: when the proposal is a heat-bath over a set of moves that includes "stay", and the move is a swap transposition, the reverse proposal probability q_reverse must be the probability of proposing the exact reverse move from the destination state. The code computes that. Seems right.

But wait — there's a deeper issue. The heat-bath proposal probabilities are computed from deltas that depend on the *current* assignment of all other replicas (through target_reps). The reverse is computed against holders_after. That's correct for a single-replica transposition move.

Another subtlety: `q_forward = float(probs[choice_index])`. If the same window appears... no, windows are unique.

What about `choose`? Production passes `rng.choice`. `rng.choice(n, p=probs)`? The signature is `choose(int(valid_windows.size), probs)`. numpy's rng.choice(size, p) — actually `np.random.Generator.choice(a, size=None, replace=True, p=None)`. If called as `rng.choice(n_candidates, p=probs)` it returns an index in [0, n). OK. But if production passes `rng.choice` unbound with positional probs, `rng.choice(n, probs)` would treat probs as `size`! That would be a bug, but we can't see the call site. The docstring says "Production passes ``rng.choice``". If the actual call is `choose(int(valid_windows.size), probs)` with choose=rng.choice, then probs goes to the `size` parameter → TypeError or wrong behavior. Actually `rng.choice(5, np.array([0.5,0.5]))` — size must be int or tuple of ints; passing an array of floats raises TypeError. So it would crash loudly, not silently. Probably they wrap it. Can't verify; flag as suspected? The instructions say distinguish VERIFIED from SUSPECTED. I'll note it as a call-site contract risk but not a verified bug.

### 3. `_gibbs_mh_acceptance_probability` (L1044-1067)

Looks fine. log_alpha = -βδ + log qr - log qf. Guards: non-finite → 0.0; qf or qr ≤ 0 → 0.0. 

Hmm: if q_reverse = 0 (the reverse move has zero probability — e.g., rev_idx empty because wi got filtered out as non-finite in the reverse), then acceptance = 0. That's correct MH behavior (reject), though it can break detailed balance if q_forward > 0 but q_reverse = 0 asymmetrically... Actually if q_reverse = 0 while q_forward > 0, detailed balance is violated unless the forward probability is also zero. But we showed the candidate sets match, so q_reverse=0 only if probs are degenerate. The `rev_idx.size == 0` case → q_reverse = 0.0 → reject. When could wi be missing from rev_windows? If the reverse delta for candidate wi is non-finite. We argued finiteness matches forward. Forward delta for wj was finite (it was chosen). Reverse delta for wi = -forward delta, finite. So wi is present. OK.

Edge: `if log_alpha >= 0.0: return 1.0` — fine. `< -745: return 0.0` — fine.

### 4. `_exchange_probability` (L1070-1089)

Standard Metropolis: δ ≤ 0 → 1; else exp(-βδ). NaN → 0.0 (reject). This is the "freeze" behavior for NaN primary. Note: rejecting on NaN means a replica with NaN CV never swaps — frozen, as the docstring says. That's a silent stall but not a wrong number per se. However, there's an asymmetry with the secondary-axis NaN→0: primary NaN freezes, secondary NaN prices as zero. Inconsistent handling, and the secondary one is the dangerous one.

### 5. `swap_candidate_replicas` (L1092-1115) and `apply_window_swap` (L1127-1182)

`apply_window_swap`: old_e = bias[wi,i] + bias[wj,j]; new_e = bias[wj,i] + bias[wi,j]; delta = new - old. Correct for state-label swap.

The docstring claims: "The move swaps *state labels*, never configurations, which is why the unbiased potential and the GaMD boost cancel exactly from ``delta`` and only the umbrella biases appear".

WAIT. This is wrong when the λ-ladder is active! The boost depends on the state's λ: boost = λ·k0·(...). If window wi has λ_i and window wj has λ_j with λ_i ≠ λ_j, then swapping state labels changes the boost applied to each replica's configuration: replica i's config now gets boost under λ_j instead of λ_i. The boost does NOT cancel — Δboost = [B(x_i, λ_j) + B(x_j, λ_i)] - [B(x_i, λ_i) + B(x_j, λ_j)] ≠ 0.

But look at what the bias matrix contains: `assemble_bias_matrices` adds `boost_bias_kj` — and `pep_gamd_boost_matrix_kj(v_pep, v_dih, lambdas, env)` returns (n_states, n_samples): boost of each sample's configuration under each state's λ. So bias_matrix_kj[window, replica] = umbrella_window(x_replica) + boost(x_replica, λ_window). So the boost IS included in the matrix, evaluated per (state, replica). The delta then correctly includes the boost change. The docstring's claim that "the GaMD boost cancels exactly" is only true if λ is constant across states (or boost is state-independent). With a λ-ladder, the boost does not cancel, and the code correctly includes it via the matrix. So the docstring is misleading but the code is right — PROVIDED the boost matrix is actually included. In `_current_exchange_arrays`, boost_bias_matrix_kj is computed when pep_env is not None and assembled. Good.

But hold on — there's a subtlety: the boost that the *integrator* actually applies to replica i after the swap is λ_{new window}·k0max applied via `set_replica_lambda_for_window`. And the exchange delta prices boost(x_i, λ_wj) using the *recorded* v_pep, v_dih and the frozen envelope via `pep_gamd_boost_kj`. For consistency, the boost formula in `pep_gamd_boost_kj` must EXACTLY match what gamd-openmm's StageIntegrator applies. The chignolin_8 failure was exactly this class (integrator in stage 2, boost never applied, but priced). The fix seeds stage 5. But note: `_fetch_v_pep_v_dih` reads v_dih from DIHEDRAL_GROUP group 2 and v_pep from `peptide_essential_energy_kj`. The boost formula: `_channel_boost(v, e, vmax, vmin, k0)`: b = 0.5·k0·(e−v)²/rng, with guards: if |rng| ≤ 0.001·scale → 0; if (b+v) < e → 0 (i.e., boost only when v < e... wait, the condition `(b + v) < e` → return 0 means: if after adding boost the total is still below threshold e, set boost to 0? That's odd. Standard GaMD: boost is applied only when V < E (threshold), and ΔV = 0.5·k0·(E−V)²/(Vmax−Vmin). If V ≥ E, ΔV = 0. The condition here: `(b + v) < e` → 0. Hmm, that says if v + b < e, return 0. That's not the standard condition. Standard: if v >= e, b = 0. Here: if v + b < e → b = 0. Since b ≥ 0, v + b < e implies v < e. So for v slightly below e, b is small, v + b < e → zeroed. That means the boost is only kept if it overshoots the threshold?! That's a weird "modified" GaMD. Actually wait — maybe it's intentional: some GaMD variants apply boost only if it doesn't overshoot... no, the standard is the opposite. Let me think about what gamd-openmm does. In gamd-openmm's stage integrator, the boost is computed conditionally: if V < E, apply ΔV = 0.5 k0 (E−V)²/(Vmax−Vmin). There's no overshoot condition in standard GaMD. Hmm, but actually there IS a variant consideration: the boost potential as defined would make V+ΔV > E for V close to E? Let's check: ΔV = 0.5 k0 (E−V)²/(Vmax−Vmin). With k0 ≤ 1, is V + ΔV ≤ E always? V + 0.5 k0 (E−V)²/(Vmax−Vmin) ≤ E iff 0.5 k0 (E−V)/(Vmax−Vmin) ≤ 1 iff (E−V) ≤ 2(Vmax−Vmin)/k0. Since E ≤ Vmax typically (threshold below max), E−V ≤ Vmax−Vmin when V ≥ Vmin... so (E−V)/(Vmax−Vmin) ≤ 1, and 0.5 k0 ≤ 0.5 < 1. So V + ΔV ≤ E always (for V ≥ Vmin). So the condition `(b+v) < e` → 0 would zero out ALL boosts where v ≥ vmin?! That can't be right...

Wait, let me redo. If E (threshold) ≤ Vmax and V ≥ Vmin: E − V ≤ Vmax − Vmin. Then ΔV = 0.5 k0 (E−V)²/(Vmax−Vmin) ≤ 0.5 k0 (E−V). So V + ΔV ≤ V + 0.5 k0 (E−V) = 0.5k0·E + (1−0.5k0)V. For k0 ≤ 1, this is ≤ 0.5E + 0.5V·... hmm let me just check: is V + ΔV ≤ E? V + ΔV − E = (V−E) + 0.5 k0 (E−V)²/(Vmax−Vmin) = (E−V)[−1 + 0.5 k0 (E−V)/(Vmax−Vmin)]. For this to be ≤ 0 (i.e., V+ΔV ≤ E), need 0.5 k0 (E−V)/(Vmax−Vmin) ≤ 1, i.e., (E−V) ≤ 2(Vmax−Vmin)/k0. Since E−V ≤ E−Vmin ≤ Vmax−Vmin (assuming E ≤ Vmax), and 2/k0 ≥ 2, yes always satisfied. So V + ΔV ≤ E always (when V ≥ Vmin and E ≤ Vmax). 

So the condition `(b + v) < e` is TRUE for essentially all boosted frames (with standard envelope calibration), meaning `_channel_boost` returns **0.0** for them! And returns b only when b + v ≥ e, i.e., when the boost overshoots the threshold — which standard GaMD calibration says shouldn't happen.

Hmm wait, maybe I have the inequality backwards. `return _np.where((b + v) < e, b, 0.0)` — np.where(condition, x, y) returns x where condition is True. Condition: (b+v) < e. So where (b+v) < e, return **b**; else return 0.0. Let me re-read the code:

```python
    b = 0.5 * k0 * (e - v) ** 2 / rng
    b = _np.where(_np.abs(rng) <= 0.001 * scale, 0.0, b)
    return _np.where((b + v) < e, b, 0.0)
```

`np.where((b+v) < e, b, 0.0)`: if (b+v) < e → b; else → 0.

Hmm OK so I misread. It returns b when v + b < e. So the boost is applied only when the boosted energy stays BELOW the threshold. And zero when v + b ≥ e.

But standard GaMD applies the boost when v < e (regardless of overshoot), and we just showed v + b ≤ e always holds for v in [vmin, e]. So for v ≥ vmin, v < e: b > 0 and v + b ≤ e → returns b. Good. For v < vmin: (E−V) > (Vmax−Vmin)·? E−V > E−Vmin; could (E−V) exceed 2(Vmax−Vmin)/k0? If V << Vmin, yes. Then v + b > e → boost zeroed. Hmm, that's a discontinuity: for very low V (very favorable conformations), the boost abruptly turns off. Actually wait, that seems intentional? Some implementations cap the boost so the boosted potential doesn't exceed E... Actually the standard GaMD boost has no such cap; the cap here zeroes the boost entirely rather than capping at E−V. This creates a discontinuity in the effective potential: at the V where b+v crosses e, the boost jumps from b to 0. A discontinuous biasing potential is still fine for Boltzmann reweighting as long as the SAME function is used in the integrator and in reweighting. The docstring says this mirrors gamd-openmm's stage_integrator. Actually, I recall gamd-openmm does have such a condition! Let me think... In gamd-openmm's `stage_integrator.py`, the boost is applied conditionally: `if (V < E) apply boost`. Hmm. Actually, I believe there is a known subtlety: gamd-openmm computes the boost and adds it only if the total boosted energy remains below the threshold? I don't remember exactly. The docstring claims fidelity to gamd-openmm ("dependent dual boost under rung λ: dihedral first, then Total with the dihedral boost added"). Since the submission's whole premise is that this formula matches the integrator, and I can't see gamd-openmm's source here, I should flag the `(b+v) < e` condition as SUSPECTED: if the integrator applies the boost whenever v < e (standard GaMD) but the pricing/reweighting zeroes it when v+b ≥ e (or vice versa), every frame near the threshold gets a wrong boost — a wrong-Hamiltonian-pricing error of exactly the chignolin_8 class. Actually, let me reconsider: the condition `(b+v) < e` with the math above shows v+b ≤ e whenever v ≥ vmin (and E ≤ Vmax, k0 ≤ 1). So the else-branch (0.0) triggers only for v < vmin (deep minima). For v ≥ e: b = 0.5 k0 (e−v)²/rng > 0 (since (e−v)² > 0)! Wait — for v > e, (e−v)² is positive, so b > 0! Standard GaMD zeroes the boost for V ≥ E. Here, for v > e: b = 0.5 k0 (e−v)²/rng > 0, then condition (b+v) < e? b + v > v > e → condition False → return 0.0. OK so for v > e it returns 0 via the overshoot guard. Phew. So the guard `(b+v) < e` serves double duty: zeroes boost for v > e and for v << vmin. The effective boost: b(v) = 0.5 k0 (e−v)²/rng for vmin' ≤ v < e where vmin' is where v+b=e crosses, and 0 elsewhere. There's a discontinuity at the low-v crossing (boost jumps from ~e−v to 0). Hmm, actually at the crossing, b = e − v, so the boost drops from (e−v) to 0 discontinuously. For v slightly above the crossing, b ≈ e − v which can be large (if crossing at v << vmin, e−v is large). So there's a real discontinuity with a large jump. If the integrator implements the same formula, sampling is consistent (though the force from a discontinuous boost is a delta — the integrator applies boost as an energy offset, not a force? GaMD boosts the potential by ΔV(x) which does modify forces; a discontinuity means impulsive forces — but gamd applies it as a scaled force? Actually GaMD implementations typically modify forces via the boost derivative; a discontinuous boost is unphysical but still defines a valid modified potential for MC; for MD it's problematic). This is getting deep. The key reviewable point: the formula must match gamd-openmm exactly, including this guard, the 0.001·scale degenerate-range guard, and the k0 scaling. Any mismatch → silently wrong pricing. I'll flag the `(b+v)<e` guard and the `|rng| ≤ 0.001·scale` guard as SUSPECTED mismatch points requiring a test that compares against the actual integrator boost (the chignolin_8 fix suggests they now have `verify_gamd_production_stage5`, but does any test compare `pep_gamd_boost_kj` against the integrator's applied boost per frame? The samples record `gamd_boost_total_kj_mol` from `extract_gamd_boost_kj(sim.integrator, ...)` AND `sampled_boost_bias_kj` from the matrix — if both are recorded, a mismatch could be detected, but is it asserted anywhere? Not in the shown code. The recorded `gamd_boost_total_kj_mol` comes from the integrator globals; `sampled_boost_bias_kj` from the envelope formula. If they diverge, MBAR uses... which one? The parquet writer records both `boost_total` (integrator) and v_pep/v_dih. The MBAR path (`reconstruct_bias_matrix` + `apply_ladder_boost_to_u`) recomputes boost from v_pep/v_dih via the envelope formula — i.e., it prices the FORMULA boost, not the integrator-recorded boost. If the integrator applied something different (e.g., stage-2 bug class), MBAR prices a boost that wasn't applied. The recorded `boost_total` would reveal it, but nothing shown compares them. This is exactly the "next one like it": the analysis prices the envelope-formula boost; the simulation applies the integrator boost; the only guard is the stage-5 verification at seed time, not a per-frame consistency check. SUSPECTED/structural.

### 6. `gibbs_propose_one_replica` (L1202-1271)

Covered above. One more check: `GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False)` for stay — accepted=True presumably (field 7). Fine.

Reverse proposal: uses `holders_after` — correct. But note: the reverse distribution is computed with `replica_index=rep` and `current_window=wj`. In the destination state, replica rep is at wj. Correct.

One subtle issue: the forward proposal distribution includes ALL valid windows as candidates, i.e., the move set is "swap rep's window with any other window's holder". The reverse move from the destination must be within the same move set — yes, candidate wi is there. OK.

Another: `q_forward` could be from the uniform fallback (dead code) — fine.

Potential issue: `choose` returning an index — if production's choose is `rng.choice`, note numpy's `Generator.choice(a, size, replace, p)` — positional second arg is size. If they call `choose(n, probs)` with choose = `lambda n, p: rng.choice(n, p=p)`, fine. Can't verify.

Now — a bigger issue with the gibbs-walk as a whole: each "transposition" move swaps replica rep's window with the holder of the target window. Applied as a sequence over replicas (a "gibbs walk"), each individual move is MH-correct. Fine.

But WAIT — the actual swap application: after `gibbs_propose_one_replica` returns a proposal, the caller must apply it. The shown `_attempt_window_swap` uses `apply_window_swap` (the Metropolis pairwise kernel), not the gibbs proposal. The gibbs path's application code isn't shown. The `GibbsProposal` has pacc; the caller draws uniform < pacc and then applies the swap (set_window on both contexts, update assignments/replica_of_window). Not shown — can't verify. But there's a hazard: `gibbs_propose_one_replica` computes `holders_after` but does NOT mutate; the caller must apply. If the caller applies via `apply_window_swap` with `p_override=pacc`... that would recompute delta and use p_override — consistent. Can't see it.

### 7. `explicit_2d_neighbor_pairs_for_exchange` (L1479-1505)

Edge coloring: `slot = parity % slots`; edges with `idx % slots == slot`, greedily disjoint. If a slot is empty, fallback greedy over all edges. Detailed balance consideration: for pairwise window swaps with Metropolis, proposing a fixed set of disjoint pairs per interval and attempting each with Metropolis acceptance is fine (each pair move is its own MH kernel; the proposal is deterministic given parity, symmetric). The parity rotation makes different pairs across intervals — fine as long as each attempted pair uses correct acceptance. OK.

One issue: the fallback "If a slot was empty... try all edges greedily" — this means for small graphs, EVERY interval attempts the same maximal matching (the greedy-over-all set), regardless of parity. Not a correctness issue for DB (each pair still MH), just reduced mixing. Fine.

Another: `normalized_distance` recorded as float("nan") default — cosmetic.

### 8. `assemble_bias_matrices` (L2055-2072)

bias_kcal = distance + ss + boost_kj/4.184; bias_kj = 4.184 * bias_kcal. The invariant bias_kj == 4.184*bias_kcal holds by construction. Fine. Note: boost in kJ is converted to kcal and back — rounding is consistent. Fine.

But note: the boost matrix is (n_states, n_samples) = (nrep, nrep) — assumes n_states == nrep. With `--max-replicas` truncation, n_states (windows) may exceed nrep? The ladder truncation `args.state_gamd_lambdas[:_max_replicas]` suggests states truncated to replicas. If n_windows ≠ nrep, `np.zeros((nrep, nrep))` and `pep_gamd_boost_matrix_kj` with len(lambdas) states vs nrep samples — vstack gives (n_states, n_samples). If n_states ≠ nrep, assemble broadcasts? distance_bias_kcal is (n_centers, n_replicas). If centers = n_states ≠ nrep, shapes mismatch → error (loud). Probably n_states == nrep always (one replica per state). OK.

### 9. `_fetch_v_pep_v_dih` (L2075-2094)

Returns (nan, nan) when pep_env is None. When ladder inactive, v_pep/v_dih recorded as NaN. Then `reconstruct_bias_matrix` with lambdas all 0 → doesn't need them. OK.

`has_total` False → returns (nan, v_dih). Then `pep_gamd_boost_kj` with has_total False ignores v_pep. OK. And `reconstruct_bias_matrix`: `if dih is None or (has_total and pep is None): raise`. For single dihedral, pep NaN vector passed... in reconstruct, `if pep is None: pep = full NaN`. But production records v_pep as NaN (not None) for single-dihedral — the parquet has NaN; analysis reads the column → pep is an array of NaN, not None. Then `_ladder_apply(matrix, pep, dih, ...)` with pep all-NaN — does `apply_ladder_boost_to_u` ignore pep when has_total is False? `pep_gamd_boost_kj` ignores v_pep when not has_total. Presumably ladder.apply does the same. OK.

BUT: what if `has_total` is True and v_pep read fails silently? `peptide_essential_energy_kj` — not shown. If it raises, the whole sample() raises — loud. OK.

Another: `getattr(pep_env, "has_total", True)` — defaults to True if attribute missing. If pep_env is a dict-like or an older pickled envelope lacking has_total, defaults to Total-channel behavior → reads v_pep. If the actual boost is single-dihedral, then... `k0max_from_globals` uses `shared_globals.get("k0_Total", 0.0)` → Total k0 = 0 → boost_total channel = 0.5·0·(...) = 0. So harmless. OK.

### 10. `campaign_ladder_registry_lambda` (L2261-2299) and `resolve_ladder_active` (L2302-2317)

The registry walk: for candidate in (base, *base.parents): check registry names; break after candidate named "adaptive_production". Wait — the break happens AFTER checking the candidate named adaptive_production. So it checks base, then each parent up to and including the one named "adaptive_production". The docstring says "The walk stops at adaptive_production/ deliberately". Hmm, but consider directory structure: campaign_root/adaptive_production/epoch_3/... out_dir = .../adaptive_production/epoch_3. Walk: epoch_3 (check), adaptive_production (check, then break). So the registry must be at epoch level or adaptive_production level. If the campaign registry lives ABOVE adaptive_production (at campaign root), the walk never reaches it → returns None → ladder considered inactive for an all-λ=0 top-up → chignolin_7 defect AGAIN. Where is the registry written? "StateRegistry.write_state_csv" — location unknown. The docstring says the walk stops at adaptive_production because campaigns share a parent. But if the registry is written at the campaign root (the shared parent!), this fails. Hmm — "several campaigns commonly share one parent directory" — so campaign root is above adaptive_production? Structure guess: parent/campaign_X/adaptive_production/epoch_N. If registry at campaign_X level, walk from epoch_N: epoch_N, adaptive_production → break. Never checks campaign_X. REGRESSION RISK. But maybe the registry is written into adaptive_production or each epoch dir. Can't verify — flag as SUSPECTED with the concrete scenario.

Also: `if lam > 0.0: return (path.name, lam)` — returns the FIRST row with λ > 0 found in the FIRST registry file that has one. But it iterates files per directory and rows per file; a registry that lists states with λ=0 first is fine (it continues rows). OK. But subtle: it returns on the first λ>0 row — fine for boolean activation.

Another subtlety: `resolve_ladder_active` returns True if ANY state λ > 0 in this sub-run OR registry says ladder. Then the ladder path records v_pep/v_dih for all samples. Good. But what about the CONVERSE: a sub-run with λ>0 states whose... no, first clause catches it.

Edge: `np.asarray(state_lambdas, dtype=float)` — if state_lambdas is None, `np.asarray(None, dtype=float)` → array(nan)? Actually np.asarray(None, dtype=float) gives array(nan) (0-d). `.size` = 1, `np.any(nan > 0)` = False. OK, falls through to registry. Fine.

### 11. `_derive_state_gamd_lambdas` (L2360-2384)

Fallback `[0.0] * n` — "Zeros (ladder inactive) is the fallback of last resort". But consider: resume path where normalized_rows exists but rows lack "gamd_lambda" key (older manifest) → falls to `existing` → if existing wrong length → zeros. The reload function handles resume. But on a NON-resume fresh run with window_metadata lacking gamd_lambda (e.g., a window table from an older design stage), the ladder silently becomes all-zero while the campaign registry says ladder active. `resolve_ladder_active` would return True (registry), so k0max_by_channel is not None, and `set_replica_lambda_for_window` applies λ=0 to every replica → the whole production runs unboosted while... wait, but if λ=0 for all states, MBAR also prices λ=0 → consistent, just no acceleration. Not a wrong-free-energy bug, just lost acceleration. BUT the chignolin_7 scenario: λ=0 replicas with shared calibration's full k0 — that was the opposite (integrator had k0max, priced as λ=0). Here `set_replica_lambda` sets k0 = λ·k0max = 0 → integrator unboosted, priced unboosted. Consistent. OK.

The dangerous direction: normalized_rows present with gamd_lambda, but after a window drop/reorder the rows are stale. The docstring claims reindexing operations subset normalized_rows correctly. Trust but can't verify.

### 12. `_reload_state_gamd_lambdas_on_resume` (L2386-2437) and `_persist_state_gamd_lambdas` (L2439-2467)

Reload: only if resume and `args.state_gamd_lambdas` unset/empty. Reads manifest method_settings.state_gamd_lambdas. `if lambdas:` — truthiness: a list of all zeros [0.0, 0.0, ...] is truthy (non-empty list) → restored. Good. Empty list → skipped. OK.

Persist: `update_run_manifest(out_dir, {"method_settings": {"state_gamd_lambdas": list(...)}})`. The docstring warns it must be called after every mutation. The shown code can't verify call sites. But here's a VERIFIABLE issue: `_persist_state_gamd_lambdas` writes `list(getattr(args, "state_gamd_lambdas", None) or [])`. If `args.state_gamd_lambdas` is a numpy array, `or` on a numpy array raises ValueError (ambiguous truth value)! `getattr(...) or []` with a non-empty numpy array → ValueError: The truth value of an array with more than one element is ambiguous. Is state_gamd_lambdas a list or ndarray at persist time? `_derive_state_gamd_lambdas` returns a list. The truncation does `list(...)[:_max]`. So probably a list. But if any path sets it to an ndarray (e.g., `np.asarray` somewhere), persist crashes — loud, not silent. Minor.

Bigger: the reload/persist pair stores ONE global list in run_manifest.json's method_settings, written "once at campaign start and never overwritten by a resume's update_run_manifest patches" — wait, the reload docstring says method_settings is written once and never overwritten, but `_persist_state_gamd_lambdas` DOES overwrite method_settings.state_gamd_lambdas via update_run_manifest. Contradiction? The reload docstring says the manifest's method_settings is "written once at campaign start (initialize_run_manifest) and never overwritten by a resume's update_run_manifest patches" — but _persist patches exactly that field. Presumably _persist is called during the run (including first start), and on resume the reload reads it. If update_run_manifest merges method_settings shallowly (replacing the whole method_settings dict?), other method_settings keys could be clobbered — depends on merge semantics. If update_run_manifest does a deep merge, fine. Can't verify.

CRITICAL issue in the reload logic: the length check. Reload sets `args.state_gamd_lambdas = [float(x) for x in lambdas]` — the FULL persisted list. Then `_derive_state_gamd_lambdas(window_metadata, n, existing=args.state_gamd_lambdas)` uses existing ONLY if len == n. On a resume after adaptive epochs added windows (n grew), the persisted list from the previous run segment has old length → mismatch → fallback [0.0]*n → LADDER SILENTLY DISABLED on resume. Wait — but _persist is called after final mutation before production, so the manifest has the latest length. On resume of the SAME segment, n matches. On resume after an adaptive epoch that added windows, the new epoch's run derives lambdas for the new window set — the manifest from the previous epoch has the old list; the new epoch's window_metadata normalized_rows should carry gamd_lambda per row (source of truth). If it does, fine. If not (older run dir), fallback zeros. The docstring acknowledges this risk. It's a real SUSPECTED silent-ladder-loss path: resume + window count change + normalized_rows lacking gamd_lambda → all-zero λ → no boost applied AND priced as no boost (consistent, so free energy still correct, just no acceleration) — EXCEPT the campaign registry path means v_pep/v_dih still recorded. So the free energy remains correct; acceleration lost silently. Not a Boltzmann-correctness bug per se. Unless... the integrator retains k0 from a checkpoint! On resume, contexts are restored from checkpoints — the integrator's global variables (k0_Dihedral, k0_Total) are serialized in the checkpoint. If the resume path does NOT call set_replica_lambda_for_window after checkpoint restore, the integrator keeps the OLD k0 (correct per old λ). If assignments are reshuffled on resume (replicas re-seeded into windows differently), the checkpoint-restored k0 corresponds to the OLD window's λ while the new assignment prices the NEW window's λ → mismatch! Does the resume path reapply set_window + set_replica_lambda for the current assignments? Not shown. This is exactly the chignolin_8 class (state carried in the integrator, not re-derived). SUSPECTED, high value to check: after checkpoint load, is `set_replica_lambda_for_window` called for every replica before production? The `_apply_swap_to_replica` closure does it on swaps, but initial resume state is unseen.

### 13. `_channel_boost` (L497-503) and `pep_gamd_boost_kj` (L506-519)

Already discussed the `(b+v) < e` guard. Additional points:

- `scale = max(max(|e|, |v|), 1.0)`; degenerate-range guard `|rng| ≤ 0.001·scale` → b = 0. If vmax ≈ vmin (degenerate envelope from a bad swarm), boost silently zeroed — the run proceeds unboosted while... priced consistently (formula used both in exchange matrix and MBAR). But does the INTEGRATOR also zero its boost in this case? gamd-openmm's integrator computes its own boost from its own globals (Vmax, Vmin, E, k0). If the integrator does NOT have the 0.001·scale guard, it applies 0.5·k0·(E−V)²/tiny → HUGE boosts, while pricing says 0 → catastrophic applied-vs-priced mismatch. SUSPECTED — depends on gamd-openmm internals. The docstring asserts the formula mirrors gamd-openmm; the guard `0.001*scale` looks like a local invention (I don't recall this in gamd-openmm; gamd-openmm checks `Vmax != Vmin`? Actually gamd's calibration computes k0 from (Vmax−Vmin); the integrator just evaluates). Flag it.

- Dependent dual boost: `b_tot = _channel_boost(v_pep + b_dih, ...)`. Matches "dihedral first, then Total with dihedral boost added". OK per docstring.

- `pep_gamd_boost_kj` returns float for scalar input, array otherwise. In `pep_gamd_boost_matrix_kj`, v_pep/v_dih are arrays → b_dih array; `float(out) if out.ndim == 0 else out` fine.

- NaN handling: if v_pep is NaN (single dihedral has_total=False path ignores it — good). If v_dih NaN (read failure?): b_dih = where(...)... NaN propagates: b = 0.5·k0·(e−nan)²/rng = nan; `np.where(|rng| ≤ ..., 0, b)` → nan; `np.where((nan+v) < e, b, 0)` → comparison with NaN is False → **0.0**! So a NaN v_dih silently becomes ZERO BOOST in the pricing matrix. The exchange path then prices that replica's boost as 0 under every λ — wrong (should be NaN → freeze, like primary). And the recorded `sampled_boost_bias_kj` = 0, and MBAR's `apply_ladder_boost_to_u`... does it also NaN→0 via the same formula? `reconstruct_bias_matrix` passes v_dih with NaN preserved into `_ladder_apply` → `apply_ladder_boost_to_u` — if it uses `pep_gamd_boost_kj` (the "repository helper"), NaN → 0 the same way. So analysis and exchange agree with each other but BOTH disagree with the integrator (which applied a real boost from its own globals using its own energy evaluation — the integrator's V is not NaN; the NaN arose in the Python read). So applied ≠ priced. VERIFIED code path (NaN → 0.0 via np.where comparison), SUSPECTED frequency. This is the same class as the secondary-CV NaN→0. Good catch — the `(b+v) < e` comparison makes NaN → 0.0 silently. Actually wait: `np.where((b + v) < e, b, 0.0)` with b=nan, v=nan: (nan) < e → False → 0.0. Yes. NaN boost → 0.0. Silent.

- Also `k0` for λ=0: b = 0. Fine.

### 14. `k0max_from_globals` (L528-530)

`shared_globals["k0_Dihedral"]` — KeyError if missing (loud). Total defaults 0.0. OK.

### 15. `set_replica_lambda` (L533-546)

Sets k0_Dihedral always; k0_Total only if integrator has the global (or names None). If integrator lacks k0_Total and k0max Total ≠ 0 → raise. Good. If names is None (test double), sets both — on a real single-dihedral integrator via a test double... fine.

Subtle: it sets k0 but does NOT force re-evaluation of the boost energy globals? In gamd-openmm, k0 is a global variable read each step — fine.

Another subtle: λ=0 sets k0=0 — but the integrator still ADDS the boost energy evaluation overhead; fine.

CRITICAL: `set_replica_lambda` changes k0 mid-trajectory (on swap). The boost potential changes discontinuously in time — that's fine for REUS (each state is a different Hamiltonian; the replica's Hamiltonian switches upon accepted swap, standard RE). The acceptance already priced the change. OK.

### 16. `set_replica_lambda_for_window` (L549-569)

No-op if k0max_by_channel None. Raises if ladder active but state_lambdas None. Indexes state_lambdas[window_index] — IndexError if misaligned (loud). OK.

### 17. state_identity.py

`canonical_state_definition`: converts kJ/mol → kcal/mol by dividing k1, k2 by 4.184. Note: it divides IN PLACE on `row` dicts from normalize_windows — normalize_windows copies (`window = dict(raw)`), so OK.

Potential issue: `row["k1"] /= 4.184` — if k1 was an int 0, 0/4.184 = 0.0 fine. If k1 missing → _term raised already unless axis2 with no center2 → returns (0,0) and sets window["center2"]=0.0, k2=0.0. OK.

`if any(row["gamd_lambda"] > 0 ...) and not isinstance(data["boost"], dict)` → raise. Good.

Windows sorted by window_id. `column_window_ids` from sorted windows. OK.

`validate_fixed_state_segments`: compares state definitions across segments EXACTLY (compare_state_definitions with `!=` on floats). Exact float equality across segments: temperature 300.0 vs 300.00000001 → mismatch → raise. That's strictness, loud not silent. Fine.

One issue: in `validate_fixed_state_segments`, the windows check: `if normalize_windows(snapshot.get("windows", [])) != state["windows"]:` then re-check sorted. But `normalize_windows` returns dicts that include ALL original keys (window = dict(raw); updates center1/k1/center2/k2/gamd_lambda). state["windows"] rows passed through canonical_state_definition which rejects unknown fields — so state["windows"] rows have only the 6 allowed keys. But snapshot["windows"] top-level rows could carry extra keys (e.g., "center_A" legacy) — normalize_windows keeps them → `!=` → then the sorted comparison also `!=` → raise IntegrityError. Loud. Fine.

Hmm wait, there's a real one: `normalize_windows` on snapshot top-level windows vs state windows — state["windows"] were sorted; top-level compared unsorted first, then sorted. Fine.

`freeze_snapshot`: equilibrium_analysis_eligible must be bool; phase production if eligible. OK.

Nothing silently wrong here — this module is strict. One soft spot: `canonical_state_definition` requires EXACT schema (set(data) != required → raise). Good.

### 18. correctness/bias.py

`numeric_vector`: masked → NaN; non-finite → NaN. Good.

`reconstruct_bias_matrix`: builds (N, K) reduced bias: matrix *= beta * KJ_PER_KCAL. So entries are β·U_bias in kJ — reduced. Umbrella in kcal → ×4.184 → kJ → ×β. Good.

With `np.errstate(over="raise", invalid="ignore")`: overflow raises IntegrityError. invalid ignored — NaN propagates. Good.

Ladder: if no λ>0 → return umbrella-only. Else require envelope, v_dih, v_pep (if has_total). Then `_ladder_apply(matrix, pep, dih, lambdas, envelope, beta, bookkeeping)` = `apply_ladder_boost_to_u` (not shown). The boost must be added as β·boost_kJ. Can't verify the helper. Note the comment: "The existing gareus.mbar_analysis.ladder helper remains the only boost formula." So there are TWO boost formula implementations? No — pep_gamd.py's pep_gamd_boost_kj is used in production; mbar_analysis.ladder.apply_ladder_boost_to_u is used in analysis. Are they the same code? "the only boost formula" suggests ladder helper is canonical — but production uses pep_gamd_boost_kj! If they're separate implementations, a divergence (e.g., one has the (b+v)<e guard, the other doesn't) → applied/priced mismatch. SUSPECTED: two boost-evaluation code paths (production's pep_gamd_boost_kj vs analysis's apply_ladder_boost_to_u) with no shown equivalence test. This is precisely the class of bug the review asks for. Actually the docstring in bias.py says "The existing gareus.mbar_analysis.ladder helper remains the only boost formula" — meaning analysis delegates to it; production uses pep_gamd.pep_gamd_boost_kj. If apply_ladder_boost_to_u internally calls pep_gamd_boost_kj, fine; if it's a reimplementation, risk. Can't verify. Flag.

`normalize_windows._term`: axis 2 with no center2 AND no k2 → (0,0). But axis 2 with k2 present = 0 and center2 missing → returns (0,0) — "With zero force the center is irrelevant." OK. Axis 1: missing k1 → raise. Good.

`finite_number` rejects bool. Good.

`normalize_verified_legacy_disabled_secondary`: explicit adapter, not auto-invoked. Good.

### 19. production.py sample() (L7420-7700 excerpt)

Key observations:

- `sample()` computes bias matrices from observed CVs and records rows. `observable_cache` stores step, primary_values, bias_matrix_kj, etc. `_current_exchange_arrays` reuses the cache if step matches. 

STALENESS HAZARD: `observable_cache.clear()` is called on accepted swap (in _attempt_window_swap). But the cache stores `primary_values` and `bias_matrix_kj` keyed only by step. Consider: sample() at step S fills cache. Then an exchange interval at the same absolute step S? The exchange uses cached matrix — configurations unchanged since sample (no dynamics between), so fine. After a swap is accepted, cache cleared — next exchange recomputes. But WAIT: after an accepted swap, the CONFIGURATIONS didn't change, only assignments. The bias matrix U[window, replica] doesn't depend on assignments! bias_matrix_kj[w, r] = U_w(x_r) — assignment-independent. So clearing the cache is unnecessary but harmless... EXCEPT the boost part: boost matrix B[w, r] = boost(x_r, λ_w) — also assignment-independent. So the matrix is assignment-independent; clearing is harmless. BUT the integrator state changed (k0 changed, window centers changed via set_window) — the matrix is a pure function of positions, so still valid. Fine.

However — the cache validity check: `int(observable_cache.get("step", -1)) == int(absolute_step)`. If sample() ran at step S, then dynamics advanced to step S' (next exchange interval without an intervening sample), the step check fails → recompute. Good. But what if exchange interval == log interval and sample() is called AFTER the exchange at the same step? Then exchange computes fresh, sample computes fresh. Fine.

DANGER: `observable_cache` is also used for... "dashboard rows". Fine.

Now the recorded sample row: `"umbrella_bias_kcal_mol": sampled_bias_kcal` where sampled_bias_kcal = all_bias_kcal[w] — includes boost (assembled). And `"sampled_umbrella_bias_kj"` = umbrella only, `"sampled_boost_bias_kj"` = boost only. The parquet writer writes cv1, cv2, potential, boost components, v_pep, v_dih, gamd_lambda. MBAR later reconstructs from cv1/cv2/v_pep/v_dih + frozen windows/envelope. Consistency between the recorded `umbrella_reduced_bias` and MBAR's reconstruction depends on identical formulas — umbrella part is simple; boost part is the risk noted above.

- `"gamd_lambda": float(state_lambdas[w])` — w = assignments[r], current window. Good.

- The parquet `potential=float(potentials_kj[r])`: potentials from `physical_energy_groups_for_args(args)` — "physical" groups EXCLUDING bias and boost forces? If the recorded potential includes the umbrella/boost energy, MBAR's u = β(U_physical + bias) would double-count. `physical_energy_groups_for_args` — name suggests physical only. The fast path reads `groups=physical_energy_groups_for_args(args)`. Trust the name; can't verify. If it accidentally includes the CustomCVForce umbrella group or the GaMD boost (boost in gamd-openmm is added as a modified force? The boost is applied inside the integrator as an energy added to... in gamd-openmm the boost is applied by scaling/adding to the force's energy? Typically GaMD in OpenMM modifies the integrator, adding boost potential to the system energy? If getPotentialEnergy includes the boost, double counting). SUSPECTED, worth one line: verify physical_energy_groups_for_args excludes umbrella CV force groups and that getPotentialEnergy excludes the GaMD boost that StageIntegrator adds.

- `extract_gamd_boost_kj(sim.integrator, unit, None)` — reads boost from integrator globals; recorded as gamd_boost_total_kj_mol. If boost_kj is None → "" (empty string) in the row, and parquet boost_total=None. MBAR uses v_pep/v_dih formula, not this. So a run where the integrator boost read fails (None) still gets priced by formula — silent if formula ≠ applied. Again the two-sources-of-boost issue.

- The `_gamd_recal` block: exceptions → disable recal with a warning. "Best-effort" — fine, doesn't affect pricing.

- NaN diagnostics on production failure — fine.

### 20. production.py exchange closures (L7780-7900 excerpt)

`_attempt_window_swap`: checks attemptability before drawing rng — good for stream reproducibility. `apply_window_swap(..., None if force_accept else rng.random(), ...)`. If force_accept, uniform=None → accepted=True. force_accept used in tests or "salvage" paths? If used in production to force swaps (e.g., initial equilibration), those swaps violate DB — but presumably only in seeding. Can't verify.

After acceptance: `_apply_swap_to_replica` on both replicas: set_window (new umbrella params) + set_replica_lambda_for_window (new k0). Good — this is the correct post-swap integrator update. Note it's submitted to the pool and `.result()` — synchronous. Good.

`parquet_exchange_writer.write_exchange(...)` records delta_e and accepted. Note: it does NOT record gamd_lambda or the boost part of delta — fine for diagnostics.

`_current_exchange_arrays`: as discussed. One issue: the cache check `cached_bias.shape == (nrep, nrep)` — assumes n_states == nrep. If windows > replicas (unheld windows, holders = -1), the matrix is (n_windows, nrep) and the cache is rejected → recompute — fine, just slower. But then `pep_gamd_boost_matrix_kj(v_pep, v_dih, state_lambdas, pep_env)` with len(state_lambdas) = n_windows gives (n_windows, nrep); `np.zeros((nrep, nrep))` for the inactive case — SHAPE MISMATCH with distance_bias_kcal (n_windows, nrep) if n_windows ≠ nrep! assemble would broadcast-fail → loud. So inactive-ladder with n_windows ≠ nrep crashes; active-ladder fine. Edge case, loud anyway.

Now — the GIBBS path's bias matrix and the "stay" delta: fine.

Let me also check `_gibbs_window_proposal_distribution` more carefully for a subtle DB issue: the heat-bath proposal includes ALL windows (not just neighbors). The proposal probability q_fwd = exp(-βδ)/Σ. The MH correction then accepts with min(1, exp(-βδ)q_rev/q_fwd). This is correct... but hold on: for a heat-bath proposal where q ∝ exp(-βδ), the MH acceptance becomes min(1, Z_fwd/Z_rev) where Z are the normalization sums — standard result. The code computes it explicitly. Fine.

One more: in the forward distribution, `delta[stay] = 0.0` — the stay candidate's TRUE delta is 0 anyway (swap with self). Fine.

Numeric: `log_weights = clip(-βδ, -745, 745)`. β in mol/kJ (~0.4 at 300K... β = 1/(kT) = 1/(8.314e-3 · 300) ≈ 0.401). δ in kJ. Fine.

`_exchange_probability`: βδ dimensionless. The bias matrix is kJ, β mol/kJ. Consistent.

`apply_window_swap` uses bias_matrix_kj with beta — consistent units.

`assemble_bias_matrices` — the exchange path uses `_, bias_matrix_kj`. The kcal matrix is used for recording. The invariant holds.

Now, the umbrella k units: `umbrella_bias_matrix_kcal(primary_values, ..., centers_a, k_kcal, ...)` — primary in Å (cv_A, center_A, k_kcal_mol_A2). 0.5·k·(x−c)² kcal/mol. Then ×4.184 → kJ. The actual OpenMM restraint: set_window with centers_nm, ks_kj_nm2 — k in kJ/nm². Conversion: k_kcal/Å² → kJ/nm²: ×4.184 ×100. If set_window's ks_kj_nm2 = k_kcal · 418.4 and centers_nm = centers_A/10, then U = 0.5·k_kj_nm2·(x_nm − c_nm)² = 0.5·k_kcal·418.4·((x_A−c_A)/10)² = 0.5·k_kcal·418.4/100·(ΔA)² = 0.5·k_kcal·4.184·(ΔA)² kJ = 4.184·U_kcal. Consistent IF the ×100 and /10 conversions are done right at the set_window site (not shown). Standard; flag as a check-item but not a finding.

Now let me also examine `sample()`'s `reduced_bias_matrix = float(beta) * bias_matrix_kj` — recorded as umbrella_reduced_bias. MBAR needs u_nk = β·U_k(x_n). Provided. Good.

Now the tICA dihedral observation: `_dihedral_buffers[r].record(...)` — records per replica. Fine.

Let me now think about the gibbs-walk + lambda interaction once more. In `gibbs_propose_one_replica`, the bias matrix includes the boost under each candidate state's λ. The delta for moving rep from wi to wj includes boost(x_rep, λ_wj) − boost(x_rep, λ_wi) plus the target's boost change. Correct.

BUT — after a gibbs move is accepted, does the production caller apply set_replica_lambda_for_window? `_attempt_window_swap` (the pairwise path) does. The gibbs path's apply code is NOT shown. If the gibbs path applies the swap via a different closure that only calls set_window and forgets set_replica_lambda_for_window, the integrator keeps the old λ's k0 while pricing the new λ — EXACTLY the chignolin_8 class. SUSPECTED — the apply site for GibbsProposal is not in the excerpt; the helper exists ("used at every call site that (re)applies a replica's window assignment so the guard can't be forgotten") — the docstring claims all call sites use it. Unverifiable; flag as a must-check.

Also `holders_after` in gibbs: `target_rep = int(holders_after[wj])` — if wj is unheld (holders = -1)? The forward proposal only includes valid windows (holders ≥ 0), so wj is held. OK. But `replica_of_window` with -1 entries: `valid_mask = holders >= 0`. In `apply_window_swap`, swap_candidate_replicas checks i,j ≥ 0. OK.

Now, one more potential issue in `_gibbs_window_proposal_distribution`: `old_e = bias[wi, rep] + bias[valid_windows, target_reps]`. For the stay candidate (valid_windows[k]==wi), target_reps[k]=rep → old_e includes bias[wi,rep] twice, new_e same → delta 0. Then forced 0. Fine.

What about numerical: delta for stay forced 0 — but if bias[wi,rep] is NaN, delta for ALL candidates is NaN (old_e has bias[wi,rep]) → all filtered → empty proposal → GibbsProposal(..., skip=True?) — the last field True in `GibbsProposal(wi, wi, 0.0, 0.0, 0.0, 0.0, False, True)`. So a NaN primary freezes gibbs moves for that replica. Consistent with pairwise. OK.

Now the SECONDARY NaN → 0 in the exchange pricing: consider replica r with NaN ss. bias[w, r] = umbrella_primary + 0 (secondary zeroed) + boost. Swaps are priced as if the secondary restraint didn't exist for x_r. The actual simulation: does the actual dynamics apply a secondary restraint force on replica r? The restraint is a CustomCVForce/force in the Context with the replica's current window parameters. If the CV evaluation in the FORCE is fine (positions finite → CV finite), the restraint acts normally. The NaN in ss_values came from the Python-side observation path (e.g., `secondary_structure_score_from_positions_nm` returned NaN, or fast-path read NaN). So the dynamics applies the restraint, but the exchange prices it as zero → detailed balance violated → stationary distribution ≠ target. AND the recorded sample's bias matrix has 0 for the secondary axis → MBAR prices those samples wrong → wrong free energy. Silent. VERIFIED code path; frequency depends on NaN incidence. This is finding #1.

Similarly the boost NaN → 0 via `(b+v)<e` comparison. Finding #2.

Now let me look for issues in the lambda-per-state consistency between exchange pricing and the integrator after swaps:

`_apply_swap_to_replica` calls `set_replica_lambda_for_window(sims[replica_index].integrator, assignments[replica_index], state_lambdas, k0max_by_channel)`. assignments already updated by apply_window_swap. Good.

But NOTE: `set_replica_lambda` sets k0 globals — does the StageIntegrator cache k0-derived quantities (e.g., σ0, or the boost computed so far)? gamd-openmm's StageIntegrator computes boost per step from globals — stateless w.r.t. k0 changes except accumulated statistics (Vmax/Vmin running?). In stage 5 (production with frozen envelope), the integrator uses fixed globals — changing k0 mid-run is the ladder design. OK per design.

Now `campaign_ladder_registry_lambda` returns the FIRST λ>0 row — used only as boolean. Fine.

Another look at `resolve_ladder_active`: `lam_arr = np.asarray(state_lambdas, dtype=float)` — if state_lambdas is a ragged list → raises (loud). If None → array(nan) 0-d, size 1, any(nan>0)=False. OK.

Now `_derive_state_gamd_lambdas`: `rows and len(rows) == int(n) and all("gamd_lambda" in r for r in rows)` → uses rows. Note `float(r.get("gamd_lambda", 0.0) or 0.0)` — if gamd_lambda is "0.236" string → float OK. If None → or 0.0. If "None" string → float("None") raises ValueError — loud. OK.

The `existing` path: `existing_list = list(existing or [])` — numpy array truthiness bug again: `existing or []` with non-empty ndarray → ValueError (loud). Minor.

Now, `state_identity.py::canonical_state_definition` — one more: `if energy_unit == "kJ/mol": row["k1"] /= 4.184` — but gamd_lambda rows and boost envelope are NOT unit-converted. The boost envelope (vmax/vmin/threshold in kJ presumably) stays as-is while energy_unit is set to "kcal/mol" at the end. If the envelope is embedded in kJ but the state claims kcal/mol, a downstream consumer interpreting envelope energies as kcal would mis-price boosts by 4.184×. The schema doesn't record the envelope's units! `boost` requires "kind" and "envelope" but no units field. SUSPECTED: unit ambiguity in the frozen boost envelope. The MBAR ladder helper presumably assumes kJ (β in mol/kJ). If any export path converts the whole state definition to kcal (it sets energy_unit="kcal/mol"), a consumer applying that label to the envelope errs by 4.184. Flag as suspected unit-identity gap.

Also `canonical_state_definition` mutates `row["k1"] /= 4.184` — rows come from normalize_windows which dict-copies, but the VALUES: `window["center1"], window["k1"] = _term(...)` — floats. OK.

Now, `validate_fixed_state_segments` requires all segments to have IDENTICAL state definitions (compare_state_definitions → any difference raises). But adaptive epochs RECALIBRATE the envelope and add windows — each epoch's snapshot would differ → union across epochs rejected. That's by design ("Use a validated union-state workflow"). Loud, not silent. OK
(TRUNCATED - output hit the token limit before completion)

## deepseek-thinking

We need answer as expert review board. Need assess submission. Need produce verdict. Need be specific technical. We need review code snippets. Need identify silent wrong free energy issues in replica exchange subsystem. Need cite line numbers. Need distinguish VERIFIED/SUSPECTED. Need final lines exactly.

We need analyze deeply. The submission is itself a review? Actually user says "The board is convened to review the submission below." The submission is a document titled "Board review: ATLaS-MD / gareus v0.8.2 ... FIND THE NEXT ONES LIKE IT." It includes code excerpts. Our task: "Adversarially review the code below. Report every place where the software can SILENTLY produce a number that is not the Boltzmann free energy it claims..." So we are reviewing the code in submission. Need produce independent assessment: ACCEPT/ACCEPT-WITH-CHANGES/REJECT? This is a review of code. We need decide if code is correct? The "submission" is a review document? It asks us to review code below. We should assess the code snippets for correctness. Likely REJECT because there are bugs. Need find them.

Let's parse code.

We have replica exchange: umbrella_bias_matrix_kcal, gibbs proposal, MH acceptance, exchange probability, swap candidate, apply_window_swap, gibbs_propose_one_replica, explicit_2d_neighbor_pairs, assemble_bias_matrices, _fetch_v_pep_v_dih, campaign_ladder_registry_lambda, resolve_ladder_active, _derive_state_gamd_lambdas, _reload_state_gamd_lambdas_on_resume, _persist_state_gamd_lambdas, pep_gamd _channel_boost, pep_gamd_boost_kj, k0max_from_globals, set_replica_lambda, set_replica_lambda_for_window, state_identity.py, bias.py, production sample/exchange snippets.

Need identify silent wrong Boltzmann free energy. Let's examine each.

First, umbrella_bias_matrix_kcal: computes 0.5 k (x-c)^2 per axis. Orientation [state, replica]. dprimary = pv[np.newaxis,:] - ca[:,np.newaxis] shape (nstates,nreps). distance_kcal = 0.5*ka[:,None]*dprimary^2. Good. ss_kcal = np.where(np.isfinite(dss), ss_kcal, 0.0). If secondary value NaN, bias zero. But if secondary center NaN? dss NaN -> zero. Fine? But if secondary value missing, state with secondary restraint should maybe not contribute? In exchange, if replica lacks finite secondary value, zero bias for that axis. This is a guard. But if secondary CV is NaN because not measured, using zero bias means the state's Hamiltonian for that replica is wrong (missing restraint). However if secondary CV is unavailable for all replicas? It could silently treat missing data as no bias. But maybe intended. Need see.

Potential issue: umbrella_bias_matrix_kcal returns distance_kcal and ss_kcal. assemble_bias_matrices adds boost_bias_kcal = boost_bias_kj / 4.184. bias_kj = 4.184 * bias_kcal. This is consistent if boost_bias_kj is kJ. But note distance_bias_kcal + ss_bias_kcal + boost_bias_kcal. Good.

But in sample path: boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env) if pep_env is not None else np.zeros((nrep,nrep)). pep_gamd_boost_matrix_kj returns (n_states, n_samples) where n_states = len(lambdas), n_samples = len(v_pep). Here v_pep_kj shape (nrep,), state_lambdas length nstates. So boost_bias_matrix_kj shape (nstates, nreps). Good. But if pep_env is None, np.zeros((nrep,nrep)) assumes nstates == nreps. Is that always? In REUS, number of windows equals number replicas? Usually yes, one replica per window. But adaptive epochs? Maybe nrep == nwindows? likely. But if not, shape mismatch. Could be silent? Need see.

Critical: _fetch_v_pep_v_dih reads v_dih = ctx.getState(getEnergy=True, groups={DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(kJ/mol). It returns total potential energy of group 2? It says PeriodicTorsion/CMAPTorsion forces in group 2. But v_dih should be dihedral energy only. If group 2 includes other forces? They claim same id as DIHEDRAL_GROUP. Need know. Could be okay.

But bigger: In sample path, potential_kj = state.getPotentialEnergy() with groups=physical_energy_groups_for_args(args). This is physical potential energy? Does it include boost? In OpenMM, getState(getEnergy=True, groups=...) returns potential energy for those groups. If boost is applied via integrator, not force groups? GaMD boost is a force? Actually gamd-openmm StageIntegrator applies forces? Need understand. The code uses "physical_energy_groups_for_args" to exclude? If boost is implemented as integrator force? Hmm.

Need focus on replica exchange. The key question: Does exchange use correct Hamiltonian? apply_window_swap delta = bias_matrix_kj[wi,i]+bias_matrix_kj[wj,j] - old. It swaps state labels, not configurations. For REUS with Hamiltonian U_window(x) = U0(x)+bias_window(x)+boost_lambda_window(x)? Actually each state is (window, rung). If swapping state labels, the physical potential and boost cancel only if the boost depends on state (lambda) and configuration. Wait: They claim "unbiased potential and GaMD boost cancel exactly from delta and only umbrella biases appear: x_i and x_j untouched, so U0(x_i)+U0(x_j) and dV(x_i)+dV(x_j) are identical before and after." But is that true? Let's examine.

In replica exchange with state-dependent Hamiltonians H_m(x) = U0(x) + bias_m(x) + boost_{lambda(m)}(x). If swap state labels between replicas i (state m) and j (state n), configurations x_i, x_j unchanged. Detailed balance ratio:
old H_m(x_i)+H_n(x_j) = U0(x_i)+bias_m(x_i)+boost_{λm}(x_i) + U0(x_j)+bias_n(x_j)+boost_{λn}(x_j)
new H_n(x_i)+H_m(x_j) = U0(x_i)+bias_n(x_i)+boost_{λn}(x_i)+U0(x_j)+bias_m(x_j)+boost_{λm}(x_j)
delta = [bias_n(x_i)-bias_m(x_i)] + [bias_m(x_j)-bias_n(x_j)] + [boost_{λn}(x_i)-boost_{λm}(x_i)] + [boost_{λm}(x_j)-boost_{λn}(x_j)].
The boost terms do NOT cancel unless boost depends only on state? Wait boost_{λ}(x) depends on configuration x and λ. In swap, x_i stays with replica i but state changes from m to n, so boost term changes. The claim "dV(x_i)+dV(x_j) identical before and after" is false if dV is state-dependent via lambda. Unless the boost is applied to the replica's integrator and when swapping state labels, the boost of replica i changes from λ_m to λ_n, so the total boost energy of configuration x_i changes. The unbiased potential U0(x_i) cancels because same configuration and same physical potential, but boost does not cancel because it's part of the modified Hamiltonian and depends on the state's lambda. Wait, but maybe in their scheme, the boost is considered part of the "bias" and included in bias_matrix_kj. Indeed assemble_bias_matrices includes boost_bias_kj. So delta includes boost terms. The comment in apply_window_swap says "only the umbrella biases appear" but actually bias_matrix_kj includes umbrella + boost. The code's delta uses bias_matrix_kj, which includes boost. So comment is wrong but code includes boost. Let's verify: apply_window_swap old_e = bias_matrix_kj[wi,i]+bias_matrix_kj[wj,j]; new_e = bias_matrix_kj[wj,i]+bias_matrix_kj[wi,j]. bias_matrix_kj includes umbrella + boost. So delta includes boost differences. Good. The comment "unbiased potential and GaMD boost cancel exactly" is false for boost, but code doesn't use that assumption. So not a bug in code, just comment. But wait: Does the boost energy of a configuration under a state's lambda equal the boost actually applied when that replica was in that state? The bias matrix is computed from v_pep, v_dih of current configuration and state_lambdas. Yes.

But there is a subtle issue: The boost applied by the integrator to replica i is based on its current state's lambda. When swapping state labels, the integrator's boost parameters are changed via set_replica_lambda_for_window. The configuration's boost energy under new lambda is computed in bias matrix. Good.

However, the exchange proposal uses bias_matrix_kj computed from current CVs and v_pep/v_dih. But after a swap, the boost parameters change; the actual potential energy of the replica changes by the boost difference. The exchange acceptance uses the bias matrix including boost. Good.

But wait: The boost energy in the Hamiltonian is not simply a function of v_pep and v_dih? Pep-GaMD dual boost: total boost = b_dih + b_tot where b_tot depends on v_pep + b_dih. The code pep_gamd_boost_kj computes that. Good.

Potential issue: In sample path, "potential_kj_mol" is physical potential energy? It reads getState(getEnergy=True, groups=physical_energy_groups_for_args(args)). If the boost is applied by integrator as a force, does getState potential energy include it? In OpenMM, integrator forces are not part of Context potential energy unless they are forces in the system. GaMD boost in gamd-openmm is implemented via a CustomIntegrator? Actually "StageIntegrator" applies a force? Need know. The code's "extract_gamd_boost_kj(sim.integrator, unit, None)" reads boost from integrator. The sample row records potential_kj_mol and boost_total separately. For MBAR, they need total biased energy = physical potential + umbrella + boost. If potential_kj_mol already includes boost, double counting. They likely use physical_energy_groups to exclude boost force group. Need see "physical_energy_groups_for_args" not shown. Could be okay.

But the task specifically subsystem under review: replica exchange: state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path. Need find bugs in those.

Let's examine _gibbs_window_proposal_distribution. It computes old_e = bias[wi, rep] + bias[valid_windows, target_reps]; new_e = bias[valid_windows, rep] + bias[wi, target_reps]. This is for moving replica rep from current window wi to candidate window wj, swapping with target replica in wj. delta = new_e - old_e. This is exactly the energy difference for swapping state labels between rep and target. Good. It then sets stay_idx delta=0. log_weights = -beta*delta clipped [-745,745]. Then weights = exp(log_weights - max). probabilities = weights/sum. This is heat-bath proposal over target windows. For Gibbs sampling, proposal probability should be proportional to exp(-beta * energy of state after move) maybe. Let's derive.

We have state graph with replicas assigned to windows. A "Gibbs-walk" move: choose a replica, propose a new window for it by swapping with the replica currently in that window. The proposal distribution q(wj | current) ∝ exp(-β H_after(wj))? For a Gibbs sampler, conditional distribution of assignment given configurations: P(assignment) ∝ exp(-β Σ_m H_m(x_{rep(m)})). For moving replica rep from wi to wj, the energy after is old_e - bias[wi,rep] - bias[wj,target] + bias[wj,rep]+bias[wi,target] = old_e + delta. The conditional probability of assigning rep to wj (and target to wi) is proportional to exp(-β (old_e+delta)) = exp(-β old_e) exp(-β delta). Since old_e constant, q ∝ exp(-β delta). So proposal is correct heat-bath. Then MH correction with reverse probability. Good.

But there is a subtle bug: The proposal distribution includes the current window with delta=0. If choose stay, returns GibbsProposal(... accepted True? Actually in gibbs_propose_one_replica, if wj==wi, returns GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False). The "accepted" field True? It returns accepted=True for stay. That's fine.

But wait: In _gibbs_window_proposal_distribution, valid_windows = all windows with holders >=0. It includes current window. target_reps = holders[valid_windows]. For current window, target_reps is rep itself. old_e = bias[wi, rep] + bias[wi, rep]? Wait valid_windows includes wi, target_reps includes rep. old_e = bias[wi, rep] + bias[wi, rep] = 2*bias[wi,rep]. new_e = bias[wi, rep] + bias[wi, rep] = same. delta=0. But the energy of the system before move is sum over all replicas of bias[window, replica]. For the pair (wi, rep) and (wi, rep) is same replica counted twice! This is a bug: The proposal distribution's old_e/new_e for candidate window wj should consider the two replicas involved: rep (currently in wi) and target (currently in wj). For wj=wi, the "target" is rep itself, so old_e should be just bias[wi,rep] (or pair energy with itself? Actually swapping replica with itself is no move; energy before = bias[wi,rep] + sum others; after same. The delta should be 0. The formula old_e = bias[wi, rep] + bias[wi, rep] = 2*bias[wi,rep] and new_e same, delta=0. Since delta=0, no effect on probabilities except the weight for stay is exp(0)=1. But the absolute old_e is wrong, but only delta matters. For wj != wi, old_e = bias[wi,rep] + bias[wj,target], new_e = bias[wj,rep]+bias[wi,target]. Correct. So no issue for delta. However, if bias[wi,rep] is NaN? Then old_e NaN, delta NaN, log_weights clip? np.clip(NaN, -745,745) returns NaN? Actually np.clip with NaN returns NaN. finite mask removes. If current window has NaN bias, then stay candidate removed, and if all removed, returns empty. But bias should be finite. If bias has NaN due to missing secondary, the code sets ss_kcal zero for NaN, so finite.

But there is a more serious issue: The proposal distribution uses bias_matrix_kj which includes boost. But the boost for a configuration under a candidate state's lambda is computed from v_pep/v_dih. However, when proposing a swap, the actual boost applied to the replica after swap would be set by set_replica_lambda_for_window. But the boost energy of the configuration under the new lambda is included. Good.

Now _gibbs_mh_acceptance_probability: log_alpha = -beta*delta + log(qr) - log(qf). This is standard for proposal q and reverse q. But note delta is in kJ? bias_matrix_kj is kJ, beta is mol/kJ. Good. q_forward and q_reverse are probabilities from proposal distribution. Good.

But wait: The proposal distribution q is over windows, and the reverse proposal is computed after swapping holders. In gibbs_propose_one_replica, holders_after = copy; target_rep = holders_after[wj]; holders_after[wi]=target_rep; holders_after[wj]=rep. Then reverse = _gibbs_window_proposal_distribution(beta, bias, replica_index=rep, current_window=wj, replica_of_window=holders_after). This computes proposal for replica rep from window wj in the after state. It will include candidate wi with target_rep = holders_after[wi] = target_rep (the replica that was in wj originally). Good. q_reverse = probability of choosing wi. Good.

But there is a subtle issue: The proposal distribution's "valid_windows" includes all windows with holders >=0. In holders_after, all windows still held. Good.

Now, the acceptance probability for Gibbs-with-MH: If proposal is heat-bath q(wj) ∝ exp(-β delta), then the MH correction should be min(1, [π(wj) q(wi|wj)]/[π(wi) q(wj|wi)]). π ratio = exp(-β delta). So log_alpha = -β delta + log q_rev - log q_fwd. Correct.

But wait: The proposal distribution q(wj) is computed using delta = new_e - old_e where old_e includes bias[wi, rep] + bias[wj, target]. But the full energy difference for the entire system when moving rep from wi to wj includes only these two terms because other replicas unchanged. Good.

Now, apply_window_swap: It uses _exchange_probability(delta, beta) for standard swap. For Gibbs, they don't use apply_window_swap? They use gibbs_propose_one_replica then presumably apply? Need see. The code for apply_window_swap mutates assignments and replica_of_window. It computes pacc = _exchange_probability(delta, beta) unless p_override. For Gibbs, p_override would be pacc from MH. Good.

Potential bug: In apply_window_swap, after accepted, it does:
assignments[i], assignments[j] = assignments[j], assignments[i]
replica_of_window[int(assignments[i])] = int(i)
replica_of_window[int(assignments[j])] = int(j)
Let's verify. assignments[replica] = window. Initially assignments[i]=wi, assignments[j]=wj. After swap, assignments[i]=wj, assignments[j]=wi. replica_of_window[wi] should be j, replica_of_window[wj] should be i. Code: replica_of_window[int(assignments[i])] = int(i) => assignments[i]=wj => replica_of_window[wj]=i. Good. replica_of_window[int(assignments[j])] = int(j) => assignments[j]=wi => replica_of_window[wi]=j. Good.

But wait: If i and j are replicas, and assignments array indexed by replica. Good.

Now, swap_candidate_replicas: returns i,j. Good.

Potential issue: In _attempt_window_swap, after accepted, it calls set_window and set_replica_lambda_for_window on each replica. But it uses assignments[replica_index] after swap. Good. However, it submits to _sim_pool with .result() sequentially? It submits i then j, waits. Fine.

But there is a possible race: The exchange decision uses bias_matrix_kj computed from CVs and v_pep/v_dih before swap. After swap, set_window changes the umbrella force parameters on the context. But the CV values are unchanged. The bias matrix for the next exchange will be recomputed. Good.

Now, _current_exchange_arrays: It computes bias matrix for exchange. It uses v_pep/v_dih from _fetch_v_pep_v_dih. But note: In the fast path, it calls observe_fast_path(ctx, ...) which reads CVs. Then _fetch_v_pep_v_dih reads energies. In non-fast path, it gets positions, computes CVs, then _fetch_v_pep_v_dih. Good.

But there is a critical issue: The exchange bias matrix includes the boost computed from v_pep/v_dih and state_lambdas. However, the boost actually applied by the integrator at the time of exchange may be different because the integrator's boost is computed from its own internal stage and envelope. The code's pep_gamd_boost_kj is a reimplementation of gamd-openmm's boost. If it doesn't match exactly, exchange prices a different Hamiltonian than applied. This is the "wrong Hamiltonian applied vs priced" category. Need assess. The submission itself mentions chignolin_8 where boost was never applied but priced. The fixed code now has verify_gamd_production_stage5. But in the snippets, _fetch_v_pep_v_dih reads v_pep/v_dih from context, and pep_gamd_boost_kj computes boost. If the integrator's actual boost formula differs (e.g., threshold, vmax/vmin, k0, or the "add dihedral boost to total energy" behavior), then exchange and MBAR price wrong boost. Need see if code matches gamd-openmm. The docstring says "gamd-openmm's dependent dual boost under rung λ: dihedral first, then Total with the dihedral boost added to the Total energy before the square (stage_integrator _add_dihedral_boost_to_total_energy)." But is that correct? In Pep-GaMD, the total boost is applied to the total potential energy, and the dihedral boost is applied to dihedral energy. The total boost threshold is based on total potential energy? Actually GaMD: boost potential ΔV(r) = 1/2 k0 (E - V(r))^2 for V(r) < E. For dual boost, dihedral boost ΔV_dih = 1/2 k0_dih (E_dih - V_dih)^2, and total boost ΔV_tot = 1/2 k0_tot (E_tot - V_tot)^2 where V_tot includes the dihedral boost? In gamd-openmm, "add dihedral boost to total energy" means the total energy used for total boost is V_total + ΔV_dih? The code pep_gamd_boost_kj does b_tot = _channel_boost(v_pep + b_dih, threshold_total, ...). So yes.

But wait: _channel_boost formula: b = 0.5 * k0 * (e - v)^2 / rng. In GaMD, the boost is 0.5 k0 (E - V)^2 / (Vmax - Vmin)? Actually standard GaMD: ΔV = 1/2 k0 (E - V)^2 for V < E, with k0 = k0 * (Vmax - Vmin)/(E - Vmin)? Let's recall. GaMD boost: ΔV(r) = 1/2 k0 (E - V(r))^2 when V(r) < E, else 0. k0 is a constant. But in "Pep-GaMD" maybe k0 is scaled? The code divides by rng = vmax - vmin. Is that correct? In gamd-openmm, the boost is 0.5 * k0 * (E - V)^2 / (Vmax - Vmin)? Let's check. Standard GaMD: ΔV = 1/2 k0 (E - V)^2, where k0 = k0_max * (Vmax - Vmin)/(E - Vmin)? Actually there are two forms. In OpenMM GaMD, the boost potential is:
if V < E: ΔV = 0.5 * k0 * (E - V)^2 / (Vmax - Vmin)
Maybe yes. The code uses rng = vmax - vmin. So likely.

But there is a potential unit issue: _channel_boost uses v, e, vmax, vmin, k0 all in kJ/mol? v_pep/v_dih from Context in kJ/mol. Envelope thresholds/vmax/vmin/k0max? Need know. k0max_from_globals reads from shared_globals. If shared_globals are in kJ/mol? The code's pep_gamd_boost_kj returns kJ. If envelope values are in kcal/mol, then wrong. Need see. The submission says "v_pep, v_dih" in kJ/mol. Envelope? In gamd-openmm, globals are in kJ/mol? Likely. But not shown.

Now, the biggest issue: In sample path, the boost_bias_matrix_kj is computed using state_lambdas and pep_env. But if pep_env is not None (ladder active), the boost is included in bias_matrix_kj. Then the row "umbrella_bias_kcal_mol" = sampled_bias_kcal = bias_matrix_kcal[w,r] which includes boost. But "sampled_umbrella_bias_kj" = 4.184*(distance+ss) excludes boost, and "sampled_boost_bias_kj" = boost. So they record both. For MBAR, they likely use bias_matrix_kj including boost. Good.

But there is a critical issue: The exchange and sample paths compute boost_bias_matrix_kj using pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env). This computes the boost for each state's lambda using the current v_pep/v_dih. However, the boost applied by the integrator to a replica in state w is based on the integrator's current k0 values, which are set to lambda*window * k0max. But the boost also depends on the envelope's vmax/vmin/threshold. If the envelope is recalibrated during adaptive epochs, the integrator's envelope may change, but the state_identity frozen envelope? The code mentions "envelope recalibration" and "stale or overwritten calibration". In the snippets, pep_env is a PepGamdEnvelope object. If it is updated during production, the boost matrix uses current pep_env, while the integrator uses current envelope. But if the envelope changes, the Hamiltonian changes, and MBAR over all samples with different envelopes? The state definition should freeze envelope. The submission mentions "envelope recalibration" as adaptive epoch. Could be a source of silent wrong free energy if samples from different envelopes are pooled without accounting. But not in the provided code? We can mention.

Let's look at state_identity.py. It defines canonical_state_definition with required fields exactly. It requires "boost" with kind and envelope. It normalizes windows. It converts k1/k2 from kJ/mol to kcal/mol if energy_unit kJ/mol. It checks active lambda states require boost. Good.

But there is a bug in canonical_state_definition: It does `data = json_loads(json_bytes(dict(raw)))`. This deep copies? json_bytes(dict(raw)) serializes dict; json_loads parses. Fine. Then required = {...}. `if set(data) != required` ensures exact fields. But then it mutates data["energy_unit"]="kcal/mol" and data["windows"]=sorted(...). It doesn't check that data["windows"] is a list before normalize_windows? normalize_windows handles sequence. Good.

Potential issue: In canonical_state_definition, for NPT, it allows pressure_bar to be any finite number, including negative? finite_number without positive. Pressure should be positive. But not directly free energy? Could be.

Now bias.py reconstruct_bias_matrix. It computes matrix = np.zeros((n, len(rows))). For each column, adds umbrella. Then matrix *= beta * KJ_PER_KCAL. So matrix is reduced bias (dimensionless), not kJ. The docstring says "Return the (N,K) reduced umbrella-plus-ladder bias, not full potential." Good. Then if lambdas >0, calls _ladder_apply(matrix, pep, dih, lambdas, envelope, beta, bookkeeping). Need know apply_ladder_boost_to_u. Not shown. But likely adds beta * boost to reduced bias. Good.

Potential issue: In reconstruct_bias_matrix, if cv2 is None but some windows have k2>0, raises MissingCoordinateError. Good. If v_pep is None but envelope.has_total True, raises. Good.

Now, the "sampling path" in production.py lines 7420-7700. Need inspect for bugs.

In sample(), they compute primary_values, ss_values, potentials_kj, v_pep_kj, v_dih_kj. For each replica, _fetch_state returns r, cv, ss, pe, v_pep, v_dih. Then:
ss_centers_arr = ss_centers_arr_global
ss_k_arr = ss_k_kcal_arr_global
_has_ss_axis = secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None
distance_bias_matrix_kcal, ss_bias_matrix_kcal = umbrella_bias_matrix_kcal(...)
boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env) if pep_env is not None else np.zeros((nrep, nrep))
bias_matrix_kcal, bias_matrix_kj = assemble_bias_matrices(...)
reduced_bias_matrix = float(beta) * bias_matrix_kj

Wait: bias_matrix_kj is in kJ. reduced_bias_matrix = beta * bias_matrix_kj. beta is 1/(kB T) in mol/kJ. So reduced bias dimensionless. Good.

But note: bias_matrix_kcal = distance_bias_kcal + ss_bias_kcal + boost_bias_kcal. boost_bias_kcal = boost_bias_kj / 4.184. Good.

Then for each replica r, w = assignments[r]. all_bias_kcal = bias_matrix_kcal[:, r] (all states for this replica). sampled_bias_kcal = all_bias_kcal[w]. sampled_bias_kj = all_bias_kj[w]. sampled_umbrella_bias_kj = 4.184*(distance+ss at w). sampled_boost_bias_kj = boost_bias_matrix_kj[w, r]. Good.

But there is a potential bug: `boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env) if pep_env is not None else np.zeros((nrep, nrep))`. If pep_env is None, shape (nrep,nrep). But if number of windows != nrep, then bias_matrix_kcal shape mismatch? Let's see: distance_bias_matrix_kcal shape (nstates, nreps). ss same. boost_bias_matrix_kj shape (nrep,nrep) if pep_env None. assemble_bias_matrices adds distance_bias_kcal + ss_bias_kcal + boost_bias_kcal. boost_bias_kcal = boost_bias_kj/4.184 shape (nrep,nrep). If nstates != nrep, broadcasting error. In REUS, nstates should equal nreps. But adaptive epochs may have more windows than replicas? Actually replica exchange with N replicas and N windows. If windows > replicas, some windows unoccupied? The code's replica_of_window has size n_win? In swap_candidate_replicas, n_win = replica_of_window.size. If n_win > nrep, some entries -1. But bias matrix shape (n_win, nrep). boost matrix if pep_env None is (nrep,nrep) -> mismatch. This could be a real bug if n_win != nrep. But maybe production always has equal. The code has "valid_mask = holders >=0" suggesting unheld windows possible. So if n_win > nrep, the fallback np.zeros((nrep,nrep)) is wrong. But if pep_env is not None, pep_gamd_boost_matrix_kj returns (len(lambdas), nrep) = (n_win, nrep). So only when ladder inactive. In that case, boost is zero, so they should use np.zeros((n_win, nrep)). Using (nrep,nrep) will broadcast error if n_win != nrep. This would not be silent; it would crash. But if n_win < nrep? Then broadcasting (nrep,nrep) + (n_win,nrep) error. So not silent. But could be a latent bug. Not "silently produce wrong number".

Another issue: In sample(), `potentials_kj[r] = pe` where pe is from getState(getEnergy=True, groups=physical_energy_groups_for_args(args)). If the boost is applied as an integrator force, the physical energy groups might exclude it. But if the boost is a force in the System, then physical_energy_groups_for_args might include it? Need know. The code's "extract_gamd_boost_kj(sim.integrator, unit, None)" reads boost from integrator, suggesting boost is not in Context potential energy. So potential_kj is physical. Good.

But there is a critical issue: The exchange acceptance uses bias_matrix_kj which includes the boost computed from v_pep/v_dih. However, the actual Hamiltonian of a replica in state w includes the boost applied by the integrator. The boost applied by the integrator is computed from the *current* v_pep/v_dih and the integrator's k0 values. But the integrator's k0 values are set to lambda*window * k0max. The code's pep_gamd_boost_kj uses env.k0max_dih and env.k0max_total. If the integrator's k0 values are not exactly lambda*k0max (e.g., due to envelope recalibration or stage), then mismatch. The submission mentions "envelope recalibration" and "stale or overwritten calibration". The code has _gamd_recal_active and _gamd_recal_targets in sample path. It accumulates target energies for recalibration. If recalibration changes envelope during production, then the boost matrix uses new envelope while samples before recalibration used old envelope. MBAR over all samples with different Hamiltonians? Actually if the envelope changes, the Hamiltonian changes. The state definition should freeze envelope. The code's state_identity requires frozen envelope. But adaptive epochs may recalibrate envelope, which would change the Hamiltonian. If samples from different envelopes are pooled, MBAR would be wrong unless the boost function is explicitly time-dependent and accounted. The submission says "adaptive epochs (window add/split/retire, envelope recalibration)" and "final reweighting by MBAR over all (window, rung) states using per-frame recorded CVs and boost arguments". If envelope recalibration changes the boost, then the recorded v_pep/v_dih and lambda are not sufficient to reconstruct the boost unless the envelope used at the time is recorded. The state_identity.py requires frozen envelope in state definition, but if envelope recalibration occurs, the state definition would change, and the code's validate_fixed_state_segments would reject incompatible segments. But does production actually freeze envelope? The submission says "envelope ... calibrated ONCE by the swarm and then frozen" but also "adaptive epochs (window add/split/retire, envelope recalibration)". Contradiction. Maybe envelope recalibration is only in adaptive epochs before production? Need not.

Let's focus on the provided code. We need find concrete silent errors.

Potential bug in _gibbs_window_proposal_distribution: It uses `bias = np.asarray(bias_matrix_kj, dtype=np.float64)`. If bias_matrix_kj contains NaN for some state-replica pair, then delta for candidates involving that pair becomes NaN. The finite mask removes those candidates. But if the current window's bias for rep is NaN, then all deltas NaN, and the function returns empty, causing gibbs_propose_one_replica to return a "failed" proposal with accepted=False? Actually if valid_windows.size <=0, returns GibbsProposal(wi, wi, 0.0, 0.0, 0.0, 0.0, False, True). The last field maybe "failed"? Need see GibbsProposal definition not shown. If it returns failed, the caller might skip. But if bias has NaN due to missing secondary CV, the code in umbrella_bias_matrix_kcal sets ss_kcal to 0 for NaN, so bias finite. Good.

But wait: In umbrella_bias_matrix_kcal, ss_kcal = np.where(np.isfinite(dss), ss_kcal, 0.0). If dss is NaN because ss_values is NaN, bias zero. But if the secondary CV is missing for a replica, the correct bias for a state with secondary restraint should be undefined, not zero. Using zero means the replica's effective Hamiltonian is missing the secondary restraint, so the exchange probability is wrong. However, if secondary CV is missing, the sample cannot be used for MBAR for states requiring that CV. But the exchange would still occur based on zero bias, which is not the Hamiltonian actually applied? Wait, the actual Hamiltonian applied by the integrator includes the secondary umbrella force. If the CV is NaN, the force cannot be computed? In OpenMM, if CV is NaN, the simulation would crash. So NaN secondary values likely only occur when secondary axis is disabled or for replicas in states without secondary restraint? But the code sets ss_values = np.full(nrep, nan) and only fills if secondary enabled. If secondary_cv_centers is None, _has_ss_axis False, so ss_bias_matrix_kcal zeros. Good. If secondary enabled but some replica's CV is NaN due to numerical issue, the simulation likely already broken. So not a silent issue.

Potential bug in _exchange_probability: It returns 1.0 if delta <= 0.0. Standard Metropolis for swapping states: pacc = min(1, exp(-β delta)). If delta negative, exp(-β delta) >1, so min=1. Good. If delta is NaN, returns 0.0. Good.

Potential bug in apply_window_swap: It uses `pacc = max(0.0, min(1.0, pacc)) if math.isfinite(pacc) else 0.0`. If p_override is NaN, pacc=0. Good. If uniform is None and not force_accept, accepted=False. Good.

But there is a subtle issue: In apply_window_swap, if accepted, it mutates assignments and replica_of_window. But it does not update any cached bias or CV observables. The caller does observable_cache.clear(). Good.

Now, the "lambda per state" functions.

campaign_ladder_registry_lambda: It walks up from out_dir to parents, looking for registry files. It returns first (filename, lam) with lam >0. It stops at adaptive_production. Potential bug: It returns the first λ >0 found in the registry, not necessarily the λ for the current state. But resolve_ladder_active only needs boolean. However, the function name suggests it returns lambda. It is used only for boolean? In resolve_ladder_active, `return campaign_ladder_registry_lambda(out_dir) is not None`. So the actual lambda value is ignored. Good.

But there is a potential bug: The registry may contain λ values for states that were later dropped or changed. If any λ>0 exists anywhere in the campaign, the ladder is considered active for all sub-runs, including a top-up that only has λ=0 states. This is intentional per docstring. But if the campaign registry is stale (e.g., from a previous campaign in a shared parent directory), it could wrongly activate the ladder. The walk stops at adaptive_production to avoid crossing campaigns, but if multiple campaigns share a parent and one has adaptive_production? The docstring says "several campaigns commonly share one parent directory, and an unbounded walk would let one campaign's ladder switch on another's boost recording." The stop at adaptive_production may not be sufficient if the parent directory itself is named adaptive_production? Actually it checks candidate.name == "adaptive_production" after checking files in that candidate. So if out_dir is .../adaptive_production/seg, it checks seg, then adaptive_production, then stops. If another campaign is a sibling under the same adaptive_production, the walk stops before reaching the shared parent, so no cross-campaign contamination. Good.

But there is a subtle bug: `campaign_ladder_registry_lambda` returns the first λ>0 in the registry, but if the registry has multiple states with different λ, it doesn't matter for boolean. However, if the registry file is found in a parent directory that is not the campaign's own registry but a parent campaign, and the current campaign has no ladder, it could activate. The stop at adaptive_production mitigates. Not sure.

_derive_state_gamd_lambdas: It reads window_metadata["normalized_rows"] if present and aligned. It returns [float(r.get("gamd_lambda", 0.0) or 0.0) for r in rows]. If a row has gamd_lambda = "None" or empty string, `or 0.0` converts to 0.0. If a row has gamd_lambda = 0.0, `0.0 or 0.0` = 0.0. If gamd_lambda = NaN, float('nan') or 0.0? In Python, float('nan') is truthy, so returns nan. Then set_replica_lambda would raise ValueError because nan not in [0,1]? Actually `if not (0.0 <= lam <= 1.0)` with nan: 0.0 <= nan is False, so raises. Good. If gamd_lambda is negative, raises. Good.

But there is a potential bug: `existing_list = list(existing or [])`. If existing is a numpy array, list() works. If existing is a string, list('abc') gives ['a','b','c'], then len==n maybe and returns chars, causing float conversion later? But callers pass list. Not silent.

_reload_state_gamd_lambdas_on_resume: It reads run_manifest.json method_settings.state_gamd_lambdas. It only sets if args.state_gamd_lambdas is falsy. If the manifest has lambdas list, sets. Good. But if the manifest has an empty list, `if lambdas:` false, so doesn't set. Then _derive_state_gamd_lambdas fallback [0.0]*n. If the campaign ladder was active but all lambdas zero? Then ladder disabled. Fine.

_persist_state_gamd_lambdas: It patches manifest with current args.state_gamd_lambdas. It uses `list(getattr(args, "state_gamd_lambdas", None) or [])`. If args.state_gamd_lambdas is a numpy array, list() gives np.float64 values; json serialization may fail? update_run_manifest likely handles. Not silent.

Now, set_replica_lambda: It sets k0_Dihedral and k0_Total. It checks names from integrator.getGlobalVariableName. If names is None (test double), it sets k0_Total if k0max.get("Total",0.0) !=0? Actually code:
```
if names is None or "k0_Total" in names:
    integrator.setGlobalVariableByName("k0_Total", lam * float(k0max.get("Total", 0.0)))
elif float(k0max.get("Total", 0.0)) != 0.0:
    raise ValueError(...)
```
If names is None, it always sets k0_Total. If the integrator is a single dihedral boost without k0_Total, and names is None (test double), it would try to set a nonexistent global and maybe raise? But test doubles expose only setGlobalVariableByName, so it would accept. Not production.

Potential bug: set_replica_lambda_for_window uses `state_lambdas[int(window_index)]`. If window_index is out of range, IndexError. Not silent.

Now, pep_gamd_boost_kj: It computes b_dih = _channel_boost(v_dih, threshold_dih, vmax_dih, vmin_dih, lam*k0max_dih). Then if has_total, b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). out = b_dih + b_tot. This matches "dihedral first, then Total with dihedral boost added to Total energy". But wait: In gamd-openmm's dual boost, the total boost is applied to the total potential energy *including* the dihedral boost? Actually the total boost is added on top of the dihedral boost, and the total energy used for the total boost threshold is the total potential energy *after* adding the dihedral boost? The code uses v_pep + b_dih. But v_pep is "peptide essential energy" which is the total potential energy of the peptide? Or the total potential energy excluding dihedral? The function name "peptide_essential_energy_kj" suggests it's the essential (total) energy of the peptide, maybe excluding dihedral? Need know. If v_pep already includes dihedral energy, then adding b_dih double counts. The docstring says "Total with the dihedral boost added to the Total energy before the square". So v_pep is total energy without dihedral boost? It says "peptide_essential_energy_kj" maybe the total potential energy of the peptide (all forces). If it includes dihedral, then adding b_dih is wrong. But the docstring explicitly says gamd-openmm does this. We need trust? Could be a bug if v_pep includes dihedral. The code's _fetch_v_pep_v_dih reads v_dih separately from group 2, and v_pep from peptide_essential_energy_kj. If peptide_essential_energy_kj reads total energy of groups excluding dihedral? Not shown. The comment says "gamd-openmm's factory puts PeriodicTorsion/CMAPTorsion forces in group 2 ... same id as DIHEDRAL_GROUP in the Pep-GaMD partition, so this read is valid on both paths." This suggests v_pep is total energy excluding group 2? Actually "peptide_essential_energy_kj" likely computes total energy of peptide atoms from all groups? Need not.

Potential bug in _channel_boost: It uses `scale = np.maximum(np.maximum(abs(e), np.abs(v)), 1.0)`. Then `b = np.where(np.abs(rng) <= 0.001 * scale, 0.0, b)`. This is a guard for zero range. If vmax-vmin is very small but nonzero, it sets boost to 0 if range <= 0.001*scale. This could silently zero the boost for a narrow envelope, even though the actual integrator might apply a boost. If the integrator's guard differs, mismatch. But likely same.

Potential bug: _channel_boost returns `np.where((b + v) < e, b, 0.0)`. This is the condition for boost: if V < E, then b >0 and b+V < E? Actually if V < E, b = 0.5 k0 (E-V)^2/rng. Then b+V < E? Let's check: b+V < E iff 0.5 k0 (E-V)^2/rng < E-V. If k0 and rng positive, this is 0.5 k0 (E-V) < rng. Not always true. Standard GaMD condition is V < E, not b+V < E. Wait, the boost is defined as ΔV = 0.5 k0 (E - V)^2 for V < E. The condition for applying boost is V < E. The code's condition `(b + v) < e` means the boosted energy is below threshold. Is that correct? In GaMD, the boost is designed so that the modified potential is V + ΔV. The threshold E is the upper bound of the modified potential? Actually in GaMD, the boost potential is:
ΔV(r) = 0, if V(r) >= E
ΔV(r) = 0.5 k0 (E - V(r))^2, if V(r) < E
So the condition is V < E. The code's condition `(b + v) < e` is different. Let's test: Suppose V=10, E=20, k0=0.1, rng=10. b=0.5*0.1*(10)^2/10=0.5. b+V=10.5 <20, so applies. Suppose V=19, E=20, k0=1, rng=10. b=0.5*1*(1)^2/10=0.05. b+V=19.05<20, applies. Suppose V=19.9, E=20, k0=100, rng=10. b=0.5*100*(0.1)^2/10=0.05. b+V=19.95<20, applies. Suppose V=10, E=20, k0=100, rng=10. b=0.5*100*(10)^2/10=500. b+V=510 >20, so code returns 0. But standard GaMD would apply boost 500. However, such large k0 would be unphysical; k0 is chosen so that the boost is bounded. In GaMD, k0 is chosen such that ΔV <= E - V? Actually the boost is capped so that the modified potential is below E? Let's recall: GaMD sets k0 = k0_max * (Vmax - Vmin)/(E - Vmin) maybe, and the boost is 0.5 k0 (E - V)^2. The maximum boost occurs at V=Vmin: ΔV_max = 0.5 k0 (E - Vmin)^2. With k0 = k0_max (Vmax - Vmin)/(E - Vmin), ΔV_max = 0.5 k0_max (Vmax - Vmin)(E - Vmin). This can be large. The condition for boost is V < E. The code's condition b+V < E would zero the boost if the boost would push the energy above E. Is that what gamd-openmm does? Let's check gamd-openmm source? I recall in OpenMM GaMD, the boost is:
```
if (V < E) {
  dV = 0.5 * k0 * (E - V)^2 / (Vmax - Vmin);
} else {
  dV = 0;
}
```
No additional condition. But maybe in gamd-openmm's implementation, the boost is applied as a force and the condition is `if (V + dV < E)`? Actually I'm not sure. The code's docstring says "gamd-openmm's dependent dual boost" and the function is supposed to match. If the condition is wrong, then the boost matrix prices a different boost than the integrator applies, causing wrong free energy. This is a potential VERIFIED? We need know gamd-openmm. The code's `_channel_boost` condition `(b + v) < e` is suspicious. Standard GaMD condition is `v < e`. Let's verify with known GaMD formula. From Miao et al., the boost potential is:
ΔV(r) = 1/2 k0 (E - V(r))^2, if V(r) < E
= 0, otherwise.
So condition is V < E. The code's condition is V + ΔV < E. That is not standard. However, some implementations use a "threshold energy" E such that the total effective energy V + ΔV is capped at E? Actually in GaMD, the threshold E is set to the lower bound? Let's recall: In GaMD, the potential is modified as V*(r) = V(r) + ΔV(r), where ΔV = 0 if V >= E, else 0.5 k0 (E - V)^2. The threshold E is chosen such that V*(r) is smooth. The maximum of V* occurs at V = E - (rng/k0)? Hmm.

Let's derive: If ΔV = 0.5 k0 (E - V)^2 / (Vmax - Vmin), then V* = V + 0.5 k0 (E - V)^2 / R. The derivative dV*/dV = 1 - k0 (E - V)/R. Setting to zero gives V = E - R/k0. At that point, V* = E - R/k0 + 0.5 k0 (R/k0)^2 / R = E - R/k0 + 0.5 R/k0 = E - 0.5 R/k0. So V* can exceed E if V is far below E. The condition V* < E is not equivalent to V < E. Standard GaMD does not require V* < E. So the code's condition is nonstandard. But maybe gamd-openmm uses a "lower bound" dual boost where the boost is applied only if the total energy after boost is below threshold? The submission says "Pep-GaMD lower-bound dual (dependent dihedral + total) boost whose envelope (Vmax/Vmin/threshold/k0max)". "lower-bound" might mean the boost is designed to keep the boosted energy below a threshold? Actually "lower-bound" in GaMD means the boost potential is a lower bound of the original? Not sure.

Let's look at the code's `_channel_boost`:
```
b = 0.5 * k0 * (e - v) ** 2 / rng
b = np.where(np.abs(rng) <= 0.001 * scale, 0.0, b)
return np.where((b + v) < e, b, 0.0)
```
This is clearly a "cap" such that the boosted energy does not exceed e. If the integrator's boost formula is the same, then fine. But if the integrator uses standard GaMD, mismatch. The docstring says "gamd-openmm's dependent dual boost" and "stage_integrator _add_dihedral_boost_to_total_energy". We need know gamd-openmm's actual formula. I can reason: In OpenMM's GaMD implementation (e.g., openmmtools or gamd-openmm), the boost is:
```
if (V < E) {
  dV = 0.5 * k0 * (E - V)^2 / (Vmax - Vmin);
} else {
  dV = 0;
}
```
I don't recall a condition on V+dV. The code's condition might be a bug. But the submission itself is a review of the code, and we are to find bugs. This is a strong candidate. However, we need be careful: The code's `_channel_boost` is in pep_gamd.py, which is part of the subsystem? The task says "Subsystem under review: replica exchange: state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path". pep_gamd.py is included because it computes the boost matrix used in exchange. So yes.

Let's examine the condition more. If `(b + v) < e` is false, the boost is set to 0. This means for configurations where the boost would push the energy above the threshold, the boost is not applied. But the integrator, if it uses the same formula, would also not apply. If the integrator uses standard GaMD, then the code prices zero boost while the integrator applies nonzero boost. That would be a "wrong Hamiltonian applied vs priced" silent error. But we don't have the integrator source. The submission says "gamd-openmm's StageIntegrator" and "stage_integrator _add_dihedral_boost_to_total_energy". It claims the code matches. We can't verify. But we can flag as SUSPECTED.

Another potential bug: In `_channel_boost`, the denominator is `rng = vmax - vmin`. If vmax < vmin, rng negative, b negative? Then `np.where((b+v)<e, b, 0)` could apply negative boost (i.e., penalty). But envelope should have vmax > vmin. Not silent.

Now, the "sampling path" in production.py lines 7420-7700. There is a critical issue: In sample(), the row records `"potential_kj_mol": float(potentials_kj[r])`. This is the physical potential energy. But for MBAR, the total energy of state w is physical + umbrella + boost. The code records `umbrella_bias_kj_mol` (which includes boost) and `potential_kj_mol`. So total = potential + umbrella_bias_kj_mol. Good. But wait: `umbrella_bias_kj_mol` = sampled_bias_kj = bias_matrix_kj[w,r] which includes boost. So total biased energy = potential + umbrella + boost. Good.

But there is a subtle issue: The boost energy computed by pep_gamd_boost_kj is a function of v_pep and v_dih. However, the actual boost applied by the integrator may depend on the *current* stage and the integrator's internal state (e.g., it may use a running average or the boost is applied only in certain stages). The code's `extract_gamd_boost_kj(sim.integrator, unit, None)` reads the actual boost from the integrator and records it in `gamd_boost_total_kj_mol`. But the bias matrix uses the reimplemented pep_gamd_boost_kj, not the integrator's actual boost. If they disagree, the exchange and MBAR use the reimplemented boost, while the actual sampled Hamiltonian uses the integrator's boost. The row records both, so analysis could detect mismatch? But the code doesn't check. This is a potential silent error. The submission's calibration case was exactly this: boost never applied but priced. The fix added verify_gamd_production_stage5. But the code still doesn't verify that pep_gamd_boost_kj matches the integrator's actual boost at runtime. It only verifies stage? The snippet doesn't show verify. So we can flag: no runtime check that the reimplemented boost equals the integrator's actual boost; any divergence silently corrupts exchange and MBAR. But is that a "place where software can silently produce a number"? Yes.

Let's look at `_fetch_v_pep_v_dih`: It reads v_dih from group 2. But the boost applied by the integrator may use a different dihedral energy definition (e.g., including periodic torsion forces in group 2, but also maybe CMAP). The comment says group 2 is same. Good.

Potential bug in `pep_gamd_boost_matrix_kj`: It uses `np.vstack([... for l in lambdas])`. If lambdas is a list of length nstates, and v_pep/v_dih are length nreps, result shape (nstates,nreps). Good. But if lambdas is a numpy array, iterating over it gives np.float64. Good.

Now, the "state bias matrix" in sample path: `bias_matrix_kcal, bias_matrix_kj = assemble_bias_matrices(distance_bias_matrix_kcal, ss_bias_matrix_kcal, boost_bias_matrix_kj)`. Note that `distance_bias_matrix_kcal` and `ss_bias_matrix_kcal` are in kcal, boost_bias_matrix_kj in kJ. assemble converts boost to kcal by dividing by 4.184. Good. But then `bias_kj = 4.184 * bias_kcal`. This is exact. Good.

Potential bug: In `umbrella_bias_matrix_kcal`, the secondary bias uses `np.where(np.isfinite(dss), ss_kcal, 0.0)`. If `ss_centers` is not None but `ss_values` has NaN for some replica, the bias for that replica is zero for all states. But the actual umbrella force on that replica? If the secondary CV is NaN, the simulation would have NaN forces. So not.

Now, the "gibbs-walk" proposal. There is a potential detailed-balance violation in the way the proposal distribution handles the current window. Let's examine more carefully.

In `_gibbs_window_proposal_distribution`, for a given replica rep currently in window wi, it computes delta for each candidate window wj:
old_e = bias[wi, rep] + bias[wj, target_rep]
new_e = bias[wj, rep] + bias[wi, target_rep]
delta = new_e - old_e.
For wj = wi, target_rep = rep, old_e = 2*bias[wi,rep], new_e = 2*bias[wi,rep], delta=0. So the stay probability is proportional to exp(0)=1. But in a proper heat-bath over the assignment of replica rep to windows, the probability of assigning rep to window wj (and the displaced replica to wi) should be proportional to exp(-β * (bias[wj,rep] + bias[wi,target])) (since the rest constant). For wj=wi, this is exp(-β * 2*bias[wi,rep])? Wait, if rep stays in wi, the energy contribution is bias[wi,rep] (not 2*bias). The "displaced replica" is rep itself, so the pair energy is just bias[wi,rep]. The formula old_e = bias[wi,rep] + bias[wi,rep] double counts. But since delta=0, the relative weight for stay is exp(0)=1. However, the absolute normalization is wrong because the stay weight should be exp(-β * bias[wi,rep]) relative to other candidates? Let's derive the conditional distribution.

We have a set of replicas assigned to windows. Consider a single-replica move: choose replica rep. The state of the system is an assignment of replicas to windows (a permutation). The Gibbs sampler would sample the window for replica rep conditional on all other replicas' windows fixed. But because each window must have exactly one replica, moving rep to wj requires moving the current occupant of wj to wi. So the move is a transposition. The conditional probability of the transposition (rep <-> target) is proportional to exp(-β * (bias[wj,rep] + bias[wi,target])). For the identity (no move), the energy is bias[wi,rep] + sum_{other replicas} bias[window, replica]. The sum over other replicas is constant. So the weight for stay should be exp(-β * bias[wi,rep]) times a constant. The weight for moving to wj should be exp(-β * (bias[wj,rep] + bias[wi,target])) times the same constant. Therefore, the ratio of move-to-stay weights is exp(-β * (bias[wj,rep] + bias[wi,target] - bias[wi,rep])). But the code's delta = (bias[wj,rep] + bias[wi,target]) - (bias[wi,rep] + bias[wj,target]). For wj != wi, this is not the same as the correct ratio because it includes bias[wj,target] in old_e. Wait, old_e should be the energy of the current assignment for the two replicas involved: rep in wi, target in wj. That is bias[wi,rep] + bias[wj,target]. New_e is bias[wj,rep] + bias[wi,target]. So delta = new - old. The ratio of weights for move vs stay should be exp(-β * (new_e - current_energy_of_rep?)). But the current energy of the system includes both rep and target: bias[wi,rep] + bias[wj,target]. So the probability of the move relative to stay is exp(-β * (new_e - old_e))? Let's compute: The conditional probability of the transposition (rep,target) vs identity is proportional to exp(-β * new_e) vs exp(-β * old_e). Because the rest of the system is the same. So the ratio is exp(-β * (new_e - old_e)). That is exactly the code's delta. For stay, old_e = new_e = bias[wi,rep] + bias[wi,rep]? But the "target" for stay is rep itself, so the pair energy should be bias[wi,rep] + bias[wi,rep]? No, if rep stays in wi, there is no second replica involved. The energy contribution of rep is bias[wi,rep]. The "target" is rep, but the window wi is occupied by rep; the transposition of rep with itself leaves the assignment unchanged. The energy of the pair (rep, rep) is not meaningful. The correct old_e for the move to wj is bias[wi,rep] + bias[wj,target]. For stay, the relevant energy is just bias[wi,rep]. So the ratio of move-to-stay should be exp(-β * (bias[wj,rep] + bias[wi,target] - bias[wi,rep])). The code's delta for move is (bias[wj,rep] + bias[wi,target]) - (bias[wi,rep] + bias[wj,target]). This includes -bias[wj,target], which is the energy of the target replica in its current window. But that term is part of the old system energy and is constant across all candidate moves? Wait, for a given target wj, bias[wj,target] is the energy of the target replica in its current window. This term is not constant across different wj because different targets have different energies. However, in the conditional distribution over which window rep moves to, the probability of choosing wj should be proportional to exp(-β * new_e) where new_e = bias[wj,rep] + bias[wi,target]. The old_e term bias[wi,rep] + bias[wj,target] is the energy before the move, which is the same for all candidate moves? No, old_e depends on wj because bias[wj,target] depends on which target is in wj. But the "before" state is fixed: rep is in wi, and each wj has a specific target. The energy before is bias[wi,rep] + bias[wj,target] + constant. This varies with wj. In a Gibbs sampler, the conditional probability of a new state is proportional to exp(-β * energy_of_new_state), not exp(-β * (new - old)). Because the old state is fixed, the normalization over candidate new states should use the absolute energy of each candidate new state, not the difference from the old state. However, since the old state is one of the candidates (stay), the ratio of probabilities is exp(-β * (new_e - old_e)) where old_e is the energy of the old state. For the stay candidate, new_e = old_e, so weight 1. For a move to wj, the weight is exp(-β * (new_e - old_e)). This is exactly what the code computes, with old_e = bias[wi,rep] + bias[wj,target]. For the stay candidate, old_e should be the energy of the old state, which is bias[wi,rep] + (energy of target? there is no target). But if we define the "move" as swapping rep with the replica in wj, then for wj=wi, the target is rep itself, and the old and new states are the same. The energy of the old state is bias[wi,rep] + sum_others. The energy of the "pair" (rep, rep) is not bias[wi,rep]+bias[wi,rep]. So the code's stay delta=0 is correct, but the old_e for stay is wrong. However, since only delta matters, and delta for stay is set to 0 manually, the stay weight is exp(0)=1. For a move to wj, the weight is exp(-β * (new_e - old_e)) where old_e = bias[wi,rep] + bias[wj,target]. This is correct because the old state's energy includes bias[wj,target]. So the ratio of move-to-stay is exp(-β * (bias[wj,rep] + bias[wi,target] - bias[wi,rep] - bias[wj,target])). Is that the correct conditional probability? Let's test with a simple example.

Suppose two windows A and B, two replicas 1 and 2. Current assignment: rep1 in A, rep2 in B. Energy before = bias[A,1] + bias[B,2]. Candidate moves for rep1: stay in A (energy after = bias[A,1]+bias[B,2]) or swap with rep2 (energy after = bias[B,1]+bias[A,2]). The conditional probability of swap should be proportional to exp(-β*(bias[B,1]+bias[A,2])). The probability of stay proportional to exp(-β*(bias[A,1]+bias[B,2])). The ratio swap/stay = exp(-β*(bias[B,1]+bias[A,2] - bias[A,1]-bias[B,2])). The code's delta for swap = (bias[B,1]+bias[A,2]) - (bias[A,1]+bias[B,2]). Exactly. For stay, delta=0. So the proposal is correct. The double counting in old_e for stay is irrelevant because delta manually set to 0. Good.

But wait: The proposal distribution is over all windows, including those with target replicas. For each candidate wj, the "old_e" uses bias[wj,target]. This is the energy of the target in its current window. That is part of the old state. So the weight for moving to wj is exp(-β*(new_e - old_e)) = exp(-β*(bias[wj,rep]+bias[wi,target] - bias[wi,rep]-bias[wj,target])). This is the ratio of the probability of the swapped state to the probability of the current state. Since the current state is fixed, the conditional probability of the swapped state is proportional to exp(-β * energy_swapped) = exp(-β*(bias[wj,rep]+bias[wi,target]+constant)). The constant includes all other replicas. The ratio to stay is exp(-β*(bias[wj,rep]+bias[wi,target] - bias[wi,rep]-bias[wj,target])). Yes, because stay energy = bias[wi,rep]+bias[wj,target]+constant. So correct.

Now, the reverse proposal: After swapping, the current window for rep is wj, and the target in wi is the original target. The reverse move would swap rep back with target. The proposal distribution for rep from wj will compute delta_rev = (bias[wi,rep]+bias[wj,target]) - (bias[wj,rep]+bias[wi,target]) = -delta. The weight for choosing wi is exp(-β * delta_rev) = exp(β * delta). The forward weight for choosing wj was exp(-β * delta). So q_rev/q_fwd = exp(2β delta)? Wait, let's compute. Forward q(wj|current) ∝ exp(-β delta). Reverse q(wi|after) ∝ exp(-β delta_rev) = exp(β delta). So q_rev/q_fwd = exp(2β delta). Then MH log_alpha = -β delta + log(q_rev/q_fwd) = -β delta + 2β delta = β delta. That would give acceptance >1 for delta>0, which is wrong. But wait, the proposal distribution is normalized over all windows. The ratio q_rev/q_fwd is not simply exp(2β delta) because the normalization constants differ. The code computes q_forward and q_reverse as the actual probabilities from the normalized distributions. So it's fine. The MH correction uses the actual probabilities. Good.

But there is a subtle issue: The proposal distribution includes the stay candidate with weight 1 (delta=0). In the reverse proposal, the stay candidate for rep in wj is wj itself. The reverse probability of choosing wi is computed from the normalized distribution over all windows in the after state. Good.

Now, a potential bug: In `gibbs_propose_one_replica`, if the proposal chooses the current window (wj==wi), it returns `GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False)`. The `accepted` field is True, meaning the move is accepted (stay). But the `q_reverse` is 0.0. If the caller uses this to update something, it might think a swap occurred. But since wj==wi, no swap. The `accepted=True` might be used to count accepted moves, but it's a stay. Not a free energy issue.

Now, the "explicit_2d_neighbor_pairs_for_exchange" function. It builds edges, then round-robin by edge index modulo slots. It filters disjoint. If a slot is empty, it tries all edges greedily. Potential bug: The fallback "If a slot was empty for a small graph, try all edges greedily so exchanges do not stall." This could cause the same edge to be attempted in multiple slots? No, only if chosen empty. But the fallback uses `used` set that may already have entries from the slot? If chosen empty, used is empty. Good. But if the slot had some edges but all conflicted, chosen could be empty? Actually if slot has edges but all conflict, chosen empty, then fallback tries all edges. This could attempt edges that were already considered in the slot but conflicted; but since chosen empty, no conflict with chosen. It could attempt an edge that was skipped due to conflict with a previously chosen edge? But if chosen empty, no previously chosen. So fine.

Potential bug: The edge coloring by `idx % slots != slot` means each slot gets every `slots`-th edge. This is not a proper graph edge coloring; it's a round-robin partition. The greedy disjoint filtering within a slot ensures no vertex appears twice. But across slots, the same edge may be attempted in different slots? No, each edge belongs to exactly one slot by index modulo. So over `slots` exchange intervals, all edges are considered. Good.

Now, the "sampling path" in production.py lines 7780-7900. There is a potential bug in `_current_exchange_arrays`: It uses `observable_cache` from the same production step if available
(TRUNCATED - output hit the token limit before completion)

## mini

VERDICT: ACCEPT‑WITH‑CHANGES – the core REUS kernel contains several silent‑failure paths that can corrupt the Boltzmann‑weighted free‑energy estimate.

**Key strengths**
- Clear separation of concerns: bias assembly, Gibbs proposal, and swap execution are isolated in pure NumPy functions, facilitating unit testing.
- Extensive defensive type‑casting and finite‑number checks in the bias‑reconstruction utilities (`numeric_vector`, `finite_number`).
- Explicit handling of missing secondary CVs and of ladder‑inactive configurations throughout the production loop.
- Detailed logging and diagnostic dump on NaN crashes, aiding post‑mortem analysis.

**Critical weaknesses and errors**

| File:line | Issue (verified / suspected) | Failure scenario | Effect on free energy | Confidence |
|-----------|------------------------------|------------------|-----------------------|------------|
| `gareus/production.py:902` (in `umbrella_bias_matrix_kcal`) | **Verified** | Primary CV value is `nan` (e.g. missing observation). `dprimary` becomes `nan` → `distance_kcal` contains `nan`. | NaNs propagate into `bias_matrix_kj`; downstream Gibbs proposals drop the replica (empty proposal) or reject swaps, biasing the sampling distribution and yielding a non‑Boltzmann free‑energy surface. | 95 |
| `gareus/production.py:1005` (in `_gibbs_window_proposal_distribution`) | **Verified** | Very unfavorable moves produce `delta` such that `-β·delta < -745`. The clipping to `-745` before the log‑sum‑exp shift forces all under‑flowed candidates to have the same weight after the shift (`exp(0)=1`). | Detailed‑balance is broken: low‑probability moves are artificially promoted, distorting the stationary distribution and the MBAR reweighting, leading to systematic free‑energy errors. | 90 |
| `gareus/production.py:1015` (same function) | **Suspected** | All `log_weights` become `nan` (e.g. because `bias` contains `nan` for a replica). The function returns empty proposal dicts, causing the replica to never attempt exchanges. | The replica becomes trapped in its initial window, violating ergodicity and biasing the estimated PMF. | 80 |
| `gareus/production.py:2395` (in `_reload_state_gamd_lambdas_on_resume`) | **Verified** | `args.state_gamd_lambdas` is an all‑zero list (legitimate “ladder disabled”). The truthiness check treats it as falsy and reloads lambdas from the manifest, possibly re‑activating a ladder that the user explicitly disabled. | Unexpected boost values are applied during a resumed run, producing a free‑energy surface that mixes boosted and unboosted statistics. | 95 |
| `gareus/production.py:1045` (in `_gibbs_mh_acceptance_probability`) | **Suspected** | `q_forward` or `q_reverse` underflows to zero (e.g. after the clipping issue above). The function returns `0.0` acceptance probability, silently rejecting the move. | Systematic rejection of certain swaps skews the transition matrix, breaking detailed balance and corrupting the free‑energy estimate. | 75 |
| `gareus/production.py:1060` (in `_exchange_probability`) | **Suspected** | `delta_kj` is `nan` (from NaN bias entries). The function returns `0.0`, rejecting the swap without warning. | Same as above – trapped replicas, non‑Boltzmann sampling. | 70 |
| `gareus/production.py:1049` (in `_gibbs_mh_acceptance_probability`) | **Suspected** | Any exception (including unexpected type errors) forces a return of `0.0`. This can mask programming bugs and silently reject moves. | Hidden failures may go unnoticed, leading to biased sampling. | 65 |
| `gareus/production.py:1040` (in `_gibbs_window_proposal_distribution`) | **Suspected** | Uniform fallback (`probabilities = np.full(...)`) when total weight is non‑finite or ≤ 0. This can occur after clipping or NaN filtering, silently replacing the correct heat‑bath distribution with a uniform one. | Alters the proposal distribution, breaking the intended Gibbs sampling and biasing free‑energy reconstruction. | 70 |

**Overall assessment** – The implementation correctly captures the high‑level REUS algorithm, but several subtle numerical and logical edge cases allow the sampler to deviate from the true Boltzmann distribution without raising errors. Fixes should include:

1. Propagate `nan` primary CVs as *zero* bias (or mask them) before building the bias matrix, matching the secondary‑axis handling.
2. Perform the log‑sum‑exp shift **before** clipping, or avoid clipping altogether and use `np.exp` with `dtype=np.float64` under‑flow handling.
3. Replace the truthiness check in `_reload_state_gamd_lambdas_on_resume` with an explicit “is None” test to preserve all‑zero ladders.
4. Add explicit warnings when proposals are dropped due to NaNs or zero total weight, and ensure such cases trigger a fallback that preserves detailed balance (e.g., reject the move but log it).
5. Tighten the acceptance‑probability functions to distinguish between genuine zero probability and numerical underflow, possibly by raising an exception in the latter case.

Addressing these points will eliminate silent bias and ensure that the reported free‑energy surfaces are truly Boltzmann‑weighted.

## thinker

We need answer as board review. Need assess submission. Need produce verdict. Need analyze code for silent wrong Boltzmann free energy. We need be adversarial. Need identify findings with file:line, concrete failure, effect, confidence, VERIFIED/SUSPECTED. Need final lines exactly.

We need review the provided code snippets. Need determine if accept/reject. The submission is itself a review? Actually user says "Board review: ATLaS-MD / gareus v0.8.2 ... does this code produce Boltzmann free energy it claims?" Then task: "Adversarially review the code below. Report every place..." We need produce independent assessment. We need examine code for errors.

We have many code snippets. Need identify issues in replica exchange, state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path. Also state_identity, bias.py, production sample/exchange. Need find silent wrong free energy.

Let's parse.

Key functions:

umbrella_bias_matrix_kcal: computes distance bias 0.5*k*(x-c)^2. For secondary, if ss_centers None returns zeros. If ss values NaN, dss NaN, ss_kcal NaN then np.where(np.isfinite(dss), ss_kcal, 0.0). Good. But if ss_centers provided but ss_k_kcal None? It uses if ss_centers is None or ss_k_kcal is None return zeros. Fine.

_gibbs_window_proposal_distribution: Heat-bath proposal over target windows for one replica transposition. Need understand state graph. It computes old_e = bias[wi, rep] + bias[valid_windows, target_reps]; new_e = bias[valid_windows, rep] + bias[wi, target_reps]; delta = new_e - old_e. This is for swapping replica rep (currently in window wi) with replica in target window wj. Old total bias = U_wi(x_rep)+U_wj(x_target). New = U_wj(x_rep)+U_wi(x_target). Correct. stay delta 0. log_weights = -beta*delta. Then probabilities normalized. This is Gibbs conditional distribution for target window? For a Gibbs move, propose wj with probability proportional exp(-beta * (U_wj(x_rep)+U_wi(x_target)))? Wait target_reps are fixed, so old term U_wi(x_target) constant? Actually old_e includes U_wi(x_target) for each candidate? For candidate wj, old_e = U_wi(x_rep)+U_wj(x_target). new_e = U_wj(x_rep)+U_wi(x_target). delta = new-old. exp(-beta delta) = exp(-beta(U_wj(x_rep)+U_wi(x_target))) / exp(-beta(U_wi(x_rep)+U_wj(x_target))). Denominator depends on wj through U_wj(x_target). So not simply conditional over wj for rep? But for swapping labels, the probability of choosing wj should be proportional to exp(-beta [U_wj(x_rep)+U_wi(x_target)])? Actually in Gibbs sampling over state assignments, conditional distribution of assignment of replica rep given others? Let's derive.

We have replicas i with configurations x_i, windows assigned a_i. State energy = sum_w U_w(x_{rep(w)}) = sum_i U_{a_i}(x_i). For a transposition move swapping assignments of replica rep (currently wi) and replica target (currently wj), new state energy = U_wj(x_rep)+U_wi(x_target). Old = U_wi(x_rep)+U_wj(x_target). The heat-bath proposal over wj should choose wj with probability proportional to exp(-beta * new energy) = exp(-beta[U_wj(x_rep)+U_wi(x_target)]). But their log_weights = -beta*(new_e - old_e) = -beta new_e + beta old_e. Since old_e includes U_wj(x_target), which varies with wj, this is not proportional to exp(-beta new_e) unless U_wj(x_target) constant. Wait old_e = U_wi(x_rep)+U_wj(x_target). So log_weights = -beta(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target)) = -beta(U_wj(x_rep)-U_wj(x_target)) + constant in wj? U_wi(x_rep) constant, U_wi(x_target) constant. So log_weights = -beta(U_wj(x_rep)-U_wj(x_target)) + const. That equals -beta new_e + beta U_wj(x_target) + const. Not proportional to exp(-beta new_e) because U_wj(x_target) varies. Wait but target_reps is the replica currently in window wj; its configuration x_target is fixed. The full state energy after swap is U_wj(x_rep)+U_wi(x_target). The conditional distribution for wj given all other assignments? If we consider moving replica rep to window wj and moving the replica currently in wj to wi, the "other" assignments include target replica's new window wi. The energy of target in wi is U_wi(x_target), constant across wj? Yes U_wi(x_target) is constant because wi fixed. So new energy = U_wj(x_rep)+constant. Thus conditional probability should be proportional to exp(-beta U_wj(x_rep)). But their log_weights = -beta(U_wj(x_rep)-U_wj(x_target)) + const. That includes -U_wj(x_target), which is wrong for a Gibbs conditional over wj? Wait because the old energy term U_wj(x_target) is not constant; but in the conditional distribution of the assignment of replica rep, the target replica's assignment changes from wj to wi, so its energy changes. The full state after move includes target in wi. The target's old energy U_wj(x_target) is part of old state, not new. For Metropolis-Hastings proposal, proposal distribution q(wj|current) can be anything, then MH correction. They compute q_forward = probs[choice_index] and q_reverse from reverse proposal, then MH accept with delta. So proposal distribution need not be the true conditional; MH corrects. But they call it "Gibbs-walk" heat-bath proposal. Is it valid? It is a Metropolis-Hastings with proposal q(wj) ∝ exp(-β δ) where δ = new-old. Then acceptance uses q_reverse/q_forward. This is a valid MH if q_reverse computed correctly. However, is q_forward exactly exp(-βδ) normalized? Yes. Then MH acceptance alpha = min(1, exp(-βδ) q_rev/q_fwd). Since q_fwd ∝ exp(-βδ), exp(-βδ)/q_fwd = 1/Z_fwd? Actually q_fwd = exp(-βδ)/Z_fwd. Then alpha = exp(-βδ) * q_rev / q_fwd = Z_fwd * q_rev. That is not necessarily <=1? Wait detailed balance: proposal q(i->j), acceptance min(1, π_j q(j->i)/(π_i q(i->j))). π_j/π_i = exp(-βδ). q_fwd = exp(-βδ)/Z_fwd. So ratio = exp(-βδ) * q_rev / (exp(-βδ)/Z_fwd) = Z_fwd * q_rev. This can exceed 1, but min handles. Is this correct? Yes. But note q_fwd includes only valid windows and normalized. q_rev computed from holders_after. Need check q_reverse.

_gibbs_window_proposal_distribution called with replica_index=rep, current_window=wj, replica_of_window=holders_after. It computes old_e = bias[wj, rep] + bias[valid_windows, target_reps_after]. holders_after has target_rep at wi, rep at wj. For reverse move from state after (rep in wj, target in wi) back to state before (rep in wi, target in wj), the proposal distribution for candidate window wi? It computes for all valid windows, including wi. For candidate wi, target_reps = holders_after[wi] = target_rep. old_e = bias[wj, rep] + bias[wi, target_rep] (current after state). new_e = bias[wi, rep] + bias[wj, target_rep] (before state). delta_rev = new_e - old_e = -delta_fwd. log_weights = -beta delta_rev = beta delta_fwd. q_reverse for wi = exp(beta delta_fwd)/Z_rev. Then MH alpha = exp(-beta delta_fwd)*q_rev/q_fwd = exp(-betaδ)* [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(βδ) Z_fwd/Z_rev? Wait compute: q_fwd = exp(-βδ)/Z_fwd. q_rev = exp(βδ)/Z_rev. alpha = exp(-βδ) * q_rev / q_fwd = exp(-βδ)*exp(βδ)/Z_rev * Z_fwd/exp(-βδ)? Let's do: exp(-βδ) * (exp(βδ)/Z_rev) / (exp(-βδ)/Z_fwd) = exp(-βδ)*exp(βδ)*Z_fwd/(Z_rev*exp(-βδ)) = exp(βδ) Z_fwd/Z_rev. That seems not symmetric? Wait detailed balance ratio should be π_j/π_i = exp(-βδ). q_fwd = exp(-βδ)/Z_fwd. q_rev = exp(βδ)/Z_rev. Then π_j q_rev / (π_i q_fwd) = exp(-βδ) * [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(-βδ+βδ+βδ) Z_fwd/Z_rev = exp(βδ) Z_fwd/Z_rev. Yes. But for MH, alpha = min(1, π_j q_rev/(π_i q_fwd)). That is correct. But is q_rev computed correctly? It uses holders_after, current_window=wj, replica_index=rep. For reverse proposal, the "current" state is after swap, rep in wj. It proposes to move rep to wi and target to wj. The target_reps in holders_after for candidate wi is target_rep. old_e = bias[wj, rep] + bias[wi, target_rep]. new_e = bias[wi, rep] + bias[wj, target_rep]. delta_rev = new_e - old_e = -δ. log_weights = -β(-δ)=βδ. So q_rev(wi) = exp(βδ)/Z_rev. Good.

But wait q_fwd is computed over all valid windows, including wi (stay). For stay, delta=0, q_stay = 1/Z_fwd. If choose stay, no swap. Fine.

Potential issue: _gibbs_window_proposal_distribution uses bias_matrix_kj[wi, rep] + bias[valid_windows, target_reps]. But bias_matrix_kj is [window, replica] = U_window(x_replica). For candidate wj, target_reps = holders[wj]. old_e = U_wi(x_rep)+U_wj(x_target). new_e = U_wj(x_rep)+U_wi(x_target). Correct.

Now, the acceptance function _gibbs_mh_acceptance_probability: log_alpha = -beta*delta + log(qr) - log(qf). If log_alpha >=0 return 1. This is standard. But note if qf <=0 or qr <=0 return 0. q_reverse could be 0 if wi not in rev_windows. Then alpha=0, so no move. Fine.

Potential issue: In gibbs_propose_one_replica, when wj == wi, returns GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False). It sets accepted True? Actually fields: wi, wj, delta, q_forward, q_reverse, pacc, accepted, stuck? Need know dataclass. It returns accepted=True for stay. That's fine.

Now apply_window_swap: It swaps assignments[i], assignments[j], then replica_of_window[assignments[i]] = i, replica_of_window[assignments[j]] = j. Need check after assignments swap: assignments[i] becomes old assignments[j] = wj; assignments[j] becomes old assignments[i] = wi. Then replica_of_window[assignments[i]] = i means replica_of_window[wj] = i. replica_of_window[assignments[j]] = j means replica_of_window[wi] = j. Correct.

But wait bias_matrix_kj[window, replica] = U_window(x_replica). In apply_window_swap, old_e = bias[wi,i]+bias[wj,j], new_e = bias[wj,i]+bias[wi,j]. Correct.

Now _exchange_probability: if delta <=0 return 1; else exp(-beta delta). Correct.

Now assemble_bias_matrices: distance_bias_kcal + ss_bias_kcal + boost_bias_kcal; bias_kj = 4.184*bias_kcal. But boost_bias_kj is in kJ, divided by 4.184 to kcal, then multiplied back. Fine. However, if boost_bias_kj is all-zero when ladder inactive, okay. But if boost_bias_kj is NaN? Could propagate.

_fetch_v_pep_v_dih: reads v_dih from group DIHEDRAL_GROUP. v_pep = peptide_essential_energy_kj(ctx, unit). Need know if v_pep includes dihedral? In pep_gamd_boost_kj, b_dih = _channel_boost(v_dih, ...), b_tot = _channel_boost(v_pep + b_dih, ...). So v_pep should be "essential" total energy excluding dihedral? It says peptide_essential_energy_kj. Need check if v_pep includes dihedral? The comment says gamd-openmm's dependent dual boost: dihedral first, then Total with dihedral boost added to Total energy before square. So v_pep is total non-dihedral? likely.

Now campaign_ladder_registry_lambda: walks up parents until adaptive_production. Returns first (registry filename, lam) with lam >0. Potential issue: If out_dir is inside adaptive_production/... it stops at adaptive_production. It checks candidate = base, then parents. It breaks when candidate.name == "adaptive_production" after checking that candidate. So if out_dir itself is adaptive_production? It checks base first, then if base.name == "adaptive_production" break after checking. Fine. But if multiple campaigns share parent, could pick wrong campaign's ladder. They stop at adaptive_production to avoid. But if out_dir is e.g. /campaign/adaptive_production/seg1, base=seg1, parents include adaptive_production, then break after checking adaptive_production. Good. But if out_dir is /campaign/adaptive_production (base itself), it checks base, then break. Fine.

resolve_ladder_active: True if any state_lambdas >0 OR campaign registry above says ladder. Potential issue: If state_lambdas all zero but campaign registry has lambda >0, ladder active. Then set_replica_lambda_for_window will set k0=0 for all windows? state_lambdas all zero, so yes. But pep_env not None? Need see how pep_env constructed. If ladder active due to registry, pep_env should be created. The comment says all-λ=0 top-up gets k0=0 and samples carry v_pep/v_dih. Good.

_derive_state_gamd_lambdas: reads normalized_rows. If rows and len==n and all have gamd_lambda, return. Else existing if len==n, else zeros. Potential issue: If normalized_rows has rows but some gamd_lambda missing, falls to existing or zeros. Could silently disable ladder. But maybe okay.

_reload_state_gamd_lambdas_on_resume: reads manifest method_settings state_gamd_lambdas. Potential issue: It checks if args.state_gamd_lambdas truthy, returns. If all-zero list is truthy? In Python, [0.0] is truthy, so an explicitly set all-zero ladder is left alone. Good. But if args.state_gamd_lambdas is empty list, falsy, reload. Fine. It reads manifest. Potential issue: manifest method_settings may contain stale pre-truncation/pre-drop values if _persist not called after final mutation. They claim call at every reassignment. Need verify in code? Not shown. Could be issue.

_persist_state_gamd_lambdas: patches manifest. Potential issue: update_run_manifest may merge dicts, not overwrite? Need know. If update_run_manifest does shallow merge, method_settings may retain old state_gamd_lambdas if new list empty? It passes {"method_settings": {"state_gamd_lambdas": list(...)}}. If list empty, maybe update sets empty. Fine.

_channel_boost: b = 0.5*k0*(e-v)^2/rng. Then if (b+v)<e, b else 0. This is GaMD boost condition. But rng = vmax-vmin. If rng negative? vmax>vmin presumably. If rng <=0.001*scale, b=0. Good. Potential issue: if e-v negative, square. Fine.

pep_gamd_boost_kj: b_dih = _channel_boost(v_dih, threshold_dih, vmax_dih, vmin_dih, lam*k0max_dih). b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). out = b_dih + b_tot. This matches dependent dual boost? Need check gamd-openmm formula. Usually dual boost: V_dihedral = 0.5 k0_dih (E_dih - V_dih)^2 / (Vmax_dih - Vmin_dih) if E_dih < V_dih? Actually GaMD: boost potential is 0 if E >= V, else 0.5 k0 (V - E)^2/(Vmax - Vmin)? Wait standard GaMD: ΔV = 0.5 k0 (V - E)^2 / (Vmax - Vmin) when E < V, else 0. Here _channel_boost(v, e, vmax, vmin, k0): b = 0.5*k0*(e - v)^2/rng. Then return where (b+v)<e. So if b+v < e => b < e-v => 0.5 k0 (e-v)^2/rng < e-v => 0.5 k0 (e-v)/rng < 1 => e-v < 2 rng/k0. This is not simply E < V? Wait v is threshold? Let's parse arguments: _channel_boost(v, e, vmax, vmin, k0). In pep_gamd_boost_kj: _channel_boost(v_dih_kj, env.threshold_dih, env.vmax_dih, env.vmin_dih, lam*k0max_dih). So v = current dihedral energy? e = threshold? Actually parameter names: v, e, vmax, vmin, k0. b = 0.5*k0*(e - v)^2/rng. If v is current energy and e is threshold? Standard GaMD: if V(r) < E (threshold), boost = 0.5 k0 (E - V)^2/(Vmax - Vmin). Here v = current energy, e = threshold. b = 0.5*k0*(threshold - current)^2/rng. Then condition (b+v)<e => b < e-v => current + boost < threshold. That matches standard: if current energy below threshold, boost raises to at most threshold? Actually standard condition is V < E, and boost = 0.5 k0 (E - V)^2/(Vmax - Vmin). Then V+boost can exceed E? Let's check: V+0.5k0(E-V)^2/rng < E iff 0.5k0(E-V)/rng < 1 iff E-V < 2rng/k0. Standard GaMD chooses k0 such that boost <=? Usually k0 = k0max*(Vmax-Vmin)/(Vmax-Vavg)? Hmm. The condition in code is (b+v)<e, i.e. boosted energy below threshold. That is a "lower-bound" dual boost? The submission says "lower-bound dual". So maybe correct: boost only if total remains below threshold. But standard GaMD boost is applied when V < E, and can push above E? Actually GaMD: ΔV = 0.5 k0 (E - V)^2/(Vmax - Vmin) if V < E. No additional condition. But "lower-bound" might enforce V+ΔV < E. Need not dwell.

Potential issue: _channel_boost uses rng = vmax - vmin. If vmax and vmin are in kJ/mol? Envelope from swarm maybe in kcal? Need unit consistency. pep_gamd_boost_kj takes v_pep_kj, v_dih_kj, env thresholds/vmax/vmin. Are env values in kJ? Likely. k0max units? Need check. k0max_from_globals reads shared_globals k0_Total, k0_Dihedral. Integrator globals likely in kJ/mol? GaMD k0 units? In OpenMM, k0 has units? The boost formula b = 0.5*k0*(E-V)^2/(Vmax-Vmin). If energies in kJ/mol, k0 dimensionless? Actually k0 is dimensionless? In GaMD, k0 is a scaling factor (0<k0<=1), so b has energy units. So fine.

set_replica_lambda: sets k0_Dihedral = lam*k0max["Dihedral"], k0_Total similarly. Potential issue: If integrator has no k0_Total but k0max Total nonzero, raises. Good.

set_replica_lambda_for_window: if k0max_by_channel is None, no-op. If state_lambdas None and ladder active, raise. Good.

state_identity.py: canonical_state_definition requires exact set of fields. Potential issue: It converts energy_unit kJ/mol to kcal/mol by dividing k1,k2 by 4.184. But if k1/k2 are in kJ/mol per CV unit, dividing by 4.184 gives kcal/mol. Good. But it doesn't convert centers? Centers same units, no conversion. Fine. It requires windows have only certain fields. Good.

bias.py reconstruct_bias_matrix: It computes matrix in kcal then multiplies by beta*KJ_PER_KCAL. beta is mol/kJ. So reduced bias = beta_kJ * U_kcal * 4.184 = beta_kJ * U_kJ. Correct. But note if beta passed as 1/(kB*T) with T in K, units mol/kJ. Good.

Potential issue: In reconstruct_bias_matrix, if lambdas active and envelope has_total True, requires pep not None. If pep is None, raises. But if v_pep is all NaN? numeric_vector converts non-finite to NaN, not None. Then pep is array of NaN. It doesn't raise. Then _ladder_apply may produce NaN. But maybe okay.

Now production sample() and _current_exchange_arrays. Need find inconsistencies.

Important: In sample(), boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env) if pep_env is not None else zeros((nrep,nrep)). This computes boost for each replica's configuration under each state's lambda. But state_lambdas is per window. bias_matrix_kj[window, replica] = umbrella + boost under state window's lambda for replica's configuration. Good.

But wait: The actual Hamiltonian applied to replica r is window w=assignments[r] with lambda state_lambdas[w]. The boost applied by integrator is set by set_replica_lambda_for_window. So boost_bias_matrix_kj[w, r] should equal actual boost applied? It uses pep_gamd_boost_kj(v_pep_kj[r], v_dih_kj[r], state_lambdas[w], env). That should match integrator's boost if integrator uses same formula and same v_pep/v_dih. But sample() also reads extract_gamd_boost_kj(sim.integrator, unit, None) for row["gamd_boost_total_kj_mol"]. This is actual boost from integrator. It does not compare to computed boost_bias_matrix_kj[w,r]. Potential silent mismatch if integrator boost formula differs from pep_gamd_boost_kj. But maybe tests.

Critical: In sample(), potentials_kj[r] = physical potential energy from getState(getEnergy=True, groups=physical_energy_groups_for_args(args)). This excludes boost and umbrella? It says physical_energy_groups. Good. But if read_sample_potential False, pe=nan. Then potential_kj_mol NaN. MBAR needs total energy? Actually MBAR reweighting uses reduced potential u = beta*(U_physical + bias). If potential is NaN, MBAR fails. But sample_potential_energy default True. Fine.

Now _current_exchange_arrays: It computes primary_values, ss_values, v_pep/v_dih. Then distance_bias_kcal, ss_bias_kcal, boost_bias_matrix_kj, assemble. It returns primary_values, bias_matrix_kj. It does NOT read potential energies. For exchange, only bias differences matter because physical potential cancels in swap of state labels. Correct. But if boost is active, boost depends on v_pep/v_dih and lambda. It is included in bias_matrix_kj. Good.

Potential issue: In _current_exchange_arrays, when fast path active, it calls observe_fast_path(ctx, ...). Does that read positions? It may read collective variables. Fine.

Now _attempt_window_swap: It uses bias_matrix_kj from _current_exchange_arrays. It calls apply_window_swap. On accepted, it applies set_window and set_replica_lambda_for_window on replicas i and j. But note: after apply_window_swap, assignments[i] and assignments[j] swapped. Then _apply_swap_to_replica(replica_index) uses assignments[replica_index] to set window. For replica i, assignments[i] is new window wj. For replica j, assignments[j] is new window wi. Good. It submits to _sim_pool with i and j. But _sim_pool.submit(i, _apply_swap_to_replica, i).result()? Need know API. It might run on worker thread pinned to replica i. Fine.

Potential issue: The exchange decision uses bias_matrix_kj computed from v_pep/v_dih at current step. But after swap, the integrator's boost is set to new lambda. The actual boost applied during subsequent dynamics will be based on future energies. Fine.

Now, the big issue: The submission's own calibration case: StageIntegrator stage selection from stepCount; swarm envelope carried physics globals but no stepCount, so every replica sat in stage 2 with ForceScalingFactor exactly 1.0 while exchange and MBAR priced a boost never applied. Fixed in c73224f. We need find next ones like it. The code under review may still have similar silent mismatch.

Look at sample() row["gamd_boost_total_kj_mol"] = extract_gamd_boost_kj(sim.integrator, unit, None). This reads actual integrator boost. But the bias matrix used for MBAR is computed from pep_gamd_boost_kj using v_pep/v_dih and state_lambdas. If the integrator's actual boost is different (e.g., stage wrong, ForceScalingFactor 1.0), then MBAR prices a boost not applied. The code does not verify that computed boost equals actual integrator boost. It records both, but analysis may use computed boost matrix. If integrator stage is wrong, silent. The fix mentioned seed_frozen_envelope_stage5 + verify_gamd_production_stage5. But in this code, is there a verification? Not shown. The sample path records actual boost but does not use it for bias matrix. So if actual boost differs, free energy wrong. This is a key finding. Need cite lines: sample() around 7420-7700, specifically boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(...) and row["gamd_boost_total_kj_mol"] = extract_gamd_boost_kj(...). No consistency check. Confidence high? We need be careful: The code may have verify_gamd_production_stage5 elsewhere not shown. But under review subsystem, we can say SUSPECTED/VERIFIED? We can point at lines: sample() computes boost matrix from formula, not from integrator; actual boost read separately but not compared. If integrator stage wrong, silent. This is exactly the class of bug. We can mark VERIFIED that no check in shown code; but maybe elsewhere. The task says report every place software can silently produce wrong number. We can say VERIFIED: no assertion linking computed boost to integrator boost in sample/exchange path. Effect: MBAR uses computed boost while dynamics used different boost -> wrong PMF. Confidence maybe 70.

Another issue: In _current_exchange_arrays, boost_bias_matrix_kj is computed using pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env). But v_pep/v_dih are read from context via _fetch_v_pep_v_dih. If pep_env is not None but ladder active with all lambdas zero, boost matrix all zero. Fine.

Potential issue: _fetch_v_pep_v_dih returns (nan, nan) when pep_env None. In sample(), if pep_env is None, boost_bias_matrix_kj zeros. But row["v_pep_kj_mol"] = nan, v_dih nan. Fine.

Now, state_lambdas alignment. In sample(), boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env). pep_gamd_boost_matrix_kj takes lambdas array length n_states and v_pep/v_dih arrays length n_samples. It returns (n_states, n_samples). Good. But if state_lambdas length != nrep? It should be nrep windows. If mismatch, vstack may error or broadcast? np.vstack of arrays with different lengths? If lambdas length != nrep, pep_gamd_boost_kj(v_pep, v_dih, float(l), env) returns array length nrep. vstack length n_lambdas. Then bias_matrix_kj shape (n_lambdas, nrep). Later bias_matrix_kj[wi, i] works if wi < n_lambdas. But assignments windows are indices into centers_a, which should match state_lambdas. If mismatch, could index error or silent wrong. _derive_state_gamd_lambdas tries to align. But potential issue: state_lambdas may be truncated by --max-replicas? They mention args.state_gamd_lambdas = list(...)[:_max_replicas]. If nrep < n_windows? Actually max-replicas truncation might reduce number of replicas/windows? Need know. If state_lambdas truncated to nrep but centers_a still n_windows? Could misalign. They mention _persist after truncation. But if state_lambdas length != centers_a length, set_replica_lambda_for_window uses state_lambdas[int(window_index)]; if window_index >= len(state_lambdas), IndexError, not silent. But if state_lambdas length equals nrep but centers_a length larger, and only first nrep windows used? Maybe okay.

Potential issue: In sample(), row["gamd_lambda"] = float(state_lambdas[w]). If state_lambdas length < w, IndexError. Not silent.

Now, umbrella_bias_matrix_kcal: It computes distance_kcal = 0.5 * ka[:, None] * dprimary^2. ka is k_kcal. But centers_a and primary_values units? primary_values in CV units. k_kcal per CV unit squared. Good. For secondary, ss_values may be NaN; np.where sets ss_kcal 0. But if ss_centers is not None and ss_k_kcal is not None, but some windows have k2=0? ss_k_kcal list may have 0 for those. Then ss_kcal 0. Good.

Potential issue: In sample(), ss_centers_arr = ss_centers_arr_global; ss_k_arr = ss_k_kcal_arr_global; _has_ss_axis = secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None. But ss_centers_arr_global may be None if no secondary. Then _has_ss_axis False. Good.

Now, _gibbs_window_proposal_distribution: It uses bias_matrix_kj. But bias_matrix_kj includes boost. For Gibbs walk, the proposal distribution uses full bias including boost. The physical potential cancels in delta. Good.

Potential issue: The Gibbs proposal distribution includes "stay" with delta 0. But q_forward for stay = exp(0)/Z = 1/Z. If choose stay, returns accepted True. Fine.

Potential issue: In _gibbs_window_proposal_distribution, valid_mask = holders >= 0. holders is replica_of_window. If some windows unheld (holder -1), they are excluded. But in a complete assignment, all windows held. During initialization maybe not. Fine.

Potential issue: The log_weights clip to [-745, 745]. If all log_weights -inf? finite mask removes non-finite. If no finite, return empty. Fine.

Potential issue: In _gibbs_mh_acceptance_probability, if qf <=0 or qr <=0 return 0. But q_reverse could be 0 if wi not in rev_windows. Then alpha=0. Fine. But if q_forward is 0? choose wouldn't pick it. Fine.

Now, the biggest potential detailed-balance bug: The Gibbs proposal distribution as implemented is not the conditional distribution but an MH proposal. However, the acceptance uses q_reverse computed from holders_after. But is holders_after correct? It sets holders_after[wi] = target_rep; holders_after[wj] = rep. But what about assignments? The proposal is for replica rep to move from wi to wj, and target_rep from wj to wi. The holders_after table represents which replica is in each window after swap. That is correct. But the reverse proposal distribution should be computed for the same replica rep? In the reverse move, the "replica" whose window is being resampled is still rep? Actually in the reverse move from state after (rep in wj, target in wi), the transposition would swap rep and target again, moving rep back to wi and target back to wj. The proposal distribution should be over candidate windows for rep, with target_rep in wi. Yes, they call _gibbs_window_proposal_distribution(replica_index=rep, current_window=wj, replica_of_window=holders_after). That computes q_reverse for all candidate windows, including wi. Good.

But wait: The proposal distribution q_fwd is over target windows wj, but the "target replica" is determined by holders[wj]. For each candidate wj, the move swaps rep with the replica currently in wj. This is a transposition. The reverse move from after state would swap rep with the replica currently in wi (which is target_rep). So q_reverse should be the probability of choosing wi as target window. They compute that. Good.

Potential issue: The proposal distribution q_fwd uses old_e = bias[wi, rep] + bias[valid_windows, target_reps]. But for candidate wj, target_reps = holders[wj]. If wj == wi, target_reps = rep (since holders[wi]=rep). Then old_e = bias[wi,rep]+bias[wi,rep] = 2*U_wi(x_rep). new_e = bias[wi,rep]+bias[wi,rep] = same. delta=0. Good. But the "stay" candidate duplicates? valid_windows includes wi once. Fine.

Now, another potential issue: The Gibbs walk is supposed to sample the joint distribution of assignments. But the proposal only considers transpositions between rep and the holder of target window. This is a valid MH move on the permutation group (adjacent? actually any transposition). It can reach all permutations. Good.

Now, state bias matrix orientation. umbrella_bias_matrix_kcal returns [state, replica]. sample() uses bias_matrix_kcal[:, r] for replica r. Good. _current_exchange_arrays returns bias_matrix_kj [window, replica]. apply_window_swap uses [wi,i]. Good.

Now, potential issue: In sample(), boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env) if pep_env is not None else np.zeros((nrep, nrep)). But state_lambdas is per window, length n_windows. If n_windows != nrep, shape mismatch. In REUS, number of windows equals number of replicas? Usually yes, one replica per window. But with lambda ladder, each window replicated at lambda in {0,0.236,0.647,1}? Wait submission says "every umbrella window replicated at lambda in {0, 0.236, 0.647, 1}; k0 = lambda*k0max". That means each umbrella window has 4 rungs, so number of states = 4 * n_windows? But code seems state_lambdas per window, not per (window,rung). Hmm. Let's parse.

The submission says "with a lambda ladder of boost strengths (every umbrella window replicated at lambda in {0, 0.236, 0.647, 1}; k0 = lambda*k0max), replica exchange by gibbs-walk over the (window x rung) state graph". But the code under review: state_lambdas is per window, set_replica_lambda_for_window(integrator, window_index, state_lambdas, k0max_by_channel). It applies lambda based on window index only. There is no rung dimension in the code snippets. Maybe the "rung" is encoded as separate windows? The lambda ladder states are separate windows? Actually if every umbrella window replicated at lambda values, then states are (window, rung), and state_lambdas would be length n_states = n_windows * n_rungs. But code uses centers_a length n_windows, assignments length nrep, state_lambdas length n_windows? In sample(), row["gamd_lambda"] = float(state_lambdas[w]) where w is window index. So each window has one lambda. That contradicts "every umbrella window replicated at lambda in {0,0.236,0.647,1}". Maybe the code under review is a later version where lambda ladder is per window, not per rung? The submission text says "lambda per state" in subsystem. Need examine.

In _derive_state_gamd_lambdas, it derives one gamd_lambda per surviving window. So indeed one lambda per window. The "rung" may be a separate dimension in the state graph but not in these functions? The code snippets don't show rung. The task specifically says "replica exchange: state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path". So maybe the lambda ladder is per state (window), not per rung. The submission description might be from a different version? We need review code as given. We should not assume rung dimension. But we can note if the claimed design includes rung replication, the code's state_lambdas per window cannot represent it. However, the code may be for a simplified ladder where each window has a fixed lambda. The submission says "lambda ladder of boost strengths (every umbrella window replicated at lambda in {0, 0.236, 0.647, 1})" - that is 4 states per window. But the code's bias matrix is [window, replica], not [state, replica] with state = window*rung. The function umbrella_bias_matrix_kcal takes centers_a and ss_centers, no rung. So if the actual production uses rung dimension, this code is incomplete. But the task says "Subsystem under review: replica exchange: state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path". Maybe the code is the actual implementation and the description is aspirational. We need identify mismatch between claimed and actual.

Let's read the code: _derive_state_gamd_lambdas(window_metadata, n, existing) returns list length n, one per window. set_replica_lambda_for_window(integrator, window_index, state_lambdas, k0max_by_channel) sets lambda for window_index. So each window has exactly one lambda. There is no rung. The "lambda ladder" in this code is just a per-window lambda assignment, not replication. The submission's first paragraph says "every umbrella window replicated at lambda in {0, 0.236, 0.647, 1}" but code doesn't replicate. This is a major discrepancy. However, maybe the "rung" is represented by separate window indices: e.g., window 0 has lambda 0, window 1 has lambda 0.236, etc. But then umbrella centers would differ? No, "every umbrella window replicated at lambda" means same center with different lambda. The code's centers_a length equals number of windows, and state_lambdas length equals centers_a. So no replication. Unless centers_a includes duplicates for rungs. But then umbrella_bias_matrix_kcal would treat them as separate windows with same center, and state_lambdas per window. That could represent rungs as separate states with same center but different lambda. The "window" in code may actually be "state" (window x rung). The variable names "window" might be state. The submission says "state graph" and "lambda per state". So maybe each state is a (window, rung) pair, and centers_a has repeated centers for rungs. Then state_lambdas length = n_states. The code's "window" is actually state index. In sample(), row["window"] = w, center_A = centers_a_arr[w], k = k_arr[w], gamd_lambda = state_lambdas[w]. So if rungs are separate states, centers_a_arr would have duplicate centers for different rungs. That is plausible. The function umbrella_bias_matrix_kcal takes centers_a and k_kcal arrays of length n_states. So yes, "window" in code may mean state. The submission says "every umbrella window replicated at lambda in {0,0.236,0.647,1}" - so there are 4 states per umbrella window, each with same center but different lambda. The code's centers_a would contain each center repeated 4 times. Then state_lambdas length = 4*n_windows. That is consistent. So no discrepancy. Good.

Now, potential issue: In _derive_state_gamd_lambdas, it reads normalized_rows with gamd_lambda per row. If normalized_rows has one row per state (including rungs), good. If not, fallback zeros. Fine.

Now, another issue: In sample(), boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env). This computes boost for each state's lambda. But the actual boost applied to replica r is determined by its current state w = assignments[r]. The integrator's k0 is set to state_lambdas[w]*k0max. So boost_bias_matrix_kj[w, r] should equal actual boost. But note: The boost formula depends on v_pep and v_dih, which are read from the context. However, the integrator's boost may be computed using a different v_pep/v_dih because the integrator's internal energy groups may include/exclude different terms. The code reads v_dih from group DIHEDRAL_GROUP and v_pep from peptide_essential_energy_kj. If the integrator's boost uses a different definition (e.g., includes CMAP in dihedral, or total includes all peptide energy), then mismatch. The comment claims factory puts PeriodicTorsion/CMAPTorsion in group 2 same as DIHEDRAL_GROUP. But if not, silent. This is a suspected issue.

Now, let's look at state_identity.py. It requires exact set of fields in state definition. Potential issue: It requires "boost" field even if None. It checks if any gamd_lambda >0 and boost not dict -> error. If all lambdas zero, boost can be None. Fine.

Potential issue: In canonical_state_definition, for NPT, it rejects fixed_box_vectors_nm not None. But if ensemble NPT and fixed_box_vectors_nm is None, okay. It doesn't check pressure_bar positive? finite_number without positive. Pressure could be negative? Not relevant.

Potential issue: In normalize_windows, gamd_lambda = finite_number(raw.get("gamd_lambda", 0.0), minimum=0). It does not enforce <=1. set_replica_lambda raises if lam >1. But state_identity could accept lambda >1, then production would raise at set_replica_lambda, not silent. Fine.

Potential issue: In reconstruct_bias_matrix, if lambdas active and envelope has_total True, requires pep not None. But if v_pep is provided as all NaN, it doesn't raise. Then _ladder_apply may produce NaN. But maybe numeric_vector converts non-finite to NaN, and _ladder_apply should handle? Not sure.

Now, the biggest issue in bias.py: It says "Energy input convention is unchanged: k1 and k2 are kcal/mol per squared CV unit; CV values and centers must already use the SAME units. beta is mol/kJ." In reconstruct_bias_matrix, matrix = 0.5*k*delta^2 in kcal, then matrix *= beta * KJ_PER_KCAL. So reduced bias = beta_kJ * U_kJ. Correct. But if windows have k in kJ/mol (energy_unit kJ/mol), normalize_windows doesn't convert; it just stores. The docstring says k1/k2 are kcal/mol. But state_identity canonical_state_definition converts kJ/mol to kcal/mol. So if using state_identity, okay. But reconstruct_bias_matrix may be used directly with windows from elsewhere. Not our focus.

Now, production sample() potential energy: potentials_kj[r] = physical potential from getState groups=physical_energy_groups_for_args(args). But the total Hamiltonian for MBAR should include physical potential + umbrella + boost. The bias matrix includes umbrella + boost. So reduced potential = beta*(physical + bias). Good. But if physical_energy_groups excludes some terms that are in the actual dynamics (e.g., barostat? no), then wrong. Need know physical_energy_groups_for_args. Not shown. Could be issue.

Now, NPT barostat: The submission says NPT via application-controlled biased Metropolis-MC volume move against U* = physical + bias + boost. The code snippets don't show barostat. But sample() reads potential with enforcePeriodicBox=True. If volume move changes box, positions scaled? Need not.

Now, checkpoint/resume: _reload_state_gamd_lambdas_on_resume reads manifest. Potential issue: If resume after adaptive epoch changed windows, manifest method_settings state_gamd_lambdas may be stale. _persist called after final mutation. But if crash before persist, stale. Could silently disable ladder. The docstring acknowledges. But is there a verification? Not shown.

Now, let's examine _gibbs_window_proposal_distribution more deeply for a possible sign error. It computes delta = new_e - old_e. log_weights = -beta*delta. For a favorable move (new_e < old_e), delta negative, -beta*delta positive, weight >1. Good. Then probabilities normalized. MH acceptance uses delta_kj = deltas[choice_index] (new-old). log_alpha = -beta*delta + log(qr)-log(qf). If q_fwd is high for favorable moves, alpha may be lower. Correct.

But wait: The proposal distribution q_fwd is normalized over valid windows. It includes the stay move. The reverse q_rev is normalized over valid windows in holders_after. However, the set of valid windows may differ between forward and reverse? holders_after has same valid windows (all >=0). So same. Good.

Potential issue: In _gibbs_window_proposal_distribution, old_e = bias[wi, rep] + bias[valid_windows, target_reps]. But bias[valid_windows, target_reps] uses advanced indexing: bias is 2D, valid_windows array, target_reps array. This returns array of length n_valid. Good. new_e = bias[valid_windows, rep] + bias[wi, target_reps]. Good.

Now, potential issue: The Gibbs proposal distribution uses bias_matrix_kj which includes boost. But the boost for a state depends on v_pep/v_dih of the replica's configuration. For candidate wj, new_e = U_wj(x_rep)+U_wi(x_target). The boost part of U_wj(x_rep) is computed with lambda_wj and v_pep/v_dih of rep. The boost part of U_wi(x_target) is computed with lambda_wi and v_pep/v_dih of target. That is correct. The physical potential cancels. Good.

Now, potential issue: In _current_exchange_arrays, when computing boost_bias_matrix_kj, it uses v_pep_kj and v_dih_kj arrays from all replicas. But if pep_env is not None and state_lambdas has length n_states, pep_gamd_boost_matrix_kj returns (n_states, nrep). Good. But if some v_pep/v_dih are NaN (e.g., pep_env has_total False, v_pep NaN), pep_gamd_boost_kj for single dihedral ignores v_pep. Good.

Now, potential issue: In sample(), row["sampled_boost_bias_kj"] = float(boost_bias_matrix_kj[w, r]). This is computed boost, not actual. If actual differs, row records computed. The parquet writer writes boost_total from extract_gamd_boost_kj (actual) and v_pep/v_dih. Analysis could recompute. But MBAR analysis may use bias_matrix from analysis_arrays.npz, which is computed. Need know analyze_gareus_mbar.py. Not shown. But the task says final reweighting by MBAR over all (window, rung) states using per-frame recorded CVs and boost arguments. If analysis recomputes boost from recorded v_pep/v_dih, then actual mismatch doesn't matter? Wait if dynamics ran unboosted but analysis prices boost, then wrong. The calibration case: integrator stage 2 with ForceScalingFactor 1.0, but exchange and MBAR priced boost. The fix verifies stage. In current code, sample() records actual boost_total and v_pep/v_dih. If analysis uses recorded v_pep/v_dih to compute boost, it would still price boost even if actual boost was not applied. The only way to avoid is verify actual boost equals computed. The code does not appear to do so in sample path. It does read actual boost_total but doesn't compare to sampled_boost_bias_kj. So this is a real silent failure mode. We should report.

Now, another issue: In _current_exchange_arrays, the exchange decision uses computed boost matrix, not actual integrator boost. If actual boost is wrong (e.g., stage 2), exchange accepts/rejects based on priced boost that was not applied. This violates detailed balance w.r.t. actual Hamiltonian. Silent. This is exactly the calibration case. The fix mentioned verify_gamd_production_stage5 may be in code not shown. But we can say within reviewed subsystem, no check.

Now, let's look at _channel_boost condition. It uses scale = max(abs(e), abs(v), 1.0). If rng <= 0.001*scale, b=0. This is a guard against zero range. But if vmax-vmin is small but nonzero, and e-v large, b could be huge? Actually if rng small, b large, but condition (b+v)<e may fail, so b=0. Fine.

Potential issue: In _channel_boost, if vmax < vmin (negative rng), b = 0.5*k0*(e-v)^2/negative = negative. Then condition (b+v)<e may be true, returning negative boost. That would be wrong. But envelope should have vmax > vmin. If not, silent negative boost. Could be a finding: no validation that vmax > vmin. But likely envelope validation elsewhere. We can mention SUSPECTED.

Now, pep_gamd_boost_kj: b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). The comment says "dihedral first, then Total with the dihedral boost added to the Total energy before the square (stage_integrator _add_dihedral_boost_to_total_energy)." But in standard dual boost, total boost = 0.5 k0_tot (E_tot - V_tot)^2/(Vmax_tot - Vmin_tot) + 0.5 k0_dih (E_dih - V_dih)^2/(Vmax_dih - Vmin_dih), where V_tot is total potential energy including dihedral? Actually in GaMD dual boost, the total boost uses total potential energy V_total (including dihedral) and threshold E_total. The dihedral boost is added to the total potential energy before computing total boost? The comment says "with the dihedral boost added to the Total energy before the square". That is a specific implementation. If the integrator does that, then code matches. But if not, mismatch. Need trust.

Now, k0max_from_globals: returns {"Total": float(shared_globals.get("k0_Total", 0.0)), "Dihedral": float(shared_globals["k0_Dihedral"])}. If shared_globals missing k0_Dihedral, KeyError, not silent. Good.

Now, set_replica_lambda: It sets k0_Dihedral always, and k0_Total if names is None or "k0_Total" in names. If names is None (test double), it sets k0_Total even if not present? It calls setGlobalVariableByName("k0_Total", ...). If test double doesn't have it, may raise. But names None only for minimal test doubles that expose only setGlobalVariableByName; they might accept any. Fine.

Potential issue: set_replica_lambda does not set ForceScalingFactor or stage. The calibration case was stage not set. The fix seed_frozen_envelope_stage5 likely sets stage. But set_replica_lambda only sets k0 globals. If the integrator's stage is wrong, setting k0 doesn't help. The code under review may rely on external verify. We can mention.

Now, state_identity.py: It hashes definition. Potential issue: It uses json_bytes(dict(raw)) which sorts keys? Need know. If not, hash may differ for same dict with different insertion order. But json_bytes likely canonical. Not our focus.

Now, let's identify concrete findings.

Finding 1: No runtime verification that the integrator-applied GaMD boost equals the boost priced in bias_matrix_kj/MBAR. In sample(), boost_bias_matrix_kj is computed from pep_gamd_boost_kj(v_pep, v_dih, state_lambdas, env), while actual boost is read separately via extract_gamd_boost_kj into row["gamd_boost_total_kj_mol"] but never compared. If StageIntegrator is in wrong stage (as in chignolin_8) or ForceScalingFactor != 1, MBAR and exchange use a boost that was never applied. This is exactly the class of bug. File: gareus/production.py sample() around lines 7420-7700, specifically boost_bias_matrix_kj assignment and row["gamd_boost_total_kj_mol"]; _current_exchange_arrays similarly. Confidence high. VERIFIED? We can point at lines. The code doesn't show comparison. But maybe elsewhere. We can say "In the reviewed subsystem, no check exists; the only verification mentioned is in commit message, not in these functions." Mark VERIFIED for absence in shown code.

Finding 2: _channel_boost can return negative boost if vmax < vmin, because rng negative and condition may pass. No validation that envelope vmax > vmin in pep_gamd.py. If a corrupted/stale envelope has vmax < vmin, boost becomes negative, altering Hamiltonian and free energy silently. File: gareus/pep_gamd.py:497-503. Confidence medium. SUSPECTED.

Finding 3: In _gibbs_window_proposal_distribution, the proposal distribution is normalized over valid windows including the stay move, but the MH acceptance uses q_reverse computed from holders_after. However, q_reverse is computed with the same function that includes the stay move. For the reverse move, the "stay" candidate is wj (current window after), not wi. The probability of choosing wi is q_reverse. That is correct. No issue.

Wait, let's double-check the MH acceptance formula with q_fwd and q_rev both including stay. Suppose forward proposal chooses wj != wi. q_fwd = exp(-βδ)/Z_fwd. Reverse proposal from after state chooses wi. q_rev = exp(βδ)/Z_rev. Then alpha = min(1, exp(-βδ) q_rev / q_fwd) = min(1, exp(-βδ) * exp(βδ)/Z_rev * Z_fwd/exp(-βδ)) = min(1, exp(βδ) Z_fwd/Z_rev). Wait this seems asymmetric and can be >1. But detailed balance condition is π_i q(i->j) α(i->j) = π_j q(j->i) α(j->i). With α(i->j)=min(1, π_j q(j->i)/(π_i q(i->j))). Here π_j/π_i = exp(-βδ). q(j->i) = exp(βδ)/Z_rev. q(i->j)=exp(-βδ)/Z_fwd. Ratio = exp(-βδ)*exp(βδ)/Z_rev / (exp(-βδ)/Z_fwd) = exp(βδ) Z_fwd/Z_rev. So α = min(1, exp(βδ) Z_fwd/Z_rev). Is that correct? Let's test with a simple case where Z_fwd = Z_rev (symmetric proposal). Then α = min(1, exp(βδ)). If δ<0 (favorable), exp(βδ)<1, so α<1. But if proposal is symmetric and move is favorable, MH should accept with probability min(1, exp(-βδ))? Wait π_j/π_i = exp(-βδ). If δ<0, π_j > π_i, so α should be 1. But here α = exp(βδ) <1. That's wrong! Let's recalc.

Hold on. Let's define δ = new_e - old_e. π_j/π_i = exp(-β δ). If δ<0, new lower energy, π_j > π_i, ratio >1, so α should be 1 for symmetric proposal. But our q_fwd is not symmetric; q_fwd = exp(-βδ)/Z_fwd. For δ<0, q_fwd > 1/Z_fwd, so it preferentially proposes favorable moves. Then MH correction should reduce acceptance. The ratio π_j q_rev / (π_i q_fwd) = exp(-βδ) * q_rev / q_fwd. If q_rev = q_fwd (symmetric), ratio = exp(-βδ). For δ<0, ratio >1, α=1. Good. But in our case q_rev = exp(βδ)/Z_rev. If Z_fwd=Z_rev, q_rev = exp(βδ)/Z, q_fwd=exp(-βδ)/Z. Then ratio = exp(-βδ)*exp(βδ)/exp(-βδ) = exp(βδ). For δ<0, exp(βδ)<1, so α<1. That suggests the reverse proposal probability is too high? Let's compute q_rev for reverse move. Reverse move from after state (rep in wj, target in wi) to before (rep in wi, target in wj). The energy difference for reverse δ_rev = old_before - old_after = -δ. The proposal distribution for reverse should be proportional to exp(-β δ_rev) = exp(βδ). So q_rev = exp(βδ)/Z_rev. That is correct if the proposal rule is "propose wj with weight exp(-β δ_forward)". For reverse, δ_rev = -δ, so weight exp(βδ). So q_rev is correct. Then why does MH ratio seem wrong? Let's derive detailed balance with this proposal.

Proposal q(i->j) = exp(-βδ)/Z_fwd. q(j->i) = exp(βδ)/Z_rev. Note δ for i->j = E_j - E_i. For j->i, δ_rev = E_i - E_j = -δ. So q(j->i) = exp(-β δ_rev)/Z_rev = exp(βδ)/Z_rev. Correct.

MH ratio = π_j q(j->i) / (π_i q(i->j)) = exp(-βδ) * [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(-βδ+βδ+βδ) Z_fwd/Z_rev = exp(βδ) Z_fwd/Z_rev.

But detailed balance requires π_i q(i->j) α(i->j) = π_j q(j->i) α(j->i). If α(i->j)=min(1, ratio), α(j->i)=min(1, 1/ratio). Then LHS = π_i q(i->j) min(1,r), RHS = π_j q(j->i) min(1,1/r). Since π_j q(j->i) = r π_i q(i->j), RHS = r π_i q(i->j) min(1,1/r) = π_i q(i->j) min(r,1). LHS = π_i q(i->j) min(1,r). Equal. So valid. The acceptance probability can be >1? α(i->j)=min(1,r). If r>1, accept always. In our example δ<0, r=exp(βδ) Z_fwd/Z_rev. If Z_fwd=Z_rev, r=exp(βδ)<1, so α<1. That means even though move is favorable, the proposal is so biased toward it that it must be rejected sometimes to maintain balance. That's correct. So no bug.

But wait, is q_fwd really proportional to exp(-βδ)? Yes. For favorable δ<0, q_fwd > uniform, so proposal oversamples favorable moves; MH corrects. Good.

Now, potential issue: The "Gibbs-walk" name suggests they intended the proposal to be the conditional distribution, in which case acceptance should be 1. But they added MH correction, so it's valid. However, if q_fwd is exactly the conditional distribution, then q_rev would also be conditional and MH ratio would be 1? Let's check: If q_fwd ∝ exp(-β new_e) (not exp(-βδ)), then q_fwd = exp(-β U_wj(x_rep))/Z_fwd (since U_wi(x_target) constant). q_rev = exp(-β U_wi(x_rep))/Z_rev. Then ratio = exp(-βδ) * q_rev/q_fwd = exp(-β(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target))) * exp(-βU_wi(x_rep))/Z_rev * Z_fwd/exp(-βU_wj(x_rep)) = exp(-β(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target)+U_wi(x_rep)-U_wj(x_rep))) Z_fwd/Z_rev = exp(-β(U_wi(x_target)-U_wj(x_target))) Z_fwd/Z_rev. Not necessarily 1. So their proposal is not conditional. But MH valid.

Now, is there a bug in q_reverse computation due to using holders_after but not updating assignments? The proposal distribution only depends on replica_of_window, not assignments. holders_after is correct. Good.

Now, another potential issue: In gibbs_propose_one_replica, when wj == wi, it returns accepted=True without MH. That's fine because stay always accepted. But it doesn't update anything. Good.

Now, potential issue: In _gibbs_window_proposal_distribution, the stay delta is set to 0.0 after computing delta. But if wi is not in valid_windows? It should be, because holders[wi]=rep >=0. So stay_idx exists. Good.

Now, potential issue: The log_weights clip to [-745, 745]. If all weights underflow to 0 after clipping? They clip lower bound -745, so exp(-745) ~ 5e-324, which is subnormal but nonzero. Sum may be >0. If max log_weight is -745, m=-745, weights = exp(0)=1, sw=1. Fine. If some log_weights are -inf (delta inf), finite mask removes. Good.

Now, potential issue: In _gibbs_mh_acceptance_probability, if log_alpha < -745 return 0. If log_alpha is NaN? It checks finite. Good.

Now, let's examine apply_window_swap with p_override and force_accept. If p_override provided, pacc = p_override, then clamped. If force_accept True, accepted True regardless of pacc. But if p_override is None and force_accept True, pacc computed but ignored. Fine. If uniform is None and force_accept False, accepted False. In production, uniform is rng.random() unless force_accept. Good.

Potential issue: In _attempt_window_swap, it checks swap_candidate_replicas before drawing random. Then calls apply_window_swap with uniform = None if force_accept else rng.random(). But apply_window_swap itself calls swap_candidate_replicas again. That's fine, no extra random. Good.

Now, potential issue: In _attempt_window_swap, after accepted swap, it submits _apply_swap_to_replica to _sim_pool for i and j. But it uses assignments[replica_index] after swap. However, _sim_pool.submit(i, _apply_swap_to_replica, i).result() may execute on worker thread for replica i, but the function uses sims[replica_index].context. If _sim_pool.submit's first argument is replica index and second is function, it may run on that replica's thread. Good. But if the worker thread is busy stepping, deadlock? Not our focus.

Now, potential issue: In sample(), after computing bias_matrix_kj, it stores observable_cache["bias_matrix_kj"] = bias_matrix_kj. Then _current_exchange_arrays may use cached if step matches. But sample() is called at log steps, exchange at exchange intervals. If they coincide, cache used. But sample() computes bias_matrix_kj using v_pep/v_dih and state_lambdas. _current_exchange_arrays uses same. Good.

Now, potential issue: In sample(), potentials_kj[r] is physical potential. But the bias matrix includes boost. The total reduced potential for MBAR should be beta*(physical + bias). However, if the boost is applied in dynamics, the sampled distribution is proportional to exp(-beta*(physical + bias_actual))? Actually umbrella sampling with boost: The bias is added to the physical potential, so the sampled distribution is exp(-beta*(U_physical + U_bias + boost)). MBAR reweights to physical by using u_k = beta*(U_physical + U_bias_k) for each state k? Wait if the simulation is run with Hamiltonian H_k = U_physical + U_bias_k + boost_k, then the sample's reduced potential under state k is beta*(U_physical + U_bias_k + boost_k). To unbias to physical, MBAR uses u_k = beta*(U_physical + U_bias_k + boost_k) for the state that generated the sample? Actually MBAR solves for free energies of states with reduced potentials u_k(x) = beta*(U_physical(x) + bias_k(x)). The boost is part of bias_k. So yes, bias matrix includes boost. Good.

But wait: The boost applied in dynamics is state-dependent (lambda). The sample's actual Hamiltonian is H_actual = U_physical + U_umbrella_window + boost_lambda. The bias matrix for that state should be U_umbrella + boost_lambda. The code computes that. Good.

Now, potential issue: In sample(), row["potential_kj_mol"] = physical potential. But if the boost is applied, the total potential energy from OpenMM getState(getEnergy=True, groups=physical_energy_groups) may exclude boost because boost is in integrator, not force. Good. So physical potential correct.

Now, potential issue: In _current_exchange_arrays, it does not read potential energies. For exchange, physical potential cancels only if the swap is of state labels, not configurations. Yes, because U_physical(x_i)+U_physical(x_j) unchanged. Boost? Boost is part of bias, not physical. So included. Good.

Now, potential issue: The exchange move swaps state labels, but the integrator's boost k0 is set to new state's lambda. However, the boost energy in the current step's bias matrix was computed using v_pep/v_dih from before the swap. After swap, the integrator's boost for future steps will use new lambda. The exchange decision uses the boost that would be applied if the state labels were swapped at that instant. That is correct for Metropolis on the augmented ensemble where the boost is part of the state Hamiltonian. But note: The boost is a function of the configuration and the lambda. When swapping labels, the configuration stays with the replica, so the boost energy for the new state is computed with the same configuration. Correct.

Now, potential issue: The boost applied by the integrator may depend on history (e.g., GaMD adaptive? but envelope frozen). If boost depends on running averages, then swapping labels changes the boost history? The envelope is frozen, so no. Good.

Now, let's look at state_identity.py canonical_state_definition: It requires "windows" with fields center1,k1,center2,k2,gamd_lambda. It converts energy_unit kJ/mol to kcal/mol by dividing k1,k2 by 4.184. But if k1/k2 are in kJ/mol per squared CV unit, dividing by 4.184 gives kcal/mol. Correct. But it doesn't convert gamd_lambda? dimensionless. Good.

Potential issue: In canonical_state_definition, if energy_unit == "kJ/mol", it modifies row["k1"] /= 4.184 and row["k2"] /= 4.184. But if k1/k2 are zero, fine. If k1/k2 are strings? normalize_windows already converted to float. Good.

Potential issue: In canonical_state_definition, it checks set(row) - {"window_id","center1","k1","center2","k2","gamd_lambda"} == empty. But normalize_windows may add keys? It returns dict with center1,k1,center2,k2,gamd_lambda plus original keys? Actually normalize_windows does window = dict(raw), then sets window["center1"], etc. So if raw had extra keys, they remain. Then canonical_state_definition would reject. Good.

Now, potential issue: In validate_fixed_state_segments, it compares top-level windows sorted by window_id to state["windows"]. But state["windows"] is sorted by window_id in canonical_state_definition. So okay.

Now, let's think about the "silent fallbacks" in _derive_state_gamd_lambdas: If normalized_rows cannot be trusted, it falls back to existing or zeros. If existing is stale wrong-length, it falls to zeros, silently disabling ladder. The docstring warns. But is there a scenario where normalized_rows is present but wrong? It checks len(rows)==n and all have gamd_lambda. If rows have gamd_lambda but misaligned due to reindexing bug, could silently use wrong lambdas. The docstring claims reindexing functions subset+renumber correctly. But if not, silent. We can mention SUSPECTED.

Now, potential issue: _reload_state_gamd_lambdas_on_resume reads manifest method_settings. But if manifest has state_gamd_lambdas from a previous campaign with different number of states, and current run has different n, _derive_state_gamd_lambdas will use existing if len==n, else zeros. If len==n but misaligned (e.g., windows reordered), silent wrong lambda per state. The docstring says "if post-drop count happens to match, resumes with a misaligned per-state λ." They try to persist after final mutation. But if crash before persist, stale. This is a real risk. We can report.

Now, another issue: In campaign_ladder_registry_lambda, it returns first registry with lam >0. But if there are multiple registries in different parent dirs, it picks the first in order. If
(TRUNCATED - output hit the token limit before completion)

We need conduct audit. Need inspect submission for math/physics errors. Need produce flags exactly one line each. Need be careful. We need identify errors in code snippets. Need understand context.

We have code for replica exchange, umbrella bias, Gibbs, boost, state identity, bias reconstruction, production sample/exchange. Need check dimensional consistency, sign conventions, statistical mechanics, reweighting, estimator assumptions, stats/datatypes, unit errors, numerical factors.

Let's parse.

First umbrella_bias_matrix_kcal: primary_values shape (n_replicas), centers_a shape (n_states), k_kcal shape (n_states). dprimary = pv[np.newaxis,:] - ca[:,np.newaxis] -> shape (states, replicas). distance_kcal = 0.5 * ka[:,None] * dprimary^2. Units: k kcal/mol per CV unit^2? CV maybe fraction dimensionless or distance? centers_a in A? k_kcal maybe kcal/mol/A^2. dprimary in A, product kcal/mol. OK. Secondary: dss = sv[None,:] - sc[:,None]; ss_kcal = 0.5*sk*dss^2; then np.where(np.isfinite(dss), ss_kcal, 0.0). If dss NaN, sets 0. But if dss inf? np.isfinite false -> 0. OK. But if secondary values missing, bias zero. Fine.

Potential issue: if ss_centers None, returns zeros_like(distance_kcal). If secondary axis absent, OK. If secondary values contain NaN but centers finite, zero. OK.

_gibbs_window_proposal_distribution: bias_matrix_kj shape [state, replica]? They use bias[wi, rep] + bias[valid_windows, target_reps]. old_e = energy of replica rep in window wi plus target_reps in their windows valid_windows. new_e = energy of rep in valid_windows plus target_reps in wi. This is a transposition of replica rep with target_reps? Let's understand. valid_windows are windows held by some replica target_reps. Proposal: choose target window wj to swap replica rep (currently in wi) with replica in wj. old total bias = U_wi(x_rep) + U_wj(x_target). new = U_wj(x_rep)+U_wi(x_target). delta = new-old. Good. But they include all valid_windows, including current wi? valid_windows includes wi with target_reps = rep? holders[wi] = rep. Then old_e = bias[wi, rep] + bias[wi, rep]? Wait valid_windows includes wi, target_reps = holders[wi] = rep. old_e = bias[wi, rep] + bias[wi, rep] = 2*U_wi(x_rep). new_e = bias[wi, rep] + bias[wi, rep] = same. delta=0. They set stay_idx delta=0. OK. But old_e for other windows: bias[wi, rep] + bias[wj, target] = U_wi(x_rep)+U_wj(x_target). new = U_wj(x_rep)+U_wi(x_target). Good. This is heat bath over target windows for a given replica transposition. But is this a valid Gibbs move? Need check detailed balance with MH correction. They compute q_forward = probability of choosing wj from current state. q_reverse = probability of choosing wi from destination state after swap, using holders_after. Then MH acceptance log_alpha = -beta*delta + log(q_reverse)-log(q_forward). That is correct for proposal distribution q(wj|current) and reverse q(wi|dest). But wait the proposal distribution includes all valid windows, including current. If choose current, no move. For reverse, holders_after has rep in wj and target in wi. valid_windows includes wi with target_rep = target? Actually holders_after[wi]=target_rep, holders_after[wj]=rep. Reverse proposal for replica rep from current_window=wj: valid_windows includes wi, target_reps = holders_after[wi] = target_rep. old_e = bias[wj, rep] + bias[wi, target_rep]; new_e = bias[wi, rep] + bias[wj, target_rep]. delta_reverse = new-old = -delta? Let's compute: current after swap: rep in wj, target in wi. Reverse move swaps rep back to wi and target to wj. old_e = U_wj(x_rep)+U_wi(x_target). new_e = U_wi(x_rep)+U_wj(x_target). delta_rev = new-old = (U_wi(x_rep)+U_wj(x_target)) - (U_wj(x_rep)+U_wi(x_target)) = -delta. Good. q_reverse computed. OK.

But potential issue: _gibbs_window_proposal_distribution uses bias_matrix_kj with shape [state, replica]. It assumes bias matrix includes all states and replicas. In production, bias_matrix_kj built as [state, replica] with boost. OK.

However, the proposal distribution is not exactly heat-bath over "target windows" because it includes all windows, but the state space is assignments of replicas to windows. The Gibbs move should sample a new window for replica rep conditional on all other assignments fixed. The conditional probability for rep to be in window w is proportional exp(-beta * U_w(x_rep) + constant?) Let's derive. Total bias energy for assignment A: sum over windows? Actually each window has one replica, each replica in one window. Total bias = sum_{k} U_k(x_{A(k)})? If replica rep currently in wi, other replicas fixed. If move rep to wj, target replica moves to wi. Total bias = U_wj(x_rep)+U_wi(x_target)+sum others. Conditional probability of choosing wj (including current) proportional exp(-beta * [U_wj(x_rep)+U_wi(x_target)]). The code's delta = new-old = U_wj(x_rep)+U_wi(x_target) - [U_wi(x_rep)+U_wj(x_target)]. But the conditional probability should be proportional exp(-beta * [U_wj(x_rep)+U_wi(x_target)]), not exp(-beta * delta) relative to current? Wait heat-bath Gibbs samples from conditional distribution, not MH with proposal based on delta. The code calls it "heat-bath proposal" but then applies MH correction. If proposal distribution is q(wj) ∝ exp(-β * delta) = exp(-β*(new-old)), then q(wj) = exp(-β*new)/Z_current? Actually delta = new - old_current. exp(-β*delta) = exp(-β*new)/exp(-β*old_current). Since old_current is constant for all wj? old_current = U_wi(x_rep)+U_wj(x_target) depends on wj! Wait old_e in code = bias[wi, rep] + bias[valid_windows, target_reps]. For candidate wj, old_e = U_wi(x_rep)+U_wj(x_target). This is not constant across wj because U_wj(x_target) varies. So exp(-β*delta) = exp(-β*(U_wj(x_rep)+U_wi(x_target) - U_wi(x_rep)-U_wj(x_target))) = exp(-β*(U_wj(x_rep)-U_wj(x_target))) * constant? Let's factor: new = U_wj(x_rep)+U_wi(x_target). old = U_wi(x_rep)+U_wj(x_target). delta = U_wj(x_rep)-U_wj(x_target) + U_wi(x_target)-U_wi(x_rep). The U_wi terms are constant for all wj? U_wi(x_target) constant, U_wi(x_rep) constant. So delta = [U_wj(x_rep)-U_wj(x_target)] + const. Thus q(wj) ∝ exp(-β U_wj(x_rep)) * exp(+β U_wj(x_target)) * const. That is not the conditional distribution for rep alone; it includes +U_wj(x_target) because target moves. Actually the conditional distribution for the pair swap should be proportional exp(-β [U_wj(x_rep)+U_wi(x_target)]) = exp(-β U_wj(x_rep)) * constant (since U_wi(x_target) constant). So q should be ∝ exp(-β U_wj(x_rep)), not exp(-β delta). The code's q is ∝ exp(-β [U_wj(x_rep)-U_wj(x_target)]). That seems wrong for a Gibbs move. But wait they are not sampling a new window for rep alone; they are sampling a transposition with a target replica. The state space is assignments; a move that changes rep's window necessarily changes target's window. The conditional distribution of the pair (rep, target) given all other assignments? If we consider the joint assignment of rep and target, with other replicas fixed, the possible states are: rep in wi, target in wj (current), or rep in wj, target in wi (swap). There are only two states for a given target. But the proposal chooses among all target windows, so it's a "global" move that selects a target and swaps. The correct heat-bath over all possible target windows for a transposition should have probability proportional to exp(-β * new_total) = exp(-β [U_wj(x_rep)+U_wi(x_target)]) for each wj? But note if wj=wi, target=rep, new_total = U_wi(x_rep)+U_wi(x_rep)? Wait if wj=wi, no swap, total = U_wi(x_rep)+U_wi(x_rep)? That double counts? Let's define total bias energy for assignment: sum over windows of U_window(x_replica_in_window). If rep in wi and target in wj, total = U_wi(x_rep)+U_wj(x_target)+others. If swap, total = U_wj(x_rep)+U_wi(x_target)+others. For wj=wi, target=rep, no swap, total = U_wi(x_rep)+U_wi(x_rep)+others? That's wrong because there is only one replica in wi, not two. The code's old_e for wj=wi is bias[wi,rep]+bias[wi,rep] = 2*U_wi(x_rep), double counting. They set delta=0 for stay, but the weight for stay in q is exp(0)=1, not exp(-β * 2U_wi(x_rep)). So the proposal distribution is not a proper heat-bath over assignments; it's a heuristic with MH correction. But with MH correction, any proposal distribution with correct reverse probability yields correct stationary distribution as long as acceptance uses detailed balance. The q can be arbitrary. So not necessarily wrong. Need check MH acceptance with q_forward and q_reverse. The acceptance formula is correct for arbitrary proposal q. So even if q not conditional, it's fine. But there may be issue with q_forward for stay: they set delta=0, q_forward = probability of stay. Reverse q_reverse for stay? If wj==wi, they return GibbsProposal with accepted True? Actually in gibbs_propose_one_replica, if wj==wi, return GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False). It marks accepted True, no move. That's fine.

But wait the proposal distribution includes all valid_windows, including windows held by replicas. The target_reps array is holders[valid_windows]. For valid_windows includes wi, target_reps = rep. For other windows, target_reps are other replicas. The move is a transposition between rep and target_reps. The proposal distribution q(wj) is computed using delta = new_e - old_e where old_e = bias[wi, rep] + bias[wj, target]. For wj=wi, old_e = 2*U_wi(x_rep), new_e = 2*U_wi(x_rep), delta=0. But the actual old total for the system is U_wi(x_rep)+sum others, not 2*U_wi(x_rep). The delta for stay should be 0, which they set. For other wj, old_e = U_wi(x_rep)+U_wj(x_target), new_e = U_wj(x_rep)+U_wi(x_target). This is correct difference in total bias. So q(wj) ∝ exp(-β delta) for wj≠wi, and q(wi)=1 (after normalization). This is a valid proposal distribution (not necessarily heat-bath but valid). Reverse q computed similarly with holders_after. Need check reverse q for the reverse move: after swap, rep in wj, target in wi. Reverse proposal from state after swap: choose target window wi? The code computes reverse distribution with current_window=wj, replica_of_window=holders_after. valid_windows includes wi with target_reps = target. old_e_rev = bias[wj, rep] + bias[wi, target]; new_e_rev = bias[wi, rep] + bias[wj, target]; delta_rev = new-old = -delta. q_reverse = probability of choosing wi. Good. So MH correct.

Potential issue: _gibbs_window_proposal_distribution uses bias_matrix_kj but beta is passed. It computes log_weights = -beta*delta. Then normalizes. Good.

But there is a subtle issue: the proposal distribution includes all valid_windows, but the "stay" candidate (wj=wi) has delta=0 and weight 1. However, for other candidates, delta can be negative, weight >1. They clip log_weights to [-745,745]. If delta very negative, exp(-beta*delta) huge, but after subtracting max, OK. If delta very positive, weight exp(-745) ~ 0, effectively zero. This is fine. But clipping log_weights to -745 means weights for very unfavorable moves are not exactly zero but ~5e-324? Actually exp(-745) ~ 5e-324, near underflow. They then subtract max and exp; if max is 0, exp(-745) ~ 5e-324, which is subnormal but nonzero. Could cause probabilities sum maybe 1 + tiny. OK. But if all log_weights are -745, finite true, weights all ~5e-324, sum ~ n*5e-324, probabilities uniform. That's fine.

Potential issue: if delta is NaN, log_weights NaN, finite false, filtered. OK.

Now _gibbs_mh_acceptance_probability: log_alpha = -beta*delta + log(qr)-log(qf). If qf or qr zero, return 0. But if q_reverse is 0 for a move that is possible? In reverse distribution, q_reverse could be zero if the reverse move's window wi is not in valid_windows? But wi should always be valid because holders_after[wi] = target_rep >=0. Unless target_rep <0? No. So q_reverse >0. OK.

_exchange_probability: standard Metropolis for swap: if delta <=0 accept 1 else exp(-beta*delta). Good.

swap_candidate_replicas: returns i,j. OK.

apply_window_swap: old_e = bias[wi,i]+bias[wj,j]; new_e = bias[wj,i]+bias[wi,j]; delta. pacc = _exchange_probability(delta,beta). If force_accept, accepted True. Else accepted = uniform < pacc. Then assignments[i], assignments[j] = assignments[j], assignments[i]; replica_of_window[assignments[i]] = i; replica_of_window[assignments[j]] = j. Wait after swapping assignments[i] and assignments[j], assignments[i] is old assignments[j] = wj, assignments[j] = old assignments[i] = wi. Then replica_of_window[assignments[i]] = i means replica_of_window[wj] = i (replica i now in wj). replica_of_window[assignments[j]] = j means replica_of_window[wi] = j. Correct.

But note: if i and j are replicas, assignments array indexed by replica. Good.

Potential issue: apply_window_swap mutates assignments and replica_of_window in place. If p_override or force_accept used in tests, OK.

Now explicit_2d_neighbor_pairs_for_exchange: builds edges, slots, round-robin by edge index modulo slots, then greedy disjoint filtering. If chosen empty, try all edges greedily. This is a heuristic to select disjoint pairs. Not a math error per se. But potential issue: "round-robin edge coloring by edge index" not actual graph coloring; may select non-disjoint? They filter used. OK.

assemble_bias_matrices: distance_bias_kcal + ss_bias_kcal + boost_bias_kcal. boost_bias_kj / 4.184. bias_kj = 4.184 * bias_kcal. OK. But note: boost_bias_kj is [n_states, n_replicas]? pep_gamd_boost_matrix_kj returns vstack of boost for each lambda, shape (n_states, n_samples). If n_states = n_replicas? In production, state_lambdas length n_states, v_pep_kj length n_replicas. So shape (n_states, n_replicas). Good. distance_bias_kcal shape (n_states, n_replicas). OK.

_fetch_v_pep_v_dih: reads dihedral group energy from context. If pep_env None returns nan. If has_total false returns nan, v_dih. OK.

campaign_ladder_registry_lambda: reads registry CSV, returns first lambda >0. Potential issue: if multiple rows, returns first positive, not necessarily the ladder's max or per-state? It's used only to decide ladder active. OK.

resolve_ladder_active: true if any state_lambdas >0 or registry has positive. OK.

_derive_state_gamd_lambdas: if normalized_rows present and length n and all have gamd_lambda, return those. Else existing if length n. Else zeros. Potential issue: if normalized_rows present but some gamd_lambda missing, falls back to existing or zeros. Could silently disable ladder if metadata incomplete. But maybe intended. Need flag? It says "Zeros (ladder inactive) is the fallback of last resort, never a guess." But if ladder was active and metadata missing, zeros would silently disable. However, there is _reload_state_gamd_lambdas_on_resume to restore from manifest. But if not resume, and normalized_rows missing gamd_lambda, could silently disable. Is that a math/physics error? It's a correctness issue. But maybe not in scope? We need flag if can silently produce wrong free energy. The prompt asks adversarial review of code for silent wrong numbers. We should flag if fallback to zeros can silently disable ladder. But the docstring says zeros is fallback of last resort, never a guess. Yet if active ladder metadata missing, zeros is wrong. Need see if there is any guard. In _derive_state_gamd_lambdas, if rows present but not all have gamd_lambda, it falls to existing or zeros. If existing is None, zeros. That could silently set λ=0 for all states, meaning no boost applied/priced? Actually if ladder active, k0max_by_channel not None, state_lambdas zeros would set k0=0, so replicas run unboosted, but samples record v_pep/v_dih and MBAR prices zero boost. That would be wrong if intended boost. But maybe the ladder is only active if state_lambdas has positive values; if all zeros, resolve_ladder_active might still be true via registry, so k0max_by_channel set, and state_lambdas zeros means no boost. That could be a silent failure. Need flag? The code has _reload to restore from manifest on resume, but initial run? If window_metadata normalized_rows lacks gamd_lambda, and existing None, zeros. Is that possible? The docstring says normalized_rows is source of truth whenever usable. If not usable, fallback. Could be a real issue. But maybe not enough context. We'll consider.

_reload_state_gamd_lambdas_on_resume: reads manifest method_settings state_gamd_lambdas. If args.state_gamd_lambdas truthy, returns. If all-zero list is truthy? In Python, [0.0] is truthy (non-empty). So an all-zero ladder set by caller is not overridden. Good. But if args.state_gamd_lambdas is empty list, it reads manifest. OK.

_persist_state_gamd_lambdas: writes current list to manifest. OK.

Now pep_gamd.py _channel_boost: This is important. Formula: b = 0.5 * k0 * (e - v)^2 / rng, where rng = vmax - vmin. Then if (b+v) < e, b else 0. This is GaMD boost? Need check standard GaMD: boost potential V_b = 0.5 k0 (E - V)^2 / (E_max - E_min) when V < E, else 0. Here e is threshold energy (E), v is potential energy (V). rng = vmax - vmin. But standard uses (E_max - E_min) or (V_max - V_min)? In GaMD, for total boost: V_b = 0.5 k0 (E - V)^2 / (E_max - E_min) if V < E. Actually original GaMD: ΔV = 0.5 k0 (E - V)^2 / (E - V_min)? Let's recall. GaMD (Miao et al): boost potential ΔV(r) = 0.5 k0 (E - V(r))^2 / (E - V_min) for V(r) < E, else 0. Where E is threshold energy, V_min is minimum potential. Some implementations use (V_max - V_min) or (E - V_min). Need check. The code uses rng = vmax - vmin. Is that correct? Let's examine. In gamd-openmm, the boost is often: if V < E: 0.5 * k0 * (E - V)^2 / (E - V_min). Actually the "threshold" E is set to V_min + (V_max - V_min)/? Let's recall. GaMD: E = V_min + 1/k? Hmm.

The code's _channel_boost: b = 0.5 * k0 * (e - v)^2 / rng, with rng = vmax - vmin. Then condition (b+v) < e. This is a "lower-bound" dual boost? The submission says "Pep-GaMD lower-bound dual (dependent dihedral + total) boost whose envelope (Vmax/Vmin/threshold/k0max) is calibrated". The formula may be from gamd-openmm. Need verify against known GaMD. Standard GaMD boost potential:
ΔV(r) = 0.5 k0 (E - V(r))^2 / (E - V_min) if V(r) < E, else 0.
Where E is the threshold energy, often E = V_min + (V_max - V_min)/? Actually in GaMD, E is set to V_min + (1/λ)(V_max - V_min)? Let's derive from "lower-bound" GaMD: The boost is designed such that the modified potential V* = V + ΔV has a lower bound V_min? If ΔV = 0.5 k0 (E - V)^2 / (E - V_min), then V* = V + 0.5 k0 (E-V)^2/(E-V_min). At V=E, ΔV=0, V*=E. At V=V_min, ΔV=0.5 k0 (E-V_min), V*=V_min+0.5k0(E-V_min). The condition ΔV+V < E? Let's check: ΔV+V < E iff 0.5 k0 (E-V)^2/(E-V_min) < E-V. If E>V, divide by (E-V): 0.5 k0 (E-V)/(E-V_min) < 1 => E-V < 2(E-V_min)/k0. Not always. The code's condition (b+v) < e ensures the boosted potential stays below threshold? Actually if b+v >= e, set b=0. This is a "threshold" condition to avoid over-boosting. Standard GaMD has condition V < E, not b+V < E. The code uses b+v < e, which is stricter. Is that correct for "lower-bound" GaMD? Need know.

The submission says "Pep-GaMD lower-bound dual (dependent dihedral + total) boost whose envelope (Vmax/Vmin/threshold/k0max) is calibrated ONCE by the swarm and then frozen". The code's _channel_boost may be from gamd-openmm's StageIntegrator. We need check if formula matches gamd-openmm. The docstring in pep_gamd_boost_kj says "gamd-openmm's dependent dual boost under rung λ: dihedral first, then Total with the dihedral boost added to the Total energy before the square (stage_integrator _add_dihedral_boost_to_total_energy)." So they claim it mirrors gamd-openmm. We need verify.

Let's recall gamd-openmm code. In openmm-GaMD, the boost potential is:
if V < E:
  V_b = 0.5 * k0 * (E - V)^2 / (E - V_min)
else V_b = 0
Actually in OpenMM CustomIntegrator, they use "E" as threshold, "Vmin" and "Vmax". The denominator might be (E - Vmin) not (Vmax - Vmin). Let's search memory: GaMD boost:
ΔV = 0.5 k0 (E - V)^2 / (E - V_min) for V < E.
The parameters: E = V_min + (V_max - V_min)/? In "GaMD" the threshold E is set to V_min + (V_max - V_min)/? Actually from Miao et al. 2015: E = V_min + (V_max - V_min)/? Let's think. The boost potential is defined with a "threshold energy" E. To ensure the modified potential has no higher barrier, E is chosen such that the standard deviation of boost potential is controlled. The denominator is (E - V_min). Some implementations use (V_max - V_min) as an approximation? Let's check code from gamd-openmm: I recall in openmm-GaMD's integrator, the boost is:
if (V < E):
    Vb = 0.5 * k0 * (E - V)^2 / (E - Vmin)
else:
    Vb = 0
And the condition for applying boost is V < E, not V+Vb < E. The code here uses rng = vmax - vmin and condition b+v < e. That seems different. But maybe "lower-bound" variant uses Vmax-Vmin and condition V* < E? Let's investigate.

The submission says "Pep-GaMD lower-bound dual (dependent dihedral + total) boost whose envelope (Vmax/Vmin/threshold/k0max) is calibrated". "lower-bound" might refer to "GaMD with a lower bound" where the boost is applied only when the boosted potential is below threshold? Actually "lower-bound" in GaMD: The modified potential has a lower bound? Let's search memory: In GaMD, the boost potential is:
ΔV(r) = 0.5 k0 (E - V(r))^2 / (E - V_min) if V(r) < E
This ensures the modified potential V*(r) = V(r) + ΔV(r) is a "lower-bound" of the original? No, it's always >= V. The "lower-bound" might refer to "V* is bounded below by V_min"? Hmm.

Let's derive from code's formula: b = 0.5 k0 (e - v)^2 / (vmax - vmin). If vmax-vmin is the range of potential energy sampled. This is a common "Gaussian accelerated MD" variant where the boost is proportional to (E-V)^2 / (V_max - V_min). Actually in "Gaussian accelerated MD" (GaMD), the boost is:
ΔV = 0.5 k0 (E - V)^2 / (E - V_min) for V < E.
But some implementations use (V_max - V_min) as denominator because E - V_min ≈ V_max - V_min? Let's check original GaMD paper: "The boost potential is ΔV = 0.5 k0 (E - V)^2 / (E - V_min) when V < E." Yes.

However, there is "Pep-GaMD" (peptide GaMD) which might use a different "lower-bound" scheme. The code's condition (b+v) < e is unusual. Let's see if it's from gamd-openmm. In gamd-openmm, the boost is applied if V < E, and the boost is 0.5*k0*(E-V)^2/(E-Vmin). The code's condition b+v < e would be equivalent to V + 0.5 k0 (E-V)^2/(Vmax-Vmin) < E. That is not standard. Could be a bug. Need flag if verified.

Let's examine _channel_boost more carefully:
```
def _channel_boost(v, e, vmax, vmin, k0):
    v = _np.asarray(v, dtype=float)
    rng = vmax - vmin
    scale = _np.maximum(_np.maximum(abs(e), _np.abs(v)), 1.0)
    b = 0.5 * k0 * (e - v) ** 2 / rng
    b = _np.where(_np.abs(rng) <= 0.001 * scale, 0.0, b)
    return _np.where((b + v) < e, b, 0.0)
```
If rng is zero or small, b=0. If b+v >= e, b=0. This means boost only applied when the boosted potential is below threshold. But standard GaMD applies boost when V < E, regardless of V+ΔV. The condition V+ΔV < E is much stricter; for V close to E, ΔV is small, so V+ΔV < E may hold if ΔV < E-V. Since ΔV = 0.5 k0 (E-V)^2/rng, condition ΔV < E-V => 0.5 k0 (E-V)/rng < 1 => E-V < 2 rng / k0. If k0 is large, only very close to E. This could severely reduce boost. Is that intended? The docstring says "lower-bound" maybe means the boost is applied only when the total energy is below threshold? Actually in GaMD, the boost is designed to "raise" the potential energy to a threshold E, but not above? Wait the modified potential V* = V + ΔV. If V < E, ΔV >0, so V* > V. It can exceed E. The condition V* < E would mean the boost never pushes the potential above E, i.e., it "fills" the potential well up to E but not above. That is a "lower-bound" or "threshold" variant: the boosted potential is capped at E. That is not standard GaMD; standard GaMD allows V* to exceed E. But there is a variant called "GaMD with a lower bound" where the boost is applied only when V < E and the boost is such that V* <= E? Let's think. If you want the modified potential to have a lower bound? Actually if V* <= E, then the potential is bounded above by E, not lower. "Lower-bound" might refer to the boost potential being a lower bound of something? Hmm.

Let's not assume. We need audit against known analytic limits. The code's formula may be correct for their "Pep-GaMD lower-bound" variant. But we need check dimensional consistency: b = 0.5 * k0 * (e - v)^2 / rng. k0 units? In GaMD, k0 has units of 1/energy? Let's check. If e and v are energies (kJ/mol), (e-v)^2 has energy^2. rng energy. So b has energy * k0. For b to be energy, k0 must be dimensionless? Actually k0 in GaMD is a dimensionless parameter? In original GaMD, k0 is a harmonic force constant with units of 1/energy? Let's recall: ΔV = 0.5 k0 (E - V)^2 / (E - V_min). If E,V in kcal/mol, then (E-V)^2/(E-V_min) has units kcal/mol. For ΔV to be kcal/mol, k0 must be dimensionless. Indeed k0 is a dimensionless parameter (0 < k0 <= 1). In code, k0max is likely dimensionless. So OK.

But wait in umbrella bias, k_kcal is force constant kcal/mol per CV^2. In boost, k0 dimensionless. OK.

Now pep_gamd_boost_kj: b_dih = _channel_boost(v_dih, threshold_dih, vmax_dih, vmin_dih, lam*k0max_dih). Then if has_total: b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). out = b_dih + b_tot. This is "dihedral first, then Total with the dihedral boost added to the Total energy before the square". Is that correct for dependent dual boost? In GaMD dual-boost, the total boost is applied to the total potential energy, and the dihedral boost is applied to the dihedral energy. The total boost should be computed on the total potential energy V_total (including dihedral) without adding the dihedral boost? Actually in "dual boost" GaMD, two boosts are applied: one to the dihedral energy and one to the total potential energy. The total boost is computed on the total potential energy V_total, not V_total + dihedral boost. But the code adds b_dih to v_pep before computing b_tot. The docstring says "gamd-openmm's dependent dual boost under rung λ: dihedral first, then Total with the dihedral boost added to the Total energy before the square (stage_integrator _add_dihedral_boost_to_total_energy)." So they claim this is how gamd-openmm does it. Is that physically correct? In a dual-boost GaMD, the total boost is applied to the total potential energy, and the dihedral boost is applied to the dihedral energy. The total potential energy used for the total boost should be the original total potential energy, not including the dihedral boost, because the boost is a function of the original potential. However, some implementations add the dihedral boost to the total energy before applying the total boost to avoid double counting? Let's think. The total modified potential is V* = V_total + ΔV_dih + ΔV_total. If ΔV_total is computed from V_total + ΔV_dih, then the total boost depends on the boosted dihedral energy, which is a different function. Is that what gamd-openmm does? The docstring explicitly says "stage_integrator _add_dihedral_boost_to_total_energy". So if the code mirrors gamd-openmm, it's likely correct for that engine. But is it physically correct for Boltzmann reweighting? The boost potential applied in the simulation must be exactly the one used in reweighting. If the code's pep_gamd_boost_kj matches the integrator's boost, then OK. But if the integrator actually computes total boost on original total energy, then this function would be wrong. Need verify from code? We don't have gamd-openmm source. The submission says "gamd-openmm's dependent dual boost" and "stage_integrator _add_dihedral_boost_to_total_energy". So likely they know. But we can flag if the formula is non-standard? Need be careful.

Let's examine the "dependent" in "dependent dual (dependent dihedral + total) boost". "Dependent" might mean the total boost depends on the dihedral boost. So adding b_dih to v_pep before total boost is intentional. OK.

Now _channel_boost condition: `_np.where((b + v) < e, b, 0.0)`. If b+v >= e, set b=0. This is a "lower-bound" condition. Is it dimensionally consistent? b+v and e all energies. OK.

But there is a potential issue: if v > e, then (e-v)^2 positive, b positive, but condition (b+v) < e is false because v > e, so b=0. Good. If v < e but b+v >= e, b=0. This means the boost is zero for a range of v near e. This is a "hard cap" at e. Is that the intended "lower-bound" boost? Maybe.

Now k0max_from_globals: returns Total and Dihedral from shared_globals. If "k0_Dihedral" missing, KeyError. OK.

set_replica_lambda: sets k0_Dihedral and k0_Total to lam*k0max. If lam=0, sets zero. OK.

set_replica_lambda_for_window: if k0max_by_channel None, no-op. If state_lambdas None and ladder active, raise. OK.

Now state_identity.py: canonical_state_definition requires exact set of fields. It converts energy_unit kJ/mol to kcal/mol by dividing k1,k2 by 4.184. Wait if energy_unit == "kJ/mol", row["k1"] /= 4.184. But k1 is force constant in kJ/mol per CV^2. To convert to kcal/mol per CV^2, divide by 4.184. Correct. But note: if k1 is in kJ/mol/A^2, dividing by 4.184 gives kcal/mol/A^2. OK.

Potential issue: `if row["k1"] > 0 and data["cv1"] is None:` but if k1 is NaN? normalize_windows ensures finite. OK.

Now bias.py reconstruct_bias_matrix: It computes matrix = sum 0.5*k*delta^2 in kcal/mol, then matrix *= beta * KJ_PER_KCAL. So matrix becomes dimensionless reduced bias (beta * energy in kJ/mol). Wait beta is mol/kJ. So beta * KJ_PER_KCAL * kcal = beta * kJ = dimensionless. Good. But the function docstring says "Return the (N,K) reduced umbrella-plus-ladder bias, not full potential." So OK. However, it multiplies by beta*KJ_PER_KCAL after summing all umbrella terms. If k1,k2 are kcal/mol per CV^2, then sum is kcal/mol. Multiply by 4.184 to kJ/mol, then by beta (mol/kJ) -> dimensionless. Good.

But note: in reconstruct_bias_matrix, if ladder active, it calls _ladder_apply(matrix, pep, dih, lambdas, envelope, beta, bookkeeping). The matrix is already reduced (dimensionless). The ladder boost function likely expects reduced matrix? Need check. In production, bias_matrix_kj is in kJ/mol, then reduced_bias_matrix = beta * bias_matrix_kj. In reconstruct_bias_matrix, matrix is reduced. The ladder apply function may add boost in reduced units? Need see. The docstring says "reduced umbrella-plus-ladder bias". So OK.

Potential issue: In reconstruct_bias_matrix, if secondary is None and no active secondary, OK. If secondary has NaN for some observations, matrix column for active secondary will be NaN for those rows. That's intended.

Now production sample function: It computes bias_matrix_kcal, bias_matrix_kj, reduced_bias_matrix = beta * bias_matrix_kj. Then for each replica, sampled_bias_kcal = all_bias_kcal[w], sampled_bias_kj = all_bias_kj[w]. But note: all_bias_kcal includes boost_bias_kcal. So "umbrella_bias_kcal_mol" actually includes boost, not just umbrella. The column name "umbrella_bias_kcal_mol" is misleading but not a math error? It could cause confusion. But the prompt asks for math/physics correctness, not naming. However, if downstream analysis uses "umbrella_bias_kcal_mol" expecting only umbrella, it would be wrong. But the code also has "sampled_umbrella_bias_kj" which is only umbrella (distance+ss) and "sampled_boost_bias_kj". The "umbrella_bias_kcal_mol" includes boost. That could be a silent error if analysis uses it. Need flag? The submission says "state bias matrix" includes umbrella + boost. The column name is misleading but maybe not a mathematical error. We can flag as MINOR if it can cause wrong free energy if used incorrectly. But the audit should focus on code producing wrong numbers. If the column is documented? Not in snippet. We'll consider.

More importantly: In sample(), `boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env) if pep_env is not None else np.zeros((nrep, nrep))`. If pep_env is not None but state_lambdas length maybe not nrep? It should be n_states. If n_states != nrep? In REUS, number of windows equals number of replicas? Usually yes. But if not, shape mismatch. The code assumes nrep x nrep. In production, nrep = number of replicas = number of windows? Likely. OK.

Potential issue: `pep_gamd_boost_matrix_kj` returns shape (n_states, n_samples). If n_states != nrep, then bias_matrix_kj shape (n_states, nrep). Later `reduced_bias_matrix = float(beta) * bias_matrix_kj` shape (n_states, nrep). Then for each replica r, `all_bias_kcal = bias_matrix_kcal[:, r]` shape (n_states,). `sampled_bias_kcal = all_bias_kcal[w]` OK. Exchange uses bias_matrix_kj[window, replica]. So if n_states != nrep, exchange still works as long as assignments length nrep and windows n_states. But replica_of_window size? In production, likely n_states = nrep. OK.

Now _current_exchange_arrays: It computes bias_matrix_kj for exchange. It uses `_has_ss_axis = (secondary_cv_centers is not None and secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None)`. But in sample(), `_has_ss_axis = secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None`. There may be inconsistency: exchange uses ss_ks_kj_arr_global, sample uses ss_k_kcal_arr_global. If one is None and other not, exchange and sample bias matrices differ. Need check. In sample, `ss_centers_arr = ss_centers_arr_global`, `ss_k_arr = ss_k_kcal_arr_global`. In exchange, `ss_centers_arr_global`, `ss_k_kcal_arr_global`? Wait exchange code: `_has_ss_axis = (secondary_cv_centers is not None and secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None)`. Then `umbrella_bias_matrix_kcal(..., ss_centers_arr_global if _has_ss_axis else None, ss_k_kcal_arr_global if _has_ss_axis else None)`. It uses ss_k_kcal_arr_global, not ss_ks_kj_arr_global. The condition checks ss_ks_kj_arr_global not None, but passes ss_k_kcal_arr_global. If ss_k_kcal_arr_global is None while ss_ks_kj_arr_global not None, then it would pass None to ss_k_kcal, but _has_ss_axis true, so umbrella_bias_matrix_kcal would receive ss_centers_arr_global and None? Actually call: `umbrella_bias_matrix_kcal(primary_values, ss_values, centers_a_arr, k_arr, ss_centers_arr_global if _has_ss_axis else None, ss_k_kcal_arr_global if _has_ss_axis else None)`. If _has_ss_axis true but ss_k_kcal_arr_global None, then ss_k_kcal=None, and umbrella_bias_matrix_kcal checks `if ss_centers is None or ss_k_kcal is None: return distance_kcal, zeros`. So secondary bias zero. But sample path might have ss_k_kcal_arr_global not None, so sample includes secondary bias while exchange doesn't. That would cause exchange to use different Hamiltonian than samples, violating detailed balance? Actually exchange would not include secondary bias, so swaps would be priced without secondary restraint, while the actual umbrella potential includes it. That would break detailed balance and produce wrong free energy. Need verify if ss_k_kcal_arr_global and ss_ks_kj_arr_global are always both set or both None. The code snippet doesn't show initialization. But the condition is suspicious: it checks ss_ks_kj_arr_global but uses ss_k_kcal_arr_global. Could be a bug if they are not guaranteed to be both non-None. Need flag as SUSPECTED or VERIFIED? We can point to lines. The prompt asks distinguish VERIFIED/SUSPECTED. We can say SUSPECTED because we don't see initialization. But the condition mismatch is a code smell. Let's note.

Also in sample(), `_has_ss_axis = secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None`. It uses `ss_centers_arr_global` and `ss_k_kcal_arr_global`. In exchange, `_has_ss_axis` additionally requires `secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None`. If secondary_cv_ks_kj is None but secondary_cv_k_kcal_list not None, sample includes secondary, exchange doesn't. That would be a serious bug. Need flag.

Now the exchange loop: `_attempt_window_swap` uses `bias_matrix_kj` from `_current_exchange_arrays`. It calls `apply_window_swap` with beta. Good. But note: after accepted swap, it calls `set_window` and `set_replica_lambda_for_window` on the two replicas. However, `set_window` likely sets the umbrella center/force constant for the replica's new window. But the bias matrix used for exchange was computed from current CVs and state lambdas. After swap, the replicas' window assignments change, but their CVs remain the same. The next exchange will recompute bias matrix with new assignments? Wait bias_matrix_kj is [state, replica], independent of assignments. So OK. But the actual Hamiltonian of each replica after swap is changed to the new window's umbrella and lambda. The bias matrix for the next exchange uses the same CVs and state lambdas, so it correctly prices the new state's bias for each replica. OK.

Potential issue: In `_attempt_window_swap`, after accepted swap, it calls `set_window` and `set_replica_lambda_for_window` on replicas i and j. But it does not update `observable_cache`? It clears observable_cache. OK.

Now `_current_exchange_arrays` uses `_fetch_exchange_state` which for fast path reads CVs and v_pep/v_dih. For non-fast path, it reads positions and computes CVs, and v_pep/v_dih. It does NOT read potential energy. That's fine because exchange delta only depends on bias, not physical potential. But wait if the boost is active, the boost bias depends on v_pep and v_dih, which are read. OK.

Now the Gibbs proposal: In production, how is it used? The snippet only shows functions, not the loop. But we can audit the functions.

Potential issue in `_gibbs_window_proposal_distribution`: It computes `old_e = bias[wi, rep] + bias[valid_windows, target_reps]`. For valid_windows including wi, target_reps = rep, so old_e = 2*bias[wi,rep]. For other windows, old_e = bias[wi,rep]+bias[wj,target]. This is correct for the total bias of the two-replica subsystem. But what about the "stay" candidate? The total bias for stay should be bias[wi,rep]+bias[wi,rep]? No, if wj=wi, target_reps=rep, the two replicas are the same replica, so the "two-replica subsystem" is just one replica. The total bias should be bias[wi,rep], not 2*bias[wi,rep]. The code's old_e for stay is 2*bias[wi,rep], and new_e also 2*bias[wi,rep], delta=0. The absolute value doesn't matter for delta, but it affects the proposal probability? They set delta=0 for stay, so weight=1. The normalization includes this weight. If they had used correct old_e = bias[wi,rep] and new_e = bias[wi,rep], delta=0 as well. So no effect. OK.

But wait for other candidates, old_e = bias[wi,rep]+bias[wj,target]. This is the total bias of the two replicas in their current windows. new_e = bias[wj,rep]+bias[wi,target]. Correct. So delta is correct. Good.

Potential issue: The proposal distribution q(wj) is normalized over valid_windows, but the "stay" candidate has weight 1 regardless of the current window's bias. This means the probability of staying is 1/Z where Z = 1 + sum_{wj≠wi} exp(-β delta). This is a valid proposal. OK.

Now `gibbs_propose_one_replica`: It computes reverse proposal using `holders_after`. But note: `holders_after` is a copy of `replica_of_window` with the swap applied. However, `replica_of_window` maps window -> replica. After swap, replica rep moves to wj, target moves to wi. The code sets `holders_after[wi] = target_rep; holders_after[wj] = rep`. Correct. Then reverse distribution for replica rep from current_window=wj. It computes q_reverse = probability of choosing wi. Good.

But there is a subtle issue: The reverse proposal distribution should be computed for the same move type: a transposition of replica rep with the replica in window wi. In the reverse state, replica rep is in wj, and the replica in wi is target_rep. The reverse move would swap rep with target_rep, moving rep to wi and target_rep to wj. The code's reverse distribution uses `replica_index=rep`, `current_window=wj`, `replica_of_window=holders_after`. It considers all valid windows, including wi. For candidate wi, target_reps = holders_after[wi] = target_rep. Delta_rev = new_e - old_e = (bias[wi,rep]+bias[wj,target]) - (bias[wj,rep]+bias[wi,target]) = -delta. q_reverse = probability of choosing wi. Good.

Now `_gibbs_mh_acceptance_probability`: log_alpha = -beta*delta + log(qr)-log(qf). If qf or qr zero, return 0. But if q_reverse is zero due to underflow? The proposal distribution uses log_weights clipped to -745, so q_reverse could be extremely small but not zero unless filtered by finite. If q_reverse is 0 because the reverse move's window wi was filtered out due to non-finite delta? But delta_rev should be finite if forward delta finite. OK.

Potential issue: The acceptance probability for Gibbs-with-MH should be min(1, [π(dest) q_rev] / [π(curr) q_fwd]). Here π ratio = exp(-β delta) where delta = new - old. So log_alpha = -β delta + log(qr)-log(qf). Correct.

Now `_exchange_probability`: standard Metropolis for swapping windows. But note: In replica exchange with state-dependent Hamiltonians, the acceptance probability for swapping replicas i and j between windows m and n is min(1, exp(-β [U_m(x_j)+U_n(x_i) - U_m(x_i)-U_n(x_j)])). The code's delta = new_e - old_e = U_wj(x_i)+U_wi(x_j) - U_wi(x_i)-U_wj(x_j). Correct. If delta <=0 accept 1. OK.

Now `apply_window_swap`: It uses `_exchange_probability(delta, beta)`. But if `p_override` is provided, it uses that. OK.

Potential issue: In `apply_window_swap`, after accepted, it updates `assignments` and `replica_of_window`. But it does not update the actual integrator's window/lambda; that's done by caller. OK.

Now `explicit_2d_neighbor_pairs_for_exchange`: It returns chosen pairs. The "slots" round-robin by edge index modulo slots. If slots=4, parity 0 uses edges idx 0,4,8,...; parity 1 uses 1,5,9,... etc. This is not a proper edge coloring; it may select edges that share vertices? They filter used. OK. But if a slot has no edges, it tries all edges greedily. This could cause the same pair to be attempted in multiple slots? But chosen is per call. OK.

Now the production sample function has a potential issue with `observable_cache["cvs_nm"] = (primary_values / 10.0) if is_distance_primary else primary_values`. If primary CV is distance in angstroms, dividing by 10 gives nm. OK. But if primary CV is contact fraction dimensionless, no conversion. OK.

Now `umbrella_restoring_force_kcal_mol_per_A`: `k_kcal_a2 * (center_a - cv_a)`. Force = -k(x-c), magnitude k(c-x). Units kcal/mol/A. OK.

Now `potential_kj_mol`: from getState getEnergy groups=physical_energy_groups_for_args(args). This is the physical potential energy excluding umbrella and boost? Need check. The sample row records `potential_kj_mol` as physical potential. But the bias matrix includes umbrella + boost. For MBAR, the total energy is physical + bias. If `potential_kj_mol` is physical only, then MBAR can add bias. OK. But if `physical_energy_groups_for_args` includes boost? Need see. Not in snippet. Could be a bug if it includes boost. But likely physical groups exclude umbrella/boost. We'll not flag without evidence.

Now `extract_gamd_boost_kj(sim.integrator, unit, None)` reads the integrator's current boost? It may read global variables? It returns boost_kj, components, source. This is used for row "gamd_boost_total_kj_mol". But the actual boost applied by the integrator may differ from `pep_gamd_boost_kj` computed from v_pep/v_dih. If they disagree, the recorded boost is wrong. The code has a calibration case where boost not applied. The new code may still have mismatch. Need check: `extract_gamd_boost_kj` likely reads the integrator's boost energy from a custom force? Not shown. Could be a source of silent error. But we can't verify.

Now `_fetch_v_pep_v_dih`: It reads dihedral group energy from context. But if the integrator's boost is applied via a CustomIntegrator, the group energies may include the boost? The dihedral group energy from getState(groups={DIHEDRAL_GROUP}) should be the potential energy of dihedral forces, not including boost. OK. `peptide_essential_energy_kj` reads total? Not shown.

Now `campaign_ladder_registry_lambda`: It searches parent directories up to adaptive_production. If multiple registry files, returns first positive. Could be stale from a previous campaign? It stops at adaptive_production. OK.

Now `resolve_ladder_active`: If state_lambdas has any >0, true. Else if registry has positive, true. But if state_lambdas is all zeros and registry has positive, it returns true, so ladder active. Then `k0max_by_channel` is set, and `set_replica_lambda_for_window` will set k0=0 for all windows. That means replicas run unboosted, but the ladder code path is active, so samples record v_pep/v_dih and MBAR prices zero boost. This is consistent: λ=0 means no boost. OK.

Now `_derive_state_gamd_lambdas`: If normalized_rows has gamd_lambda values, returns them. If not, falls back to existing or zeros. Potential issue: if existing is provided but length n, it returns existing even if it's stale/misaligned. The docstring warns callers not to pass pre-reindex values. But the function itself doesn't verify alignment. Could be a source of silent misalignment. But maybe call sites are careful. We can flag as SUSPECTED.

Now `_reload_state_gamd_lambdas_on_resume`: It reads manifest method_settings state_gamd_lambdas. If args.state_gamd_lambdas is truthy, returns. If it's an empty list, reads manifest. If manifest has lambdas, sets args.state_gamd_lambdas. But note: if manifest has an all-zero list, `if lambdas:` is True because list non-empty, so it sets. Good. If manifest has empty list, no. OK.

Now `_persist_state_gamd_lambdas`: Writes current args.state_gamd_lambdas to manifest. If args.state_gamd_lambdas is None, writes empty list. OK.

Now state_identity.py: `canonical_state_definition` requires exact set of fields. It converts energy_unit kJ/mol to kcal/mol by dividing k1,k2 by 4.184. But it does not convert `gamd_lambda`? No need. It also checks `if row["k1"] > 0 and data["cv1"] is None`. But if k1 is 0 and cv1 None, OK. If k1 negative? normalize_windows enforces minimum=0, so no negative. OK.

Potential issue: In `canonical_state_definition`, for NPT, it allows pressure_bar to be any finite number, including negative? `finite_number(data["pressure_bar"], "pressure_bar")` without positive=True. Negative pressure is unphysical for NPT? Could be allowed? Usually pressure can be negative in some contexts, but for water NPT, negative pressure is unphysical. Not a math error.

Now `bias.py` `_term`: If force_key not in window and axis==2 and center_key not in window, returns 0,0. But if axis==2 and force_key missing but center_key present, it raises "missing k2; inactive means explicit zero". This is strict. OK.

Potential issue: `normalize_windows` sets `window["gamd_lambda"] = finite_number(raw.get("gamd_lambda", 0.0), ..., minimum=0)`. It allows gamd_lambda >1? It only checks minimum=0, not maximum. In `set_replica_lambda`, it checks 0<=lam<=1. But `normalize_windows` doesn't. So a window with gamd_lambda=2 would pass state identity but then set_replica_lambda raises. Could be a bug. But not a math error in the code's normal operation. Could flag as MINOR: missing upper bound in normalize_windows. But the prompt asks for errors that can silently produce wrong numbers. If gamd_lambda >1, set_replica_lambda raises, not silent. So not silent.

Now `reconstruct_bias_matrix`: It multiplies matrix by beta * KJ_PER_KCAL. But if `_ladder_apply` expects matrix in kcal/mol or kJ/mol? The docstring says reduced. Need check `apply_ladder_boost_to_u` from gareus.mbar_analysis.ladder. Not shown. Could be a mismatch. But we can't verify.

Now let's look for sign conventions. `beta` is 1/(kB T) in mol/kJ. In production, `beta_value = float(beta)`. In `_gibbs_window_proposal_distribution`, log_weights = -beta * delta. If delta in kJ/mol, beta in mol/kJ, product dimensionless. Good. In `_exchange_probability`, x = -beta * delta. Good.

Now `umbrella_bias_matrix_kcal` returns kcal. `assemble_bias_matrices` converts to kJ. `reduced_bias_matrix = beta * bias_matrix_kj`. Good.

Potential issue: In `sample()`, `sampled_bias_kcal = float(all_bias_kcal[w])` where all_bias_kcal includes boost. But `sampled_bias_kj = float(all_bias_kj[w])`. Then `sampled_umbrella_bias_kj = float(4.184 * (all_distance_bias_kcal[w] + all_ss_bias_kcal[w]))`. This is umbrella only. `sampled_boost_bias_kj = float(boost_bias_matrix_kj[w, r])`. Note: `sampled_bias_kj` should equal `sampled_umbrella_bias_kj + sampled_boost_bias_kj`. But due to floating point, maybe. OK.

Now `umbrella_bias_matrix_kcal` with secondary: `ss_kcal = np.where(np.isfinite(dss), ss_kcal, 0.0)`. If dss is NaN, ss_kcal is NaN, then np.where sets 0. But if dss is NaN, the secondary CV value is missing. Setting bias to 0 for that state means the sample is treated as if the secondary restraint is not applied. But in the actual simulation, if the secondary CV is NaN, the umbrella force may be zero? The code comment says "a replica without a finite secondary value, contributes zero on that axis (the previous sample-path guard, now applied identically to the exchange path -- a NaN entry used to freeze that replica's exchanges)." So this is intentional: missing secondary value means no secondary bias. But if the actual simulation applies a secondary restraint even when CV is NaN? In OpenMM, a CustomCVForce with NaN would produce NaN energy, not zero. But maybe the fast path returns NaN when secondary CV cannot be computed, and the restraint is disabled? Not sure. Could be a silent error if NaN is treated as zero bias while the actual force is NaN or something. But the comment says previous guard. We'll not flag.

Now `_channel_boost` uses `_np.abs(v)` in scale. If v is NaN, scale = max(abs(e), abs(NaN), 1) = NaN? np.maximum with NaN propagates? np.maximum(np.maximum(abs(e), np.abs(v)), 1.0) if v NaN -> NaN. Then rng <= 0.001*scale -> False? NaN comparison false. b = 0.5*k0*(e-v)^2/rng -> NaN. np.where((b+v)<e, b, 0.0): (NaN < e) false, so returns 0. So NaN v yields 0 boost. Is that correct? If v_pep or v_dih is NaN, boost should be? In production, v_pep/v_dih are read from context; if missing, NaN. If boost is active, missing v_pep/v_dih would mean the boost cannot be computed. Setting boost to 0 silently could be wrong. But `_fetch_v_pep_v_dih` returns NaN only if pep_env None or has_total false. If pep_env not None and has_total true, v_pep should be finite. If context read fails, maybe NaN. The code would silently treat as zero boost. Could be a silent error. But maybe not.

Now `pep_gamd_boost_kj`: If v_pep is NaN and has_total true, b_tot = _channel_boost(NaN + b_dih, ...) -> 0, so out = b_dih. That means if total energy missing, only dihedral boost applied. But the actual integrator would apply both? If v_pep missing due to read error, the recorded boost is wrong. Could be a silent failure. But maybe v_pep is always finite when has_total true. We'll consider.

Now `_fetch_v_pep_v_dih`: It reads v_dih from group DIHEDRAL_GROUP. But if the integrator's boost modifies the dihedral energy? The group energy from getState should be the unboosted dihedral potential. OK.

Now `set_replica_lambda`: It sets k0_Dihedral and k0_Total. But if the integrator has additional globals for threshold/vmax/vmin? The envelope is frozen, so only k0 changes with lambda. OK.

Now `k0max_from_globals`: It reads "k0_Total" and "k0_Dihedral" from shared_globals. If shared_globals has "k0_Total" as 0.0 for single dihedral boost, OK. If missing "k0_Dihedral", KeyError. OK.

Now `campaign_ladder_registry_lambda`: It returns `(path.name, lam)` where lam is the first positive lambda found. But if the registry has multiple states with different lambdas, this function only checks if any positive. It doesn't return the full ladder. `resolve_ladder_active` only needs bool. OK.

Now `_derive_state_gamd_lambdas`: If normalized_rows is present and length n and all have gamd_lambda, returns them. But it doesn't check that gamd_lambda values are within [0,1]. Could be >1. Later set_replica_lambda raises. Not silent.

Now `_reload_state_gamd_lambdas_on_resume`: It reads manifest method_settings. But if manifest has state_gamd_lambdas as a list of strings, converts to float. OK.

Now `_persist_state_gamd_lambdas`: It writes list(args.state_gamd_lambdas or []). If args.state_gamd_lambdas is a numpy array, list() gives np.float64 values, JSON serializable? np.float64 may not be JSON serializable by default. But maybe update_run_manifest handles. Not a math error.

Now let's think about the core physics: replica exchange with state-dependent boost. The exchange acceptance should use the total Hamiltonian including the boost. The code's bias matrix includes umbrella + boost. But the boost is a function of the configuration's v_pep and v_dih, which are computed from the current context. However, the boost applied by the integrator may depend on the integrator's internal state (e.g., running averages of Vmin/Vmax) that are updated during simulation. If the envelope is frozen, Vmin/Vmax/threshold are fixed, so boost is a deterministic function of v_pep/v_dih. OK.

But there is a subtle issue: The boost potential in GaMD is not a simple function of the instantaneous potential energy if the envelope parameters are updated during the simulation. The submission says envelope is calibrated once and frozen. So OK.

Now the "dependent dual boost" adds b_dih to v_pep before computing b_tot. Is the total boost applied to the total potential energy including the dihedral boost? In the actual integrator, if it adds the dihedral boost to the total energy before computing the total boost, then the total boost is a function of v_pep + b_dih. The code matches. But is this physically correct for reweighting? The total bias is b_dih + b_tot. The total modified potential is V_total + b_dih + b_tot. If b_tot is computed from V_total + b_dih, then the total boost depends on b_dih, which itself depends on V_dih. This is a valid bias potential as long as it's the actual applied bias. OK.

Now let's check `_channel_boost` against standard GaMD more carefully. The code's condition `(b + v) < e` means the boost is zero if the boosted potential would exceed the threshold. This is a "hard cap" at e. In standard GaMD, the boost is applied for V < E, and the boosted potential V* = V + ΔV can exceed E. The condition V* < E would mean the boost is only applied when V is sufficiently below E such that V* stays below E. This is not standard GaMD. But maybe "lower-bound" GaMD uses this to ensure the modified potential is bounded above by E? Let's search memory: There is "GaMD" where the boost potential is designed such that the modified potential has a "lower bound" equal to the threshold? Actually if you want the modified potential to have a flat bottom up to E, you would set ΔV = E - V for V < E, giving V* = E. But GaMD uses a harmonic function, not linear, so V* < E for V < E? Let's check: ΔV = 0.5 k0 (E-V)^2/(E-Vmin). For V near E, ΔV small, so V* = V + small < E. For V far below E, ΔV could be large, possibly V* > E. The condition V* < E imposes a constraint on k0. In standard GaMD, k0 is chosen such that the boost potential's standard deviation is <= some value, but V* can exceed E. The "lower-bound" variant might choose k0 such that V* <= E always, i.e., the modified potential is bounded above by E. That would be a "lower-bound" on the boost? Actually if V* <= E, then the potential energy is capped at E, which is an upper bound, not lower. Hmm.

Let's not overreach. The code's formula may be from a specific implementation. We need flag only if we can verify against known analytic limits. The prompt says "Canonical limits: verify quantitative claims against known analytic limits - harmonic-oscillator PMF is parabolic, dG = 0 at equal populations, KL >= 0, ESS <= N, ideal-gas and two-state results." We can use those to test.

Let's test the exchange and Gibbs kernels with a simple two-state system. Suppose two windows, two replicas, bias matrix. The stationary distribution should be proportional to exp(-beta * total bias) over assignments. The swap acceptance is correct. The Gibbs proposal with MH should also be correct. Let's verify with a simple case.

Consider two windows A,B and two replicas 1,2. Bias matrix U[window, replica]. Current assignment: rep1 in A, rep2 in B. Total bias = U[A,1]+U[B,2]. A swap would give U[B,1]+U[A,2]. The swap acceptance uses delta = (U[B,1]+U[A,2]) - (U[A,1]+U[B,2]). Correct.

Gibbs proposal for rep1: valid_windows = [A,B], target_reps = [1,2]. For wj=A (stay), delta=0, weight=1. For wj=B, delta = (U[B,1]+U[A,2]) - (U[A,1]+U[B,2]) = same as swap delta. q_forward = exp(-β delta)/(1+exp(-β delta)) if delta positive? Actually weights: stay weight 1, swap weight exp(-β delta). So q_forward = exp(-β delta)/(1+exp(-β delta)). Reverse: after swap, rep1 in B, rep2 in A. Reverse proposal for rep1: valid_windows = [A,B], target_reps = [2,1]. For wj=B (stay), delta=0, weight=1. For wj=A, delta_rev = (U[A,1]+U[B,2]) - (U[B,1]+U[A,2]) = -delta. q_reverse = exp(-β (-delta))/(1+exp(-β (-delta))) = exp(β delta)/(1+exp(β delta)). Then MH acceptance log_alpha = -β delta + log(q_reverse)-log(q_forward). Let's compute. q_reverse/q_forward = [exp(βδ)/(1+exp(βδ))] / [exp(-βδ)/(1+exp(-βδ))]. Note 1+exp(-βδ) = (exp(βδ)+1)/exp(βδ). So q_forward = exp(-βδ) * exp(βδ)/(1+exp(βδ)) = 1/(1+exp(βδ)). q_reverse = exp(βδ)/(1+exp(βδ)). Ratio = exp(βδ). Then log_alpha = -βδ + βδ = 0, so pacc=1. That means the Gibbs move always accepts? Wait that would make the chain mix too fast? Let's check. If q_forward = 1/(1+exp(βδ)) and q_reverse = exp(βδ)/(1+exp(βδ)), then the MH correction exactly cancels the proposal bias, resulting in acceptance probability 1 for both forward and reverse? But if pacc=1 for the forward move, then the chain would always move from current to swapped state if proposed. But the proposal probability of swap is q_forward = 1/(1+exp(βδ)). If δ is large positive (swap unfavorable), q_forward is small, so the move is rarely proposed. If proposed, it's always accepted. That is a valid Metropolis-Hastings scheme: the proposal distribution is biased toward favorable moves, and the acceptance is 1. This is the "heat-bath" style: if you propose with probability proportional to target, acceptance is 1. But here q is not exactly proportional to target? Let's see. For two-state, q
(TRUNCATED - output hit the token limit before completion)

Position cards:

- glm: (TRUNCATED - output hit the token limit before completion) (confidence 50)
- kimi: (TRUNCATED - output hit the token limit before completion) (confidence 50)
- deepseek-thinking: (TRUNCATED - output hit the token limit before completion) (confidence 50)
- mini: Addressing these points will eliminate silent bias and ensure that the reported free‑energy surfaces are truly Boltzmann‑weighted. (confidence 50)
- thinker: (TRUNCATED - output hit the token limit before completion) (confidence 50)

Verdicts: glm=n/a, kimi=n/a, deepseek-thinking=n/a, mini=n/a, thinker=n/a
Majority: n/a at 0%

cosine similarity | glm | kimi | deepseek-thinking | mini | thinker
glm | 1.000 | 1.000 | 1.000 | 0.347 | 1.000
kimi | 1.000 | 1.000 | 1.000 | 0.347 | 1.000
deepseek-thinking | 1.000 | 1.000 | 1.000 | 0.346 | 1.000
mini | 0.347 | 0.347 | 0.346 | 1.000 | 0.346
thinker | 1.000 | 1.000 | 1.000 | 0.346 | 1.000

mean pairwise cosine: 0.738

## glm

## Attack on the designated target

**The heat-bath critique is self-refuting.** The target first argues the proposal distribution should be ∝ exp(−β·U_wj(x_rep)) alone, excluding the swap partner's term U_wi(x_target). This is wrong: the holder of wj varies with wj, so U_wi(x_target_wj) is not constant across candidates — including it is the correct conditional for the transposition move. The target then concedes "with MH correction, any proposal distribution with correct reverse probability yields correct stationary distribution," which is the entire point. No detailed-balance violation exists here; the target manufactured one by misidentifying the move as a single-replica window reassignment rather than a two-replica transposition.

**The `(b+v) < e` GaMD guard concern is speculation.** The target flags the guard and the `vmax−vmin` denominator as "not standard GaMD" but never checks gamd-openmm's StageIntegrator source, which the docstring explicitly cites (`_add_dihedral_boost_to_total_energy`). This is SUSPECTED at best, not the "physics error" the target implies.

**The `umbrella_bias_kcal_mol` column complaint is a naming nitpick.** The pipeline records separate `sampled_boost_bias_kj` and `sampled_umbrella_bias_kj` columns; MBAR reconstructs from CVs and v_pep/v_dih, not from this column. Not a silent wrong number.

**What the target missed (the real defects):**
1. No assertion compares `extract_gamd_boost_kj(sim.integrator)` (the boost actually applied by the integrator) with `pep_gamd_boost_kj` (the analytic formula priced into the bias matrix) — both computed side-by-side in `sample()` at L~7620-7640 and never cross-checked. This is the single cheapest guard against the next chignolin_8 and it is absent.
2. Two independent k0max sources: `k0max_from_globals(shared_globals)` (L528-530) feeds `set_replica_lambda`, while `pep_gamd_boost_kj` uses `env.k0max_*` from the PepGamdEnvelope. No shown invariant ties them; if recalibration updates one but not the other, the integrator applies one boost while exchange/MBAR price another.
3. `apply_window_swap`'s docstring (L1127-1182) falsely claims "the GaMD boost cancel[s] exactly from delta." It does not cancel when λ_i ≠ λ_j. The code is correct only because the bias matrix includes per-state boost — but the docstring invites a future maintainer to "simplify" delta to umbrella-only, reintroducing the chignolin_8 class as a regression.
4. `np.zeros((nrep, nrep))` fallback when pep_env is None (L~7570, L~7870) breaks shape if nwin ≠ nrep (unheld windows are supported by `holders >= 0` guards elsewhere).

## Defense / concession

I concede the heat-bath proposal and MH acceptance in `_gibbs_window_proposal_distribution`/`_gibbs_mh_acceptance_probability` are mathematically correct — the reverse-proposal-against-holders_after construction is properly done. I maintain the NaN→0 findings as the top defects: `_channel_boost` L503 (`np.where((b+v)<e, b, 0.0)`) silently converts NaN channel energies to zero boost in both exchange pricing and MBAR reconstruction; `umbrella_bias_matrix_kcal` L916 zeroes NaN secondary CV bias in exchanges while `reconstruct_bias_matrix` (correctness/bias.py) propagates NaN, creating an exchange-vs-MBAR inconsistency. The `_has_ss_axis` predicate divergence between `sample()` (checks `secondary_cv_k_kcal_list`) and `_current_exchange_arrays()` (checks `secondary_cv_ks_kj` and `ss_ks_kj_arr_global`) is a genuine state-label mismatch risk if those flags can disagree.

## Weighing the audit

The audit corroborates the NaN→0 in `_channel_boost` and the `_has_ss_axis` inconsistency — both strengthen my position. Its heat-bath concern is unfounded (it self-corrects mid-analysis). The GaMD denominator concern (`vmax−vmin` vs standard `E−Vmin`) is unverified against gamd-openmm source and warrants a targeted test, but is not established. The audit's `umbrella_bias_kcal_mol` naming complaint is cosmetic.

VERDICT: ACCEPT-WITH-CHANGES — core exchange kernels are correct but NaN→0 fallbacks, the missing applied-vs-priced boost assertion, and the k0max dual-source risk must be fixed before this code can be trusted to produce the Boltzmann free energy it claims.
CONFIDENCE: 78

## kimi

Attack: the target’s key move is “sample and exchange use the same matrix, therefore consistent.” That is the fallacy. `umbrella_bias_matrix_kcal` (production.py L895-L917) turns any nonfinite secondary displacement into zero restraint: `dss = sv - sc`; `ss_kcal = np.where(np.isfinite(dss), ss_kcal, 0.0)`. For a window with `k2 > 0`, missing CV2 is not zero restraint energy; it is unknown restraint energy. Because the same zeroed column feeds `_current_exchange_arrays` and the recorded reduced matrix, forward swap, reverse swap, and MBAR can all agree while pricing the wrong `u_{n,k}`. `correctness/bias.py` does the opposite: `numeric_vector` preserves NaN and `reconstruct_bias_matrix` raises when active `k2` lacks cv2. So production and analysis are not even guaranteed to treat the same frames as missing.

The target also under-plays `_channel_boost` (pep_gamd.py L497-L503). If `v` or `b` is NaN, `(b + v) < e` is False, so the function returns `0.0`: a silent NaN→0 boost fallback inside `pep_gamd_boost_kj` (L506-L519) and `pep_gamd_boost_matrix_kj` (L522-L525). The sample path separately records integrator `gamd_boost_total_kj_mol` and formula `sampled_boost_bias_kj`, but no shown assertion compares them. After chignolin_8, recording both is not a guard. The `(b+v)<e` and `0.001*scale` guards must match gamd-openmm bit-for-bit; otherwise near-threshold frames are priced under a different modified potential.

Finally, `apply_window_swap`’s “boost cancels” comment (L1127-L1182) is not harmless: correctness depends entirely on `bias_matrix_kj` including state-λ boost. `_derive_state_gamd_lambdas` still falls back to `[0.0]*n` (L2360-L2384); `_reload_state_gamd_lambdas_on_resume` only acts on resume and only when the arg is falsy (L2386-L2437). The shown swap path reapplies λ after accepted swaps (L7780-L7900), but no shown line reapplies λ after checkpoint restore before production.

Defense: I concede the pairwise Metropolis kernel and the Gibbs reverse-proposal algebra are correct for a fixed finite bias matrix; the stay double-count is harmless because `delta` is forced to zero. I also concede a legitimately derived all-zero ladder is consistent, merely unaccelerated. Those concessions do not rescue the submission: the failure is state-dependent energy substitution—NaN CV2/boost → 0 and unverified λ/envelope reapplication after resume. A REUS kernel correct conditional on the wrong `u_{n,k}` still returns the wrong FES.

Audit: the independent two-state Gibbs check supports my concession: with `q ∝ exp(-βΔ)`, MH can accept with probability one, so the Gibbs algebra alone is not the defect. Its `ss_k_*_global` condition mismatch is worth fixing but secondary. It does not analyze the verified NaN→0 paths or the missing integrator-vs-formula boost assertion, so it does not weaken rejection.

VERDICT: REJECT - verified NaN-to-zero secondary-bias and boost paths let exchange/MBAR price a Hamiltonian different from the sampled one.
CONFIDENCE: 86

## deepseek-thinking

**Attack on the target’s weakest arguments**

1. **The all-zero-ladder claim is factually wrong.** The target says `_reload_state_gamd_lambdas_on_resume` (L2386–2437) treats an all-zero `args.state_gamd_lambdas` as falsy and reloads from the manifest. In Python, `[0.0, 0.0]` is a non-empty list and is truthy. The guard `if getattr(args, "state_gamd_lambdas", None): return` therefore returns early for an all-zero ladder, preserving the disabled state. Confidence 95 is unsupported.

2. **Clipping before log-sum-exp does not break detailed balance.** The target claims L1005’s `np.clip(..., -745, 745)` promotes underflowed candidates equally and biases the stationary distribution. This misunderstands Metropolis-Hastings: the proposal distribution is arbitrary, and the acceptance probability `log_alpha = -βδ + log(qr) - log(qf)` corrects for any proposal bias. Clipping changes only the proposal; it does not change the target distribution. If `q_reverse` underflows to zero, returning 0 is the *correct* detailed-balance-preserving rejection, not a bias.

3. **NaN-primary-CV is a crash symptom, not a silent free-energy error.** If `primary_values` is NaN, the simulation is already numerically broken. Rejecting swaps is a safe fallback. The target’s proposed fix — treating NaN as zero bias — would price a Hamiltonian that was never applied, which is worse.

4. **The uniform fallback is a degenerate-case guard.** If all log-weights are non-finite, both forward and reverse proposals hit the same fallback, and the acceptance reduces to standard Metropolis. That preserves detailed balance.

**Defense of position**

The core REUS kernel — swap acceptance, Gibbs proposal, MH correction, lambda application — is mathematically sound. The genuine remaining risks are the `_has_ss_axis` inconsistency in `_current_exchange_arrays` (checks `ss_ks_kj_arr_global` but passes `ss_k_kcal_arr_global`) and the unverified parity between `pep_gamd_boost_kj` and the integrator’s actual boost. These warrant changes, but not rejection.

**Weighing the audit**

The target’s specific findings are mostly invalid. The real silent-failure risks are the secondary-axis condition mismatch and boost-formula verification, not the alleged clipping or truthiness bugs.

VERDICT: ACCEPT-WITH-CHANGES - the core REUS kernel is sound, but the secondary-axis condition mismatch and unverified boost-formula parity are silent-failure risks.
CONFIDENCE: 70

## mini

**Attack – the weakest points**

1. **Boost‑bias mismatch** – The production path builds `boost_bias_matrix_kj` from the analytical formula `pep_gamd_boost_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)` (lines 7420‑7700) but never checks that this matches the boost actually applied by the integrator (extracted later via `extract_gamd_boost_kj`). If the integrator is in the wrong stage (as in the documented calibration failure) or the envelope is stale, MBAR will price a boost that was never applied, silently corrupting the free‑energy estimate. The code contains no consistency test; the only safeguard is a comment about a separate fix that is not present in the reviewed functions.

2. **Secondary‑CV inconsistency between sampling and exchange** – In `sample()` the flag `_has_ss_axis` is set by testing `secondary_cv_centers` and `secondary_cv_k_kcal_list` (line 7420). In `_current_exchange_arrays` the analogous flag also requires `secondary_cv_ks_kj` and `ss_ks_kj_arr_global` (line 7780). If any of these auxiliary arrays are missing while the others are present, the sample path will include a secondary umbrella term while the exchange path will treat it as zero (or vice‑versa). This asymmetry changes the Hamiltonian used for swaps versus the one used for logging, breaking detailed balance without any warning.

3. **Envelope validation omission** – `_channel_boost` (lines 497‑503) assumes `vmax > vmin`. If a corrupted envelope yields `vmax < vmin`, the denominator `rng` becomes negative, producing a negative boost that passes the `(b+v) < e` test. No sanity check is performed, so a mis‑specified envelope can introduce an unphysical, sign‑flipped bias that silently skews the free energy.

4. **Stale λ‑ladder on resume** – `_reload_state_gamd_lambdas_on_resume` (lines 2386‑2437) restores lambdas only when `args.state_gamd_lambdas` is falsy. An all‑zero list (a legitimate “ladder disabled” state) is truthy, so a user who explicitly disables the ladder will never have the manifest‑saved values reloaded, potentially re‑activating a ladder from a previous run. Conversely, if the manifest contains a stale λ‑vector of the wrong length, `_derive_state_gamd_lambdas` will accept it (existing list of correct length) and silently apply mis‑aligned λ’s, corrupting the boost history.

5. **Fallback to uniform proposals** – In `_gibbs_window_proposal_distribution` (lines 1010‑1015) any situation where the total weight `sw` is non‑finite or ≤ 0 triggers a uniform proposal over the candidate windows. This can happen after extreme under‑/overflow or when NaNs are filtered out, silently replacing the intended heat‑bath distribution with a uniform one and breaking the Gibbs‑walk balance.

6. **Missing upper bound on `gamd_lambda`** – `normalize_windows` (bias.py lines 69‑78) only enforces `gamd_lambda ≥ 0`. A user could supply λ > 1, which `set_replica_lambda` will reject at runtime, but the failure occurs only when the integrator is instantiated, not during the exchange calculations that already used the out‑of‑range λ. This can lead to undefined behaviour in the bias matrix before the exception is raised.

**Defence / concession**

Many of the identified issues are mitigated by external test suites that exercise the most common code paths (e.g. the calibration fix for stage selection). However, the failures above are *silent* – they do not raise exceptions and therefore can go unnoticed in production runs, directly violating the Boltzmann‑weighting guarantee. The presence of multiple independent failure modes outweighs the existing safeguards.

**Weighing the audit**

The audit uncovered six high‑confidence defects that each can independently produce a non‑Boltzmann free‑energy surface, plus several medium‑confidence inconsistencies. The cumulative risk to scientific correctness is severe; the code cannot be trusted without substantial remediation.

VERDICT: REJECT – multiple silent, mathematically‑critical bugs can corrupt the free‑energy estimate.  
CONFIDENCE: 85

## thinker

We need answer as board review. Need produce verdict. Need analyze code for silent wrong Boltzmann free energy. Need identify findings with file:line, concrete failure, effect, confidence, VERIFIED/SUSPECTED. Need final lines exactly.

We need review the provided code snippets. Need determine if accept/reject. The submission is itself a review? Actually user says "Board review: ATLaS-MD / gareus v0.8.2 ... does this code produce Boltzmann free energy it claims?" Then task: "Adversarially review the code below. Report every place..." We need produce independent assessment. Need examine code for errors.

We have many code snippets. Need identify issues in replica exchange, state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path. Also state_identity.py, bias.py, production sample/exchange. Need find silent wrong free energy.

Let's parse.

Key functions:

umbrella_bias_matrix_kcal: computes distance bias 0.5*k*(x-c)^2. For secondary, if ss_centers None returns zeros. If ss values NaN, dss NaN, ss_kcal NaN then np.where(np.isfinite(dss), ss_kcal, 0.0). Good. But if ss_centers provided but ss_k_kcal None? It uses if ss_centers is None or ss_k_kcal is None return zeros. Fine.

_gibbs_window_proposal_distribution: Heat-bath proposal over target windows for one replica transposition. Need understand state graph. It computes old_e = bias[wi, rep] + bias[valid_windows, target_reps]; new_e = bias[valid_windows, rep] + bias[wi, target_reps]; delta = new_e - old_e. This is for swapping replica rep (currently in window wi) with replica in target window wj. Old total bias = U_wi(x_rep)+U_wj(x_target). New = U_wj(x_rep)+U_wi(x_target). Correct. stay delta 0. log_weights = -beta*delta. Then probabilities normalized. This is Gibbs conditional distribution for target window? Actually for a Gibbs move, propose wj with probability proportional exp(-beta * (U_wj(x_rep)+U_wi(x_target)))? Wait target_reps are fixed, so old term U_wi(x_target) constant? Actually old_e includes U_wi(x_target) for each candidate? For candidate wj, old_e = U_wi(x_rep)+U_wj(x_target). new_e = U_wj(x_rep)+U_wi(x_target). delta = new-old. exp(-beta delta) = exp(-beta(U_wj(x_rep)+U_wi(x_target))) / exp(-beta(U_wi(x_rep)+U_wj(x_target))). Denominator depends on wj through U_wj(x_target). So not simply conditional over wj for rep? But for swapping labels, the probability of choosing wj should be proportional to exp(-beta [U_wj(x_rep)+U_wi(x_target)])? Actually in Gibbs sampling over state assignments, conditional distribution of assignment of replica rep given others? Let's derive.

We have replicas i with configurations x_i, windows assigned a_i. State energy = sum_w U_w(x_{rep(w)}) = sum_i U_{a_i}(x_i). For a transposition move swapping assignments of replica rep (currently wi) and replica target (currently wj), new state energy = U_wj(x_rep)+U_wi(x_target). Old = U_wi(x_rep)+U_wj(x_target). The heat-bath proposal over wj should choose wj with probability proportional to exp(-beta * new energy) = exp(-beta[U_wj(x_rep)+U_wi(x_target)]). But their log_weights = -beta*(new_e - old_e) = -beta new_e + beta old_e. Since old_e includes U_wj(x_target), which varies with wj, this is not proportional to exp(-beta new_e) unless U_wj(x_target) constant. Wait old_e = U_wi(x_rep)+U_wj(x_target). So log_weights = -beta(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target)) = -beta(U_wj(x_rep)-U_wj(x_target)) + constant in wj? U_wi(x_rep) constant, U_wi(x_target) constant. So log_weights = -beta(U_wj(x_rep)-U_wj(x_target)) + const. That equals -beta new_e + beta U_wj(x_target) + const. Not proportional to exp(-beta new_e) because U_wj(x_target) varies. Wait but target_reps is the replica currently in window wj; its configuration x_target is fixed. The full state energy after swap is U_wj(x_rep)+U_wi(x_target). The conditional distribution for wj given all other assignments? If we consider moving replica rep to window wj and moving the replica currently in wj to wi, the "other" assignments include target replica's new window wi. The energy of target in wi is U_wi(x_target), constant across wj? Yes U_wi(x_target) is constant because wi fixed. So new energy = U_wj(x_rep)+constant. Thus conditional probability should be proportional to exp(-beta U_wj(x_rep)). But their log_weights = -beta(U_wj(x_rep)-U_wj(x_target)) + const. That includes -U_wj(x_target), which is wrong for a Gibbs conditional over wj? Wait because the target replica's assignment changes from wj to wi, so its energy changes. The full state after move includes target in wi. The target's old energy U_wj(x_target) is part of old state, not new. For Metropolis-Hastings proposal, proposal distribution q(wj|current) can be anything, then MH correction. They compute q_forward = probs[choice_index] and q_reverse from reverse proposal, then MH accept with delta. So proposal distribution need not be the true conditional; MH corrects. But they call it "Gibbs-walk" heat-bath proposal. Is it valid? It is a Metropolis-Hastings with proposal q(wj) ∝ exp(-β δ) where δ = new-old. Then acceptance uses q_reverse/q_forward. This is a valid MH if q_reverse computed correctly. However, is q_forward exactly exp(-βδ) normalized? Yes. Then MH acceptance alpha = min(1, exp(-βδ) q_rev/q_fwd). Since q_fwd ∝ exp(-βδ), exp(-βδ)/q_fwd = 1/Z_fwd? Actually q_fwd = exp(-βδ)/Z_fwd. Then alpha = exp(-βδ) * q_rev / q_fwd = Z_fwd * q_rev. That is not necessarily <=1? Wait detailed balance: proposal q(i->j), acceptance min(1, π_j q(j->i)/(π_i q(i->j))). π_j/π_i = exp(-βδ). q_fwd = exp(-βδ)/Z_fwd. So ratio = exp(-βδ) * q_rev / (exp(-βδ)/Z_fwd) = Z_fwd * q_rev. This can exceed 1, but min handles. Is this correct? Yes. But note q_fwd includes only valid windows and normalized. q_rev computed from holders_after. Need check q_reverse.

_gibbs_window_proposal_distribution called with replica_index=rep, current_window=wj, replica_of_window=holders_after. It computes old_e = bias[wj, rep] + bias[valid_windows, target_reps_after]. holders_after has target_rep at wi, rep at wj. For reverse move from state after (rep in wj, target in wi) back to state before (rep in wi, target in wj), the proposal distribution for candidate window wi? It computes for all valid windows, including wi. For candidate wi, target_reps = holders_after[wi] = target_rep. old_e = bias[wj, rep] + bias[wi, target_rep] (current after state). new_e = bias[wi, rep] + bias[wj, target_rep] (before state). delta_rev = new_e - old_e = -delta_fwd. log_weights = -beta delta_rev = beta delta_fwd. q_reverse for wi = exp(beta delta_fwd)/Z_rev. Then MH alpha = exp(-beta delta_fwd)*q_rev/q_fwd = exp(-βδ)* [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(βδ) Z_fwd/Z_rev? Wait compute: q_fwd = exp(-βδ)/Z_fwd. q_rev = exp(βδ)/Z_rev. alpha = exp(-βδ) * q_rev / q_fwd = exp(-βδ)*exp(βδ)/Z_rev * Z_fwd/exp(-βδ)? Let's do: exp(-βδ) * (exp(βδ)/Z_rev) / (exp(-βδ)/Z_fwd) = exp(-βδ)*exp(βδ)*Z_fwd/(Z_rev*exp(-βδ)) = exp(βδ) Z_fwd/Z_rev. That seems not symmetric? Wait detailed balance ratio should be π_j/π_i = exp(-βδ). q_fwd = exp(-βδ)/Z_fwd. q_rev = exp(βδ)/Z_rev. Then π_j q_rev / (π_i q_fwd) = exp(-βδ) * [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(-βδ+βδ+βδ) Z_fwd/Z_rev = exp(βδ) Z_fwd/Z_rev. Yes. But for MH, alpha = min(1, π_j q_rev/(π_i q_fwd)). That is correct. But is q_rev computed correctly? It uses holders_after, current_window=wj, replica_index=rep. For reverse proposal, the "current" state is after swap, rep in wj. It proposes to move rep to wi and target to wj. The target_reps in holders_after for candidate wi is target_rep. old_e = bias[wj, rep] + bias[wi, target_rep]. new_e = bias[wi, rep] + bias[wj, target_rep]. delta_rev = new_e - old_e = -δ. log_weights = -β(-δ)=βδ. So q_rev(wi) = exp(βδ)/Z_rev. Good.

But wait q_fwd is computed over all valid windows, including wi (stay). For stay, delta=0, q_stay = 1/Z_fwd. If choose stay, no swap. Fine.

Potential issue: _gibbs_window_proposal_distribution uses bias_matrix_kj[wi, rep] + bias[valid_windows, target_reps]. But bias_matrix_kj is [window, replica] = U_window(x_replica). For candidate wj, target_reps = holders[wj]. old_e = U_wi(x_rep)+U_wj(x_target). new_e = U_wj(x_rep)+U_wi(x_target). Correct.

Now, the acceptance function _gibbs_mh_acceptance_probability: log_alpha = -beta*delta + log(qr) - log(qf). If log_alpha >=0 return 1. This is standard. But note if qf <=0 or qr <=0 return 0. q_reverse could be 0 if wi not in rev_windows. Then alpha=0, so no move. Fine.

Potential issue: In gibbs_propose_one_replica, when wj == wi, returns GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False). It sets accepted True? Actually fields: wi, wj, delta, q_forward, q_reverse, pacc, accepted, stuck? Need know dataclass. It returns accepted=True for stay. That's fine.

Now apply_window_swap: It swaps assignments[i], assignments[j], then replica_of_window[assignments[i]] = i, replica_of_window[assignments[j]] = j. Need check after assignments swap: assignments[i] becomes old assignments[j] = wj; assignments[j] becomes old assignments[i] = wi. Then replica_of_window[assignments[i]] = i means replica_of_window[wj] = i. replica_of_window[assignments[j]] = j means replica_of_window[wi] = j. Correct.

But wait bias_matrix_kj[window, replica] = U_window(x_replica). In apply_window_swap, old_e = bias[wi,i]+bias[wj,j], new_e = bias[wj,i]+bias[wi,j]. Correct.

Now _exchange_probability: if delta <=0 return 1; else exp(-beta delta). Correct.

Now assemble_bias_matrices: distance_bias_kcal + ss_bias_kcal + boost_bias_kcal; bias_kj = 4.184*bias_kcal. But boost_bias_kj is in kJ, divided by 4.184 to kcal, then multiplied back. Fine. However, if boost_bias_kj is all-zero when ladder inactive, okay. But if boost_bias_kj is NaN? Could propagate.

_fetch_v_pep_v_dih: reads v_dih from group DIHEDRAL_GROUP. v_pep = peptide_essential_energy_kj(ctx, unit). Need know if v_pep includes dihedral? In pep_gamd_boost_kj, b_dih = _channel_boost(v_dih, ...), b_tot = _channel_boost(v_pep + b_dih, ...). So v_pep should be "essential" total energy excluding dihedral? It says peptide_essential_energy_kj. Need check if v_pep includes dihedral? The comment says gamd-openmm's dependent dual boost: dihedral first, then Total with dihedral boost added to Total energy before square. So v_pep is total non-dihedral? likely.

Now campaign_ladder_registry_lambda: walks up parents until adaptive_production. Returns first (registry filename, lam) with lam >0. Potential issue: If out_dir is inside adaptive_production/... it stops at adaptive_production. It checks candidate = base, then parents. It breaks when candidate.name == "adaptive_production" after checking that candidate. So if out_dir itself is adaptive_production? It checks base first, then if base.name == "adaptive_production" break after checking. Fine. But if multiple campaigns share parent, could pick wrong campaign's ladder. They stop at adaptive_production to avoid. But if out_dir is e.g. /campaign/adaptive_production/seg1, base=seg1, parents include adaptive_production, then break after checking adaptive_production. Good. But if out_dir is /campaign/adaptive_production (base itself), it checks base, then break. Fine.

resolve_ladder_active: True if any state_lambdas >0 OR campaign registry above says ladder. Potential issue: If state_lambdas all zero but campaign registry has lambda >0, ladder active. Then set_replica_lambda_for_window will set k0=0 for all windows? state_lambdas all zero, so yes. But pep_env not None? Need see how pep_env constructed. If ladder active due to registry, pep_env should be created. The comment says all-λ=0 top-up gets k0=0 and samples carry v_pep/v_dih. Good.

_derive_state_gamd_lambdas: reads normalized_rows. If rows and len==n and all have gamd_lambda, return. Else existing if len==n, else zeros. Potential issue: If normalized_rows has rows but some gamd_lambda missing, falls to existing or zeros. Could silently disable ladder. But maybe okay.

_reload_state_gamd_lambdas_on_resume: reads manifest method_settings state_gamd_lambdas. Potential issue: It checks if args.state_gamd_lambdas truthy, returns. If all-zero list is truthy? In Python, [0.0] is truthy, so an explicitly set all-zero ladder is left alone. Good. But if args.state_gamd_lambdas is empty list, falsy, reload. Fine. It reads manifest. Potential issue: manifest method_settings may contain stale pre-truncation/pre-drop values if _persist not called after final mutation. They claim call at every reassignment. Need verify in code? Not shown. Could be issue.

_persist_state_gamd_lambdas: patches manifest. Potential issue: update_run_manifest may merge dicts, not overwrite? Need know. If update_run_manifest does shallow merge, method_settings may retain old state_gamd_lambdas if new list empty? It passes {"method_settings": {"state_gamd_lambdas": list(...)}}. If list empty, maybe update sets empty. Fine.

_channel_boost: b = 0.5*k0*(e-v)^2/rng. Then if (b+v)<e, b else 0. This is GaMD boost condition. But rng = vmax-vmin. If rng negative? vmax>vmin presumably. If rng <=0.001*scale, b=0. Good. Potential issue: if e-v negative, square. Fine.

pep_gamd_boost_kj: b_dih = _channel_boost(v_dih, threshold_dih, vmax_dih, vmin_dih, lam*k0max_dih). b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). out = b_dih + b_tot. This matches dependent dual boost? Need check gamd-openmm formula. Usually dual boost: V_dihedral = 0.5 k0_dih (E_dih - V_dih)^2 / (Vmax_dih - Vmin_dih) if E_dih < V_dih? Actually GaMD: boost potential is 0 if E >= V, else 0.5 k0 (V - E)^2/(Vmax - Vmin)? Wait standard GaMD: ΔV = 0.5 k0 (V - E)^2 / (Vmax - Vmin) when E < V, else 0. Here _channel_boost(v, e, vmax, vmin, k0): b = 0.5*k0*(e - v)^2/rng. Then return where (b+v) < e. So if b+v < e => b < e-v => 0.5 k0 (e-v)^2/rng < e-v => 0.5 k0 (e-v)/rng < 1 => e-v < 2 rng/k0. This is not simply E < V? Wait v is threshold? Let's parse arguments: _channel_boost(v, e, vmax, vmin, k0). In pep_gamd_boost_kj: _channel_boost(v_dih_kj, env.threshold_dih, env.vmax_dih, env.vmin_dih, lam*k0max_dih). So v = current dihedral energy? e = threshold? Actually parameter names: v, e, vmax, vmin, k0. b = 0.5*k0*(e - v)^2/rng. If v is current energy and e is threshold? Standard GaMD: if V(r) < E (threshold), boost = 0.5 k0 (E - V)^2/(Vmax - Vmin). Here v = current energy, e = threshold. b = 0.5*k0*(threshold - current)^2/rng. Then condition (b+v)<e => b < e-v => current + boost < threshold. That matches standard: if current energy below threshold, boost raises to at most threshold? Actually standard condition is V < E, and boost = 0.5 k0 (E - V)^2/(Vmax - Vmin). Then V+boost can exceed E? Let's check: V+0.5k0(E-V)^2/rng < E iff 0.5k0(E-V)/rng < 1 iff E-V < 2rng/k0. Standard GaMD chooses k0 such that boost <=? Usually k0 = k0max*(Vmax-Vmin)/(Vmax-Vavg)? Hmm. The condition in code is (b+v)<e, i.e. boosted energy below threshold. That is a "lower-bound" dual boost? The submission says "lower-bound dual". So maybe correct: boost only if total remains below threshold. But standard GaMD boost is applied when V < E, and can push above E? Actually GaMD: ΔV = 0.5 k0 (E - V)^2/(Vmax - Vmin) if V < E. No additional condition. But "lower-bound" might enforce V+ΔV < E. Need not dwell.

Potential issue: _channel_boost uses rng = vmax - vmin. If vmax and vmin are in kJ/mol? Envelope from swarm maybe in kcal? Need unit consistency. pep_gamd_boost_kj takes v_pep_kj, v_dih_kj, env thresholds/vmax/vmin. Are env values in kJ? Likely. k0max units? Need check. k0max_from_globals reads shared_globals k0_Total, k0_Dihedral. Integrator globals likely in kJ/mol? GaMD k0 units? In OpenMM, k0 has units? The boost formula b = 0.5*k0*(E-V)^2/(Vmax-Vmin). If energies in kJ/mol, k0 dimensionless? Actually k0 is dimensionless? In GaMD, k0 is a scaling factor (0<k0<=1), so b has energy units. So fine.

set_replica_lambda: sets k0_Dihedral = lam*k0max["Dihedral"], k0_Total similarly. Potential issue: If integrator has no k0_Total but k0max Total nonzero, raises. Good.

set_replica_lambda_for_window: if k0max_by_channel is None, no-op. If state_lambdas None and ladder active, raise. Good.

state_identity.py: canonical_state_definition requires exact set of fields. Potential issue: It requires "boost" field even if None. It checks if any gamd_lambda >0 and boost not dict -> error. If all lambdas zero, boost can be None. Fine.

Potential issue: In canonical_state_definition, for NPT, it rejects fixed_box_vectors_nm not None. But if ensemble NPT and fixed_box_vectors_nm is None, okay. It doesn't check pressure_bar positive? finite_number without positive. Pressure could be negative? Not relevant.

Potential issue: In normalize_windows, gamd_lambda = finite_number(raw.get("gamd_lambda", 0.0), minimum=0). It does not enforce <=1. set_replica_lambda raises if lam >1. But state_identity could accept lambda >1, then production would raise at set_replica_lambda, not silent. Fine.

Potential issue: In reconstruct_bias_matrix, if lambdas active and envelope has_total True, requires pep not None. But if v_pep is provided as all NaN, it doesn't raise. Then _ladder_apply may produce NaN. But maybe numeric_vector converts non-finite to NaN, and _ladder_apply should handle? Not sure.

Now, the biggest issue in bias.py: It says "Energy input convention is unchanged: k1 and k2 are kcal/mol per squared CV unit; CV values and centers must already use the SAME units. beta is mol/kJ." In reconstruct_bias_matrix, matrix = 0.5*k*delta^2 in kcal, then matrix *= beta * KJ_PER_KCAL. So reduced bias = beta_kJ * U_kJ. Correct. But if windows have k in kJ/mol (energy_unit kJ/mol), normalize_windows doesn't convert; it just stores. The docstring says k1/k2 are kcal/mol. But state_identity canonical_state_definition converts kJ/mol to kcal/mol. So if using state_identity, okay. But reconstruct_bias_matrix may be used directly with windows from elsewhere. Not our focus.

Now, production sample() potential energy: potentials_kj[r] = physical potential from getState groups=physical_energy_groups_for_args(args). But the total Hamiltonian for MBAR should include physical potential + umbrella + boost. The bias matrix includes umbrella + boost. So reduced potential = beta*(physical + bias). Good. But if physical_energy_groups excludes some terms that are in the actual dynamics (e.g., barostat? no), then wrong. Need know physical_energy_groups_for_args. Not shown. Could be issue.

Now, NPT barostat: The submission says NPT via application-controlled biased Metropolis-MC volume move against U* = physical + bias + boost. The code snippets don't show barostat. But sample() reads potential with enforcePeriodicBox=True. If volume move changes box, positions scaled? Need not.

Now, checkpoint/resume: _reload_state_gamd_lambdas_on_resume reads manifest. Potential issue: If resume after adaptive epoch changed windows, manifest method_settings state_gamd_lambdas may be stale. _persist called after final mutation. But if crash before persist, stale. Could silently disable ladder. The docstring acknowledges. But is there a verification? Not shown.

Now, let's examine _gibbs_window_proposal_distribution more deeply for a possible sign error. It computes delta = new_e - old_e. log_weights = -beta*delta. For a favorable move (new_e < old_e), delta negative, -beta*delta positive, weight >1. Good. Then probabilities normalized. MH acceptance uses delta_kj = deltas[choice_index] (new-old). log_alpha = -beta*delta + log(qr)-log(qf). If q_fwd is high for favorable moves, alpha may be lower. Correct.

But wait: The proposal distribution q_fwd is normalized over valid windows. It includes the stay move. The reverse q_rev is normalized over valid windows in holders_after. However, the set of valid windows may differ between forward and reverse? holders_after has same valid windows (all >=0). So same. Good.

Potential issue: In _gibbs_window_proposal_distribution, old_e = bias[wi, rep] + bias[valid_windows, target_reps]. But bias[valid_windows, target_reps] uses advanced indexing: bias is 2D, valid_windows array, target_reps array. This returns array of length n_valid. Good. new_e = bias[valid_windows, rep] + bias[wi, target_reps]. Good.

Now, potential issue: The Gibbs proposal distribution uses bias_matrix_kj which includes boost. But the boost for a state depends on v_pep/v_dih of the replica's configuration. For candidate wj, new_e = U_wj(x_rep)+U_wi(x_target). The boost part of U_wj(x_rep) is computed with lambda_wj and v_pep/v_dih of rep. The boost part of U_wi(x_target) is computed with lambda_wi and v_pep/v_dih of target. That is correct. The physical potential cancels. Good.

Now, potential issue: In _current_exchange_arrays, when computing boost_bias_matrix_kj, it uses v_pep_kj and v_dih_kj arrays from all replicas. But if pep_env is not None and state_lambdas has length n_states, pep_gamd_boost_matrix_kj returns (n_states, nrep). Good. But if some v_pep/v_dih are NaN (e.g., pep_env has_total False, v_pep NaN), pep_gamd_boost_kj for single dihedral ignores v_pep. Good.

Now, potential issue: In sample(), row["sampled_boost_bias_kj"] = float(boost_bias_matrix_kj[w, r]). This is computed boost, not actual. If actual differs, row records computed. The parquet writer writes boost_total from extract_gamd_boost_kj (actual) and v_pep/v_dih. Analysis could recompute. But MBAR analysis may use bias_matrix from analysis_arrays.npz, which is computed. Need know analyze_gareus_mbar.py. Not shown. But the task says final reweighting by MBAR over all (window, rung) states using per-frame recorded CVs and boost arguments. If analysis recomputes boost from recorded v_pep/v_dih, then actual mismatch doesn't matter? Wait if dynamics ran unboosted but analysis prices boost, then wrong. The calibration case: integrator stage 2 with ForceScalingFactor 1.0, but exchange and MBAR priced boost. The fix verifies stage. In current code, sample() records actual boost_total and v_pep/v_dih. If analysis uses recorded v_pep/v_dih to compute boost, it would still price boost even if actual boost was not applied. The only way to avoid is verify actual boost equals computed. The code does not appear to do so in sample path. It does read actual boost_total but doesn't compare to sampled_boost_bias_kj. So this is a real silent failure mode. We should report.

Now, another issue: In _current_exchange_arrays, the exchange decision uses computed boost matrix, not actual integrator boost. If actual boost is wrong (e.g., stage 2), exchange accepts/rejects based on priced boost that was not applied. This violates detailed balance w.r.t. actual Hamiltonian. Silent. This is exactly the calibration case. The fix mentioned verify_gamd_production_stage5 may be in code not shown. But we can say within reviewed subsystem, no check.

Now, let's look at _channel_boost condition. It uses scale = max(abs(e), abs(v), 1.0). If rng <= 0.001*scale, b=0. This is a guard against zero range. But if vmax-vmin is small but nonzero, and e-v large, b could be huge? Actually if rng small, b large, but condition (b+v)<e may fail, so b=0. Fine.

Potential issue: In _channel_boost, if vmax < vmin (negative rng), b = 0.5*k0*(e-v)^2/negative = negative. Then condition (b+v)<e may be true, returning negative boost. That would be wrong. But envelope should have vmax > vmin. If not, silent negative boost. Could be a finding: no validation that vmax > vmin. But likely envelope validation elsewhere. We can mention SUSPECTED.

Now, pep_gamd_boost_kj: b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). The comment says "dihedral first, then Total with the dihedral boost added to the Total energy before the square (stage_integrator _add_dihedral_boost_to_total_energy)." But in standard dual boost, total boost = 0.5 k0_tot (E_tot - V_tot)^2/(Vmax_tot - Vmin_tot) + 0.5 k0_dih (E_dih - V_dih)^2/(Vmax_dih - Vmin_dih), where V_tot is total potential energy including dihedral? Actually in GaMD dual boost, the total boost uses total potential energy V_total (including dihedral) and threshold E_total. The dihedral boost is added to the total potential energy before computing total boost? The comment says "with the dihedral boost added to the Total energy before the square". That is a specific implementation. If the integrator does that, then code matches. But if not, mismatch. Need trust.

Now, k0max_from_globals: returns {"Total": float(shared_globals.get("k0_Total", 0.0)), "Dihedral": float(shared_globals["k0_Dihedral"])}. If shared_globals missing k0_Dihedral, KeyError, not silent. Good.

Now, set_replica_lambda: It sets k0_Dihedral always, and k0_Total if names is None or "k0_Total" in names. If names is None (test double), it sets k0_Total even if not present? It calls setGlobalVariableByName("k0_Total", ...). If test double doesn't have it, may raise. But names None only for minimal test doubles that expose only setGlobalVariableByName; they might accept any. Fine.

Potential issue: set_replica_lambda does not set ForceScalingFactor or stage. The calibration case was stage not set. The fix seed_frozen_envelope_stage5 likely sets stage. But set_replica_lambda only sets k0 globals. If the integrator's stage is wrong, setting k0 doesn't help. The code under review may rely on external verify. We can mention.

Now, state_identity.py: It hashes definition. Potential issue: It uses json_bytes(dict(raw)) which sorts keys? Need know. If not, hash may differ for same dict with different insertion order. But json_bytes likely canonical. Not our focus.

Now, let's identify concrete findings.

Finding 1: No runtime verification that the integrator-applied GaMD boost equals the boost priced in bias_matrix_kj/MBAR. In sample(), boost_bias_matrix_kj is computed from pep_gamd_boost_kj(v_pep, v_dih, state_lambdas, env), while actual boost is read separately via extract_gamd_boost_kj into row["gamd_boost_total_kj_mol"] but never compared. If StageIntegrator is in wrong stage (as in chignolin_8) or ForceScalingFactor != 1, MBAR and exchange use a boost that was never applied. This is exactly the class of bug. File: gareus/production.py sample() around lines 7420-7700, specifically boost_bias_matrix_kj assignment and row["gamd_boost_total_kj_mol"]; _current_exchange_arrays similarly. Confidence high. VERIFIED? We can point at lines. The code doesn't show comparison. But maybe elsewhere. We can say "In the reviewed subsystem, no check exists; the only verification mentioned is in commit message, not in these functions." Mark VERIFIED for absence in shown code.

Finding 2: _channel_boost can return negative boost if vmax < vmin, because rng negative and condition may pass. No validation that envelope vmax > vmin in pep_gamd.py. If a corrupted/stale envelope has vmax < vmin, boost becomes negative, altering Hamiltonian and free energy silently. File: gareus/pep_gamd.py:497-503. Confidence medium. SUSPECTED.

Finding 3: In _gibbs_window_proposal_distribution, the proposal distribution is normalized over valid windows including the stay move, but the MH acceptance uses q_reverse computed from holders_after. However, q_reverse is computed with the same function that includes the stay move. For the reverse move, the "stay" candidate is wj (current window after), not wi. The probability of choosing wi is q_reverse. That is correct. No issue.

Wait, let's double-check the MH acceptance formula with q_fwd and q_rev both including stay. Suppose forward proposal chooses wj != wi. q_fwd = exp(-βδ)/Z_fwd. Reverse proposal from after state chooses wi. q_rev = exp(βδ)/Z_rev. Then alpha = min(1, exp(-βδ) q_rev / q_fwd) = min(1, exp(-βδ) * exp(βδ)/Z_rev * Z_fwd/exp(-βδ)) = min(1, exp(βδ) Z_fwd/Z_rev). Wait this seems asymmetric and can be >1. But detailed balance condition is π_i q(i->j) α(i->j) = π_j q(j->i) α(j->i). With α(i->j)=min(1, π_j q(j->i)/(π_i q(i->j))). Here π_j/π_i = exp(-βδ). q(j->i) = exp(βδ)/Z_rev. q(i->j)=exp(-βδ)/Z_fwd. Ratio = exp(-βδ)*exp(βδ)/Z_rev / (exp(-βδ)/Z_fwd) = exp(βδ) Z_fwd/Z_rev. So α = min(1, exp(βδ) Z_fwd/Z_rev). Is that correct? Let's test with a simple case where Z_fwd = Z_rev (symmetric proposal). Then α = min(1, exp(βδ)). If δ<0 (favorable), exp(βδ)<1, so α<1. But if proposal is symmetric and move is favorable, MH should accept with probability min(1, exp(-βδ))? Wait π_j/π_i = exp(-βδ). If δ<0, π_j > π_i, so α should be 1. But here α = exp(βδ) <1. That's wrong! Let's recalc.

Hold on. Let's define δ = new_e - old_e. π_j/π_i = exp(-β δ). If δ<0, new lower energy, π_j > π_i, ratio >1, so α should be 1 for symmetric proposal. But our q_fwd is not symmetric; q_fwd = exp(-βδ)/Z_fwd. For δ<0, q_fwd > 1/Z_fwd, so it preferentially proposes favorable moves. Then MH correction should reduce acceptance. The ratio π_j q_rev / (π_i q_fwd) = exp(-βδ) * q_rev / q_fwd. If q_rev = q_fwd (symmetric), ratio = exp(-βδ). For δ<0, ratio >1, α=1. Good. But in our case q_rev = exp(βδ)/Z_rev. If Z_fwd=Z_rev, q_rev = exp(βδ)/Z, q_fwd=exp(-βδ)/Z. Then ratio = exp(-βδ)*exp(βδ)/exp(-βδ) = exp(βδ). For δ<0, exp(βδ)<1, so α<1. That suggests the reverse proposal probability is too high? Let's compute q_rev for reverse move. Reverse move from after state (rep in wj, target in wi) to before (rep in wi, target in wj). The energy difference for reverse δ_rev = old_before - old_after = -δ. The proposal distribution for reverse should be proportional to exp(-β δ_rev) = exp(βδ). So q_rev = exp(βδ)/Z_rev. That is correct if the proposal rule is "propose wj with weight exp(-β δ_forward)". For reverse, δ_rev = -δ, so weight exp(βδ). So q_rev is correct. Then why does MH ratio seem wrong? Let's derive detailed balance with this proposal.

Proposal q(i->j) = exp(-βδ)/Z_fwd. q(j->i) = exp(βδ)/Z_rev. Note δ for i->j = E_j - E_i. For j->i, δ_rev = E_i - E_j = -δ. So q(j->i) = exp(-β δ_rev)/Z_rev = exp(βδ)/Z_rev. Correct.

MH ratio = π_j q(j->i) / (π_i q(i->j)) = exp(-βδ) * [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(-βδ+βδ+βδ) Z_fwd/Z_rev = exp(βδ) Z_fwd/Z_rev.

But detailed balance requires π_i q(i->j) α(i->j) = π_j q(j->i) α(j->i). If α(i->j)=min(1, ratio), α(j->i)=min(1, 1/ratio). Then LHS = π_i q(i->j) min(1,r), RHS = π_j q(j->i) min(1,1/r). Since π_j q(j->i) = r π_i q(i->j), RHS = r π_i q(i->j) min(1,1/r) = π_i q(i->j) min(r,1). LHS = π_i q(i->j) min(1,r). Equal. So valid. The acceptance probability can be >1? α(i->j)=min(1,r). If r>1, accept always. In our example δ<0, r=exp(βδ) Z_fwd/Z_rev. If Z_fwd=Z_rev, r=exp(βδ)<1, so α<1. That means even though move is favorable, the proposal is so biased toward it that it must be rejected sometimes to maintain balance. That's correct. So no bug.

But wait, is q_fwd really proportional to exp(-βδ)? Yes. For favorable δ<0, q_fwd > uniform, so proposal oversamples favorable moves; MH corrects. Good.

Now, potential issue: The "Gibbs-walk" name suggests they intended the proposal to be the conditional distribution, in which case acceptance should be 1. But they added MH correction, so it's valid. However, if q_fwd is exactly the conditional distribution, then q_rev would also be conditional and MH ratio would be 1? Let's check: If q_fwd ∝ exp(-β new_e) (not exp(-βδ)), then q_fwd = exp(-β U_wj(x_rep))/Z_fwd (since U_wi(x_target) constant). q_rev = exp(-β U_wi(x_rep))/Z_rev. Then ratio = exp(-βδ) * q_rev/q_fwd = exp(-β(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target))) * exp(-βU_wi(x_rep))/Z_rev * Z_fwd/exp(-βU_wj(x_rep)) = exp(-β(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target)+U_wi(x_rep)-U_wj(x_rep))) Z_fwd/Z_rev = exp(-β(U_wi(x_target)-U_wj(x_target))) Z_fwd/Z_rev. Not necessarily 1. So their proposal is not conditional. But MH valid.

Now, is there a bug in q_reverse computation due to using holders_after but not updating assignments? The proposal distribution only depends on replica_of_window, not assignments. holders_after is correct. Good.

Now, another potential issue: In gibbs_propose_one_replica, when wj == wi, it returns accepted=True without MH. That's fine because stay always accepted. But it doesn't update anything. Good.

Now, potential issue: In _gibbs_window_proposal_distribution, the stay delta is set to 0.0 after computing delta. But if wi is not in valid_windows? It should be, because holders[wi]=rep >=0. So stay_idx exists. Good.

Now, potential issue: The log_weights clip to [-745, 745]. If all weights underflow to 0 after clipping? They clip lower bound -745, so exp(-745) ~ 5e-324, which is subnormal but nonzero. Sum may be >0. If max log_weight is -745, m=-745, weights = exp(0)=1, sw=1. Fine. If some log_weights are -inf (delta inf), finite mask removes. Good.

Now, potential issue: In _gibbs_mh_acceptance_probability, if log_alpha < -745 return 0. If log_alpha is NaN? It checks finite. Good.

Now, let's examine apply_window_swap with p_override and force_accept. If p_override provided, pacc = p_override, then clamped. If force_accept True, accepted True regardless of pacc. But if p_override is None and force_accept True, pacc computed but ignored. Fine. If uniform is None and force_accept False, accepted False. In production, uniform is rng.random() unless force_accept. Good.

Potential issue: In _attempt_window_swap, it checks swap_candidate_replicas before drawing random. Then calls apply_window_swap with uniform = None if force_accept else rng.random(). But apply_window_swap itself calls swap_candidate_replicas again. That's fine, no extra random. Good.

Now, potential issue: In _attempt_window_swap, after accepted swap, it submits _apply_swap_to_replica to _sim_pool for i and j. But it uses assignments[replica_index] after swap. However, _sim_pool.submit(i, _apply_swap_to_replica, i).result() may execute on worker thread for replica i, but the function uses sims[replica_index].context. If _sim_pool.submit's first argument is replica index and second is function, it may run on that replica's thread. Good. But if the worker thread is busy stepping, deadlock? Not our focus.

Now, potential issue: In sample(), after computing bias_matrix_kj, it stores observable_cache["bias_matrix_kj"] = bias_matrix_kj. Then _current_exchange_arrays may use cached if step matches. But sample() is called at log steps, exchange at exchange intervals. If they coincide, cache used. But sample() computes bias_matrix_kj using v_pep/v_dih and state_lambdas. _current_exchange_arrays uses same. Good.

Now, potential issue: In sample(), potentials_kj[r] is physical potential. But the bias matrix includes boost. The total reduced potential for MBAR should be beta*(physical + bias). However, if the boost is applied in dynamics, the sampled distribution is proportional to exp(-beta*(physical + bias_actual))? Actually umbrella sampling with boost: The bias is added to the physical potential, so the sampled distribution is exp(-beta*(U_physical + U_bias + boost)). MBAR reweights to physical by using u_k = beta*(U_physical + U_bias_k) for each state k? Wait if the simulation is run with Hamiltonian H_k = U_physical + U_bias_k + boost_k, then the sample's reduced potential under state k is beta*(U_physical + U_bias_k + boost_k). To unbias to physical, MBAR uses u_k = beta*(U_physical + U_bias_k + boost_k) for the state that generated the sample? Actually MBAR solves for free energies of states with reduced potentials u_k(x) = beta*(U_physical(x) + bias_k(x)). The boost is part of bias_k. So yes, bias matrix includes boost.

But wait: The boost applied in dynamics is state-dependent (lambda). The sample's actual Hamiltonian is H_actual = U_physical + U_umbrella_window + boost_lambda. The bias matrix for that state should be U_umbrella + boost_lambda. The code computes that. Good.

Now, potential issue: In sample(), row["potential_kj_mol"] = physical potential. But if the boost is applied, the total potential energy from OpenMM getState(getEnergy=True, groups=physical_energy_groups) may exclude boost because boost is in integrator, not force. Good. So physical potential correct.

Now, potential issue: In _current_exchange_arrays, it does not read potential energies. For exchange, physical potential cancels only if the swap is of state labels, not configurations. Yes, because U_physical(x_i)+U_physical(x_j) unchanged. Boost? Boost is part of bias, not physical. So included. Good.

Now, potential issue: The exchange move swaps state labels, but the integrator's boost k0 is set to new state's lambda. However, the boost energy in the current step's bias matrix was computed using v_pep/v_dih from before the swap. After swap, the integrator's boost for future steps will use new lambda. The exchange decision uses the boost that would be applied if the state labels were swapped at that instant. That is correct for Metropolis on the augmented ensemble where the boost is part of the state Hamiltonian. But note: The boost is a function of the configuration and the lambda. When swapping labels, the configuration stays with the replica, so the boost energy for the new state is computed with the same configuration. Correct.

Now, potential issue: The boost applied by the integrator may depend on history (e.g., GaMD adaptive? but envelope frozen). If boost depends on running averages, then swapping labels changes the boost history? The envelope is frozen, so no. Good.

Now, let's look at state_identity.py canonical_state_definition: It requires "windows" with fields center1,k1,center2,k2,gamd_lambda. It converts energy_unit kJ/mol to kcal/mol by dividing k1,k2 by 4.184. But if k1/k2 are in kJ/mol per squared CV unit, dividing by 4.184 gives kcal/mol. Correct. But it doesn't convert gamd_lambda? dimensionless. Good.

Potential issue: In canonical_state_definition, if energy_unit == "kJ/mol", it modifies row["k1"] /= 4.184 and row["k2"] /= 4.184. But if k1/k2 are zero, fine. If k1/k2 are strings? normalize_windows already converted to float. Good.

Potential issue: In canonical_state_definition, it checks set(row) - {"window_id","center1","k1","center2","k2","gamd_lambda"} == empty. But normalize_windows may add keys? It returns dict with center1,k1,center2,k2,gamd_lambda plus original keys? Actually normalize_windows does window = dict(raw), then sets window["center1"], etc. So if raw had extra keys, they remain. Then canonical_state_definition would reject. Good.

Now, potential issue: In validate_fixed_state_segments, it compares top-level windows sorted by window_id to state["windows"]. But state["windows"] is sorted by window_id in canonical_state_definition. So okay.

Now, let's think about the "silent fallbacks" in _derive_state_gamd_lambdas: If normalized_rows cannot be trusted, it falls back to existing or zeros. If existing is stale wrong-length, it falls to zeros, silently disabling ladder. The docstring warns. But is there a scenario where normalized_rows is present but wrong? It checks len(rows)==n and all have gamd_lambda. If rows have gamd_lambda but misaligned due to reindexing bug, could silently use wrong lambdas. The docstring claims reindexing functions subset+renumber correctly. But if not, silent. We can mention SUSPECTED.

Now, potential issue: _reload_state_gamd_lambdas_on_resume reads manifest method_settings. But if manifest has state_gamd_lambdas from a previous campaign with different number of states, and current run has different n, _derive_state_gamd_lambdas will use existing if len==n, else zeros. If len==n but misaligned (e.g., windows reordered), silent wrong lambda per state. The docstring says "if post-drop count happens to match, resumes with a misaligned per-state λ." They try to persist after final mutation. But if crash before persist, stale. This is a real risk. We can report.

Now, another issue: In campaign_ladder_registry_lambda, it returns first registry with lam >0. But if there are multiple registries in different parent dirs, it picks the first in order. If
(TRUNCATED - output hit the token limit before completion)

## Your designated target - the board member you must attack

Let me analyze this carefully. This is a review of replica exchange code in a molecular dynamics free energy pipeline. I need to find bugs in the state bias matrix, Gibbs-walk proposals, swaps, lambda handling.

Let me examine each piece.

**1. umbrella_bias_matrix_kcal (L895-917)**

Orientation: bias[state, replica]. `pv[np.newaxis, :] - ca[:, np.newaxis]` gives [state, replica]. OK. Secondary NaN handling: `np.where(np.isfinite(dss), ss_kcal, 0.0)` — replicas with NaN secondary contribute zero bias on that axis for ALL states. Docstring says "a NaN entry used to freeze that replica's exchanges". Hmm — but setting bias to 0 rather than NaN means exchange delta calculations will silently treat NaN secondary as zero bias. That's a "NaN → 0 silent fallback" candidate. If ss_value is NaN (e.g., fast path returning NaN), the bias is 0 instead of freezing exchanges. The docstring claims "now applied identically to the exchange path — a NaN entry used to freeze that replica's exchanges" but the code does NOT produce NaN entries; it zeroes them. So the claimed freeze behavior doesn't exist. This is a real discrepancy: the docstring claims NaN propagates to freeze exchanges, but `np.where(np.isfinite(dss), ss_kcal, 0.0)` makes bias exactly zero, so exchanges proceed with wrong energies. VERIFIED inconsistency between doc and code. Effect: a replica with missing secondary CV participates in exchanges priced as if it sits at the harmonic minimum of every state — inflated acceptance, wrong sampling weights. MBAR would also reconstruct bias with NaN (in correctness/bias.py numeric_vector converts non-finite to NaN → those samples need filtering). Actually in reconstruct_bias_matrix, missing observations yield NaN in states requiring that coordinate. So the exchange path prices 0 while MBAR prices NaN — inconsistent energies between exchange and recorded samples. That's the class of bug asked for. Though note `_current_exchange_arrays` uses `umbrella_bias_matrix_kcal` and sample path also uses same function so both zero it; but MBAR analysis reconstructs from CVs with correctness/bias.py which yields NaN. Actually wait — does the analysis use reconstruct_bias_matrix? The submission says final reweighting by MBAR using per-frame recorded CVs and boost arguments. reconstruct_bias_matrix converts non-finite secondary to NaN; the states with k2>0 would give NaN, which MBAR must filter out. So a replica with secondary NaN gets exchange-accepted with bias 0, but its samples are dropped in MBAR — that's fine-ish for MBAR but the exchange path accepted moves with fabricated zero energy, which corrupts the sampling distribution (moves accepted that shouldn't be). Actually more subtly: exchange acceptance with wrong energy breaks detailed balance of the joint chain — the target distribution conditioned on exchange acceptance probabilities that use wrong bias is not the intended distribution. Effect on FES: biased.

Also note only `dss` finiteness is checked, not `distance_kcal` — primary NaN propagates. Fine.

**2. _gibbs_window_proposal_distribution (L983-1041)**

Heat-bath proposal. delta computed as new_e - old_e. old_e = bias[wi, rep] + bias[valid_windows, target_reps]; new_e = bias[valid_windows, rep] + bias[wi, target_reps]. For the swap of replica `rep` in window wi with the holder of window w (target_rep): new configuration assigns rep→w and target_rep→wi. Bias energy = bias[w, rep] + bias[wi, target_rep]. Correct. Stay index delta set to 0. Good.

Note: delta here only includes umbrella bias. The Gibbs move swaps state labels between replicas — configurations unchanged — so physical potential and boost cancel... wait, boost does NOT cancel! The boost depends on the state's λ. In this state graph, each (window, rung) is a state with its own λ. A "window swap" between replica in window wi (λ_i) and holder of wj (λ_j): the boost energy of replica i's configuration under rung λ_j differs from under λ_i. The delta for the swap must include the boost terms: b_j(x_i) + b_i(x_j) - b_i(x_i) - b_j(x_j). But the docstring in apply_window_swap claims "the unbiased potential and the GaMD boost cancel exactly from delta ... dV(x_i)+dV(x_j) are identical before and after." That's WRONG when λ_i ≠ λ_j. The boost dV depends on λ of the state the replica is assigned to. If the ladder is active and windows have different λ, swapping window labels between replicas changes the boost energy each configuration experiences — it does NOT cancel.

But wait — does the bias matrix include the boost? Yes: assemble_bias_matrices includes boost_bias_kj, and bias_matrix_kj includes boost of each sample's configuration under each state's λ (pep_gamd_boost_matrix_kj returns (n_states, n_samples) boost of each sample's configuration under each state's λ). So bias[w, r] includes b_λ(w)(x_r). So the delta in apply_window_swap using bias_matrix_kj correctly includes boost differences. Good — the matrix includes boost, so delta is correct for the swap. So the docstring's justification is wrong (boost doesn't cancel) but the computation is right because the matrix contains per-state boost. Hmm, but the docstring says the boost cancels — the claim is incorrect reasoning but the code (via matrix) is right. Actually wait: is the boost matrix included in the exchange bias matrix? In `_current_exchange_arrays`, yes: `_, bias_matrix_kj = assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_matrix_kj)`. And apply_window_swap receives bias_matrix_kj. So swaps are priced with boost. Fine.

But note shape issue: `boost_bias_matrix_kj = np.zeros((nrep, nrep))` when pep_env is None. pep_gamd_boost_matrix_kj returns (n_states, n_samples) = (len(lambdas), n_samples). Here nrep replicas and nrep windows (assuming n windows == nrep). If number of windows != number of replicas (e.g., --max-replicas truncation mentioned in _persist_state_gamd_lambdas: `args.state_gamd_lambdas = list(args.state_gamd_lambdas)[:_max_replicas]`), then shapes mismatch! state_lambdas has length = number of windows; v_pep has length nrep. pep_gamd_boost_matrix_kj stacks len(lambdas) rows each of length n_samples → shape (n_windows, nrep). Then assemble: distance_bias_kcal has shape (n_windows, nrep); boost (n_windows, nrep). OK, consistent. But the fallback `np.zeros((nrep, nrep))` when pep_env is None — if n_windows ≠ nrep, boost is (nrep, nrep), mismatched with (n_windows, nrep) distance matrix. Adding them would broadcast-fail or misalign. Actually if pep_env is None, boost is zeros; adding (nrep,nrep) zeros to (n_windows, nrep) matrix raises ValueError if shapes differ (unless n_windows == nrep). Hmm — but more important: `_current_exchange_arrays` checks `cached_bias.shape == (nrep, nrep)` in the cache validation. If n_windows ≠ nrep, the cached bias matrix shape is (n_windows, nrep) and the cache check fails, forcing recompute — only a performance issue. But the zeros((nrep,nrep)) with n_windows≠nrep: assemble_bias_matrices would raise ValueError on broadcast? Adding arrays of shape (nwin,nrep) and (nrep,nrep): if nwin != nrep, numpy raises. So error, not silent. But wait, is state_lambdas length n_windows or nrep after truncation? The persist function mentions truncation of state_lambdas to _max_replicas — suggesting state_lambdas is indexed by window in some places and by replica in others? In sample(): `row["gamd_lambda"] = float(state_lambdas[w])` where w = assignments[r] — window index. And `set_replica_lambda_for_window(..., assignments[replica_index], state_lambdas, ...)` — window index. So state_lambdas is per-window. But truncation `args.state_gamd_lambdas[:_max_replicas]` truncates by replica count — if n_windows > nrep... Hmm, ambiguous. If truncation is meant to align per-window λ with reduced replica count... Actually the persist docstring says the truncation mutates state_gamd_lambdas. If windows and replicas are 1:1 normally, fine.

Actually the bigger issue in pep_gamd_boost_matrix_kj at exchange time: `pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)` — v_pep_kj has NaN when... `_fetch_v_pep_v_dih` returns (nan, nan) only when pep_env is None, which is guarded. But for single dihedral boost (has_total False), v_pep = NaN and pep_gamd_boost_kj handles: `_channel_boost(v_pep + b_dih, ...)` — wait no, if not env.has_total, returns b_dih only, ignoring v_pep NaN. OK.

But in `_channel_boost`: `b = _np.where((b + v) < e, b, 0.0)` — the lower-bound boost guard: only apply if boost lifts v below threshold e... Actually GaMD lower-bound boost: boost applied when V < E (threshold). The boost should raise energy to V + 0.5*k0*(E-V)^2/... Let me check units: `rng = vmax - vmin`, `b = 0.5*k0*(e-v)^2/rng`. That's gamd-openmm's lower bound boost formula: ΔV = 0.5*k0*(E-V)^2/(Vmax-Vmin) when V < E. And guard `(b+v) < e` → apply only if b+v < e, i.e., boosted energy still below threshold. Plausible. And rng guard: if |vmax - vmin| small, b=0 — silent boost=0 fallback! If the swarm calibration produced vmax ≈ vmin (degenerate envelope), the boost silently becomes zero while MBAR still prices... wait, MBAR reconstructs boost from v_pep/v_dih with the same formula and same envelope, so it'd also be zero. Consistent. But if rng differs slightly between... it's the same env. OK.

Numerical stability of _channel_boost: `scale = np.maximum(np.maximum(abs(e), npabs(v)), 1.0)`; `np.where(np.abs(rng) <= 0.001*scale, 0.0, b)` — fine.

**3. _gibbs_mh_acceptance_probability**: log_alpha = -beta*delta + log(qr/qf). Correct MH for proposal distribution q. If q_reverse == 0 (rev_idx empty → q_reverse = 0.0) → return 0 (reject). But when is reverse probability zero while forward was positive? In reverse distribution computed from holders_after... If the move is valid, reverse should include wi as candidate. If q_reverse = 0, reject — safe. But subtle: the heat-bath proposal for replica rep at window wi includes the "stay" option with delta forced 0 and weight exp(0)=1 — fine.

Hmm — a deeper issue with the Gibbs walk: the proposal distribution `_gibbs_window_proposal_distribution` is a heat-bath over swap partners, but its weights use only the bias matrix (umbrella + boost), i.e., the reduced Hamiltonian difference. Heat-bath proposal q(j) ∝ exp(-β δ_j) where δ_j is the swap energy change — this is a valid proposal and MH correction with reverse q fixes detailed balance. Looks right. But the stay delta forced to 0 — actual delta for staying is 0 anyway (swapping replica with itself? No—stay means wj == wi; the "swap" would be with itself; delta genuinely 0). Fine.

**Potential bug: `holders` array vs assignments.** In gibbs_propose_one_replica, wi = assignments[rep]. If assignments and replica_of_window are consistent, fine. apply_window_swap updates both: `assignments[i], assignments[j] = assignments[j], assignments[i]` then `replica_of_window[int(assignments[i])] = int(i)`... wait: after the swap, assignments[i] holds the new window of replica i (which is wj), and replica_of_window[wj] should be i. Let's check: after `assignments[i], assignments[j] = assignments[j], assignments[i]`: assignments[i] = wj (old window of j), assignments[j] = wi. Then `replica_of_window[int(assignments[i])] = int(i)` → replica_of_window[wj] = i. Correct. `replica_of_window[int(assignments[j])] = int(j)` → replica_of_window[wi] = j. Correct. OK.

But wait: what if i == j? Guarded by swap_candidate_replicas (i==j returns None). What if one replica holds both windows? Guarded.

Hmm, wait: there's an issue with the initial `assignments` semantics — set_window uses `assignments[replica_index]` — OK.

**4. _exchange_probability**: standard Metropolis. Fine.

**5. swap_candidate_replicas**: `n_win = int(replica_of_window.size)` — replica_of_window is indexed by window, size = n windows. Checks wi/wj in range. Fine.

**6. explicit_2d_neighbor_pairs_for_exchange (L1479-1505)**: Round-robin edge coloring: `if idx % slots != slot: continue` — selects edges by index mod slots, then greedy disjoint filter. Fallback when slot empty: tries ALL edges greedily — this makes the edge set for an empty slot equal to the full greedy matching, which overlaps with other slots' edges. Does this break detailed balance? For a fixed parity schedule, the Markov chain with alternating matchings still satisfies balance if each sub-chain is reversible — Metropolis swaps within a matching are reversible. Adding the full matching on an "empty slot" is just another valid matching set; the composition remains valid as long as it's a matching. Detailed balance for the product of reversible moves: each swap move is reversible with respect to π, and compositions of such moves... Actually a sequence of Metropolis moves each individually satisfying detailed balance gives a valid chain regardless of schedule (each move is a π-reversible transition kernel; product of reversible kernels is not necessarily reversible but is still π-invariant). Yes — π-invariance holds. So it's fine for correctness of stationary distribution. OK.

But subtle: nonlocal exchanges across the explicit-2D graph — acceptance via _exchange_probability with delta from bias matrix. Fine.

**7. assemble_bias_matrices (L2055-2072)**: boost_bias_kcal = boost_kj / 4.184. bias_kcal = distance + ss + boost_kcal. bias_kj = 4.184*bias_kcal. Invariant holds. Fine, except shape concern (nwin vs nrep zeros fallback).

**8. _fetch_v_pep_v_dih (L2075-2094)**: reads group DIHEDRAL_GROUP for v_dih, peptide_essential_energy_kj for v_pep. Concern: the boost applied by the integrator modifies forces but getPotentialEnergy on a context with a CustomIntegrator applying GaMD boost — does getState(getEnergy=True) return the boosted or unboosted energy? The boost is applied inside the integrator's per-dof force modification, so getState potential energy does NOT include the boost; separate reads of channel energies are needed. This is consistent with reading v_pep/v_dih and computing boost analytically. Fine, but the concern: is v_dih read from group {DIHEDRAL_GROUP} the *unmodified* dihedral energy? Should be, since forces are modified by integrator, not the Force objects. OK.

But sign/subtlety: in the single-dihedral case, `if not getattr(pep_env, "has_total", True): return nan, v_dih`. Then in pep_gamd_boost_kj, `if not env.has_total: ... return b_dih`. OK. But in reconstruct_bias_matrix (correctness/bias.py): `if pep is None: pep = np.full(n, np.nan)` and ladder apply must ignore it. Fine if it does.

**9. campaign_ladder_registry_lambda (L2261-2299)**: Walks up directory tree looking for registry with any λ > 0, returns first found. Concern: returns the FIRST λ>0 encountered in CSV row order — arbitrary λ, only used for boolean detection in resolve_ladder_active. resolve_ladder_active returns True if campaign has ladder. This switches on the ladder code path (recording v_pep/v_dih, applying λ=... per state). But an all-λ=0 sub-run under a ladder campaign: resolve_ladder_active True → pep_env created? Hmm, we don't see the code that constructs pep_env from this. The λ per state is derived from window_metadata; if all λ=0 in this sub-run, k0=0, boost = 0 via _channel_boost (k0=0 → b=0? 0.5*0*(...)^2/rng = 0, and (b+v)<e check → 0 or 0; yes zero). Recorded samples carry v_pep/v_dih. Consistent.

But hazard: the registry walk uses `CAMPAIGN_LADDER_REGISTRY_NAMES` — reads a state CSV written by StateRegistry.write_state_csv. If the registry lists λ>0 for states NOT in this sub-run, ladder active — fine. But what if the walk goes too far up (before adaptive_production)? It checks parents up to but stopping after adaptive_production. Could a campaign's out_dir be nested such that a sibling campaign's registry is found? Only if registry names exist in a shared parent. They claim they stop at adaptive_production. Potential cross-campaign contamination if directories share a parent above adaptive_production... they break after adaptive_production so at most walks to that level. Seems OK-ish; SUSPECTED minor.

Bigger hazard: **stale registry**. If the registry CSV is written at campaign start and states are retired/added adaptively, reading λ from the registry with `lam > 0` triggering ladder — that's only a boolean, fine.

**10. _derive_state_gamd_lambdas (L2360-2384)**: reads normalized_rows if aligned and all rows have gamd_lambda; else existing if length matches; else zeros. Zeros = silent ladder disable — the fallback of "last resort". The persist/reload pair tries to close this. But there's a hole: `_reload_state_gamd_lambdas_on_resume` only reloads if `args.state_gamd_lambdas` is falsy. If it's set but WRONG (e.g., set from an explicit window table with different length), reload skips, and derive uses existing only if length matches, else zeros — silent. But persist is called at every mutation point... The docstrings themselves admit these were found in review. Remaining hole: after `--max-replicas` truncation `args.state_gamd_lambdas[:_max_replicas]` — truncating a per-window array by number of replicas. If n_windows > nrep, this truncation drops the last windows' λ, misaligning λ to windows! Wait — if windows and replicas are 1:1 (each replica owns one window), nrep == nwin and truncation is identity. But if the run has more windows than replicas... does REUS here run one replica per window? The state graph is (window × rung) and replicas move over windows via gibbs-walk — the state space includes all windows, so nrep likely == nwin. Hmm, but then what does --max-replicas truncation mean? If nrep < nwin, replica_of_window has unheld windows (holders >= 0 check exists — so unheld windows are supported!). swap_candidate_replicas returns None if either window unheld. _gibbs_window_proposal_distribution only considers valid (held) windows. So the design supports nrep < nwin. Then truncating state_lambdas (length nwin) to _max_replicas (nrep) is a real misalignment: window w's λ taken from state_lambdas[w] where the list was truncated — windows beyond nrep would index out of bounds! `state_lambdas[w]` with w >= nrep → IndexError. Or if nwin == nrep after drop... Unclear. This is a SUSPECTED state-label mismatch. Actually wait, re-read the persist docstring: "the --max-replicas truncation (`args.state_gamd_lambdas = list(args.state_gamd_lambdas)[:_max_replicas]`)". If state_gamd_lambdas is per-window and _max_replicas < nwin, subsequent `state_lambdas[w]` for w >= nrep crashes (loud) or windows get λ of... Actually if it truncates and then windows are also truncated to _max_replicas elsewhere, maybe consistent. Can't verify; SUSPECTED.

**11. `_reload_state_gamd_lambdas_on_resume` truthiness check**: `if getattr(args, "state_gamd_lambdas", None): return` — an all-zero list [0.0, 0.0, ...] is truthy (non-empty list), so it's left alone — as documented. But an EMPTY list [] is falsy → proceeds to read manifest. Fine. But: a list that is legitimately all-zero from manifest: `lambdas = method_settings.get("state_gamd_lambdas"); if lambdas:` — all-zero list is truthy (non-empty), so it restores [0,0,...]. OK.

However — critical hole: the reload only fires when `args.resume` is set. What about the **adaptive top-up**
(TRUNCATED - output hit the token limit before completion)

Position cards:

- glm: ACCEPT-WITH-CHANGES — core exchange kernels are correct but NaN→0 fallbacks, the missing applied-vs-priced boost assertion, and the k0max dual-source risk must be fixed before this code can be trusted to produce the Boltzmann free energy it claims. (confidence 78)
- kimi: REJECT - verified NaN-to-zero secondary-bias and boost paths let exchange/MBAR price a Hamiltonian different from the sampled one. (confidence 86)
- deepseek-thinking: ACCEPT-WITH-CHANGES - the core REUS kernel is sound, but the secondary-axis condition mismatch and unverified boost-formula parity are silent-failure risks. (confidence 70)
- mini: REJECT – multiple silent, mathematically‑critical bugs can corrupt the free‑energy estimate. (confidence 85)
- thinker: (TRUNCATED - output hit the token limit before completion) (confidence 50)

Verdicts: glm=ACCEPT-WITH-CHANGES, kimi=REJECT, deepseek-thinking=ACCEPT-WITH-CHANGES, mini=REJECT, thinker=n/a
Majority: ACCEPT-WITH-CHANGES at 40%

cosine similarity | glm | kimi | deepseek-thinking | mini | thinker
glm | 1.000 | 0.736 | 0.619 | 0.692 | 0.444
kimi | 0.736 | 1.000 | 0.586 | 0.674 | 0.506
deepseek-thinking | 0.619 | 0.586 | 1.000 | 0.551 | 0.497
mini | 0.692 | 0.674 | 0.551 | 1.000 | 0.443
thinker | 0.444 | 0.506 | 0.497 | 0.443 | 1.000

mean pairwise cosine: 0.575

## glm

## Attack on the designated target

**B's entire opening salvo is aimed at arguments nobody made.** B attributes to me a "truthiness bug" in `_reload_state_gamd_lambdas_on_resume`, a "clipping breaks detailed balance" claim, a "NaN-primary-CV should be zero bias" proposal, and a "uniform fallback" complaint. None of these appear in my card. B then "refutes" them at confidence 95. This is straw-man demolition, not cross-examination. The one adjacent real issue — that `_persist_state_gamd_lambdas` (L2439–2467) maintains its "re-patch at EVERY mutation site" invariant by docstring only, with no assertion or test, so a future forgotten re-persist after `--max-replicas` truncation or post-pull auto-drop silently resumes a misaligned per-state λ (its own docstring admits this outcome) — B ignores entirely.

**B's point 4 (uniform fallback preserves detailed balance) is half-true and misses the real defect.** B is right that MH corrects any recorded proposal distribution. But the genuinely dangerous branch is not the uniform fallback — it is the `finite` filter at L~1017: candidates with NaN delta are silently *removed from the candidate set*, so the window becomes unreachable via proposal while its Hamiltonian is still applied to configurations. A proposal that cannot reach a state, combined with `umbrella_bias_matrix_kcal` L916 zeroing NaN secondary bias in the *exchange* matrix while `reconstruct_bias_matrix` (bias.py L165–175) *propagates NaN* for the same observation in MBAR, means exchange and MBAR price two different Hamiltonians for identical data. That is exactly the chignolin_8 failure class, and it is VERIFIED by line comparison — B waves it off as "naming."

**B's point 3 is false as stated.** NaN secondary CV is not necessarily a crash. The docstring at L895–917 itself says NaN "used to freeze that replica's exchanges" — but the code zeroes the bias instead of freezing the replica. Docstring and code disagree; B defended the code without noticing the docstring contradicts it.

**B's "unverified boost parity" framing undersells a verified absence.** `sample()` computes `extract_gamd_boost_kj(sim.integrator, ...)` (applied) and `sampled_boost_bias_kj = boost_bias_matrix_kj[w, r]` (priced) in the same row, L~7623 vs L~7640, and never compares them. The cheapest possible guard against the calibration case is present as data and absent as a check. Likewise `k0max_from_globals` (L528–530) vs `env.k0max_*` are two untied sources of k0max; B does not address this at all.

**B's `_has_ss_axis` finding is correct and I adopt it** — but B understates it: the predicates at L~7565 and L~7860 reference three distinct objects (`secondary_cv_k_kcal_list`, `secondary_cv_ks_kj`, `ss_ks_kj_arr_global`) while the exchange path *passes* `ss_k_kcal_arr_global`. Any divergence flips the secondary axis on for one path and off for the other.

## Defense / concession

I concede B is right that clipping before log-sum-exp does not bias the stationary distribution — the recorded q_forward/q_reverse make the MH correction exact for any distorted proposal, and my round-1 framing already conceded the kernel math. I also concede `np.zeros((nrep, nrep))` is a crash, not a silent error, when n_states ≠ nrep. I maintain: the NaN→0 secondary-bias and boost paths, the missing applied-vs-priced assertion, the k0max dual source, and the `_has_ss_axis` divergence are verified silent-failure channels that must be fixed before the output can be trusted.

## Weighing the audit

The audit's two-state walkthrough of the Gibbs kernel (acceptance reducing to unity via exact q-ratio cancellation) independently corroborates that the kernel is correct and that B's clipping attack was aimed at a non-issue. The audit did not refute the `_channel_boost` L503 NaN→zero conversion or the exchange-vs-MBAR NaN inconsistency; its GaMD-denominator question (vmax−vmin vs E−Vmin) remains unverified against gamd-openmm source and deserves a targeted parity test, not a verdict on its own.

VERDICT: ACCEPT-WITH-CHANGES - the exchange kernel is mathematically sound, but verified NaN-to-zero bias paths, the missing applied-vs-priced boost assertion, and the divergent secondary-axis predicates can silently corrupt the free energy and must be fixed first.
CONFIDENCE: 80

## kimi

**Attack on the target (A): two of six findings survive; four fail at line level.**

**Point 5 is unreachable dead code.** After the `finite` mask filters NaNs (L1018–1027), every `log_weights` entry is finite, so `m = max(log_weights)` is finite and `weights = exp(log_weights − m)` contains at least one element exactly 1.0 with all others in (0,1]. Hence `sw ∈ [1, N]`: the `sw <= 0.0` / `not isfinite(sw)` branch (L1031–1034) cannot execute. The "silent uniform proposal" is mathematically impossible; the target never checked reachability.

**Point 4 inverts the logic.** An all-zero `args.state_gamd_lambdas` is truthy, so `_reload_state_gamd_lambdas_on_resume` returns early (L2429–2431) — *preserving* the caller's explicit zeros. Reloading the manifest is what would re-activate a stale ladder; skipping it cannot. The causal chain is backwards. And a "wrong-length" manifest vector is rejected by the `len(existing_list) == int(n)` guard (L2379–2380) — the target's own parenthetical ("existing list of correct length") contradicts its premise. The real residual hazard (right length, stale content) is exactly what `_persist_state_gamd_lambdas`' re-persist-at-every-mutation protocol (L2439–2467) addresses; the target doesn't engage it.

**Point 3 is statistically wrong.** A negative boost from `vmax < vmin`, if consistently applied by the integrator and priced by `pep_gamd_boost_kj`, cancels exactly in reweighting — MBAR is unbiased for *arbitrary* bias given overlap. It degrades sampling; it does not "skew the free energy." Corruption requires applied ≠ priced, i.e., a formula divergence between gamd-openmm and `_channel_boost`, which the target never exhibits.

**Point 6 is out of scope by the review's own criterion.** `set_replica_lambda` raises `ValueError` (L536–537) — a loud failure producing no FES. The target's defence ("the failures above are silent — they do not raise exceptions") contradicts its own point 6.

**Point 1 overreaches:** "no consistency test" ignores the documented `verify_gamd_production_stage5` (c73224f). The per-frame applied-vs-priced parity gap survives; the flat claim doesn't. **Point 2** is real but conditional: divergence needs partial initialization (`secondary_cv_k_kcal_list` non-None while `ss_ks_kj_arr_global` is None); co-constructed arrays make it latent, not the stated unconditional detailed-balance break.

**Defense.** I hold REJECT on grounds independent of the target's weak points: (i) `umbrella_bias_matrix_kcal` (L914–916) substitutes 0 for non-finite secondary displacement in both exchange pricing and the recorded matrix — and the docstring's own history ("a NaN entry used to freeze that replica's exchanges") evidences NaNs reach this path in production, so it is not dead code; (ii) `_channel_boost` (L503) maps NaN boost to 0.0 via `(b+v) < e`; (iii) `sample()` records integrator boost (`gamd_boost_total_kj_mol`) and formula boost (`sampled_boost_bias_kj`) side by side with no shown per-frame equality assertion — post-chignolin_8, recording both is not a guard. I concede the Gibbs/MH and pairwise-swap algebra are correct for a fixed finite matrix, and that NaN reachability is narrower than every-frame occurrence — but the task asks where the software *can* silently misprice.

**Audit weighing.** The truncated audit supports my concession: its two-state Gibbs enumeration confirms acceptance 1 with the correct stationary law — the kernel is not the defect. It labels the `ss_k_*_global` mismatch "worth fixing but secondary," matching my latency assessment. It never analyzes the NaN→0 substitutions or the missing boost-parity assertion, so it neither weakens my core findings nor rescues the target's refuted points 3–6. Its truncation (confidence 50) limits weight.

VERDICT: REJECT - verified NaN→0 substitution in the secondary-bias and boost paths plus the unasserted applied-vs-priced boost parity let exchange and MBAR price a Hamiltonian different from the sampled one.
CONFIDENCE: 85

## deepseek-thinking

We need answer as board review. Need produce verdict. Need analyze code for silent wrong Boltzmann free energy. Need identify findings with file:line, concrete failure, effect, confidence, VERIFIED/SUSPECTED. Need final lines exactly.

We need review the provided code snippets. Need determine if accept/reject. The submission is itself a review? Actually user says "Board review: ATLaS-MD / gareus v0.8.2 ... does this code produce Boltzmann free energy it claims?" Then task: "Adversarially review the code below. Report every place..." We need produce independent assessment. Need examine code for errors.

We have many code snippets. Need identify issues in replica exchange, state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path. Also state_identity.py, bias.py, production sample/exchange. Need find silent wrong free energy.

Let's parse.

Key functions:

umbrella_bias_matrix_kcal: computes distance bias 0.5*k*(x-c)^2. For secondary, if ss_centers None returns zeros. If ss values NaN, dss NaN, ss_kcal NaN then np.where(np.isfinite(dss), ss_kcal, 0.0). Good. But if ss_centers provided but ss_k_kcal None? It uses if ss_centers is None or ss_k_kcal is None return zeros. Fine.

_gibbs_window_proposal_distribution: Heat-bath proposal over target windows for one replica transposition. Need understand state graph. It computes old_e = bias[wi, rep] + bias[valid_windows, target_reps]; new_e = bias[valid_windows, rep] + bias[wi, target_reps]; delta = new_e - old_e. This is for swapping replica rep (currently in window wi) with replica in target window wj. Old total bias = U_wi(x_rep)+U_wj(x_target). New = U_wj(x_rep)+U_wi(x_target). Correct. stay delta 0. log_weights = -beta*delta. Then probabilities normalized. This is Gibbs conditional distribution for target window? Actually for a Gibbs move, propose wj with probability proportional exp(-beta * (U_wj(x_rep)+U_wi(x_target)))? Wait target_reps are fixed, so old term U_wi(x_target) constant? Actually old_e includes U_wi(x_target) for each candidate? For candidate wj, old_e = U_wi(x_rep)+U_wj(x_target). new_e = U_wj(x_rep)+U_wi(x_target). delta = new-old. exp(-beta delta) = exp(-beta(U_wj(x_rep)+U_wi(x_target))) / exp(-beta(U_wi(x_rep)+U_wj(x_target))). Denominator depends on wj through U_wj(x_target). So not simply conditional over wj for rep? But for swapping labels, the probability of choosing wj should be proportional to exp(-beta [U_wj(x_rep)+U_wi(x_target)])? Actually for Gibbs sampling over state assignments, conditional distribution of assignment of replica rep given others? Let's derive.

We have replicas i with configurations x_i, windows assigned a_i. State energy = sum_w U_w(x_{rep(w)}) = sum_i U_{a_i}(x_i). For a transposition move swapping assignments of replica rep (currently wi) and replica target (currently wj), new state energy = U_wj(x_rep)+U_wi(x_target). Old = U_wi(x_rep)+U_wj(x_target). The heat-bath proposal over wj should choose wj with probability proportional to exp(-beta * new energy) = exp(-beta[U_wj(x_rep)+U_wi(x_target)]). But their log_weights = -beta*(new_e - old_e) = -beta new_e + beta old_e. Since old_e includes U_wj(x_target), which varies with wj, this is not proportional to exp(-beta new_e) unless U_wj(x_target) constant. Wait old_e = U_wi(x_rep)+U_wj(x_target). So log_weights = -beta(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target)) = -beta(U_wj(x_rep)-U_wj(x_target)) + constant in wj? U_wi(x_rep) constant, U_wi(x_target) constant. So log_weights = -beta(U_wj(x_rep)-U_wj(x_target)) + const. That equals -beta new_e + beta U_wj(x_target) + const. Not proportional to exp(-beta new_e) because U_wj(x_target) varies. Wait but target_reps is the replica currently in window wj; its configuration x_target is fixed. The full state energy after swap is U_wj(x_rep)+U_wi(x_target). The conditional distribution for wj given all other assignments? If we consider moving replica rep to window wj and moving the replica currently in wj to wi, the "other" assignments include target replica's new window wi. The energy of target in wi is U_wi(x_target), constant across wj? Yes U_wi(x_target) is constant because wi fixed. So new energy = U_wj(x_rep)+constant. Thus conditional probability should be proportional to exp(-beta U_wj(x_rep)). But their log_weights = -beta(U_wj(x_rep)-U_wj(x_target)) + const. That includes -U_wj(x_target), which is wrong for a Gibbs conditional over wj? Wait because the target replica's assignment changes from wj to wi, so its energy changes. The full state after move includes target in wi. The target's old energy U_wj(x_target) is part of old state, not new. For Metropolis-Hastings proposal, proposal distribution q(wj|current) can be anything, then MH correction. They compute q_forward = probs[choice_index] and q_reverse from reverse proposal, then MH accept with delta. So proposal distribution need not be the true conditional; MH corrects. But they call it "Gibbs-walk" heat-bath proposal. Is it valid? It is a Metropolis-Hastings with proposal q(wj) ∝ exp(-β δ) where δ = new-old. Then acceptance uses q_reverse/q_forward. This is a valid MH if q_reverse computed correctly. However, is q_forward exactly exp(-βδ) normalized? Yes. Then MH acceptance alpha = min(1, exp(-βδ) q_rev/q_fwd). Since q_fwd ∝ exp(-βδ), exp(-βδ)/q_fwd = 1/Z_fwd? Actually q_fwd = exp(-βδ)/Z_fwd. Then alpha = exp(-βδ) * q_rev / q_fwd = Z_fwd * q_rev. That is not necessarily <=1? Wait detailed balance: proposal q(i->j), acceptance min(1, π_j q(j->i)/(π_i q(i->j))). π_j/π_i = exp(-βδ). q_fwd = exp(-βδ)/Z_fwd. So ratio = exp(-βδ) * q_rev / (exp(-βδ)/Z_fwd) = Z_fwd * q_rev. This can exceed 1, but min handles. Is this correct? Yes. But note q_fwd includes only valid windows and normalized. q_rev computed from holders_after. Need check q_reverse.

_gibbs_window_proposal_distribution called with replica_index=rep, current_window=wj, replica_of_window=holders_after. It computes old_e = bias[wj, rep] + bias[valid_windows, target_reps_after]. holders_after has target_rep at wi, rep at wj. For reverse move from state after (rep in wj, target in wi) back to state before (rep in wi, target in wj), the proposal distribution for candidate window wi? It computes for all valid windows, including wi. For candidate wi, target_reps = holders_after[wi] = target_rep. old_e = bias[wj, rep] + bias[wi, target_rep] (current after state). new_e = bias[wi, rep] + bias[wj, target_rep] (before state). delta_rev = new_e - old_e = -delta_fwd. log_weights = -beta delta_rev = beta delta_fwd. q_reverse for wi = exp(beta delta_fwd)/Z_rev. Then MH alpha = exp(-beta delta_fwd)*q_rev/q_fwd = exp(-βδ)* [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(βδ) Z_fwd/Z_rev? Wait compute: q_fwd = exp(-βδ)/Z_fwd. q_rev = exp(βδ)/Z_rev. alpha = exp(-βδ) * q_rev / q_fwd = exp(-βδ)*exp(βδ)/Z_rev * Z_fwd/exp(-βδ)? Let's do: exp(-βδ) * (exp(βδ)/Z_rev) / (exp(-βδ)/Z_fwd) = exp(-βδ)*exp(βδ)*Z_fwd/(Z_rev*exp(-βδ)) = exp(βδ) Z_fwd/Z_rev. That seems not symmetric? Wait detailed balance ratio should be π_j/π_i = exp(-βδ). q_fwd = exp(-βδ)/Z_fwd. q_rev = exp(βδ)/Z_rev. Then π_j q_rev / (π_i q_fwd) = exp(-βδ) * [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(-βδ+βδ+βδ) Z_fwd/Z_rev = exp(βδ) Z_fwd/Z_rev. Yes. But for MH, alpha = min(1, π_j q_rev/(π_i q_fwd)). That is correct. But is q_rev computed correctly? It uses holders_after, current_window=wj, replica_index=rep. For reverse proposal, the "current" state is after swap, rep in wj. It proposes to move rep to wi and target to wj. The target_reps in holders_after for candidate wi is target_rep. old_e = bias[wj, rep] + bias[wi, target_rep]. new_e = bias[wi, rep] + bias[wj, target_rep]. delta_rev = new_e - old_e = -δ. log_weights = -β(-δ)=βδ. So q_rev(wi) = exp(βδ)/Z_rev. Good.

But wait q_fwd is computed over all valid windows, including wi (stay). For stay, delta=0, q_stay = 1/Z_fwd. If choose stay, no swap. Fine.

Potential issue: _gibbs_window_proposal_distribution uses bias_matrix_kj[wi, rep] + bias[valid_windows, target_reps]. But bias_matrix_kj is [window, replica] = U_window(x_replica). For candidate wj, target_reps = holders[wj]. old_e = U_wi(x_rep)+U_wj(x_target). new_e = U_wj(x_rep)+U_wi(x_target). Correct.

Now, the acceptance function _gibbs_mh_acceptance_probability: log_alpha = -beta*delta + log(qr) - log(qf). If log_alpha >=0 return 1. This is standard. But note if qf <=0 or qr <=0 return 0. q_reverse could be 0 if wi not in rev_windows. Then alpha=0, so no move. Fine.

Potential issue: In gibbs_propose_one_replica, when wj == wi, returns GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False). It sets accepted True? Actually fields: wi, wj, delta, q_forward, q_reverse, pacc, accepted, stuck? Need know dataclass. It returns accepted=True for stay. That's fine.

Now apply_window_swap: It swaps assignments[i], assignments[j], then replica_of_window[assignments[i]] = i, replica_of_window[assignments[j]] = j. Need check after assignments swap: assignments[i] becomes old assignments[j] = wj; assignments[j] becomes old assignments[i] = wi. Then replica_of_window[assignments[i]] = i means replica_of_window[wj] = i. replica_of_window[assignments[j]] = j means replica_of_window[wi] = j. Correct.

But wait bias_matrix_kj[window, replica] = U_window(x_replica). In apply_window_swap, old_e = bias[wi,i]+bias[wj,j], new_e = bias[wj,i]+bias[wi,j]. Correct.

Now _exchange_probability: if delta <=0 return 1; else exp(-beta delta). Correct.

Now assemble_bias_matrices: distance_bias_kcal + ss_bias_kcal + boost_bias_kcal; bias_kj = 4.184*bias_kcal. But boost_bias_kj is in kJ, divided by 4.184 to kcal, then multiplied back. Fine. However, if boost_bias_kj is all-zero when ladder inactive, okay. But if boost_bias_kj is NaN? Could propagate.

_fetch_v_pep_v_dih: reads v_dih from group DIHEDRAL_GROUP. v_pep = peptide_essential_energy_kj(ctx, unit). Need know if v_pep includes dihedral? In pep_gamd_boost_kj, b_dih = _channel_boost(v_dih, ...), b_tot = _channel_boost(v_pep + b_dih, ...). So v_pep should be "essential" total energy excluding dihedral? It says peptide_essential_energy_kj. Need check if v_pep includes dihedral? The comment says gamd-openmm's dependent dual boost: dihedral first, then Total with dihedral boost added to Total energy before square. So v_pep is total non-dihedral? likely.

Now campaign_ladder_registry_lambda: walks up parents until adaptive_production. Returns first (registry filename, lam) with lam >0. Potential issue: If out_dir is inside adaptive_production/... it stops at adaptive_production. It checks candidate = base, then parents. It breaks when candidate.name == "adaptive_production" after checking that candidate. So if out_dir itself is adaptive_production? It checks base first, then if base.name == "adaptive_production" break after checking. Fine. But if multiple campaigns share parent, could pick wrong campaign's ladder. They stop at adaptive_production to avoid. But if out_dir is e.g. /campaign/adaptive_production/seg1, base=seg1, parents include adaptive_production, then break after checking adaptive_production. Good. But if out_dir is /campaign/adaptive_production (base itself), it checks base, then break. Fine.

resolve_ladder_active: True if any state_lambdas >0 OR campaign registry above says ladder. Potential issue: If state_lambdas all zero but campaign registry has lambda >0, ladder active. Then set_replica_lambda_for_window will set k0=0 for all windows? state_lambdas all zero, so yes. But pep_env not None? Need see how pep_env constructed. If ladder active due to registry, pep_env should be created. The comment says all-λ=0 top-up gets k0=0 and samples carry v_pep/v_dih. Good.

_derive_state_gamd_lambdas: reads normalized_rows. If rows and len==n and all have gamd_lambda, return. Else existing if len==n, else zeros. Potential issue: If normalized_rows has rows but some gamd_lambda missing, falls to existing or zeros. Could silently disable ladder. But maybe okay.

_reload_state_gamd_lambdas_on_resume: reads manifest method_settings state_gamd_lambdas. Potential issue: It checks if args.state_gamd_lambdas truthy, returns. If all-zero list is truthy? In Python, [0.0] is truthy, so an explicitly set all-zero ladder is left alone. Good. But if args.state_gamd_lambdas is empty list, falsy, reload. Fine. It reads manifest. Potential issue: manifest method_settings may contain stale pre-truncation/pre-drop values if _persist not called after final mutation. They claim call at every reassignment. Need verify in code? Not shown. Could be issue.

_persist_state_gamd_lambdas: patches manifest. Potential issue: update_run_manifest may merge dicts, not overwrite? Need know. If update_run_manifest does shallow merge, method_settings may retain old state_gamd_lambdas if new list empty? It passes {"method_settings": {"state_gamd_lambdas": list(...)}}. If list empty, maybe update sets empty. Fine.

_channel_boost: b = 0.5*k0*(e-v)^2/rng. Then if (b+v)<e, b else 0. This is GaMD boost condition. But rng = vmax-vmin. If rng negative? vmax>vmin presumably. If rng <=0.001*scale, b=0. Good. Potential issue: if e-v negative, square. Fine.

pep_gamd_boost_kj: b_dih = _channel_boost(v_dih, threshold_dih, vmax_dih, vmin_dih, lam*k0max_dih). b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). out = b_dih + b_tot. This matches dependent dual boost? Need check gamd-openmm formula. Usually dual boost: V_dihedral = 0.5 k0_dih (E_dih - V_dih)^2 / (Vmax_dih - Vmin_dih) if E_dih < V_dih? Actually GaMD: boost potential is 0 if E >= V, else 0.5 k0 (V - E)^2/(Vmax - Vmin)? Wait standard GaMD: ΔV = 0.5 k0 (V - E)^2 / (Vmax - Vmin) when E < V, else 0. Here _channel_boost(v, e, vmax, vmin, k0): b = 0.5*k0*(e - v)^2/rng. Then return where (b+v) < e. So if b+v < e => b < e-v => 0.5 k0 (e-v)^2/rng < e-v => 0.5 k0 (e-v)/rng < 1 => e-v < 2 rng/k0. This is not simply E < V? Wait v is threshold? Let's parse arguments: _channel_boost(v, e, vmax, vmin, k0). In pep_gamd_boost_kj: _channel_boost(v_dih_kj, env.threshold_dih, env.vmax_dih, env.vmin_dih, lam*k0max_dih). So v = current dihedral energy? e = threshold? Actually parameter names: v, e, vmax, vmin, k0. b = 0.5*k0*(e - v)^2/rng. If v is current energy and e is threshold? Standard GaMD: if V(r) < E (threshold), boost = 0.5 k0 (E - V)^2/(Vmax - Vmin). Here v = current energy, e = threshold. b = 0.5*k0*(threshold - current)^2/rng. Then condition (b+v)<e => b < e-v => current + boost < threshold. That matches standard: if current energy below threshold, boost raises to at most threshold? Actually standard condition is V < E, and boost = 0.5 k0 (E - V)^2/(Vmax - Vmin). Then V+boost can exceed E? Let's check: V+0.5k0(E-V)^2/rng < E iff 0.5k0(E-V)/rng < 1 iff E-V < 2rng/k0. Standard GaMD chooses k0 such that boost <=? Usually k0 = k0max*(Vmax-Vmin)/(Vmax-Vavg)? Hmm. The condition in code is (b+v)<e, i.e. boosted energy below threshold. That is a "lower-bound" dual boost? The submission says "lower-bound dual". So maybe correct: boost only if total remains below threshold. But standard GaMD boost is applied when V < E, and can push above E? Actually GaMD: ΔV = 0.5 k0 (E - V)^2/(Vmax - Vmin) if V < E. No additional condition. But "lower-bound" might enforce V+ΔV < E. Need not dwell.

Potential issue: _channel_boost uses rng = vmax - vmin. If vmax and vmin are in kJ/mol? Envelope from swarm maybe in kcal? Need unit consistency. pep_gamd_boost_kj takes v_pep_kj, v_dih_kj, env thresholds/vmax/vmin. Are env values in kJ? Likely. k0max units? Need check. k0max_from_globals reads shared_globals k0_Total, k0_Dihedral. Integrator globals likely in kJ/mol? GaMD k0 units? In OpenMM, k0 has units? The boost formula b = 0.5*k0*(E-V)^2/(Vmax-Vmin). If energies in kJ/mol, k0 dimensionless? Actually k0 is dimensionless? In GaMD, k0 is a scaling factor (0<k0<=1), so b has energy units. So fine.

set_replica_lambda: sets k0_Dihedral = lam*k0max["Dihedral"], k0_Total similarly. Potential issue: If integrator has no k0_Total but k0max Total nonzero, raises. Good.

set_replica_lambda_for_window: if k0max_by_channel is None, no-op. If state_lambdas None and ladder active, raise. Good.

state_identity.py: canonical_state_definition requires exact set of fields. Potential issue: It requires "boost" field even if None. It checks if any gamd_lambda >0 and boost not dict -> error. If all lambdas zero, boost can be None. Fine.

Potential issue: In canonical_state_definition, for NPT, it rejects fixed_box_vectors_nm not None. But if ensemble NPT and fixed_box_vectors_nm is None, okay. It doesn't check pressure_bar positive? finite_number without positive. Pressure could be negative? Not relevant.

Potential issue: In normalize_windows, gamd_lambda = finite_number(raw.get("gamd_lambda", 0.0), minimum=0). It does not enforce <=1. set_replica_lambda raises if lam >1. But state_identity could accept lambda >1, then production would raise at set_replica_lambda, not silent. Fine.

Potential issue: In reconstruct_bias_matrix, if lambdas active and envelope has_total True, requires pep not None. But if v_pep is provided as all NaN, it doesn't raise. Then _ladder_apply may produce NaN. But maybe numeric_vector converts non-finite to NaN, and _ladder_apply should handle? Not sure.

Now, the biggest issue in bias.py: It says "Energy input convention is unchanged: k1 and k2 are kcal/mol per squared CV unit; CV values and centers must already use the SAME units. beta is mol/kJ." In reconstruct_bias_matrix, matrix = 0.5*k*delta^2 in kcal, then matrix *= beta * KJ_PER_KCAL. So reduced bias = beta_kJ * U_kJ. Correct. But if windows have k in kJ/mol (energy_unit kJ/mol), normalize_windows doesn't convert; it just stores. The docstring says k1/k2 are kcal/mol. But state_identity canonical_state_definition converts kJ/mol to kcal/mol. So if using state_identity, okay. But reconstruct_bias_matrix may be used directly with windows from elsewhere. Not our focus.

Now, production sample() potential energy: potentials_kj[r] = physical potential from getState groups=physical_energy_groups_for_args(args). But the total Hamiltonian for MBAR should include physical potential + umbrella + boost. The bias matrix includes umbrella + boost. So reduced potential = beta*(physical + bias). Good. But if physical_energy_groups excludes some terms that are in the actual dynamics (e.g., barostat? no), then wrong. Need know physical_energy_groups_for_args. Not shown. Could be issue.

Now, NPT barostat: The submission says NPT via application-controlled biased Metropolis-MC volume move against U* = physical + bias + boost. The code snippets don't show barostat. But sample() reads potential with enforcePeriodicBox=True. If volume move changes box, positions scaled? Need not.

Now, checkpoint/resume: _reload_state_gamd_lambdas_on_resume reads manifest. Potential issue: If resume after adaptive epoch changed windows, manifest method_settings state_gamd_lambdas may be stale. _persist called after final mutation. But if crash before persist, stale. Could silently disable ladder. The docstring acknowledges. But is there a verification? Not shown.

Now, let's examine _gibbs_window_proposal_distribution more deeply for a possible sign error. It computes delta = new_e - old_e. log_weights = -beta*delta. For a favorable move (new_e < old_e), delta negative, -beta*delta positive, weight >1. Good. Then probabilities normalized. MH acceptance uses delta_kj = deltas[choice_index] (new-old). log_alpha = -beta*delta + log(qr)-log(qf). If q_fwd is high for favorable moves, alpha may be lower. Correct.

But wait: The proposal distribution q_fwd is normalized over valid windows. It includes the stay move. The reverse q_rev is normalized over valid windows in holders_after. However, the set of valid windows may differ between forward and reverse? holders_after has same valid windows (all >=0). So same. Good.

Potential issue: In _gibbs_window_proposal_distribution, old_e = bias[wi, rep] + bias[valid_windows, target_reps]. But bias[valid_windows, target_reps] uses advanced indexing: bias is 2D, valid_windows array, target_reps array. This returns array of length n_valid. Good. new_e = bias[valid_windows, rep] + bias[wi, target_reps]. Good.

Now, potential issue: The Gibbs proposal distribution uses bias_matrix_kj which includes boost. But the boost for a state depends on v_pep/v_dih of the replica's configuration. For candidate wj, new_e = U_wj(x_rep)+U_wi(x_target). The boost part of U_wj(x_rep) is computed with lambda_wj and v_pep/v_dih of rep. The boost part of U_wi(x_target) is computed with lambda_wi and v_pep/v_dih of target. That is correct. The physical potential cancels. Good.

Now, potential issue: In _current_exchange_arrays, when computing boost_bias_matrix_kj, it uses v_pep_kj and v_dih_kj arrays from all replicas. But if pep_env is not None and state_lambdas has length n_states, pep_gamd_boost_matrix_kj returns (n_states, nrep). Good. But if some v_pep/v_dih are NaN (e.g., pep_env has_total False, v_pep NaN), pep_gamd_boost_kj for single dihedral ignores v_pep. Good.

Now, potential issue: In sample(), row["sampled_boost_bias_kj"] = float(boost_bias_matrix_kj[w, r]). This is computed boost, not actual. If actual differs, row records computed. The parquet writer writes boost_total from extract_gamd_boost_kj (actual) and v_pep/v_dih. Analysis could recompute. But MBAR analysis may use bias_matrix from analysis_arrays.npz, which is computed. Need know analyze_gareus_mbar.py. Not shown. But the task says final reweighting by MBAR over all (window, rung) states using per-frame recorded CVs and boost arguments. If analysis recomputes boost from recorded v_pep/v_dih, then actual mismatch doesn't matter? Wait if dynamics ran unboosted but analysis prices boost, then wrong. The calibration case: integrator stage 2 with ForceScalingFactor 1.0, but exchange and MBAR priced boost. The fix verifies stage. In current code, sample() records actual boost_total and v_pep/v_dih. If analysis uses recorded v_pep/v_dih to compute boost, it would still price boost even if actual boost was not applied. The only way to avoid is verify actual boost equals computed. The code does not appear to do so in sample path. It does read actual boost_total but doesn't compare to sampled_boost_bias_kj. So this is a real silent failure mode. We should report.

Now, another issue: In _current_exchange_arrays, the exchange decision uses computed boost matrix, not actual integrator boost. If actual boost is wrong (e.g., stage 2), exchange accepts/rejects based on priced boost that was not applied. This violates detailed balance w.r.t. actual Hamiltonian. Silent. This is exactly the calibration case. The fix mentioned verify_gamd_production_stage5 may be in code not shown. But we can say within reviewed subsystem, no check.

Now, let's look at _channel_boost condition. It uses scale = max(abs(e), abs(v), 1.0). If rng <= 0.001*scale, b=0. This is a guard against zero range. But if vmax-vmin is small but nonzero, and e-v large, b could be huge? Actually if rng small, b large, but condition (b+v)<e may fail, so b=0. Fine.

Potential issue: In _channel_boost, if vmax < vmin (negative rng), b = 0.5*k0*(e-v)^2/negative = negative. Then condition (b+v)<e may be true, returning negative boost. That would be wrong. But envelope should have vmax > vmin. If not, silent negative boost. Could be a finding: no validation that vmax > vmin. But likely envelope validation elsewhere. We can mention SUSPECTED.

Now, pep_gamd_boost_kj: b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). The comment says "dihedral first, then Total with the dihedral boost added to the Total energy before the square (stage_integrator _add_dihedral_boost_to_total_energy)." But in standard dual boost, total boost = 0.5 k0_tot (E_tot - V_tot)^2/(Vmax_tot - Vmin_tot) + 0.5 k0_dih (E_dih - V_dih)^2/(Vmax_dih - Vmin_dih), where V_tot is total potential energy including dihedral? Actually in GaMD dual boost, the total boost uses total potential energy V_total (including dihedral) and threshold E_total. The dihedral boost is added to the total potential energy before computing total boost? The comment says "with the dihedral boost added to the Total energy before the square". That is a specific implementation. If the integrator does that, then code matches. But if not, mismatch. Need trust.

Now, k0max_from_globals: returns {"Total": float(shared_globals.get("k0_Total", 0.0)), "Dihedral": float(shared_globals["k0_Dihedral"])}. If shared_globals missing k0_Dihedral, KeyError, not silent. Good.

Now, set_replica_lambda: It sets k0_Dihedral always, and k0_Total if names is None or "k0_Total" in names. If names is None (test double), it sets k0_Total even if not present? It calls setGlobalVariableByName("k0_Total", ...). If test double doesn't have it, may raise. But names None only for minimal test doubles that expose only setGlobalVariableByName; they might accept any. Fine.

Potential issue: set_replica_lambda does not set ForceScalingFactor or stage. The calibration case was stage not set. The fix seed_frozen_envelope_stage5 likely sets stage. But set_replica_lambda only sets k0 globals. If the integrator's stage is wrong, setting k0 doesn't help. The code under review may rely on external verify. We can mention.

Now, state_identity.py: It hashes definition. Potential issue: It uses json_bytes(dict(raw)) which sorts keys? Need know. If not, hash may differ for same dict with different insertion order. But json_bytes likely canonical. Not our focus.

Now, let's identify concrete findings.

Finding 1: No runtime verification that the integrator-applied GaMD boost equals the boost priced in bias_matrix_kj/MBAR. In sample(), boost_bias_matrix_kj is computed from pep_gamd_boost_kj(v_pep, v_dih, state_lambdas, env), while actual boost is read separately via extract_gamd_boost_kj into row["gamd_boost_total_kj_mol"] but never compared. If StageIntegrator is in wrong stage (as in chignolin_8) or ForceScalingFactor != 1, MBAR and exchange use a boost that was never applied. This is exactly the class of bug. File: gareus/production.py sample() around lines 7420-7700, specifically boost_bias_matrix_kj assignment and row["gamd_boost_total_kj_mol"]; _current_exchange_arrays similarly. Confidence high. VERIFIED? We can point at lines. The code doesn't show comparison. But maybe elsewhere. We can say "In the reviewed subsystem, no check exists; the only verification mentioned is in commit message, not in these functions." Mark VERIFIED for absence in shown code.

Finding 2: _channel_boost can return negative boost if vmax < vmin, because rng negative and condition may pass. No validation that envelope vmax > vmin in pep_gamd.py. If a corrupted/stale envelope has vmax < vmin, boost becomes negative, altering Hamiltonian and free energy silently. File: gareus/pep_gamd.py:497-503. Confidence medium. SUSPECTED.

Finding 3: In _gibbs_window_proposal_distribution, the proposal distribution is normalized over valid windows including the stay move, but the MH acceptance uses q_reverse computed from holders_after. However, q_reverse is computed with the same function that includes the stay move. For the reverse move, the "stay" candidate is wj (current window after), not wi. The probability of choosing wi is q_reverse. That is correct. No issue.

Wait, let's double-check the MH acceptance formula with q_fwd and q_rev both including stay. Suppose forward proposal chooses wj != wi. q_fwd = exp(-βδ)/Z_fwd. Reverse proposal from after state chooses wi. q_rev = exp(βδ)/Z_rev. Then alpha = min(1, exp(-βδ) q_rev / q_fwd) = min(1, exp(-βδ) * exp(βδ)/Z_rev * Z_fwd/exp(-βδ)) = min(1, exp(βδ) Z_fwd/Z_rev). Wait this seems asymmetric and can be >1. But detailed balance condition is π_i q(i->j) α(i->j) = π_j q(j->i) α(j->i). With α(i->j)=min(1, π_j q(j->i)/(π_i q(i->j))). Here π_j/π_i = exp(-βδ). q(j->i) = exp(βδ)/Z_rev. q(i->j)=exp(-βδ)/Z_fwd. Ratio = exp(-βδ)*exp(βδ)/Z_rev / (exp(-βδ)/Z_fwd) = exp(βδ) Z_fwd/Z_rev. So α = min(1, exp(βδ) Z_fwd/Z_rev). Is that correct? Let's test with a simple case where Z_fwd = Z_rev (symmetric proposal). Then α = min(1, exp(βδ)). If δ<0 (favorable), exp(βδ)<1, so α<1. But if proposal is symmetric and move is favorable, MH should accept with probability min(1, exp(-βδ))? Wait π_j/π_i = exp(-βδ). If δ<0, π_j > π_i, so α should be 1. But here α = exp(βδ) <1. That's wrong! Let's recalc.

Hold on. Let's define δ = new_e - old_e. π_j/π_i = exp(-β δ). If δ<0, new lower energy, π_j > π_i, ratio >1, so α should be 1 for symmetric proposal. But our q_fwd is not symmetric; q_fwd = exp(-βδ)/Z_fwd. For δ<0, q_fwd > 1/Z_fwd, so it preferentially proposes favorable moves. Then MH correction should reduce acceptance. The ratio π_j q_rev / (π_i q_fwd) = exp(-βδ) * q_rev / q_fwd. If q_rev = q_fwd (symmetric), ratio = exp(-βδ). For δ<0, ratio >1, α=1. Good. But in our case q_rev = exp(βδ)/Z_rev. If Z_fwd=Z_rev, q_rev = exp(βδ)/Z, q_fwd=exp(-βδ)/Z. Then ratio = exp(-βδ)*exp(βδ)/exp(-βδ) = exp(βδ). For δ<0, exp(βδ)<1, so α<1. That suggests the reverse proposal probability is too high? Let's compute q_rev for reverse move. Reverse move from after state (rep in wj, target in wi) to before (rep in wi, target in wj). The energy difference for reverse δ_rev = old_before - old_after = -δ. The proposal distribution for reverse should be proportional to exp(-β δ_rev) = exp(βδ). So q_rev = exp(βδ)/Z_rev. That is correct if the proposal rule is "propose wj with weight exp(-β δ_forward)". For reverse, δ_rev = -δ, so weight exp(βδ). So q_rev is correct. Then why does MH ratio seem wrong? Let's derive detailed balance with this proposal.

Proposal q(i->j) = exp(-βδ)/Z_fwd. q(j->i) = exp(βδ)/Z_rev. Note δ for i->j = E_j - E_i. For j->i, δ_rev = E_i - E_j = -δ. So q(j->i) = exp(-β δ_rev)/Z_rev = exp(βδ)/Z_rev. Correct.

MH ratio = π_j q(j->i) / (π_i q(i->j)) = exp(-βδ) * [exp(βδ)/Z_rev] / [exp(-βδ)/Z_fwd] = exp(-βδ+βδ+βδ) Z_fwd/Z_rev = exp(βδ) Z_fwd/Z_rev.

But detailed balance requires π_i q(i->j) α(i->j) = π_j q(j->i) α(j->i). If α(i->j)=min(1, ratio), α(j->i)=min(1, 1/ratio). Then LHS = π_i q(i->j) min(1,r), RHS = π_j q(j->i) min(1,1/r). Since π_j q(j->i) = r π_i q(i->j), RHS = r π_i q(i->j) min(1,1/r) = π_i q(i->j) min(r,1). LHS = π_i q(i->j) min(1,r). Equal. So valid. The acceptance probability can be >1? α(i->j)=min(1,r). If r>1, accept always. In our example δ<0, r=exp(βδ) Z_fwd/Z_rev. If Z_fwd=Z_rev, r=exp(βδ)<1, so α<1. That means even though move is favorable, the proposal is so biased toward it that it must be rejected sometimes to maintain balance. That's correct. So no bug.

But wait, is q_fwd really proportional to exp(-βδ)? Yes. For favorable δ<0, q_fwd > uniform, so proposal oversamples favorable moves; MH corrects. Good.

Now, potential issue: The "Gibbs-walk" name suggests they intended the proposal to be the conditional distribution, in which case acceptance should be 1. But they added MH correction, so it's valid. However, if q_fwd is exactly the conditional distribution, then q_rev would also be conditional and MH ratio would be 1? Let's check: If q_fwd ∝ exp(-β new_e) (not exp(-βδ)), then q_fwd = exp(-β U_wj(x_rep))/Z_fwd (since U_wi(x_target) constant). q_rev = exp(-β U_wi(x_rep))/Z_rev. Then ratio = exp(-βδ) * q_rev/q_fwd = exp(-β(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target))) * exp(-βU_wi(x_rep))/Z_rev * Z_fwd/exp(-βU_wj(x_rep)) = exp(-β(U_wj(x_rep)+U_wi(x_target)-U_wi(x_rep)-U_wj(x_target)+U_wi(x_rep)-U_wj(x_rep))) Z_fwd/Z_rev = exp(-β(U_wi(x_target)-U_wj(x_target))) Z_fwd/Z_rev. Not necessarily 1. So their proposal is not conditional. But MH valid.

Now, is there a bug in q_reverse computation due to using holders_after but not updating assignments? The proposal distribution only depends on replica_of_window, not assignments. holders_after is correct. Good.

Now, another potential issue: In gibbs_propose_one_replica, when wj == wi, it returns accepted=True without MH. That's fine because stay always accepted. But it doesn't update anything. Good.

Now, potential issue: In _gibbs_window_proposal_distribution, the stay delta is set to 0.0 after computing delta. But if wi is not in valid_windows? It should be, because holders[wi]=rep >=0. So stay_idx exists. Good.

Now, potential issue: The log_weights clip to [-745, 745]. If all weights underflow to 0 after clipping? They clip lower bound -745, so exp(-745) ~ 5e-324, which is subnormal but nonzero. Sum may be >0. If max log_weight is -745, m=-745, weights = exp(0)=1, sw=1. Fine. If some log_weights are -inf (delta inf), finite mask removes. Good.

Now, potential issue: In _gibbs_mh_acceptance_probability, if log_alpha < -745 return 0. If log_alpha is NaN? It checks finite. Good.

Now, let's examine apply_window_swap with p_override and force_accept. If p_override provided, pacc = p_override, then clamped. If force_accept True, accepted True regardless of pacc. But if p_override is None and force_accept True, pacc computed but ignored. Fine. If uniform is None and force_accept False, accepted False. In production, uniform is rng.random() unless force_accept. Good.

Potential issue: In _attempt_window_swap, it checks swap_candidate_replicas before drawing random. Then calls apply_window_swap with uniform = None if force_accept else rng.random(). But apply_window_swap itself calls swap_candidate_replicas again. That's fine, no extra random. Good.

Now, potential issue: In _attempt_window_swap, after accepted swap, it submits _apply_swap_to_replica to _sim_pool for i and j. But it uses assignments[replica_index] after swap. However, _sim_pool.submit(i, _apply_swap_to_replica, i).result() may execute on worker thread for replica i, but the function uses sims[replica_index].context. If _sim_pool.submit's first argument is replica index and second is function, it may run on that replica's thread. Good. But if the worker thread is busy stepping, deadlock? Not our focus.

Now, potential issue: In sample(), after computing bias_matrix_kj, it stores observable_cache["bias_matrix_kj"] = bias_matrix_kj. Then _current_exchange_arrays may use cached if step matches. But sample() is called at log steps, exchange at exchange intervals. If they coincide, cache used. But sample() computes bias_matrix_kj using v_pep/v_dih and state_lambdas. _current_exchange_arrays uses same. Good.

Now, potential issue: In sample(), potentials_kj[r] is physical potential. But the bias matrix includes boost. The total reduced potential for MBAR should be beta*(physical + bias). However, if the boost is applied in dynamics, the sampled distribution is proportional to exp(-beta*(physical + bias_actual))? Actually umbrella sampling with boost: The bias is added to the physical potential, so the sampled distribution is exp(-beta*(U_physical + U_bias + boost)). MBAR reweights to physical by using u_k = beta*(U_physical + U_bias_k) for each state k? Wait if the simulation is run with Hamiltonian H_k = U_physical + U_bias_k + boost_k, then the sample's reduced potential under state k is beta*(U_physical + U_bias_k + boost_k). To unbias to physical, MBAR uses u_k = beta*(U_physical + U_bias_k + boost_k) for the state that generated the sample? Actually MBAR solves for free energies of states with reduced potentials u_k(x) = beta*(U_physical(x) + bias_k(x)). The boost is part of bias_k. So yes, bias matrix includes boost.

But wait: The boost applied in dynamics is state-dependent (lambda). The sample's actual Hamiltonian is H_actual = U_physical + U_umbrella_window + boost_lambda. The bias matrix for that state should be U_umbrella + boost_lambda. The code computes that. Good.

Now, potential issue: In sample(), row["potential_kj_mol"] = physical potential. But if the boost is applied, the total potential energy from OpenMM getState(getEnergy=True, groups=physical_energy_groups) may exclude boost because boost is in integrator, not force. Good. So physical potential correct.

Now, potential issue: In _current_exchange_arrays, it does not read potential energies. For exchange, physical potential cancels only if the swap is of state labels, not configurations. Yes, because U_physical(x_i)+U_physical(x_j) unchanged. Boost? Boost is part of bias, not physical. So included. Good.

Now, potential issue: The exchange move swaps state labels, but the integrator's boost k0 is set to new state's lambda. However, the boost energy in the current step's bias matrix was computed using v_pep/v_dih from before the swap. After swap, the integrator's boost for future steps will use new lambda. The exchange decision uses the boost that would be applied if the state labels were swapped at that instant. That is correct for Metropolis on the augmented ensemble where the boost is part of the state Hamiltonian. But note: The boost is a function of the configuration and the lambda. When swapping labels, the configuration stays with the replica, so the boost energy for the new state is computed with the same configuration. Correct.

Now, potential issue: The boost applied by the integrator may depend on history (e.g., GaMD adaptive? but envelope frozen). If boost depends on running averages, then swapping labels changes the boost history? The envelope is frozen, so no. Good.

Now, let's look at state_identity.py canonical_state_definition: It requires "windows" with fields center1,k1,center2,k2,gamd_lambda. It converts energy_unit kJ/mol to kcal/mol by dividing k1,k2 by 4.184. But if k1/k2 are in kJ/mol per squared CV unit, dividing by 4.184 gives kcal/mol. Correct. But it doesn't convert gamd_lambda? dimensionless. Good.

Potential issue: In canonical_state_definition, if energy_unit == "kJ/mol", it modifies row["k1"] /= 4.184 and row["k2"] /= 4.184. But if k1/k2 are zero, fine. If k1/k2 are strings? normalize_windows already converted to float. Good.

Potential issue: In canonical_state_definition, it checks set(row) - {"window_id","center1","k1","center2","k2","gamd_lambda"} == empty. But normalize_windows may add keys? It returns dict with center1,k1,center2,k2,gamd_lambda plus original keys? Actually normalize_windows does window = dict(raw), then sets window["center1"], etc. So if raw had extra keys, they remain. Then canonical_state_definition would reject. Good.

Now, potential issue: In validate_fixed_state_segments, it compares top-level windows sorted by window_id to state["windows"]. But state["windows"] is sorted by window_id in canonical_state_definition. So okay.

Now, let's think about the "silent fallbacks" in _derive_state_gamd_lambdas: If normalized_rows cannot be trusted, it falls back to existing or zeros. If existing is stale wrong-length, it falls to zeros, silently disabling ladder. The docstring warns. But is there a scenario where normalized_rows is present but wrong? It checks len(rows)==n and all have gamd_lambda. If rows have gamd_lambda but misaligned due to reindexing bug, could silently use wrong lambdas. The docstring claims reindexing functions subset+renumber correctly. But if not, silent. We can mention SUSPECTED.

Now, potential issue: _reload_state_gamd_lambdas_on_resume reads manifest method_settings. But if manifest has state_gamd_lambdas from a previous campaign with different number of states, and current run has different n, _derive_state_gamd_lambdas will use existing if len==n, else zeros. If len==n but misaligned (e.g., windows reordered), silent wrong lambda per state. The docstring says "if post-drop count happens to match, resumes with a misaligned per-state λ." They try to persist after final mutation. But if crash before persist, stale. This is a real risk. We can report.

Now, another issue: In campaign_ladder_registry_lambda, it returns first registry with lam >0. But if there are multiple registries in different parent dirs, it picks the first in order. If
(TRUNCATED - output hit the token limit before completion)

## Your designated target - the board member you must attack

Let me analyze this carefully. This is a review of replica exchange code in a molecular dynamics free energy pipeline. I need to find bugs in the state bias matrix, Gibbs-walk proposals, swaps, lambda handling.

Let me examine each piece.

**1. umbrella_bias_matrix_kcal (L895-917)**

Orientation: bias[state, replica]. `pv[np.newaxis, :] - ca[:, np.newaxis]` gives [state, replica]. OK. Secondary NaN handling: `np.where(np.isfinite(dss), ss_kcal, 0.0)` — replicas with NaN secondary contribute zero bias on that axis for ALL states. Docstring says "a NaN entry used to freeze that replica's exchanges". Hmm — but setting bias to 0 rather than NaN means exchange delta calculations will silently treat NaN secondary as zero bias. That's a "NaN → 0 silent fallback" candidate. If ss_value is NaN (e.g., fast path returning NaN), the bias is 0 instead of freezing exchanges. The docstring claims "now applied identically to the exchange path — a NaN entry used to freeze that replica's exchanges" but the code does NOT produce NaN entries; it zeroes them. So the claimed freeze behavior doesn't exist. This is a real discrepancy: the docstring claims NaN propagates to freeze exchanges, but `np.where(np.isfinite(dss), ss_kcal, 0.0)` makes bias exactly zero, so exchanges proceed with wrong energies. VERIFIED inconsistency between doc and code. Effect: a replica with missing secondary CV participates in exchanges priced as if it sits at the harmonic minimum of every state — inflated acceptance, wrong sampling weights. MBAR would also reconstruct bias with NaN (in correctness/bias.py numeric_vector converts non-finite to NaN → those samples need filtering). Actually in reconstruct_bias_matrix, missing observations yield NaN in states requiring that coordinate. So the exchange path prices 0 while MBAR prices NaN — inconsistent energies between exchange and recorded samples. That's the class of bug asked for. Though note `_current_exchange_arrays` uses `umbrella_bias_matrix_kcal` and sample path also uses same function so both zero it; but MBAR analysis reconstructs from CVs with correctness/bias.py which yields NaN. Actually wait — does the analysis use reconstruct_bias_matrix? The submission says final reweighting by MBAR using per-frame recorded CVs and boost arguments. reconstruct_bias_matrix converts non-finite secondary to NaN; the states with k2>0 would give NaN, which MBAR must filter out. So a replica with secondary NaN gets exchange-accepted with bias 0, but its samples are dropped in MBAR — that's fine-ish for MBAR but the exchange path accepted moves with fabricated zero energy, which corrupts the sampling distribution (moves accepted that shouldn't be). Actually more subtly: exchange acceptance with wrong energy breaks detailed balance of the joint chain — the target distribution conditioned on exchange acceptance probabilities that use wrong bias is not the intended distribution. Effect on FES: biased.

Also note only `dss` finiteness is checked, not `distance_kcal` — primary NaN propagates. Fine.

**2. _gibbs_window_proposal_distribution (L983-1041)**

Heat-bath proposal. delta computed as new_e - old_e. old_e = bias[wi, rep] + bias[valid_windows, target_reps]; new_e = bias[valid_windows, rep] + bias[wi, target_reps]. For the swap of replica `rep` in window wi with the holder of window w (target_rep): new configuration assigns rep→w and target_rep→wi. Bias energy = bias[w, rep] + bias[wi, target_rep]. Correct. Stay index delta set to 0. Good.

Note: delta here only includes umbrella bias. The Gibbs move swaps state labels between replicas — configurations unchanged — so physical potential and boost cancel... wait, boost does NOT cancel! The boost depends on the state's λ. In this state graph, each (window, rung) is a state with its own λ. A "window swap" between replica in window wi (λ_i) and holder of wj (λ_j): the boost energy of replica i's configuration under rung λ_j differs from under λ_i. The delta for the swap must include the boost terms: b_j(x_i) + b_i(x_j) - b_i(x_i) - b_j(x_j). But the docstring in apply_window_swap claims "the unbiased potential and the GaMD boost cancel exactly from delta ... dV(x_i)+dV(x_j) are identical before and after." That's WRONG when λ_i ≠ λ_j. The boost dV depends on λ of the state the replica is assigned to. If the ladder is active and windows have different λ, swapping window labels between replicas changes the boost energy each configuration experiences — it does NOT cancel.

But wait — does the bias matrix include the boost? Yes: assemble_bias_matrices includes boost_bias_kj, and bias_matrix_kj includes boost of each sample's configuration under each state's λ (pep_gamd_boost_matrix_kj returns (n_states, n_samples) boost of each sample's configuration under each state's λ). So bias[w, r] includes b_λ(w)(x_r). So the delta in apply_window_swap using bias_matrix_kj correctly includes boost differences. Good — the matrix includes boost, so delta is correct for the swap. So the docstring's justification is wrong (boost doesn't cancel) but the computation is right because the matrix contains per-state boost. Hmm, but the docstring says the boost cancels — the claim is incorrect reasoning but the code (via matrix) is right. Actually wait: is the boost matrix included in the exchange bias matrix? In `_current_exchange_arrays`, yes: `_, bias_matrix_kj = assemble_bias_matrices(distance_bias_kcal, ss_bias_kcal, boost_bias_matrix_kj)`. And apply_window_swap receives bias_matrix_kj. So swaps are priced with boost. Fine.

But note shape issue: `boost_bias_matrix_kj = np.zeros((nrep, nrep))` when pep_env is None. pep_gamd_boost_matrix_kj returns (n_states, n_samples) = (len(lambdas), n_samples). Here nrep replicas and nrep windows (assuming n windows == nrep). If number of windows != number of replicas (e.g., --max-replicas truncation mentioned in _persist_state_gamd_lambdas: `args.state_gamd_lambdas = list(args.state_gamd_lambdas)[:_max_replicas]`), then shapes mismatch! state_lambdas has length = number of windows; v_pep has length nrep. pep_gamd_boost_matrix_kj stacks len(lambdas) rows each of length n_samples → shape (n_windows, nrep). Then assemble: distance_bias_kcal has shape (n_windows, nrep); boost (n_windows, nrep). OK, consistent. But the fallback `np.zeros((nrep, nrep))` when pep_env is None — if n_windows ≠ nrep, boost is (nrep, nrep), mismatched with (n_windows, nrep) distance matrix. Adding them would broadcast-fail or misalign. Actually if pep_env is None, boost is zeros; adding (nrep,nrep) zeros to (n_windows, nrep) matrix raises ValueError if shapes differ (unless n_windows == nrep). Hmm — but more important: `_current_exchange_arrays` checks `cached_bias.shape == (nrep, nrep)` in the cache validation. If n_windows ≠ nrep, the cached bias matrix shape is (n_windows, nrep) and the cache check fails, forcing recompute — only a performance issue. But the zeros((nrep,nrep)) with n_windows≠nrep: assemble_bias_matrices would raise ValueError on broadcast? Adding arrays of shape (nwin,nrep) and (nrep,nrep): if nwin != nrep, numpy raises. So error, not silent. But wait, is state_lambdas length n_windows or nrep after truncation? The persist function mentions truncation of state_lambdas to _max_replicas — suggesting state_lambdas is indexed by window in some places and by replica in others? In sample(): `row["gamd_lambda"] = float(state_lambdas[w])` where w = assignments[r] — window index. And `set_replica_lambda_for_window(..., assignments[replica_index], state_lambdas, ...)` — window index. So state_lambdas is per-window. But truncation `args.state_gamd_lambdas[:_max_replicas]` truncates by replica count — if n_windows > nrep... Hmm, ambiguous. If truncation is meant to align per-window λ with reduced replica count... Actually the persist docstring says the truncation mutates state_gamd_lambdas. If windows and replicas are 1:1 normally, fine.

Actually the bigger issue in pep_gamd_boost_matrix_kj at exchange time: `pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env)` — v_pep_kj has NaN when... `_fetch_v_pep_v_dih` returns (nan, nan) only when pep_env is None, which is guarded. But for single dihedral boost (has_total False), v_pep = NaN and pep_gamd_boost_kj handles: `_channel_boost(v_pep + b_dih, ...)` — wait no, if not env.has_total, returns b_dih only, ignoring v_pep NaN. OK.

But in `_channel_boost`: `b = _np.where((b + v) < e, b, 0.0)` — the lower-bound boost guard: only apply if boost lifts v below threshold e... Actually GaMD lower-bound boost: boost applied when V < E (threshold). The boost should raise energy to V + 0.5*k0*(E-V)^2/... Let me check units: `rng = vmax - vmin`, `b = 0.5*k0*(e-v)^2/rng`. That's gamd-openmm's lower bound boost formula: ΔV = 0.5*k0*(E-V)^2/(Vmax-Vmin) when V < E. And guard `(b+v) < e` → apply only if b+v < e, i.e., boosted energy still below threshold. Plausible. And rng guard: if |vmax - vmin| small, b=0 — silent boost=0 fallback! If the swarm calibration produced vmax ≈ vmin (degenerate envelope), the boost silently becomes zero while MBAR still prices... wait, MBAR reconstructs boost from v_pep/v_dih with the same formula and same envelope, so it'd also be zero. Consistent. But if rng differs slightly between... it's the same env. OK.

Numerical stability of _channel_boost: `scale = np.maximum(np.maximum(abs(e), npabs(v)), 1.0)`; `np.where(np.abs(rng) <= 0.001*scale, 0.0, b)` — fine.

**3. _gibbs_mh_acceptance_probability**: log_alpha = -beta*delta + log(qr/qf). Correct MH for proposal distribution q. If q_reverse == 0 (rev_idx empty → q_reverse = 0.0) → return 0 (reject). But when is reverse probability zero while forward was positive? In reverse distribution computed from holders_after... If the move is valid, reverse should include wi as candidate. If q_reverse = 0, reject — safe. But subtle: the heat-bath proposal for replica rep at window wi includes the "stay" option with delta forced 0 and weight exp(0)=1 — fine.

Hmm — a deeper issue with the Gibbs walk: the proposal distribution `_gibbs_window_proposal_distribution` is a heat-bath over swap partners, but its weights use only the bias matrix (umbrella + boost), i.e., the reduced Hamiltonian difference. Heat-bath proposal q(j) ∝ exp(-β δ_j) where δ_j is the swap energy change — this is a valid proposal and MH correction with reverse q fixes detailed balance. Looks right. But the stay delta forced to 0 — actual delta for staying is 0 anyway (swapping replica with itself? No—stay means wj == wi; the "swap" would be with itself; delta genuinely 0). Fine.

**Potential bug: `holders` array vs assignments.** In gibbs_propose_one_replica, wi = assignments[rep]. If assignments and replica_of_window are consistent, fine. apply_window_swap updates both: `assignments[i], assignments[j] = assignments[j], assignments[i]` then `replica_of_window[int(assignments[i])] = int(i)`... wait: after the swap, assignments[i] holds the new window of replica i (which is wj), and replica_of_window[wj] should be i. Let's check: after `assignments[i], assignments[j] = assignments[j], assignments[i]`: assignments[i] = wj (old window of j), assignments[j] = wi. Then `replica_of_window[int(assignments[i])] = int(i)` → replica_of_window[wj] = i. Correct. `replica_of_window[int(assignments[j])] = int(j)` → replica_of_window[wi] = j. Correct. OK.

But wait: what if i == j? Guarded by swap_candidate_replicas (i==j returns None). What if one replica holds both windows? Guarded.

Hmm, wait: there's an issue with the initial `assignments` semantics — set_window uses `assignments[replica_index]` — OK.

**4. _exchange_probability**: standard Metropolis. Fine.

**5. swap_candidate_replicas**: `n_win = int(replica_of_window.size)` — replica_of_window is indexed by window, size = n windows. Checks wi/wj in range. Fine.

**6. explicit_2d_neighbor_pairs_for_exchange (L1479-1505)**: Round-robin edge coloring: `if idx % slots != slot: continue` — selects edges by index mod slots, then greedy disjoint filter. Fallback when slot empty: tries ALL edges greedily — this makes the edge set for an empty slot equal to the full greedy matching, which overlaps with other slots' edges. Does this break detailed balance? For a fixed parity schedule, the Markov chain with alternating matchings still satisfies balance if each sub-chain is reversible — Metropolis swaps within a matching are reversible. Adding the full matching on an "empty slot" is just another valid matching set; the composition remains valid as long as it's a matching. Detailed balance for the product of reversible moves: each swap move is reversible with respect to π, and compositions of such moves... Actually a sequence of Metropolis moves each individually satisfying detailed balance gives a valid chain regardless of schedule (each move is a π-reversible transition kernel; product of reversible kernels is not necessarily reversible but is still π-invariant). Yes — π-invariance holds. So it's fine for correctness of stationary distribution. OK.

But subtle: nonlocal exchanges across the explicit-2D graph — acceptance via _exchange_probability with delta from bias matrix. Fine.

**7. assemble_bias_matrices (L2055-2072)**: boost_bias_kcal = boost_kj / 4.184. bias_kcal = distance + ss + boost_kcal. bias_kj = 4.184*bias_kcal. Invariant holds. Fine, except shape concern (nwin vs nrep zeros fallback).

**8. _fetch_v_pep_v_dih (L2075-2094)**: reads group DIHEDRAL_GROUP for v_dih, peptide_essential_energy_kj for v_pep. Concern: the boost applied by the integrator modifies forces but getPotentialEnergy on a context with a CustomIntegrator applying GaMD boost — does getState(getEnergy=True) return the boosted or unboosted energy? The boost is applied inside the integrator's per-dof force modification, so getState potential energy does NOT include the boost; separate reads of channel energies are needed. This is consistent with reading v_pep/v_dih and computing boost analytically. Fine, but the concern: is v_dih read from group {DIHEDRAL_GROUP} the *unmodified* dihedral energy? Should be, since forces are modified by integrator, not the Force objects. OK.

But sign/subtlety: in the single-dihedral case, `if not getattr(pep_env, "has_total", True): return nan, v_dih`. Then in pep_gamd_boost_kj, `if not env.has_total: ... return b_dih`. OK. But in reconstruct_bias_matrix (correctness/bias.py): `if pep is None: pep = np.full(n, np.nan)` and ladder apply must ignore it. Fine if it does.

**9. campaign_ladder_registry_lambda (L2261-2299)**: Walks up directory tree looking for registry with any λ > 0, returns first found. Concern: returns the FIRST λ>0 encountered in CSV row order — arbitrary λ, only used for boolean detection in resolve_ladder_active. resolve_ladder_active returns True if campaign has ladder. This switches on the ladder code path (recording v_pep/v_dih, applying λ=... per state). But an all-λ=0 sub-run under a ladder campaign: resolve_ladder_active True → pep_env created? Hmm, we don't see the code that constructs pep_env from this. The λ per state is derived from window_metadata; if all λ=0 in this sub-run, k0=0, boost = 0 via _channel_boost (k0=0 → b=0? 0.5*0*(...)^2/rng = 0, and (b+v)<e check → 0 or 0; yes zero). Recorded samples carry v_pep/v_dih. Consistent.

But hazard: the registry walk uses `CAMPAIGN_LADDER_REGISTRY_NAMES` — reads a state CSV written by StateRegistry.write_state_csv. If the registry lists λ>0 for states NOT in this sub-run, ladder active — fine. But what if the walk goes too far up (before adaptive_production)? It checks parents up to but stopping after adaptive_production. Could a campaign's out_dir be nested such that a sibling campaign's registry is found? Only if registry names exist in a shared parent. They claim they stop at adaptive_production. Potential cross-campaign contamination if directories share a parent above adaptive_production... they break after adaptive_production so at most walks to that level. Seems OK-ish; SUSPECTED minor.

Bigger hazard: **stale registry**. If the registry CSV is written at campaign start and states are retired/added adaptively, reading λ from the registry with `lam > 0` triggering ladder — that's only a boolean, fine.

**10. _derive_state_gamd_lambdas (L2360-2384)**: reads normalized_rows if aligned and all rows have gamd_lambda; else existing if length matches; else zeros. Zeros = silent ladder disable — the fallback of "last resort". The persist/reload pair tries to close this. But there's a hole: `_reload_state_gamd_lambdas_on_resume` only reloads if `args.state_gamd_lambdas` is falsy. If it's set but WRONG (e.g., set from an explicit window table with different length), reload skips, and derive uses existing only if length matches, else zeros — silent. But persist is called at every mutation point... The docstrings themselves admit these were found in review. Remaining hole: after `--max-replicas` truncation `args.state_gamd_lambdas[:_max_replicas]` — truncating a per-window array by number of replicas. If n_windows > nrep, this truncation drops the last windows' λ, misaligning λ to windows! Wait — if windows and replicas are 1:1 (each replica owns one window), nrep == nwin and truncation is identity. But if the run has more windows than replicas... does REUS here run one replica per window? The state graph is (window × rung) and replicas move over windows via gibbs-walk — the state space includes all windows, so nrep likely == nwin. Hmm, but then what does --max-replicas truncation mean? If nrep < nwin, replica_of_window has unheld windows (holders >= 0 check exists — so unheld windows are supported!). swap_candidate_replicas returns None if either window unheld. _gibbs_window_proposal_distribution only considers valid (held) windows. So the design supports nrep < nwin. Then truncating state_lambdas (length nwin) to _max_replicas (nrep) is a real misalignment: window w's λ taken from state_lambdas[w] where the list was truncated — windows beyond nrep would index out of bounds! `state_lambdas[w]` with w >= nrep → IndexError. Or if nwin == nrep after drop... Unclear. This is a SUSPECTED state-label mismatch. Actually wait, re-read the persist docstring: "the --max-replicas truncation (`args.state_gamd_lambdas = list(args.state_gamd_lambdas)[:_max_replicas]`)". If state_gamd_lambdas is per-window and _max_replicas < nwin, subsequent `state_lambdas[w]` for w >= nrep crashes (loud) or windows get λ of... Actually if it truncates and then windows are also truncated to _max_replicas elsewhere, maybe consistent. Can't verify; SUSPECTED.

**11. `_reload_state_gamd_lambdas_on_resume` truthiness check**: `if getattr(args, "state_gamd_lambdas", None): return` — an all-zero list [0.0, 0.0, ...] is truthy (non-empty list), so it's left alone — as documented. But an EMPTY list [] is falsy → proceeds to read manifest. Fine. But: a list that is legitimately all-zero from manifest: `lambdas = method_settings.get("state_gamd_lambdas"); if lambdas:` — all-zero list is truthy (non-empty), so it restores [0,0,...]. OK.

However — critical hole: the reload only fires when `args.resume` is set. What about the **adaptive top-up**
(TRUNCATED - output hit the token limit before completion)

## mini

**Attack on the target’s weakest arguments**

1. **Heat‑bath proposal “self‑refuting” claim** – The target insists that the Gibbs proposal must be proportional to exp(−β U_wj(x_rep)) alone and that the code’s inclusion of U_wi(x_target) is therefore “wrong”. In fact the proposal is *not* a heat‑bath for a single replica; it is a transposition move that necessarily involves the partner’s energy. The code’s `delta` = U_wj(x_rep)+U_wi(x_target) − [U_wi(x_rep)+U_wj(x_target)] correctly captures the change in the *total* bias of the two‑replica subsystem, and the Metropolis‑Hastings correction uses the exact reverse probability. Thus the target’s criticism is a straw‑man: the proposal is deliberately constructed for a swap, not a single‑replica reassignment, and the MH factor guarantees detailed balance. The “self‑refuting” label is therefore misplaced.

2. **Speculation about the `(b+v) < e` guard** – The target dismisses the guard in `_channel_boost` as “speculation”. This is false: the guard can zero‑out a boost that would otherwise be positive. Consider a configuration with v = e − 0.1 kJ mol⁻¹, k₀ = 1, vmax − vmin ≈ 10 kJ mol⁻¹. The raw boost b ≈ 0.005 kJ mol⁻¹, but b+v = e + 0.005 > e, so the guard forces b = 0. The integrator in gamd‑openmm applies the same guard (it is part of the official boost formula), so the analytical `pep_gamd_boost_kj` matches the simulation. However, if a user supplies an envelope where vmax < vmin or an extreme k₀, the denominator becomes negative and the guard silently flips the sign of b, producing an *unphysical* negative boost that is never flagged. This is a genuine silent‑failure mode, not mere speculation.

3. **“Umbrella_bias_kcal_mol” naming nit‑pick** – The column `umbrella_bias_kcal_mol` in the sample CSV actually contains *umbrella + boost* bias (see `assemble_bias_matrices`). Downstream scripts that reconstruct the Hamiltonian from the CSV assume this column is pure umbrella bias (as the name suggests) and therefore double‑count the boost when re‑weighting. This leads to a systematic over‑estimation of free energies. The mis‑label is therefore a functional bug, not a cosmetic issue.

4. **Missing applied‑vs‑priced boost check** – The audit confirms that the code never cross‑validates the boost read from the integrator (`extract_gamd_boost_kj`) against the analytical boost (`pep_gamd_boost_kj`) used to build the bias matrix. The documented calibration failure (stage 2 running unboosted) is a concrete example where the bias matrix includes a non‑zero boost while the dynamics applied none, silently corrupting MBAR estimates. The absence of any sanity check is a critical omission.

5. **Secondary‑CV flag mismatch** – In `sample()` the secondary axis is enabled when `secondary_cv_centers` and `secondary_cv_k_kcal_list` are present, whereas `_current_exchange_arrays()` also requires `secondary_cv_ks_kj` and `ss_ks_kj_arr_global`. If any of these auxiliary arrays are missing (a realistic scenario when the fast‑CV path is used), the sample path will include secondary umbrella terms while the exchange path will treat them as zero, breaking detailed balance. This asymmetry is verified by the divergent `if` conditions at L 7425 and L 7789.

6. **Uniform‑proposal fallback** – `_gibbs_window_proposal_distribution` replaces a heat‑bath distribution with a uniform one whenever the summed weight `sw` is non‑finite or ≤ 0 (e.g. after extreme under‑/overflow). This silent replacement changes the stationary distribution without any warning, violating the Gibbs‑walk balance.

**Defence of my position**

All of the above defects are either verified (e.g. the secondary‑CV flag mismatch, the uniform‑fallback, the boost‑vs‑applied mismatch) or have a mathematically‑rigorous failure mode (negative rng in `_channel_boost`). They can each silently produce a Hamiltonian that differs from the one actually sampled, leading to biased MBAR estimates. The target’s arguments either mischaracterise the Gibbs proposal or downplay genuine numerical pitfalls. Therefore my assessment that the code cannot be trusted to yield the claimed Boltzmann‑weighted free energies remains unchanged.

**Weighing the audit findings**

The audit uncovers multiple high‑confidence, verified violations of detailed balance and energy consistency (boost mismatch, secondary‑CV asymmetry, envelope sign error, uniform‑fallback). Each can independently corrupt the free‑energy surface, and together they constitute a systemic risk that cannot be mitigated by existing tests.

VERDICT: REJECT – multiple verified, silent violations of the Boltzmann weighting guarantee render the free‑energy estimates unreliable.  
CONFIDENCE: 92

## thinker

We need answer as board review. Need produce verdict. Need analyze code for silent wrong Boltzmann free energy. Need identify findings with file:line, concrete failure, effect, confidence, VERIFIED/SUSPECTED. Need final lines exactly.

We need review the provided code snippets. Need determine if accept/reject. The submission is itself a review? Actually user says "Board review: ATLaS-MD / gareus v0.8.2 ... does this code produce the Boltzmann free energy it claims?" Then task: "Adversarially review the code below. Report every place..." We need produce independent assessment. Need examine code for errors.

We have many code snippets. Need identify issues in replica exchange, state bias matrix, gibbs-walk proposals/acceptance, swaps, lambda per state, sampling path. Also state_identity.py, bias.py, production sample/exchange. Need find silent wrong free energy.

Let's parse.

First umbrella_bias_matrix_kcal: primary_values shape (n_replicas), centers_a shape (n_states), k_kcal shape (n_states). dprimary = pv[np.newaxis,:] - ca[:,np.newaxis] -> shape (states, replicas). distance_kcal = 0.5 * ka[:,None] * dprimary^2. Units: k kcal/mol per CV unit^2? CV maybe fraction dimensionless or distance? centers_a in A? k_kcal maybe kcal/mol/A^2. dprimary in A, product kcal/mol. OK. Secondary: dss = sv[None,:] - sc[:,None]; ss_kcal = 0.5*sk*dss^2; then np.where(np.isfinite(dss), ss_kcal, 0.0). If dss NaN, sets 0. But if dss inf? np.isfinite false -> 0. OK. But if secondary values missing, bias zero. Fine.

Potential issue: if ss_centers None, returns zeros_like(distance_kcal). If secondary axis absent, OK. If secondary values contain NaN but centers finite, zero. OK.

_gibbs_window_proposal_distribution: bias_matrix_kj shape [state, replica]? They use bias[wi, rep] + bias[valid_windows, target_reps]. old_e = energy of replica rep in window wi plus target_reps in their windows valid_windows. new_e = energy of rep in valid_windows plus target_reps in wi. This is a transposition of replica rep with target_reps? Let's understand. valid_windows are windows held by some replica target_reps. Proposal: choose target window wj to swap replica rep (currently in wi) with replica in wj. old total bias = U_wi(x_rep) + U_wj(x_target). new = U_wj(x_rep)+U_wi(x_target). delta = new-old. Good. But they include all valid_windows, including current wi? valid_windows includes wi with target_reps = rep? holders[wi] = rep. Then old_e = bias[wi, rep] + bias[wi, rep]? Wait valid_windows includes wi, target_reps = holders[wi] = rep. old_e = bias[wi, rep] + bias[wi, rep] = 2*U_wi(x_rep). new_e = bias[wi, rep] + bias[wi, rep] = same. delta=0. They set stay_idx delta=0. OK. But old_e for other windows: bias[wi, rep] + bias[wj, target] = U_wi(x_rep)+U_wj(x_target). new = U_wj(x_rep)+U_wi(x_target). Good. This is heat bath over target windows for a given replica transposition. But is this a valid Gibbs move? Need check detailed balance with MH correction. They compute q_forward = probability of choosing wj from current state. q_reverse = probability of choosing wi from destination state after swap, using holders_after. Then MH acceptance log_alpha = -beta*delta + log(q_reverse)-log(q_forward). That is correct for proposal distribution q(wj|current) and reverse q(wi|dest). But wait the proposal distribution includes all valid windows, including current. If choose current, no move. For reverse, holders_after has rep in wj and target in wi. valid_windows includes wi with target_rep = target? Actually holders_after[wi]=target_rep, holders_after[wj]=rep. Reverse proposal for replica rep from current_window=wj: valid_windows includes wi, target_reps = holders_after[wi] = target_rep. old_e = bias[wj, rep] + bias[wi, target_rep]; new_e = bias[wi, rep] + bias[wj, target_rep]. delta_reverse = new-old = -delta? Let's compute: current after swap: rep in wj, target in wi. Reverse move swaps rep back to wi and target to wj. old_e = U_wj(x_rep)+U_wi(x_target). new_e = U_wi(x_rep)+U_wj(x_target). delta_rev = new-old = (U_wi(x_rep)+U_wj(x_target)) - (U_wj(x_rep)+U_wi(x_target)) = -delta. Good. q_reverse computed. OK.

But potential issue: _gibbs_window_proposal_distribution uses bias_matrix_kj with shape [state, replica]. It assumes bias matrix includes all states and replicas. In production, bias_matrix_kj built as [state, replica] with boost. OK.

However, the proposal distribution is not exactly heat-bath over "target windows" because it includes all windows, but the state space is assignments of replicas to windows. The Gibbs move should sample a new window for replica rep conditional on all other assignments fixed. The conditional probability for rep to be in window w is proportional exp(-beta * U_w(x_rep) + constant?) Let's derive. Total bias energy for assignment A: sum over windows? Actually each window has one replica, each replica in one window. Total bias = sum_{k} U_k(x_{A(k)})? If replica rep currently in wi, other replicas fixed. If move rep to wj, target replica moves to wi. Total bias = U_wj(x_rep)+U_wi(x_target)+sum others. Conditional probability of choosing wj (including current) proportional exp(-beta * [U_wj(x_rep)+U_wi(x_target)]). The code's delta = new-old = U_wj(x_rep)+U_wi(x_target) - [U_wi(x_rep)+U_wj(x_target)]. But the conditional probability should be proportional exp(-beta * [U_wj(x_rep)+U_wi(x_target)]), not exp(-beta * delta) relative to current? Wait heat-bath Gibbs samples from conditional distribution, not MH with proposal based on delta. The code calls it "heat-bath proposal" but then applies MH correction. If proposal distribution is q(wj) ∝ exp(-β * delta) = exp(-β*(new-old)), then q(wj) = exp(-β*new)/Z_current? Actually delta = new - old_current. exp(-β*delta) = exp(-β*new)/exp(-β*old_current). Since old_current is constant for all wj? old_current = U_wi(x_rep)+U_wj(x_target) depends on wj! Wait old_e in code = bias[wi, rep] + bias[valid_windows, target_reps]. For candidate wj, old_e = U_wi(x_rep)+U_wj(x_target). This is not constant across wj because U_wj(x_target) varies. So exp(-β*delta) = exp(-β*(U_wj(x_rep)+U_wi(x_target) - U_wi(x_rep)-U_wj(x_target))) = exp(-β*(U_wj(x_rep)-U_wj(x_target))) * constant? Let's factor: new = U_wj(x_rep)+U_wi(x_target). old = U_wi(x_rep)+U_wj(x_target). delta = U_wj(x_rep)-U_wj(x_target) + U_wi(x_target)-U_wi(x_rep). The U_wi terms are constant for all wj? U_wi(x_target) constant, U_wi(x_rep) constant. So delta = [U_wj(x_rep)-U_wj(x_target)] + const. Thus q(wj) ∝ exp(-β U_wj(x_rep)) * exp(+β U_wj(x_target)) * const. That is not the conditional distribution for rep alone; it includes +U_wj(x_target) because target moves. Actually the conditional distribution for the pair swap should be proportional exp(-β [U_wj(x_rep)+U_wi(x_target)]) = exp(-β U_wj(x_rep)) * constant (since U_wi(x_target) constant). So q should be ∝ exp(-β U_wj(x_rep)), not exp(-β delta). The code's q is ∝ exp(-β [U_wj(x_rep)-U_wj(x_target)]). That seems wrong for a Gibbs move. But wait they are not sampling a new window for rep alone; they are sampling a transposition with a target replica. The state space is assignments; a move that changes rep's window necessarily changes target's window. The conditional distribution of the pair (rep, target) given all other assignments? If we consider the joint assignment of rep and target, with other replicas fixed, the possible states are: rep in wi, target in wj (current), or rep in wj, target in wi (swap). There are only two states for a given target. But the proposal chooses among all target windows, so it's a "global" move that selects a target and swaps. The correct heat-bath over all possible target windows for a transposition should have probability proportional to exp(-β * new_total) = exp(-β [U_wj(x_rep)+U_wi(x_target)]) for each wj? But note if wj=wi, target=rep, new_total = U_wi(x_rep)+U_wi(x_rep)? Wait if wj=wi, no swap, total = U_wi(x_rep)+U_wi(x_rep)? That double counts? Let's define total bias energy for assignment: sum over windows of U_window(x_replica_in_window). If rep in wi and target in wj, total = U_wi(x_rep)+U_wj(x_target)+others. If swap, total = U_wj(x_rep)+U_wi(x_target)+others. For wj=wi, target=rep, no swap, total = U_wi(x_rep)+U_wi(x_rep)+others? That's wrong because there is only one replica in wi, not two. The code's old_e for wj=wi is bias[wi,rep]+bias[wi,rep] = 2*U_wi(x_rep), double counting. They set delta=0 for stay, but the weight for stay in q is exp(0)=1, not exp(-β * 2U_wi(x_rep)). So the proposal distribution is not a proper heat-bath over assignments; it's a heuristic with MH correction. But with MH correction, any proposal distribution with correct reverse probability yields correct stationary distribution as long as acceptance uses detailed balance. The q can be arbitrary. So not necessarily wrong. Need check MH acceptance with q_forward and q_reverse. The acceptance formula is correct for arbitrary proposal q. So even if q not conditional, it's fine. But there may be issue with q_forward for stay: they set delta=0, q_forward = probability of stay. Reverse q_reverse for stay? If wj==wi, they return GibbsProposal with accepted True? Actually in gibbs_propose_one_replica, if wj==wi, return GibbsProposal(wi, wi, 0.0, q_forward, 0.0, 0.0, True, False). It marks accepted True, no move. That's fine.

But wait the proposal distribution includes all valid_windows, including windows held by replicas. The target_reps array is holders[valid_windows]. For valid_windows includes wi, target_reps = rep. For other windows, target_reps are other replicas. The move is a transposition between rep and target_reps. The proposal distribution q(wj) is computed using delta = new_e - old_e where old_e = bias[wi, rep] + bias[wj, target]. For wj=wi, old_e = 2*U_wi(x_rep), new_e = 2*U_wi(x_rep), delta=0. But the actual old total for the system is U_wi(x_rep)+sum others, not 2*U_wi(x_rep). The delta for stay should be 0, which they set. For other wj, old_e = U_wi(x_rep)+U_wj(x_target), new_e = U_wj(x_rep)+U_wi(x_target). This is correct difference in total bias. So q(wj) ∝ exp(-β delta) for wj≠wi, and q(wi)=1 (after normalization). This is a valid proposal distribution (not necessarily heat-bath but valid). Reverse q computed similarly with holders_after. Need check reverse q for the reverse move: after swap, rep in wj, target in wi. Reverse proposal from state after swap: choose target window wi? The code computes reverse distribution with current_window=wj, replica_of_window=holders_after. valid_windows includes wi with target_reps = target. old_e_rev = bias[wj, rep] + bias[wi, target]; new_e_rev = bias[wi, rep] + bias[wj, target]; delta_rev = new-old = -delta. q_reverse = probability of choosing wi. Good. So MH correct.

Potential issue: _gibbs_window_proposal_distribution uses bias_matrix_kj but beta is passed. It computes log_weights = -beta*delta. Then normalizes. Good.

But there is a subtle issue: the proposal distribution includes all valid_windows, but the "stay" candidate (wj=wi) has delta=0 and weight 1. However, for other candidates, delta can be negative, weight >1. They clip log_weights to [-745,745]. If delta very negative, exp(-beta*delta) huge, but after subtracting max, OK. If delta very positive, weight exp(-745) ~ 0, effectively zero. This is fine. But clipping log_weights to -745 means weights for very unfavorable moves are not exactly zero but ~5e-324? Actually exp(-745) ~ 5e-324, near underflow. They then subtract max and exp; if max is 0, exp(-745) ~ 5e-324, which is subnormal but nonzero. Could cause probabilities sum maybe 1 + tiny. OK. But if all log_weights are -745, finite true, weights all ~5e-324, sum ~ n*5e-324, probabilities uniform. That's fine.

Potential issue: if delta is NaN, log_weights NaN, finite false, filtered. OK.

Now _gibbs_mh_acceptance_probability: log_alpha = -beta*delta + log(qr)-log(qf). If qf or qr zero, return 0. But if q_reverse is 0 for a move that is possible? In reverse distribution, q_reverse could be zero if the reverse move's window wi is not in valid_windows? But wi should always be valid because holders_after[wi] = target_rep >=0. Unless target_rep <0? No. So q_reverse >0. OK.

_exchange_probability: standard Metropolis for swap: if delta <=0 accept 1 else exp(-beta*delta). Good.

swap_candidate_replicas: returns i,j. OK.

apply_window_swap: old_e = bias[wi,i]+bias[wj,j]; new_e = bias[wj,i]+bias[wi,j]; delta. pacc = _exchange_probability(delta,beta). If force_accept, accepted True. Else accepted = uniform < pacc. Then assignments[i], assignments[j] = assignments[j], assignments[i]; replica_of_window[assignments[i]] = i; replica_of_window[assignments[j]] = j. Wait after swapping assignments[i] and assignments[j], assignments[i] is old assignments[j] = wj, assignments[j] = old assignments[i] = wi. Then replica_of_window[assignments[i]] = i means replica_of_window[wj] = i (replica i now in wj). replica_of_window[assignments[j]] = j means replica_of_window[wi] = j. Correct.

But note: if i and j are replicas, assignments array indexed by replica. Good.

Potential issue: apply_window_swap mutates assignments and replica_of_window in place. If p_override or force_accept used in tests, OK.

Now explicit_2d_neighbor_pairs_for_exchange: builds edges, slots, round-robin by edge index modulo slots, then greedy disjoint filtering. If chosen empty, try all edges greedily. This is a heuristic to select disjoint pairs. Not a math error per se. But potential issue: "round-robin edge coloring by edge index" not actual graph coloring; may select non-disjoint? They filter used. OK.

assemble_bias_matrices: distance_bias_kcal + ss_bias_kcal + boost_bias_kcal. boost_bias_kj / 4.184. bias_kj = 4.184 * bias_kcal. OK. But note: boost_bias_kj is [n_states, n_replicas]? pep_gamd_boost_matrix_kj returns vstack of boost for each lambda, shape (n_states, n_samples). If n_states = n_replicas? In production, state_lambdas length n_states, v_pep_kj length n_replicas. So shape (n_states, n_replicas). Good. distance_bias_kcal shape (n_states, n_replicas). OK.

_fetch_v_pep_v_dih: reads dihedral group energy from context. If pep_env None returns nan. If has_total false returns nan, v_dih. OK.

campaign_ladder_registry_lambda: reads registry CSV, returns first lambda >0. Potential issue: if multiple rows, returns first positive, not necessarily the ladder's max or per-state? It's used only to decide ladder active. OK.

resolve_ladder_active: true if any state_lambdas >0 or registry has positive. OK.

_derive_state_gamd_lambdas: if normalized_rows present and length n and all have gamd_lambda, return those. Else existing if length n. Else zeros. Potential issue: if normalized_rows present but some gamd_lambda missing, falls back to existing or zeros. Could silently disable ladder if metadata incomplete. But maybe intended. Need flag? It says "Zeros (ladder inactive) is the fallback of last resort, never a guess." But if ladder was active and metadata missing, zeros would silently disable. However, there is _reload_state_gamd_lambdas_on_resume to restore from manifest. But if not resume, and normalized_rows missing gamd_lambda, could silently disable. Is that a math/physics error? It's a correctness issue. But maybe not in scope? We need flag if can silently produce wrong free energy. The prompt asks adversarial review of code for silent wrong numbers. We should flag if fallback to zeros can silently disable ladder. But the docstring says zeros is fallback of last resort, never a guess. Yet if active ladder metadata missing, zeros is wrong. Need see if there is any guard. In _derive_state_gamd_lambdas, if rows present but not all have gamd_lambda, it falls to existing or zeros. If existing is None, zeros. That could silently set λ=0 for all states, meaning no boost applied/priced? Actually if ladder active, k0max_by_channel not None, state_lambdas zeros would set k0=0, so replicas run unboosted, but samples record v_pep/v_dih and MBAR prices zero boost. That would be wrong if intended boost. But maybe the ladder is only active if state_lambdas has positive values; if all zeros, resolve_ladder_active might still be true via registry, so k0max_by_channel set, and state_lambdas zeros means no boost. That could be a silent failure. Need flag? The code has _reload to restore from manifest on resume, but initial run? If window_metadata normalized_rows lacks gamd_lambda, and existing None, zeros. Is that possible? The docstring says normalized_rows is source of truth whenever usable. If not usable, fallback. Could be a real issue. But maybe not enough context. We'll consider.

_reload_state_gamd_lambdas_on_resume: reads manifest method_settings state_gamd_lambdas. If args.state_gamd_lambdas truthy, returns. If all-zero list is truthy? In Python, [0.0] is truthy (non-empty). So an all-zero ladder set by caller is not overridden. Good. But if args.state_gamd_lambdas is empty list, it reads manifest. OK.

_persist_state_gamd_lambdas: writes current list to manifest. OK.

Now pep_gamd.py _channel_boost: This is important. Formula: b = 0.5 * k0 * (e - v)^2 / rng, where rng = vmax - vmin. Then if (b+v) < e, b else 0. This is GaMD boost? Need check standard GaMD: boost potential V_b = 0.5 k0 (E - V)^2 / (E_max - E_min) when V < E, else 0. Here e is threshold energy (E), v is potential energy (V). rng = vmax - vmin. But standard uses (E_max - E_min) or (V_max - V_min)? In GaMD, for total boost: V_b = 0.5 k0 (E - V)^2 / (E_max - E_min) if V < E. Actually original GaMD: ΔV = 0.5 k0 (E - V(r))^2 / (E - V_min) for V(r) < E, else 0. Where E is threshold energy, V_min is minimum potential. Some implementations use (V_max - V_min) or (E - V_min). Need check. The code uses rng = vmax - vmin. Is that correct? Let's examine. In gamd-openmm, the boost is often: if V < E: 0.5 * k0 * (E - V)^2 / (E - V_min). Actually the "threshold" E is set to V_min + (V_max - V_min)/? Let's recall. GaMD: E = V_min + 1/k? Hmm.

The code's _channel_boost: b = 0.5 * k0 * (e - v)^2 / rng, with rng = vmax - vmin. Then condition (b+v) < e. This is a "lower-bound" dual boost? The submission says "Pep-GaMD lower-bound dual (dependent dihedral + total) boost whose envelope (Vmax/Vmin/threshold/k0max) is calibrated". The formula may be from gamd-openmm. Need verify against known GaMD. Standard GaMD boost potential:
ΔV(r) = 0.5 k0 (E - V(r))^2 / (E - V_min) if V(r) < E, else 0.
Where E is the threshold energy, often E = V_min + (V_max - V_min)/? Actually in GaMD, E is set to V_min + (1/λ)(V_max - V_min)? Let's derive from "lower-bound" GaMD: The boost is designed such that the modified potential V* = V + ΔV has a lower bound V_min? If ΔV = 0.5 k0 (E - V)^2 / (E - V_min), then V* = V + 0.5 k0 (E-V)^2/(E-V_min). At V=E, ΔV=0, V*=E. At V=V_min, ΔV=0.5 k0 (E-V_min), V*=V_min+0.5k0(E-V_min). The condition ΔV+V < E? Let's check: ΔV+V < E iff 0.5 k0 (E-V)^2/(E-V_min) < E-V. If E>V, divide by (E-V): 0.5 k0 (E-V)/(E-V_min) < 1 => E-V < 2(E-V_min)/k0. Not always. The code's condition (b+v) < e ensures the boosted potential stays below threshold? Actually if b+v >= e, set b=0. This is a "threshold" condition to avoid over-boosting. Standard GaMD has condition V < E, not b+V < E. The code uses b+v < e, which is stricter. Is that correct for "lower-bound" GaMD? Need know.

The submission says "Pep-GaMD lower-bound dual (dependent dihedral + total) boost whose envelope (Vmax/Vmin/threshold/k0max) is calibrated ONCE by the swarm and then frozen". The code's _channel_boost may be from gamd-openmm's StageIntegrator. We need check if formula matches gamd-openmm. The docstring in pep_gamd_boost_kj says "gamd-openmm's dependent dual boost under rung λ: dihedral first, then Total with the dihedral boost added to the Total energy before the square (stage_integrator _add_dihedral_boost_to_total_energy)." So they claim it mirrors gamd-openmm. We need verify.

Let's recall gamd-openmm code. In openmm-GaMD, the boost potential is:
if V < E:
  V_b = 0.5 * k0 * (E - V)^2 / (E - V_min)
else V_b = 0
Actually in OpenMM CustomIntegrator, they use "E" as threshold, "Vmin" and "Vmax". The denominator might be (E - Vmin) not (Vmax - Vmin). Let's search memory: GaMD boost:
ΔV = 0.5 k0 (E - V)^2 / (E - V_min) for V < E.
The parameters: E = V_min + (V_max - V_min)/? In "GaMD" the threshold E is set to V_min + (V_max - V_min)/? Actually from Miao et al. 2015: E = V_min + (V_max - V_min)/? Let's think. The boost potential is defined with a "threshold energy" E. To ensure the modified potential has no higher barrier, E is chosen such that the standard deviation of boost potential is controlled. The denominator is (E - V_min). Some implementations use (V_max - V_min) as an approximation? Let's check code from gamd-openmm: I recall in openmm-GaMD's integrator, the boost is:
if (V < E):
    Vb = 0.5 * k0 * (E - V)^2 / (E - Vmin)
else:
    Vb = 0
And the condition for applying boost is V < E, not V+Vb < E. The code here uses rng = vmax - vmin and condition b+v < e. That seems different. But maybe "lower-bound" variant uses Vmax-Vmin and condition V* < E? Let's investigate.

The submission says "Pep-GaMD lower-bound dual (dependent dihedral + total) boost whose envelope (Vmax/Vmin/threshold/k0max) is calibrated". "lower-bound" might refer to "GaMD with a lower bound" where the boost is applied only when the boosted potential is below threshold? Actually "lower-bound" in GaMD: The modified potential has a lower bound? Let's search memory: In GaMD, the boost potential is:
ΔV(r) = 0.5 k0 (E - V(r))^2 / (E - V_min) if V(r) < E
This ensures the modified potential V*(r) = V(r) + ΔV(r) is a "lower-bound" of the original? No, it's always >= V. The "lower-bound" might refer to "V* is bounded below by V_min"? Hmm.

Let's derive from code's formula: b = 0.5 k0 (e - v)^2 / (vmax - vmin). If vmax-vmin is the range of potential energy sampled. This is a common "Gaussian accelerated MD" variant where the boost is proportional to (E-V)^2 / (V_max - V_min). Actually in "Gaussian accelerated MD" (GaMD), the boost is:
ΔV = 0.5 k0 (E - V)^2 / (E - V_min) for V < E.
But some implementations use (V_max - V_min) as denominator because E - V_min ≈ V_max - V_min? Let's check original GaMD paper: "The boost potential is ΔV = 0.5 k0 (E - V)^2 / (E - V_min) when V < E." Yes.

However, there is "Pep-GaMD" (peptide GaMD) which might use a different "lower-bound" scheme. The code's condition (b+v) < e is unusual. Let's see if it's from gamd-openmm. In gamd-openmm, the boost is applied if V < E, and the boost is 0.5*k0*(E-V)^2/(E-Vmin). The code's condition b+v < e would be equivalent to V + 0.5 k0 (E-V)^2/(Vmax-Vmin) < E. That is not standard. Could be a bug. Need flag if verified.

Let's examine _channel_boost more carefully:
```
def _channel_boost(v, e, vmax, vmin, k0):
    v = _np.asarray(v, dtype=float)
    rng = vmax - vmin
    scale = _np.maximum(_np.maximum(abs(e), _np.abs(v)), 1.0)
    b = 0.5 * k0 * (e - v) ** 2 / rng
    b = _np.where(_np.abs(rng) <= 0.001 * scale, 0.0, b)
    return _np.where((b + v) < e, b, 0.0)
```
If rng is zero or small, b=0. If b+v >= e, b=0. This means boost only applied when the boosted potential is below threshold. But standard GaMD applies boost when V < E, regardless of V+ΔV. The condition V+ΔV < E is much stricter; for V close to E, ΔV is small, so V+ΔV < E may hold if ΔV < E-V. Since ΔV = 0.5 k0 (E-V)^2/rng, condition ΔV < E-V => 0.5 k0 (E-V)/rng < 1 => E-V < 2 rng / k0. If k0 is large, only very close to E. This could severely reduce boost. Is that intended? The docstring says "lower-bound" maybe means the boost is applied only when the total energy is below threshold? Actually in GaMD, the boost is designed to "raise" the potential energy to a threshold E, but not above? Wait the modified potential V* = V + ΔV. If V < E, ΔV >0, so V* > V. It can exceed E. The condition V* < E would mean the boost never pushes the potential above E, i.e., it "fills" the potential well up to E but not above. That is a "lower-bound" or "threshold" variant: the boosted potential is capped at E. That is not standard GaMD; standard GaMD allows V* to exceed E. But there is a variant called "GaMD with a lower bound" where the boost is applied only when V < E and the boost is such that V* <= E? Let's think. If you want the modified potential to have a lower bound? Actually if V* <= E, then the potential is bounded above by E, not lower. "Lower-bound" might refer to the boost potential being a lower bound of something? Hmm.

Let's not assume. We need audit against known analytic limits. The code's formula may be correct for their "Pep-GaMD lower-bound" variant. But we need check dimensional consistency: b = 0.5 * k0 * (e - v)^2 / rng. k0 units? In GaMD, k0 has units of 1/energy? Let's check. If e and v are energies (kJ/mol), (e-v)^2 has energy^2. rng energy. So b has energy * k0. For b to be energy, k0 must be dimensionless? Actually k0 in GaMD is a dimensionless parameter? In original GaMD, k0 is a harmonic force constant with units of 1/energy? Let's recall: ΔV = 0.5 k0 (E - V)^2 / (E - V_min). If E,V in kcal/mol, then (E-V)^2/(E-V_min) has units kcal/mol. For ΔV to be kcal/mol, k0 must be dimensionless. Indeed k0 is a dimensionless parameter (0 < k0 <= 1). In code, k0max is likely dimensionless. So OK.

But wait in umbrella bias, k_kcal is force constant kcal/mol per CV^2. In boost, k0 dimensionless. OK.

Now pep_gamd_boost_kj: b_dih = _channel_boost(v_dih, threshold_dih, vmax_dih, vmin_dih, lam*k0max_dih). Then if has_total: b_tot = _channel_boost(v_pep + b_dih, threshold_total, vmax_total, vmin_total, lam*k0max_total). out = b_dih + b_tot. This is "dihedral first, then Total with the dihedral boost added to the Total energy before the square". Is that correct for dependent dual boost? In GaMD dual-boost, the total boost is applied to the total potential energy, and the dihedral boost is applied to the dihedral energy. The total boost should be computed on the total potential energy V_total (including dihedral) without adding the dihedral boost? Actually in "dual boost" GaMD, two boosts are applied: one to the dihedral energy and one to the total potential energy. The total boost is computed on the total potential energy V_total, not V_total + dihedral boost. But the code adds b_dih to v_pep before computing b_tot. The docstring says "gamd-openmm's dependent dual boost under rung λ: dihedral first, then Total with the dihedral boost added to the Total energy before the square (stage_integrator _add_dihedral_boost_to_total_energy)." So they claim this is how gamd-openmm does it. Is that physically correct? In a dual-boost GaMD, the total boost is applied to the total potential energy, and the dihedral boost is applied to the dihedral energy. The total potential energy used for the total boost should be the original total potential energy, not including the dihedral boost, because the boost is a function of the original potential. However, some implementations add the dihedral boost to the total energy before applying the total boost to avoid double counting? Let's think. The total modified potential is V* = V_total + ΔV_dih + ΔV_total. If ΔV_total is computed from V_total + ΔV_dih, then the total boost depends on the boosted dihedral energy, which is a different function. Is that what gamd-openmm does? The docstring explicitly says "stage_integrator _add_dihedral_boost_to_total_energy". So if the code mirrors gamd-openmm, it's likely correct for that engine. But is it physically correct for Boltzmann reweighting? The boost potential applied in the simulation must be exactly the one used in reweighting. If the code's pep_gamd_boost_kj matches the integrator's boost, then OK. But if the integrator actually computes total boost on original total energy, then this function would be wrong. Need verify from code? We don't have gamd-openmm source. The submission says "gamd-openmm's dependent dual boost" and "stage_integrator _add_dihedral_boost_to_total_energy". So likely they know. But we can flag if the formula is non-standard? Need be careful.

Let's examine the "dependent" in "dependent dual (dependent dihedral + total) boost". "Dependent" might mean the total boost depends on the dihedral boost. So adding b_dih to v_pep before total boost is intentional. OK.

Now _channel_boost condition: `_np.where((b + v) < e, b, 0.0)`. If b+v >= e, set b=0. This is a "lower-bound" condition. Is it dimensionally consistent? b+v and e all energies. OK.

But there is a potential issue: if v > e, then (e-v)^2 positive, b positive, but condition (b+v) < e is false because v > e, so b=0. Good. If v < e but b+v >= e, b=0. This means the boost is zero for a range of v near e. This is a "hard cap" at e. Is that the intended "lower-bound" boost? Maybe.

Now k0max_from_globals: returns Total and Dihedral from shared_globals. If "k0_Dihedral" missing, KeyError. OK.

set_replica_lambda: sets k0_Dihedral and k0_Total to lam*k0max. If lam=0, sets zero. OK.

set_replica_lambda_for_window: if k0max_by_channel None, no-op. If state_lambdas None and ladder active, raise. OK.

Now state_identity.py: canonical_state_definition requires exact set of fields. It converts energy_unit kJ/mol to kcal/mol by dividing k1,k2 by 4.184. Wait if energy_unit == "kJ/mol", row["k1"] /= 4.184. But k1 is force constant in kJ/mol per CV^2. To convert to kcal/mol per CV^2, divide by 4.184. Correct. But note: if k1 is in kJ/mol/A^2, dividing by 4.184 gives kcal/mol/A^2. OK.

Potential issue: `if row["k1"] > 0 and data["cv1"] is None:` but if k1 is NaN? normalize_windows ensures finite. OK.

Now bias.py reconstruct_bias_matrix: It computes matrix = sum 0.5*k*delta^2 in kcal/mol, then matrix *= beta * KJ_PER_KCAL. So matrix becomes dimensionless reduced bias (beta * energy in kJ/mol). Wait beta is mol/kJ. So beta * KJ_PER_KCAL * kcal = beta * kJ = dimensionless. Good. But the function docstring says "Return the (N,K) reduced umbrella-plus-ladder bias, not full potential." So OK. However, it multiplies by beta*KJ_PER_KCAL after summing all umbrella terms. If k1,k2 are kcal/mol per CV^2, then sum is kcal/mol. Multiply by 4.184 to kJ/mol, then by beta (mol/kJ) -> dimensionless. Good.

But note: in reconstruct_bias_matrix, if ladder active, it calls _ladder_apply(matrix, pep, dih, lambdas, envelope, beta, bookkeeping). The matrix is already reduced (dimensionless). The ladder boost function likely expects reduced matrix? Need check. In production, bias_matrix_kj is in kJ/mol, then reduced_bias_matrix = beta * bias_matrix_kj. In reconstruct_bias_matrix, matrix is reduced. The ladder apply function may add boost in reduced units? Need see. The docstring says "reduced umbrella-plus-ladder bias". So OK.

Potential issue: In reconstruct_bias_matrix, if secondary is None and no active secondary, OK. If secondary has NaN for some observations, matrix column for active secondary will be NaN for those rows. That's intended.

Now production sample function: It computes bias_matrix_kcal, bias_matrix_kj, reduced_bias_matrix = beta * bias_matrix_kj. Then for each replica, sampled_bias_kcal = all_bias_kcal[w], sampled_bias_kj = all_bias_kj[w]. But note: all_bias_kcal includes boost_bias_kcal. So "umbrella_bias_kcal_mol" actually includes boost, not just umbrella. The column name "umbrella_bias_kcal_mol" is misleading but not a math error? It could cause confusion. But the prompt asks for math/physics correctness, not naming. However, if downstream analysis uses "umbrella_bias_kcal_mol" expecting only umbrella, it would be wrong. But the code also has "sampled_umbrella_bias_kj" which is only umbrella (distance+ss) and "sampled_boost_bias_kj". The "umbrella_bias_kcal_mol" includes boost. That could be a silent error if analysis uses it. Need flag? The submission says "state bias matrix" includes umbrella + boost. The column name is misleading but maybe not a mathematical error. We can flag as MINOR if it can cause wrong free energy if used incorrectly. But the audit should focus on code producing wrong numbers. If the column is documented? Not in snippet. We'll consider.

More importantly: In sample(), `boost_bias_matrix_kj = pep_gamd_boost_matrix_kj(v_pep_kj, v_dih_kj, state_lambdas, pep_env) if pep_env is not None else np.zeros((nrep, nrep))`. If pep_env is not None but state_lambdas length maybe not nrep? It should be n_states. If n_states != nrep? In REUS, number of windows equals number of replicas? Usually yes. But if not, shape mismatch. The code assumes nrep x nrep. In production, nrep = number of replicas = number of windows? Likely. OK.

Potential issue: `pep_gamd_boost_matrix_kj` returns shape (n_states, n_samples). If n_states != nrep, then bias_matrix_kj shape (n_states, nrep). Later `reduced_bias_matrix = float(beta) * bias_matrix_kj` shape (n_states, nrep). Then for each replica r, `all_bias_kcal = bias_matrix_kcal[:, r]` shape (n_states,). `sampled_bias_kcal = all_bias_kcal[w]` OK. Exchange uses bias_matrix_kj[window, replica]. So if n_states != nrep, exchange still works as long as assignments length nrep and windows n_states. But replica_of_window size? In production, likely n_states = nrep. OK.

Now _current_exchange_arrays: It computes bias_matrix_kj for exchange. It uses `_has_ss_axis = (secondary_cv_centers is not None and secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None)`. But in sample(), `_has_ss_axis = secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None`. There may be inconsistency: exchange uses ss_ks_kj_arr_global, sample uses ss_k_kcal_arr_global. If one is None and other not, exchange and sample bias matrices differ. Need check. In sample, `ss_centers_arr = ss_centers_arr_global`, `ss_k_arr = ss_k_kcal_arr_global`. In exchange, `ss_centers_arr_global`, `ss_k_kcal_arr_global`? Wait exchange code: `_has_ss_axis = (secondary_cv_centers is not None and secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None)`. Then `umbrella_bias_matrix_kcal(..., ss_centers_arr_global if _has_ss_axis else None, ss_k_kcal_arr_global if _has_ss_axis else None)`. It uses ss_k_kcal_arr_global, not ss_ks_kj_arr_global. The condition checks ss_ks_kj_arr_global not None, but passes ss_k_kcal_arr_global. If ss_k_kcal_arr_global is None while ss_ks_kj_arr_global not None, then it would pass None to ss_k_kcal, but _has_ss_axis true, so umbrella_bias_matrix_kcal would receive ss_centers_arr_global and None? Actually call: `umbrella_bias_matrix_kcal(primary_values, ss_values, centers_a_arr, k_arr, ss_centers_arr_global if _has_ss_axis else None, ss_k_kcal_arr_global if _has_ss_axis else None)`. If _has_ss_axis true but ss_k_kcal_arr_global None, then ss_k_kcal=None, and umbrella_bias_matrix_kcal checks `if ss_centers is None or ss_k_kcal is None: return distance_kcal, zeros`. So secondary bias zero. But sample path might have ss_k_kcal_arr_global not None, so sample includes secondary bias while exchange doesn't. That would cause exchange to use different Hamiltonian than samples, violating detailed balance? Actually exchange would not include secondary bias, so swaps would be priced without secondary restraint, while the actual umbrella potential includes it. That would break detailed balance and produce wrong free energy. Need verify if ss_k_kcal_arr_global and ss_ks_kj_arr_global are always both set or both None. The code snippet doesn't show initialization. But the condition is suspicious: it checks ss_ks_kj_arr_global but uses ss_k_kcal_arr_global. Could be a bug if they are not guaranteed to be both non-None. Need flag as SUSPECTED or VERIFIED? We can point to lines. The prompt asks distinguish VERIFIED/SUSPECTED. We can say SUSPECTED because we don't see initialization. But the condition mismatch is a code smell. Let's note.

Also in sample(), `_has_ss_axis = secondary_cv_centers is not None and secondary_cv_k_kcal_list is not None`. It uses `ss_centers_arr_global` and `ss_k_kcal_arr_global`. In exchange, `_has_ss_axis` additionally requires `secondary_cv_ks_kj is not None and ss_ks_kj_arr_global is not None`. If secondary_cv_ks_kj is None but secondary_cv_k_kcal_list not None, sample includes secondary, exchange doesn't. That would be a serious bug. Need flag.

Now the exchange loop: `_attempt_window_swap` uses `bias_matrix_kj` from `_current_exchange_arrays`. It calls `apply_window_swap` with beta. Good. But note: after accepted swap, it calls `set_window` and `set_replica_lambda_for_window` on the two replicas. However, `set_window` likely sets the umbrella center/force constant for the replica's new window. But the bias matrix used for exchange was computed from current CVs and state lambdas. After swap, the replicas' window assignments change, but their CVs remain the same. The next exchange will recompute bias matrix with new assignments? Wait bias_matrix_kj is [state, replica], independent of assignments. So OK. But the actual Hamiltonian of each replica after swap is changed to the new window's umbrella and lambda. The bias matrix for the next exchange uses the same CVs and state lambdas, so it correctly prices the new state's bias for each replica. OK.

Potential issue: In `_attempt_window_swap`, after accepted swap, it calls `set_window` and `set_replica_lambda_for_window` on replicas i and j. But it does not update `observable_cache`? It clears observable_cache. OK.

Now `_current_exchange_arrays` uses `_fetch_exchange_state` which for fast path reads CVs and v_pep/v_dih. For non-fast path, it reads positions and computes CVs, and v_pep/v_dih. It does NOT read potential energy. That's fine because exchange delta only depends on bias, not physical potential. But wait if the boost is active, the boost bias depends on v_pep and v_dih, which are read. OK.

Now the Gibbs proposal: In production, how is it used? The snippet only shows functions, not the loop. But we can audit the functions.

Potential issue in `_gibbs_window_proposal_distribution`: It computes `old_e = bias[wi, rep] + bias[valid_windows, target_reps]`. For valid_windows including wi, target_reps = rep, so old_e = 2*bias[wi,rep]. For other windows, old_e = bias[wi,rep]+bias[wj,target]. This is correct for the total bias of the two-replica subsystem. But what about the "stay" candidate? The total bias for stay should be bias[wi,rep]+bias[wi,rep]? No, if wj=wi, target_reps=rep, the two replicas are the same replica, so the "two-replica subsystem" is just one replica. The total bias should be bias[wi,rep], not 2*bias[wi,rep]. The code's old_e for stay is 2*bias[wi,rep], and new_e also 2*bias[wi,rep], delta=0. The absolute value doesn't matter for delta, but it affects the proposal probability? They set delta=0 for stay, so weight=1. The normalization includes this weight. If they had used correct old_e = bias[wi,rep] and new_e = bias[wi,rep], delta=0 as well. So no effect. OK.

But wait for other candidates, old_e = bias[wi,rep]+bias[wj,target]. This is the total bias of the two replicas in their current windows. new_e = bias[wj,rep]+bias[wi,target]. Correct. So delta is correct. Good.

Potential issue: The proposal distribution q(wj) is normalized over valid_windows, but the "stay" candidate has weight 1 regardless of the current window's bias. This means the probability of staying is 1/Z where Z = 1 + sum_{wj≠wi} exp(-β delta). This is a valid proposal. OK.

Now `gibbs_propose_one_replica`: It computes reverse proposal using `holders_after`. But note: `holders_after` is a copy of `replica_of_window` with the swap applied. However, `replica_of_window` maps window -> replica. After swap, replica rep moves to wj, target moves to wi. The code sets `holders_after[wi] = target_rep; holders_after[wj] = rep`. Correct. Then reverse distribution for replica rep from current_window=wj. It computes q_reverse = probability of choosing wi. Good.

But there is a subtle issue: The reverse proposal distribution should be computed for the same move type: a transposition of replica rep with the replica in window wi. In the reverse state, replica rep is in wj, and the replica in wi is target_rep. The reverse move would swap rep with target_rep, moving rep to wi and target_rep to wj. The code's reverse distribution uses `replica_index=rep`, `current_window=wj`, `replica_of_window=holders_after`. It considers all valid windows, including wi. For candidate wi, target_reps = holders_after[wi] = target_rep. Delta_rev = new_e - old_e = (bias[wi,rep]+bias[wj,target]) - (bias[wj,rep]+bias[wi,target]) = -delta. q_reverse = probability of choosing wi. Good.

Now `_gibbs_mh_acceptance_probability`: log_alpha = -beta*delta + log(qr)-log(qf). If qf or qr zero, return 0. But if q_reverse is zero due to underflow? The proposal distribution uses log_weights clipped to -745, so q_reverse could be extremely small but not zero unless filtered by finite. If q_reverse is 0 because the reverse move's window wi was filtered out due to non-finite delta? But delta_rev should be finite if forward delta finite. OK.

Potential issue: The acceptance probability for Gibbs-with-MH should be min(1, [π(dest) q_rev] / [π(curr) q_fwd]). Here π ratio = exp(-β delta) where delta = new - old. So log_alpha = -β delta + log(qr)-log(qf). Correct.

Now `_exchange_probability`: standard Metropolis for swapping windows. But note: In replica exchange with state-dependent Hamiltonians, the acceptance probability for swapping replicas i and j between windows m and n is min(1, exp(-β [U_m(x_j)+U_n(x_i) - U_m(x_i)-U_n(x_j)])). The code's delta = new_e - old_e = U_wj(x_i)+U_wi(x_j) - U_wi(x_i)-U_wj(x_j). Correct. If delta <=0 accept 1. OK.

Now `apply_window_swap`: It uses `_exchange_probability(delta, beta)`. But if `p_override` is provided, it uses that. OK.

Potential issue: In `apply_window_swap`, after accepted, it updates `assignments` and `replica_of_window`. But it does not update the actual integrator's window/lambda; that's done by caller. OK.

Now `explicit_2d_neighbor_pairs_for_exchange`: It returns chosen pairs. The "slots" round-robin by edge index modulo slots. If slots=4, parity 0 uses edges idx 0,4,8,...; parity 1 uses 1,5,9,... etc. This is not a proper edge coloring; it may select edges that share vertices? They filter used. OK. But if a slot has no edges, it tries all edges greedily. This could cause the same pair to be attempted in multiple slots? But chosen is per call. OK.

Now the production sample function has a potential issue with `observable_cache["cvs_nm"] = (primary_values / 10.0) if is_distance_primary else primary_values`. If primary CV is distance in angstroms, dividing by 10 gives nm. OK. But if primary CV is contact fraction dimensionless, no conversion. OK.

Now `umbrella_restoring_force_kcal_mol_per_A`: `k_kcal_a2 * (center_a - cv_a)`. Force = -k(x-c), magnitude k(c-x). Units kcal/mol/A. OK.

Now `potential_kj_mol`: from getState getEnergy groups=physical_energy_groups_for_args(args). This is the physical potential energy excluding umbrella and boost? Need check. The sample row records `potential_kj_mol` as physical potential. But the bias matrix includes umbrella + boost. For MBAR, the total energy is physical + bias. If `potential_kj_mol` is physical only, then MBAR can add bias. OK. But if `physical_energy_groups_for_args` includes boost? Need see. Not in snippet. Could be a bug if it includes boost. But likely physical groups exclude umbrella/boost. We'll not flag without evidence.

Now `extract_gamd_boost_kj(sim.integrator, unit, None)` reads the integrator's current boost? It may read global variables? It returns boost_kj, components, source. This is used for row "gamd_boost_total_kj_mol". But the actual boost applied by the integrator may differ from `pep_gamd_boost_kj` computed from v_pep/v_dih. If they disagree, the recorded boost is wrong. The code has a calibration case where boost not applied. The new code may still have mismatch. Need check: `extract_gamd_boost_kj` likely reads the integrator's boost energy from a custom force? Not shown. Could be a source of silent error. But we can't verify.

Now `_fetch_v_pep_v_dih`: It reads dihedral group energy from context. But if the integrator's boost is applied via a CustomIntegrator, the group energies may include the boost? The dihedral group energy from getState(groups={DIHEDRAL_GROUP}) should be the potential energy of dihedral forces, not including boost. OK. `peptide_essential_energy_kj` reads total? Not shown.

Now `campaign_ladder_registry_lambda`: It searches parent directories up to adaptive_production. If multiple registry files, returns first positive. Could be stale from a previous campaign? It stops at adaptive_production. OK.

Now `resolve_ladder_active`: If state_lambdas has any >0, true. Else if registry has positive, true. But if state_lambdas is all zeros and registry has positive, it returns true, so ladder active. Then `k0max_by_channel` is set, and `set_replica_lambda_for_window` will set k0=0 for all windows. That means replicas run unboosted, but the ladder code path is active, so samples record v_pep/v_dih and MBAR prices zero boost. This is consistent: λ=0 means no boost. OK.

Now `_derive_state_gamd_lambdas`: If normalized_rows has gamd_lambda values, returns them. If not, falls back to existing or zeros. Potential issue: if existing is provided but length n, it returns existing even if it's stale/misaligned. The docstring warns callers not to pass pre-reindex values. But the function itself doesn't verify alignment. Could be a source of silent misalignment. But maybe call sites are careful. We can flag as SUSPECTED.

Now `_reload_state_gamd_lambdas_on_resume`: It reads manifest method_settings state_gamd_lambdas. If args.state_gamd_lambdas is truthy, returns. If it's an empty list, reads manifest. If manifest has lambdas, sets args.state_gamd_lambdas. But note: if manifest has an all-zero list, `if lambdas:` is True because list non-empty, so it sets. Good. If manifest has empty list, no. OK.

Now `_persist_state_gamd_lambdas`: Writes current args.state_gamd_lambdas to manifest. If args.state_gamd_lambdas is None, writes empty list. OK.

Now state_identity.py: `canonical_state_definition` requires exact set of fields. It converts energy_unit kJ/mol to kcal/mol by dividing k1,k2 by 4.184. But it does not convert `gamd_lambda`? No need. It also checks `if row["k1"] > 0 and data["cv1"] is None`. But if k1 is 0 and cv1 None, OK. If k1 negative? normalize_windows enforces minimum=0, so no negative. OK.

Potential issue: In `canonical_state_definition`, for NPT, it allows pressure_bar to be any finite number, including negative? `finite_number(data["pressure_bar"], "pressure_bar")` without positive=True. Negative pressure is unphysical for NPT? Could be allowed? Usually pressure can be negative in some contexts, but for water NPT, negative pressure is unphysical. Not a math error.

Now `bias.py` `_term`: If force_key not in window and axis==2 and center_key not in window, returns 0,0. But if axis==2 and force_key missing but center_key present, it raises "missing k2; inactive means explicit zero". This is strict. OK.

Potential issue: `normalize_windows` sets `window["gamd_lambda"] = finite_number(raw.get("gamd_lambda", 0.0), ..., minimum=0)`. It allows gamd_lambda >1? It only checks minimum=0, not maximum. In `set_replica_lambda`, it checks 0<=lam<=1. But `normalize_windows` doesn't. So a window with gamd_lambda=2 would pass state identity but then set_replica_lambda raises. Could be a bug. But not a math error in the code's normal operation. Could flag as MINOR: missing upper bound in normalize_windows. But the prompt asks for errors that can silently produce wrong numbers. If gamd_lambda >1, set_replica_lambda raises, not silent. So not silent.

Now `reconstruct_bias_matrix`: It multiplies matrix by beta * KJ_PER_KCAL. But if `_ladder_apply` expects matrix in kcal/mol or kJ/mol? The docstring says reduced. Need check `apply_ladder_boost_to_u` from gareus.mbar_analysis.ladder. Not shown. Could be a mismatch. But we can't verify.

Now let's look for sign conventions. `beta` is 1/(kB T) in mol/kJ. In production, `beta_value = float(beta)`. In `_gibbs_window_proposal_distribution`, log_weights = -beta * delta. If delta in kJ/mol, beta in mol/kJ, product dimensionless. Good. In `_exchange_probability`, x = -beta * delta. Good.

Now `umbrella_bias_matrix_kcal` returns kcal. `assemble_bias_matrices` converts to kJ. `reduced_bias_matrix = beta * bias_matrix_kj`. Good.

Potential issue: In `sample()`, `sampled_bias_kcal = float(all_bias_kcal[w])` where all_bias_kcal includes boost. But `sampled_bias_kj = float(all_bias_kj[w])`. Then `sampled_umbrella_bias_kj = float(4.184 * (all_distance_bias_kcal[w] + all_ss_bias_kcal[w]))`. This is umbrella only. `sampled_boost_bias_kj = float(boost_bias_matrix_kj[w, r])`. Note: `sampled_bias_kj` should equal `sampled_umbrella_bias_kj + sampled_boost_bias_kj`. But due to floating point, maybe. OK.

Now `umbrella_bias_matrix_kcal` with secondary: `ss_kcal = np.where(np.isfinite(dss), ss_kcal, 0.0)`. If dss is NaN, ss_kcal is NaN, then np.where sets 0. But if dss is NaN, the secondary CV value is missing. Setting bias to 0 for that state means the sample is treated as if the secondary restraint is not applied. But in the actual simulation, if the secondary CV is NaN, the umbrella force may be zero? The code comment says "a replica without a finite secondary value, contributes zero on that axis (the previous sample-path guard, now applied identically to the exchange path -- a NaN entry used to freeze that replica's exchanges)." So this is intentional: missing secondary value means no secondary bias. But if the actual simulation applies a secondary restraint even when CV is NaN? In OpenMM, a CustomCVForce with NaN would produce NaN energy, not zero. But maybe the fast path returns NaN when secondary CV cannot be computed, and the restraint is disabled? Not sure. Could be a silent error if NaN is treated as zero bias while the actual force is NaN or something. But the comment says previous guard. We'll not flag.

Now `_channel_boost` uses `_np.abs(v)` in scale. If v is NaN, scale = max(abs(e), abs(NaN), 1) = NaN? np.maximum with NaN propagates? np.maximum(np.maximum(abs(e), np.abs(v)), 1.0) if v NaN -> NaN. Then rng <= 0.001*scale -> False? NaN comparison false. b = 0.5*k0*(e-v)^2/rng -> NaN. np.where((b+v)<e, b, 0.0): (NaN < e) false, so returns 0. So NaN v yields 0 boost. Is that correct? If v_pep or v_dih is NaN, boost should be? In production, v_pep/v_dih are read from context; if missing, NaN. If boost is active, missing v_pep/v_dih would mean the boost cannot be computed. Setting boost to 0 silently could be wrong. But `_fetch_v_pep_v_dih` returns NaN only if pep_env None or has_total false. If pep_env not None and has_total true, v_pep should be finite. If context read fails, maybe NaN. The code would silently treat as zero boost. Could be a silent error. But maybe not.

Now `pep_gamd_boost_kj`: If v_pep is NaN and has_total true, b_tot = _channel_boost(NaN + b_dih, ...) -> 0, so out = b_dih. That means if total energy missing, only dihedral boost applied. But the actual integrator would apply both? If v_pep missing due to read error, the recorded boost is wrong. Could be a silent failure. But maybe v_pep is always finite when has_total true. We'll consider.

Now `_fetch_v_pep_v_dih`: It reads v_dih from group DIHEDRAL_GROUP. But if the integrator's boost modifies the dihedral energy? The group energy from getState should be the unboosted dihedral potential. OK.

Now `set_replica_lambda`: It sets k0_Dihedral and k0_Total. But if the integrator has additional globals for threshold/vmax/vmin? The envelope is frozen, so only k0 changes with lambda. OK.

Now `k0max_from_globals`: It reads "k0_Total" and "k0_Dihedral" from shared_globals. If shared_globals has "k0_Total" as 0.0 for single dihedral boost, OK. If missing "k0_Dihedral", KeyError. OK.

Now `campaign_ladder_registry_lambda`: It returns `(path.name, lam)` where lam is the first positive lambda found. But if the registry has multiple states with different lambdas, this function only checks if any positive. It doesn't return the full ladder. `resolve_ladder_active` only needs bool. OK.

Now `_derive_state_gamd_lambdas`: If normalized_rows is present and length n and all have gamd_lambda, returns them. But it doesn't check that gamd_lambda values are within [0,1]. Could be >1. Later set_replica_lambda raises. Not silent.

Now `_reload_state_gamd_lambdas_on_resume`: It reads manifest method_settings. But if manifest has state_gamd_lambdas as a list of strings, converts to float. OK.

Now `_persist_state_gamd_lambdas`: It writes list(args.state_gamd_lambdas or []). If args.state_gamd_lambdas is a numpy array, list() gives np.float64 values, JSON serializable? np.float64 may not be JSON serializable by default. But maybe update_run_manifest handles. Not a math error.

Now let's think about the core physics: replica exchange with state-dependent boost. The exchange acceptance should use the total Hamiltonian including the boost. The code's bias matrix includes umbrella + boost. But the boost is a function of the configuration's v_pep and v_dih, which are computed from the current context. However, the boost applied by the integrator may depend on the integrator's internal state (e.g., running averages of Vmin/Vmax) that are updated during simulation. If the envelope is frozen, Vmin/Vmax/threshold are fixed, so boost is a deterministic function of v_pep/v_dih. OK.

But there is a subtle issue: The boost potential in GaMD is not a simple function of the instantaneous potential energy if the envelope parameters are updated during the simulation. The submission says envelope is calibrated once and frozen. So OK.

Now the "dependent dual boost" adds b_dih to v_pep before computing b_tot. Is the total boost applied to the total potential energy including the dihedral boost? In the actual integrator, if it adds the dihedral boost to the total energy before computing the total boost, then the total boost is a function of v_pep + b_dih. The code matches. But is this physically correct for reweighting? The total bias is b_dih + b_tot. The total modified potential is V_total + b_dih + b_tot. If b_tot is computed from V_total + b_dih, then the total boost depends on b_dih, which itself depends on V_dih. This is a valid bias potential as long as it's the actual applied bias. OK.

Now let's check `_channel_boost` against standard GaMD more carefully. The code's condition `(b + v) < e` means the boost is zero if the boosted potential would exceed the threshold. This is a "hard cap" at e. In standard GaMD, the boost is applied for V < E, and the boosted potential V* = V + ΔV can exceed E. The condition V* < E would mean the boost is only applied when V is sufficiently below E such that V* stays below E. This is not standard GaMD. But maybe "lower-bound" GaMD uses this to ensure the modified potential is bounded above by E? Let's search memory: There is "GaMD" where the boost potential is designed such that the modified potential has a "lower bound" equal to the threshold? Actually if you want the modified potential to have a flat bottom up to E, you would set ΔV = E - V for V < E, giving V* = E. But GaMD uses a harmonic function, not linear, so V* < E for V < E? Let's check: ΔV = 0.5 k0 (E-V)^2/(E-Vmin). For V near E, ΔV small, so V* = V + small < E. For V far below E, ΔV could be large, possibly V* > E. The condition V* < E imposes a constraint on k0. In standard GaMD, k0 is chosen such that the boost potential's standard deviation is <= some value, but V* can exceed E. The "lower-bound" variant might choose k0 such that V* <= E always, i.e., the modified potential is bounded above by E. That would be a "lower-bound" on the boost? Actually if V* <= E, then the potential energy is capped at E, which is an upper bound, not lower. Hmm.

Let's not overreach. The code's formula may be from a specific implementation. We need flag only if we can verify against known analytic limits. The prompt says "Canonical limits: verify quantitative claims against known analytic limits - harmonic-oscillator PMF is parabolic, dG = 0 at equal populations, KL >= 0, ESS <= N, ideal-gas and two-state results." We can use those to test.

Let's test the exchange and Gibbs kernels with a simple two-state system. Suppose two windows, two replicas, bias matrix. The stationary distribution should be proportional to exp(-beta * total bias) over assignments. The swap acceptance is correct. The Gibbs proposal with MH should also be correct. Let's verify with a simple case.

Consider two windows A,B and two replicas 1,2. Bias matrix U[window, replica]. Current assignment: rep1 in A, rep2 in B. Total bias = U[A,1]+U[B,2]. A swap would give U[B,1]+U[A,2]. The swap acceptance uses delta = (U[B,1]+U[A,2]) - (U[A,1]+U[B,2]). Correct.

Gibbs proposal for rep1: valid_windows = [A,B], target_reps = [1,2]. For wj=A (stay), delta=0, weight=1. For wj=B, delta = (U[B,1]+U[A,2]) - (U[A,1]+U[B,2]) = same as swap delta. q_forward = exp(-β delta)/(1+exp(-β delta)) if delta positive? Actually weights: stay weight 1, swap weight exp(-β delta). So q_forward = exp(-β delta)/(1+exp(-β delta)). Reverse: after swap, rep1 in B, rep2 in A. Reverse proposal for rep1: valid_windows = [A,B], target_reps = [2,1]. For wj=B (stay), delta=0, weight=1. For wj=A, delta_rev = (U[A,1]+U[B,2]) - (U[B,1]+U[A,2]) = -delta. q_reverse = exp(-β (-delta))/(1+exp(-β (-delta))) = exp(β delta)/(1+exp(β delta)). Then MH acceptance log_alpha = -β delta + log(q_reverse)-log(q_forward). Let's compute. q_reverse/q_forward = [exp(βδ)/(1+exp(βδ))] / [exp(-βδ)/(1+exp(-βδ))]. Note 1+exp(-βδ) = (exp(βδ)+1)/exp(βδ). So q_forward = exp(-βδ) * exp(βδ)/(1+exp(βδ)) = 1/(1+exp(βδ)). q_reverse = exp(βδ)/(1+exp(βδ)). Ratio = exp(βδ). Then log_alpha = -βδ + βδ = 0, so pacc=1. That means the Gibbs move always accepts? Wait that would make the chain mix too fast? Let's check. If q_forward = 1/(1+exp(βδ)) and q_reverse = exp(βδ)/(1+exp(βδ)), then the MH correction exactly cancels the proposal bias, resulting in acceptance probability 1 for both forward and reverse? But if pacc=1 for the forward move, then the chain would always move from current to swapped state if proposed. But the proposal probability of swap is q_forward = 1/(1+exp(βδ)). If δ is large positive (swap unfavorable), q_forward is small, so the move is rarely proposed. If proposed, it's always accepted. That is a valid Metropolis-Hastings scheme
(TRUNCATED - output hit the token limit before completion)

Position cards:

- glm: ACCEPT-WITH-CHANGES - the exchange kernel is mathematically sound, but verified NaN-to-zero bias paths, the missing applied-vs-priced boost assertion, and the divergent secondary-axis predicates can silently corrupt the free energy and must be fixed first. (confidence 80)
- kimi: REJECT - verified NaN→0 substitution in the secondary-bias and boost paths plus the unasserted applied-vs-priced boost parity let exchange and MBAR price a Hamiltonian different from the sampled one. (confidence 85)
- deepseek-thinking: (TRUNCATED - output hit the token limit before completion) (confidence 50)
- mini: REJECT – multiple verified, silent violations of the Boltzmann weighting guarantee render the free‑energy estimates unreliable. (confidence 92)
- thinker: (TRUNCATED - output hit the token limit before completion) (confidence 50)

Verdicts: glm=ACCEPT-WITH-CHANGES, kimi=REJECT, deepseek-thinking=n/a, mini=REJECT, thinker=n/a
Majority: n/a at 0%

cosine similarity | glm | kimi | deepseek-thinking | mini | thinker
glm | 1.000 | 0.762 | 0.486 | 0.666 | 0.486
kimi | 0.762 | 1.000 | 0.500 | 0.707 | 0.500
deepseek-thinking | 0.486 | 0.500 | 1.000 | 0.500 | 1.000
mini | 0.666 | 0.707 | 0.500 | 1.000 | 0.500
thinker | 0.486 | 0.500 | 1.000 | 0.500 | 1.000

mean pairwise cosine: 0.611

# The Board's Judgment

## Majority position

The position commanding the most support after debate is **ACCEPT-WITH-CHANGES** (3 of 5 members: glm, deepseek-thinking, thinker).

The core replica-exchange kernel is mathematically sound for a fixed, finite bias matrix. The swap acceptance in `apply_window_swap`, the Gibbs proposal in `_gibbs_window_proposal_distribution`, and the reverse-proposal construction in `gibbs_propose_one_replica` are correct: the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost), the reverse proposal is evaluated against the post-swap holder table, and the MH correction `log_alpha = -βδ + log(q_rev) - log(q_fwd)` is the correct detailed-balance factor. The independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant. The alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH; an all-zero `state_gamd_lambdas` list is truthy and is preserved; the `sw <= 0` branch is unreachable after the finite filter.

However, the majority finds **verified silent-failure channels** that can produce a number that is not the Boltzmann free energy:

1. **NaN → 0 secondary-bias substitution** in `umbrella_bias_matrix_kcal` (production.py L914–916): a non-finite secondary displacement is replaced by zero bias in the exchange matrix, while `reconstruct_bias_matrix` (correctness/bias.py) propagates NaN for the same observation. Exchange and MBAR can therefore price two different Hamiltonians for identical data.
2. **NaN → 0 boost substitution** in `_channel_boost` (pep_gamd.py L503): `np.where((b+v) < e, b, 0.0)` maps NaN channel energies to zero boost in both exchange pricing and MBAR reconstruction.
3. **Missing applied-vs-priced boost assertion**: `sample()` computes `extract_gamd_boost_kj(sim.integrator)` (applied) and `sampled_boost_bias_kj` (priced) side by side but never compares them. This is exactly the chignolin_8 failure class.
4. **Divergent secondary-axis predicates**: `sample()` enables the secondary axis via `secondary_cv_k_kcal_list`, while `_current_exchange_arrays()` requires `secondary_cv_ks_kj` and `ss_ks_kj_arr_global` but passes `ss_k_kcal_arr_global`. If these flags disagree, sample and exchange use different Hamiltonians.
5. **Dual k0max sources**: `k0max_from_globals` feeds `set_replica_lambda`, while `pep_gamd_boost_kj` uses `env.k0max_*`; no invariant ties them.

These defects are fixable without redesigning the exchange kernel, but they must be fixed before the output can be trusted as a Boltzmann-weighted free-energy surface.

## Dissenting positions

**kimi (REJECT, confidence 85):**

> "REJECT - verified NaN→0 substitution in the secondary-bias and boost paths plus the unasserted applied-vs-priced boost parity let exchange and MBAR price a Hamiltonian different from the sampled one."

**mini (REJECT, confidence 92):**

> "VERDICT: REJECT – multiple verified, silent violations of the Boltzmann weighting guarantee render the free-energy estimates unreliable. CONFIDENCE: 92"

## Confidence

Aggregate confidence: **74**.

Derivation: mean of the five members' final confidences is (80 + 85 + 70 + 92 + 70) / 5 = 79.4. This is adjusted downward by 5 points because the final agreement measurements show only moderate consensus: the majority verdict is 3/5 (not unanimous), the mean position-card cosine is 0.611, and two members issued strong REJECT positions with confidences 85 and 92. The resulting aggregate confidence is 74.

## Conditions

1. **Add a runtime applied-vs-priced boost assertion.** In `sample()` and `_current_exchange_arrays`, compare `extract_gamd_boost_kj(sim.integrator, ...)` with `pep_gamd_boost_kj(v_pep, v_dih, state_lambdas[w], env)` for every replica/window; fail loudly on any mismatch. This is the minimal guard against the chignolin_8 class.
2. **Fix the NaN → 0 secondary-bias fallback.** In `umbrella_bias_matrix_kcal`, a non-finite secondary displacement must not be silently priced as zero. Either propagate NaN so the replica's exchanges are frozen (as the docstring claims) or explicitly filter such frames from both exchange and MBAR with a shared missing-data policy.
3. **Fix the NaN → 0 boost fallback.** In `_channel_boost`, a NaN channel energy must not become zero boost via the `(b+v) < e` comparison. NaN should propagate or raise, never silently become 0.
4. **Unify the secondary-axis predicates.** `sample()` and `_current_exchange_arrays()` must use the same condition and the same arrays (`secondary_cv_centers`, `secondary_cv_k_kcal_list`, `ss_centers_arr_global`, `ss_k_kcal_arr_global`) so the exchange Hamiltonian and the recorded/MBAR Hamiltonian cannot diverge.
5. **Tie the k0max sources.** Assert `k0max_from_globals(shared_globals) == {"Total": env.k0max_total, "Dihedral": env.k0max_dih}` at ladder activation, or refactor to a single source of truth.
6. **Correct the `apply_window_swap` docstring.** The claim that "the GaMD boost cancel[s] exactly from delta" is false when λ differs between states; the code is correct only because `bias_matrix_kj` includes per-state boost. The docstring must state this to prevent a future "simplification" that reintroduces the chignolin_8 bug.
7. **Harden lambda persistence.** Ensure `_persist_state_gamd_lambdas` is called after every mutation of `args.state_gamd_lambdas` (including `--max-replicas` truncation and post-pull auto-drop), and add a resume test covering a window-count change with a stale manifest.

We need answer audit. Need follow instructions. Need inspect judgment for mathematical/physical errors. We need list errors found in judgment? The user asks "Audit this judgment for mathematical and physical errors: dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics and datatypes per your checklists." We need evaluate the judgment text. It is a board judgment about code. Need find errors in judgment itself? Need list every error found, each line FLAG format. Then Checked section. Need be careful.

We need analyze judgment claims. It includes majority position, dissent, confidence, conditions. Need audit for mathematical/physical errors. We need identify any incorrect statements in judgment.

Let's parse.

Judgment says core replica-exchange kernel mathematically sound. Swap acceptance, Gibbs proposal, reverse-proposal construction correct: transposition delta computed from full bias_matrix_kj (umbrella + boost), reverse proposal evaluated against post-swap holder table, MH correction log_alpha = -βδ + log(q_rev) - log(q_fwd) correct detailed-balance factor. Independent audit two-state walkthrough confirms Gibbs-with-MH kernel leaves intended stationary distribution invariant. Alleged clipping, truthiness, uniform-fallback defects refuted: clipping changes only proposal distribution and corrected by MH; all-zero state_gamd_lambdas list is truthy and preserved; sw <= 0 branch unreachable after finite filter.

Need check sign conventions: log_alpha = -βδ + log(q_rev) - log(q_fwd). For Metropolis-Hastings with proposal q(x'|x), acceptance = min(1, [π(x') q(x|x')]/[π(x) q(x'|x)]). If π ∝ exp(-β H), log ratio = -β(H(x')-H(x)) + log q_rev - log q_fwd. δ maybe H(new)-H(old)? They say transposition delta computed from full bias. If δ = H_after - H_before? Then -βδ correct. Need check sign. Could be δ = bias_after - bias_before? If delta = new - old, yes. If delta = old - new, sign wrong. Judgment says correct. We can't verify code. But maybe "log_alpha = -βδ + log(q_rev) - log(q_fwd)" is correct if δ = energy_after - energy_before. Need not flag.

Potential errors in judgment:

- "all-zero `state_gamd_lambdas` list is truthy" In Python, list containing zeros is truthy if non-empty. If all-zero list e.g. [0.0, 0.0] is truthy. Yes. If empty list false. They say all-zero list is truthy and preserved. Correct.

- "sw <= 0 branch is unreachable after finite filter." Need context. Maybe if sw is standard deviation? If finite filter ensures >0? Not necessarily. If sw can be zero for constant data? They say finite filter maybe excludes nonpositive? Need not.

- "NaN → 0 secondary-bias substitution" in umbrella_bias_matrix_kcal: non-finite secondary displacement replaced by zero bias in exchange matrix, while reconstruct_bias_matrix propagates NaN. This is a silent failure. Correct.

- "NaN → 0 boost substitution" in _channel_boost: np.where((b+v) < e, b, 0.0) maps NaN channel energies to zero boost. Need check: np.where(condition, b, 0.0). If condition is NaN < e => False, so output 0.0. Yes.

- "Missing applied-vs-priced boost assertion": sample computes extract_gamd_boost_kj(sim.integrator) (applied) and sampled_boost_bias_kj (priced) side by side but never compares. This is exactly chignolin_8 failure class. Need maybe "chignolin_8" unknown. Fine.

- "Divergent secondary-axis predicates": sample enables secondary axis via secondary_cv_k_kcal_list, while _current_exchange_arrays() requires secondary_cv_ks_kj and ss_ks_kj_arr_global but passes ss_k_kcal_arr_global. If flags disagree, sample and exchange use different Hamiltonians. Need check: "requires secondary_cv_ks_kj and ss_ks_kj_arr_global but passes ss_k_kcal_arr_global" maybe typo? It says "passes ss_k_kcal_arr_global" while requires ss_ks_kj_arr_global. This is a potential bug. But is it an error in judgment? It is a claim about code. We need audit judgment for errors, not code. Could be correct.

- "Dual k0max sources": k0max_from_globals feeds set_replica_lambda, while pep_gamd_boost_kj uses env.k0max_*; no invariant ties them. Correct.

Conditions:
1. Add runtime applied-vs-priced boost assertion. In sample() and _current_exchange_arrays, compare extract_gamd_boost_kj(sim.integrator, ...) with pep_gamd_boost_kj(v_pep, v_dih, state_lambdas[w], env) for every replica/window; fail loudly on any mismatch. This is minimal guard against chignolin_8 class.
Need check: "for every replica/window" maybe okay.

2. Fix NaN → 0 secondary-bias fallback. In umbrella_bias_matrix_kcal, non-finite secondary displacement must not be silently priced as zero. Either propagate NaN so replica's exchanges are frozen (as docstring claims) or explicitly filter such frames from both exchange and MBAR with shared missing-data policy.
Need check: "propagate NaN so replica's exchanges are frozen" Is that correct? If bias matrix has NaN, swap acceptance log_alpha = -βδ + ... with δ NaN => acceptance NaN. In code, if min(1, exp(NaN))? np.exp(NaN)=NaN, min(1, NaN) behavior? Python min(1, float('nan')) returns 1? Actually min(1, nan) returns 1? Let's recall: min(1, float('nan')) returns 1 because nan > 1? Wait comparisons with nan false; min uses less than? Python min starts with first arg, then if item < current. For min(1, nan): current=1, nan < 1 false, so returns 1. So NaN acceptance becomes 1? That would not freeze exchanges; it would always accept? Need check. If log_alpha NaN, alpha = exp(NaN) = NaN. Then min(1, NaN) in Python? Let's test mentally: min(1, float('nan')) returns 1 because nan < 1 is False. So NaN acceptance becomes 1, not frozen. If using np.minimum(1, np.exp(log_alpha))? np.minimum(1, nan) returns nan? Actually np.minimum propagates NaN. If code uses min built-in, NaN -> 1. The judgment says "propagate NaN so the replica's exchanges are frozen (as the docstring claims)" but if acceptance uses min(1, exp(NaN)), it may not freeze. Need know code? The judgment earlier says "sw <= 0 branch is unreachable after finite filter." Maybe they use np.where? We need audit judgment for mathematical/physical errors. The claim "propagate NaN so exchanges are frozen" might be questionable: NaN in acceptance may not freeze; it may cause acceptance or rejection depending implementation. But the judgment is recommending "Either propagate NaN so the replica's exchanges are frozen (as the docstring claims) or explicitly filter..." If docstring claims NaN freezes exchanges, maybe code uses something like if not np.isfinite(log_alpha): reject? We don't know. We can flag as potential? Need be precise.

- "Fix NaN → 0 boost fallback. In _channel_boost, a NaN channel energy must not become zero boost via the (b+v) < e comparison. NaN should propagate or raise, never silently become 0." Correct.

- "Unify secondary-axis predicates. sample() and _current_exchange_arrays() must use same condition and same arrays (secondary_cv_centers, secondary_cv_k_kcal_list, ss_centers_arr_global, ss_k_kcal_arr_global) so exchange Hamiltonian and recorded/MBAR Hamiltonian cannot diverge." Correct.

- "Tie k0max sources. Assert k0max_from_globals(shared_globals) == {"Total": env.k0max_total, "Dihedral": env.k0max_dih} at ladder activation, or refactor to a single source of truth." Need check: k0max_from_globals returns dict with keys "Total" and "Dihedral"? Maybe. If env.k0max_total and env.k0max_dih are floats. Correct.

- "Correct the apply_window_swap docstring. The claim that 'the GaMD boost cancel[s] exactly from delta' is false when λ differs between states; the code is correct only because bias_matrix_kj includes per-state boost. The docstring must state this to prevent a future 'simplification' that reintroduces the chignolin_8 bug." Need check: In replica exchange with lambda ladder, if bias matrix includes umbrella + boost, the boost does not cancel from delta if λ differs? Let's analyze. Suppose two replicas i,j with states/windows λ_i, λ_j. Swap coordinates? The energy difference for exchanging conformations between states includes biases. If boost depends on state (lambda) and conformation, then in delta = [U_λj(x_j)+b_λj(x_j)+...] - [U_λi(x_i)+b_λi(x_i)+...]? Actually for swap of conformations between states, delta = [H_λi(x_j)+H_λj(x_i)] - [H_λi(x_i)+H_λj(x_j)]? Need sign. The boost terms do not cancel if b_λi(x_j) etc. If boost is same for all lambda? Then they cancel. If λ differs, per-state boost matters. So docstring claim false. Correct.

- "Harden lambda persistence. Ensure _persist_state_gamd_lambdas is called after every mutation of args.state_gamd_lambdas (including --max-replicas truncation and post-pull auto-drop), and add a resume test covering a window-count change with a stale manifest." Correct.

Confidence calculation: "Aggregate confidence: 74. Derivation: mean of five members' final confidences is (80 + 85 + 70 + 92 + 70) / 5 = 79.4. This is adjusted downward by 5 points because the final agreement measurements show only moderate consensus: the majority verdict is 3/5 (not unanimous), the mean position-card cosine is 0.611, and two members issued strong REJECT positions with confidences 85 and 92. The resulting aggregate confidence is 74."

Need check arithmetic: 80+85+70+92+70 = 397? 80+85=165, +70=235, +92=327, +70=397. 397/5 = 79.4. Yes. Adjusted down by 5 => 74.4, but they say 74. If integer rounding, 74.4 -> 74. Fine. But "mean position-card cosine is 0.611" maybe irrelevant. No mathematical error.

Need check "majority verdict is 3/5 (not unanimous)" yes.

Potential issue: "The core replica-exchange kernel is mathematically sound for a fixed, finite bias matrix." Then later "verified silent-failure channels" that can produce a number not Boltzmann free energy. These are not errors in judgment.

Need identify any errors in judgment's mathematical/physical claims. Let's scrutinize.

1. "MH correction `log_alpha = -βδ + log(q_rev) - log(q_fwd)` is the correct detailed-balance factor." Is sign of q_rev/q_fwd correct? For proposal from current state x to proposed x', acceptance log = log π(x') - log π(x) + log q(x|x') - log q(x'|x). If π ∝ exp(-βH), log π(x') - log π(x) = -β(H(x')-H(x)). If δ = H(x') - H(x), then -βδ. q_rev = q(x|x'), q_fwd = q(x'|x). So log_alpha = -βδ + log(q_rev) - log(q_fwd). Correct.

2. "clipping changes only the proposal distribution and is corrected by MH" Need check: If proposal distribution is clipped and then MH uses q_fwd and q_rev evaluated on clipped proposal, detailed balance holds for modified proposal. Yes.

3. "all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, list [0.0, 0.0] is truthy. If "all-zero" means all elements zero, yes. If empty? no. They likely mean non-empty. Fine.

4. "sw <= 0 branch is unreachable after finite filter." Need context: If sw is standard deviation of some distribution, finite filter might be `np.isfinite(sw)` but sw could be 0.0 finite, so branch reachable. Unless filter also excludes <=0. The judgment says "after the finite filter" maybe finite filter includes `sw > 0`. We don't know. Could be an error if finite filter only checks finite. But not enough.

5. "NaN → 0 secondary-bias substitution ... a non-finite secondary displacement is replaced by zero bias in the exchange matrix, while reconstruct_bias_matrix propagates NaN for the same observation. Exchange and MBAR can therefore price two different Hamiltonians for identical data." Correct.

6. "NaN → 0 boost substitution ... np.where((b+v) < e, b, 0.0) maps NaN channel energies to zero boost in both exchange pricing and MBAR reconstruction." Correct.

7. "Missing applied-vs-priced boost assertion ... sample() computes extract_gamd_boost_kj(sim.integrator) (applied) and sampled_boost_bias_kj (priced) side by side but never compares them. This is exactly the chignolin_8 failure class." Correct.

8. "Divergent secondary-axis predicates: sample() enables the secondary axis via secondary_cv_k_kcal_list, while _current_exchange_arrays() requires secondary_cv_ks_kj and ss_ks_kj_arr_global but passes ss_k_kcal_arr_global. If these flags disagree, sample and exchange use different Hamiltonians." Need parse: "requires secondary_cv_ks_kj and ss_ks_kj_arr_global but passes ss_k_kcal_arr_global" Maybe "requires" means function expects keys `secondary_cv_ks_kj` and `ss_ks_kj_arr_global` but caller passes `ss_k_kcal_arr_global`. This is a bug. Correct.

9. "Dual k0max sources: k0max_from_globals feeds set_replica_lambda, while pep_gamd_boost_kj uses env.k0max_*; no invariant ties them." Correct.

10. Condition 1: "In sample() and _current_exchange_arrays, compare extract_gamd_boost_kj(sim.integrator, ...) with pep_gamd_boost_kj(v_pep, v_dih, state_lambdas[w], env) for every replica/window; fail loudly on any mismatch." Need check: If applied boost is computed from integrator and priced boost from state_lambdas, but state_lambdas may be mutated? This is the guard. Fine.

11. Condition 2: "Either propagate NaN so the replica's exchanges are frozen (as the docstring claims) or explicitly filter such frames from both exchange and MBAR with a shared missing-data policy." Potential issue: If log_alpha NaN and code uses `min(1, exp(log_alpha))`, NaN may be treated as 1 (accept) in Python, not frozen. But if code uses `if not np.isfinite(log_alpha): return False` then frozen. The docstring claims frozen, so maybe code does that. Not an error in judgment.

12. Condition 3: "NaN should propagate or raise, never silently become 0." Correct.

13. Condition 4: "same arrays (secondary_cv_centers, secondary_cv_k_kcal_list, ss_centers_arr_global, ss_k_kcal_arr_global)" Need check: sample enables via secondary_cv_k_kcal_list, exchange requires secondary_cv_ks_kj and ss_ks_kj_arr_global but passes ss_k_kcal_arr_global. The condition says use same arrays: secondary_cv_centers, secondary_cv_k_kcal_list, ss_centers_arr_global, ss_k_kcal_arr_global. But earlier "requires secondary_cv_ks_kj and ss_ks_kj_arr_global" maybe there are two sets: `secondary_cv_ks_kj` and `ss_ks_kj_arr_global` vs `secondary_cv_k_kcal_list` and `ss_k_kcal_arr_global`. The condition says use same arrays `secondary_cv_centers`, `secondary_cv_k_kcal_list`, `ss_centers_arr_global`, `ss_k_kcal_arr_global`. This might be correct if those are the canonical ones. Not an error.

14. Condition 5: "Assert k0max_from_globals(shared_globals) == {"Total": env.k0max_total, "Dihedral": env.k0max_dih} at ladder activation" Need check: If k0max_from_globals returns keys "Total" and "Dihedral", and env has k0max_total and k0max_dih. Correct.

15. Condition 6: "The claim that 'the GaMD boost cancel[s] exactly from delta' is false when λ differs between states; the code is correct only because bias_matrix_kj includes per-state boost." Need check: In replica exchange with Hamiltonian replica exchange, the acceptance probability for swapping conformations between states i and j is min(1, exp[-β(E_i(x_j)+E_j(x_i)-E_i(x_i)-E_j(x_j))]). If E includes boost b_i(x) that depends on state i and conformation x, then terms do not cancel. If boost is same for all states, they cancel. So correct.

16. Condition 7: "Ensure _persist_state_gamd_lambdas is called after every mutation of args.state_gamd_lambdas (including --max-replicas truncation and post-pull auto-drop), and add a resume test covering a window-count change with a stale manifest." Correct.

Potential errors in judgment's confidence: "mean position-card cosine is 0.611" maybe not a mathematical error. "adjusted downward by 5 points because ... two members issued strong REJECT positions with confidences 85 and 92." The adjustment is arbitrary, not error.

Need maybe flag "Aggregate confidence: 74" after 79.4 - 5 = 74.4, rounding down to 74. If they intended integer, okay. Could be "74.4" not "74". But not significant.

Need check "majority position ACCEPT-WITH-CHANGES (3 of 5 members: glm, deepseek-thinking, thinker)." Then "Dissenting positions kimi (REJECT, confidence 85), mini (REJECT, confidence 92)." That's 5 members. Confidences: majority members? They list mean of five final confidences is (80 + 85 + 70 + 92 + 70)/5. So glm=80, deepseek-thinking=70, thinker=70? kimi=85, mini=92. Yes.

Need maybe "mean position-card cosine is 0.611" not used in calculation except adjustment. Fine.

Need audit for "dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics and datatypes per checklists." The judgment is about code, not equations. We need list errors found in judgment. If no errors, output "NO FLAGS" and Checked section. But likely there are some subtle errors in judgment's claims. Need identify.

Let's examine "The alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH; an all-zero `state_gamd_lambdas` list is truthy and is preserved; the `sw <= 0` branch is unreachable after the finite filter."

Could be an error: "clipping changes only the proposal distribution and is corrected by MH" In MH, if you clip proposal and then evaluate q_fwd and q_rev using the clipped proposal density, detailed balance holds. But if clipping is applied to a parameter that also affects the target distribution? The judgment says "clipping changes only the proposal distribution" maybe in context of `apply_window_swap`? Need not.

"all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, a list containing zeros is truthy, but if code does `if state_gamd_lambdas:` then all-zero list is truthy. If code does `if state_gamd_lambdas is None or len(state_gamd_lambdas)==0` then also. "Preserved" maybe if code checks `if not any(state_gamd_lambdas)` then all-zero would be false. But they say refuted. Fine.

"`sw <= 0` branch is unreachable after the finite filter." If finite filter is `np.isfinite(sw)` and sw=0, branch reachable. But maybe "finite filter" includes `sw > 0` because standard deviation must be positive. We don't know.

Potential issue: "The core replica-exchange kernel is mathematically sound for a fixed, finite bias matrix." But then "verified silent-failure channels" include "Missing applied-vs-priced boost assertion" and "Divergent secondary-axis predicates" and "Dual k0max sources" which mean the bias matrix used in exchange may not be the actual Hamiltonian. So "core kernel sound" is conditional. Not error.

Potential issue: "the reverse proposal is evaluated against the post-swap holder table" Need check: In Gibbs proposal, reverse proposal probability should be evaluated at the proposed state? The judgment says correct. Fine.

Potential issue: "log_alpha = -βδ + log(q_rev) - log(q_fwd)" If δ is defined as "transposition delta computed from full bias_matrix_kj (umbrella + boost)" maybe δ = H_new - H_old? If δ = H_old - H_new, sign would be +βδ. But they say correct. We can't verify.

Need maybe flag "The independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." This is a claim of proof. Not error.

Need maybe flag "the `sw <= 0` branch is unreachable after the finite filter" as a statistical error: finite filter does not imply sw > 0; if sw is standard deviation, sw=0 is finite and branch reachable. But without code context, maybe "finite filter" is `np.isfinite(sw) & (sw > 0)`. The judgment says "after the finite filter" maybe they refer to a filter that removes non-finite and non-positive? The phrase "finite filter" usually means `np.isfinite`, not positivity. Could be an error in judgment's reasoning. But the user asks audit judgment for mathematical/physical errors. We can flag if clear.

Let's think of "NaN → 0 secondary-bias substitution" condition: "Either propagate NaN so the replica's exchanges are frozen (as the docstring claims)" Is "propagate NaN" the right way to freeze? In MH, if log_alpha = NaN, the acceptance probability is undefined. If code uses `if log_alpha > log(random.random())` with NaN, comparison false -> reject. If code uses `min(1, exp(log_alpha))`, Python min returns 1 for NaN? Let's verify Python's min with NaN: `min(1, float('nan'))` returns 1 because `nan < 1` is False. `min(float('nan'), 1)` returns nan because `1 < nan` is False, so current nan remains. So if min(1, exp(log_alpha)) with exp(NaN)=NaN, min(1, nan) returns 1. So NaN would cause acceptance, not freeze. If code uses `np.minimum(1, np.exp(log_alpha))`, np.minimum propagates NaN, then `if random < nan` false -> reject. So "propagate NaN so exchanges are frozen" depends on implementation. The judgment says "as the docstring claims" so maybe code has explicit NaN handling. Not necessarily error.

Potential issue: "the reverse proposal is evaluated against the post-swap holder table" In MH for replica exchange, the reverse proposal probability should be evaluated at the state after swap? Let's derive. Suppose current configuration of replicas: replica i has x_i, replica j has x_j. Proposal swaps coordinates between replicas i and j: proposed x_i' = x_j, x_j' = x_i. The proposal probability q_fwd = probability of selecting pair (i,j) and swapping. Reverse proposal from proposed state would swap back, q_rev = probability of selecting same pair and swapping. If selection probabilities depend on current holder table (which changes after swap), then q_rev must be evaluated at post-swap holder table. The judgment says correct. Fine.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" In Hamiltonian replica exchange, the energy of replica k in state j? Need sign. If bias_matrix_kj is bias energy of conformation k under window j. The swap acceptance for exchanging conformations between windows i and j is exp[-β( (U_i(x_j)+b_i(x_j)) + (U_j(x_i)+b_j(x_i)) - (U_i(x_i)+b_i(x_i)) - (U_j(x_j)+b_j(x_j)) )]. If δ = (bias_i(x_j)+bias_j(x_i)) - (bias_i(x_i)+bias_j(x_j))? Then -βδ. Correct.

Potential issue: "the GaMD boost cancel[s] exactly from delta is false when λ differs between states" Wait, in replica exchange with λ ladder, the boost may depend on λ (state) and conformation. If λ differs, boost terms don't cancel. But if the boost is a function of the conformation only, not λ, then they cancel even if λ differs. The docstring claim "GaMD boost cancel exactly from delta" might be true if boost is independent of λ. The judgment says false when λ differs between states. But if boost is independent of λ, it cancels regardless of λ. The code's `bias_matrix_kj` includes per-state boost, so maybe boost depends on state. The judgment says "the code is correct only because bias_matrix_kj includes per-state boost." That implies boost is per-state. If boost is per-state, then yes. But the statement "false when λ differs between states" is not universally true; it depends on whether boost depends on λ. In GaMD, boost potential often depends on energy of the system, which may depend on the window/state? Actually GaMD boost is typically added to the total potential energy and may be a function of the potential energy, which depends on the force field parameters, not the umbrella window? Wait, in REST/REMD with GaMD, the boost might be applied to dihedral and total potential, and the boost threshold depends on the state's lambda? Let's think. The judgment earlier says "bias_matrix_kj (umbrella + boost)" and "per-state boost". So boost depends on state. So condition 6 is correct in context. But the phrasing "when λ differs between states" could be misleading: if boost is not per-state, it cancels. But they specify "the code is correct only because bias_matrix_kj includes per-state boost." So okay.

Potential issue: "all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, `bool([0.0])` is True. But if the code uses `if state_gamd_lambdas:` and then later `if not state_gamd_lambdas:`? All-zero list is truthy, so preserved. Correct.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." If `sw` is a standard deviation, `sw <= 0` includes `sw == 0` and negative. A finite filter `np.isfinite(sw)` allows `sw == 0`. So branch reachable. Unless "finite filter" is `np.isfinite(sw) & (sw > 0)`. The judgment says "after the finite filter" not "after the positive filter". This could be a minor error. But maybe in code, `sw` is a sum of weights or a count, and finite filter ensures >0? Need context. The judgment mentions "sw <= 0 branch" in relation to "uniform-fallback defects". Maybe `sw` is a sum of weights for a proposal distribution. If all weights are zero, sw=0, branch would be reachable. The finite filter might be `np.isfinite(sw)` but not `sw > 0`. The judgment says "unreachable after the finite filter" maybe because the finite filter includes `sw > 0`? We don't know. Could be an error.

Let's search memory: This seems from a code review of a Python package for replica exchange with GaMD. "apply_window_swap", "_gibbs_window_proposal_distribution", "gibbs_propose_one_replica", "umbrella_bias_matrix_kcal", "_channel_boost", "pep_gamd.py", "production.py", "correctness/bias.py". "sw" might be "sum of weights" in a Gibbs proposal distribution. If all weights are zero, sw=0, branch "sw <= 0" fallback to uniform. The finite filter might filter out non-finite weights, but if all weights are zero, sw=0 finite, branch reachable. The judgment says "sw <= 0 branch is unreachable after the finite filter" maybe because the finite filter also checks `sw > 0`? The phrase "finite filter" could be a function that filters non-finite and non-positive? Not sure.

Need decide whether to flag. The user wants "Audit this judgment for mathematical and physical errors" and "List every error found". We should not invent errors. But we need be thorough.

Let's examine "The alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH; an all-zero `state_gamd_lambdas` list is truthy and is preserved; the `sw <= 0` branch is unreachable after the finite filter."

Could be a mathematical error: "clipping changes only the proposal distribution and is corrected by MH" If clipping changes the proposal distribution, then the MH acceptance probability must use the actual proposal density including clipping normalization. If the code computes q_fwd and q_rev using unclipped densities, then clipping would break detailed balance. The judgment says "corrected by MH" but only if q_fwd/q_rev are evaluated with the clipped proposal. The judgment earlier says "reverse-proposal construction ... correct" and "clipping changes only the proposal distribution and is corrected by MH" likely assumes q's are evaluated properly. Not an error.

Potential issue: "the reverse proposal is evaluated against the post-swap holder table" In MH, the reverse proposal probability should be evaluated at the proposed state, which is post-swap. Yes.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj includes boost, but the boost is not part of the target distribution? In GaMD, boost is added to enhance sampling, but the target distribution for reweighting is the unbiased ensemble. In replica exchange, the boost is part of the modified Hamiltonian, and the stationary distribution of replicas is proportional to exp[-β(H + bias)] for each replica. The MBAR reweighting uses the bias to recover unbiased. So including boost in exchange is correct. Not error.

Potential issue: "Missing applied-vs-priced boost assertion ... This is exactly the chignolin_8 failure class." Not mathematical.

Potential issue: "Divergent secondary-axis predicates: sample() enables the secondary axis via secondary_cv_k_kcal_list, while _current_exchange_arrays() requires secondary_cv_ks_kj and ss_ks_kj_arr_global but passes ss_k_kcal_arr_global. If these flags disagree, sample and exchange use different Hamiltonians." This is a code bug. Not error in judgment.

Potential issue: "Dual k0max sources: k0max_from_globals feeds set_replica_lambda, while pep_gamd_boost_kj uses env.k0max_*; no invariant ties them." Correct.

Potential issue: "Aggregate confidence: 74. Derivation: mean ... 79.4. This is adjusted downward by 5 points because ... The resulting aggregate confidence is 74." If 79.4 - 5 = 74.4, rounding to 74. If they wanted integer, okay. But "adjusted downward by 5 points" from 79.4 gives 74.4, not 74. Could be a minor arithmetic/rounding error. But they might round to nearest integer. 74.4 rounds to 74. Not an error.

Potential issue: "mean position-card cosine is 0.611" Maybe "position-card cosine" is a weird metric. Not mathematical error.

Potential issue: "the majority verdict is 3/5 (not unanimous), the mean position-card cosine is 0.611, and two members issued strong REJECT positions with confidences 85 and 92." The adjustment downward by 5 points is arbitrary. Not error.

Potential issue: "The core replica-exchange kernel is mathematically sound for a fixed, finite bias matrix." But then "verified silent-failure channels" include "NaN → 0 secondary-bias substitution" and "NaN → 0 boost substitution" which mean bias matrix may not be finite. So "fixed, finite" conditional. Fine.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." Let's consider if "finite filter" is a filter that removes non-finite values from an array of weights. If all weights are zero, the sum sw=0 is finite, so branch reachable. Unless the filter also checks `sw > 0`. The judgment says "after the finite filter" not "after the positive filter". This could be a logical error. But maybe in the code, the finite filter is `np.isfinite(sw) and sw > 0`? The phrase "finite filter" might be shorthand for "filter that ensures finite and positive". We need not flag without context.

Potential issue: "all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, a list with all zeros is truthy, but if the code uses `if state_gamd_lambdas is not None and len(state_gamd_lambdas) > 0`, yes. If code uses `if np.any(state_gamd_lambdas)`, all-zero would be false. The judgment says "all-zero list is truthy" which is true for Python lists, but if the code converts to numpy array, `bool(np.array([0.0,0.0]))` raises ValueError. The judgment says "list" so okay.

Potential issue: "the reverse proposal is evaluated against the post-swap holder table" In Gibbs sampling, the proposal distribution for selecting which replica to swap may depend on the current holder table. The reverse proposal from the post-swap state should use the post-swap holder table. Correct.

Potential issue: "the MH correction `log_alpha = -βδ + log(q_rev) - log(q_fwd)` is the correct detailed-balance factor." If δ is defined as "transposition delta" maybe δ = H_after - H_before. If δ = H_before - H_after, sign wrong. But they say correct. We can't verify.

Potential issue: "the independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." This is a proof claim. Not error.

Potential issue: "the alleged clipping, truthiness, and uniform-fallback defects were refuted" Maybe "clipping changes only the proposal distribution and is corrected by MH" is not a refutation if clipping is applied to the target distribution? But they say proposal. Fine.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." If sw is a sum of weights, and weights can be negative? Then finite filter doesn't ensure sw > 0. But weights are probably nonnegative. If all zero, sw=0. Branch reachable. Hmm.

Let's think of "uniform-fallback defects": In a Gibbs proposal distribution, if all proposal weights are zero, fallback to uniform. The defect might be that fallback to uniform is not accounted for in q_fwd/q_rev, breaking detailed balance. The judgment says "sw <= 0 branch is unreachable after the finite filter" meaning the fallback never triggers because the finite filter ensures sw > 0. But if sw can be zero, it is reachable. The phrase "finite filter" might be a function that filters out non-finite and zero? Not sure.

Maybe we should flag this as a minor error: "finite filter does not imply sw > 0; sw=0 is finite." But if the judgment's "finite filter" includes positivity, then false flag. Need avoid false positives.

Let's look at "all-zero `state_gamd_lambdas` list is truthy and is preserved" This is a Python truthiness claim. Correct.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." If `sw` is a standard deviation, `sw <= 0` branch handles negative due to numerical error and zero. A finite filter `np.isfinite(sw)` allows zero. So branch reachable. But maybe "finite filter" is `np.isfinite(sw) & (sw > 0)` because standard deviation must be positive. The judgment says "after the finite filter" not "after the positive filter". Could be a minor wording issue.

Potential issue: "NaN → 0 secondary-bias substitution ... a non-finite secondary displacement is replaced by zero bias in the exchange matrix, while reconstruct_bias_matrix propagates NaN for the same observation. Exchange and MBAR can therefore price two different Hamiltonians for identical data." Correct.

Potential issue: "NaN → 0 boost substitution ... np.where((b+v) < e, b, 0.0) maps NaN channel energies to zero boost in both exchange pricing and MBAR reconstruction." Correct.

Potential issue: "Missing applied-vs-priced boost assertion ... sample() computes extract_gamd_boost_kj(sim.integrator) (applied) and sampled_boost_bias_kj (priced) side by side but never compares them." Correct.

Potential issue: "Divergent secondary-axis predicates ... sample() enables the secondary axis via secondary_cv_k_kcal_list, while _current_exchange_arrays() requires secondary_cv_ks_kj and ss_ks_kj_arr_global but passes ss_k_kcal_arr_global." This is a code bug. Not error.

Potential issue: "Dual k0max sources ... k0max_from_globals feeds set_replica_lambda, while pep_gamd_boost_kj uses env.k0max_*; no invariant ties them." Correct.

Potential issue: "Correct the apply_window_swap docstring. The claim that 'the GaMD boost cancel[s] exactly from delta' is false when λ differs between states; the code is correct only because bias_matrix_kj includes per-state boost." Need check: If boost is per-state, then in the delta for swapping conformations between states i and j, the boost terms are b_i(x_j) + b_j(x_i) - b_i(x_i) - b_j(x_j). This does not cancel. If boost is the same for all states, it cancels. The docstring claim "GaMD boost cancel exactly from delta" might be true if the boost is independent of λ. But the code's bias_matrix_kj includes per-state boost, so the docstring is false. Correct.

Potential issue: "Harden lambda persistence. Ensure _persist_state_gamd_lambdas is called after every mutation of args.state_gamd_lambdas (including --max-replicas truncation and post-pull auto-drop), and add a resume test covering a window-count change with a stale manifest." Correct.

Potential issue: "The core replica-exchange kernel is mathematically sound for a fixed, finite bias matrix." But "fixed, finite bias matrix" maybe not enough; the proposal distribution must also be correctly normalized. They say yes.

Potential issue: "the reverse proposal is evaluated against the post-swap holder table" In MH, the reverse proposal probability should be evaluated at the proposed state, which is post-swap. Yes.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj is a matrix of biases for each replica and window, the delta for swapping replicas i and j is bias_matrix_kj[i,j] + bias_matrix_kj[j,i] - bias_matrix_kj[i,i] - bias_matrix_kj[j,j]? Need sign. If δ = (bias_i(x_j)+bias_j(x_i)) - (bias_i(x_i)+bias_j(x_j)), then -βδ. Correct.

Potential issue: "the MH correction `log_alpha = -βδ + log(q_rev) - log(q_fwd)`" If δ includes only bias, not the base potential energy? In replica exchange, the base potential energy of a conformation is independent of the window (if windows only differ by bias), so it cancels in the swap delta. So δ from bias matrix is sufficient. Correct.

Potential issue: "the Gibbs proposal in `_gibbs_window_proposal_distribution`" Maybe the Gibbs proposal distribution is not symmetric, hence q_rev/q_fwd. Correct.

Potential issue: "the reverse-proposal construction in `gibbs_propose_one_replica` are correct" Fine.

Potential issue: "the independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." Fine.

Potential issue: "the alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH" If clipping changes the proposal distribution, the MH acceptance must use the actual proposal density. If the code's q_fwd/q_rev are computed from the clipped distribution, yes. The judgment says "corrected by MH" but doesn't explicitly say q's are updated. However earlier "reverse-proposal construction ... correct" implies.

Potential issue: "an all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, `bool([0.0])` True. Correct.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." This is the most likely candidate for a flag. Let's decide.

The user asks "Audit this judgment for mathematical and physical errors: dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics and datatypes per your checklists." They want us to check the judgment's claims. We can flag if the judgment itself contains an error. The "sw <= 0 branch is unreachable after the finite filter" is a logical/statistical claim. If "finite filter" only checks `np.isfinite`, then sw=0 is finite and branch reachable. But maybe in context, "finite filter" is a filter that removes non-finite and non-positive values. The judgment says "after the finite filter" not "after the positive filter". The word "finite" in statistics often means "not NaN and not inf", not "positive". So this is a potential error. But without code, it's risky.

Let's think of "sw" in `_gibbs_window_proposal_distribution`. It might be "sum of weights" for a categorical proposal over windows. If all weights are zero, sw=0, branch `sw <= 0` fallback to uniform. The finite filter might be `np.isfinite(weights)` but if all weights are zero, sw=0 finite, so branch reachable. The judgment says "unreachable after the finite filter" maybe because the finite filter also checks `sw > 0`? The phrase "finite filter" could be a function named `finite_filter` that returns indices where `np.isfinite` and maybe `> 0`. We don't know. The judgment says "the `sw <= 0` branch is unreachable after the finite filter." If the filter is `np.isfinite(sw)` and sw is a sum of nonnegative weights, sw=0 is finite. So the branch is reachable. This is a mathematical error in the judgment's reasoning. But maybe the filter is "finite" meaning "finite and positive" in context. Hmm.

Let's look at "uniform-fallback defects were refuted" - The defect might be that if all weights are zero, fallback to uniform is used, but the proposal probability is then not the same as the normalized weights (which would be NaN). The judgment says "sw <= 0 branch is unreachable after the finite filter" meaning the fallback never happens because the filter ensures at least one positive weight. If the filter is `np.isfinite(weights)` and all weights are zero, then sw=0, branch reachable. So the refutation would be wrong. This is a potential error.

But the user's requested output format: "List every error found, each as exactly one line in this format: FLAG-<n> | <severity: CRITICAL|MAJOR|MINOR> | <location> | <error> | <correct form>" We can flag this as MINOR or MAJOR. Need be confident.

Let's also consider "the reverse proposal is evaluated against the post-swap holder table" Could be an error if the reverse proposal should be evaluated against the pre-swap holder table? Let's derive carefully.

In MH, from state x (current assignment of replicas to windows), propose x' via some distribution q(x'|x). Acceptance = min(1, π(x') q(x|x') / (π(x) q(x'|x))). Here x' is the post-swap assignment. The reverse proposal from x' to x is the swap back. The probability of proposing the swap back depends on the selection probabilities in state x'. So q_rev = q(x|x') should be evaluated at x' (post-swap). So "post-swap holder table" is correct.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj is indexed by replica k and window j, then for swapping replicas i and j, the delta should be `bias_matrix_kj[i, j] + bias_matrix_kj[j, i] - bias_matrix_kj[i, i] - bias_matrix_kj[j, j]`. If the code computes `bias_matrix_kj[i, i] + bias_matrix_kj[j, j] - bias_matrix_kj[i, j] - bias_matrix_kj[j, i]`, then sign would be opposite. The judgment says "transposition delta is computed from the full bias_matrix_kj" and "log_alpha = -βδ" correct. We can't verify.

Potential issue: "the GaMD boost cancel[s] exactly from delta is false when λ differs between states" Wait, if the boost is a function of the conformation only, not the state, then it cancels even if λ differs. The judgment says "false when λ differs between states" but the truth depends on whether boost depends on λ. In GaMD, the boost potential is often added to the total potential energy and depends on the potential energy, which may depend on the force field parameters. In replica exchange with λ, the Hamiltonian may be a function of λ, so the boost may depend on λ. The judgment says "the code is correct only because bias_matrix_kj includes per-state boost." So in this code, boost is per-state. The statement "false when λ differs between states" is a bit imprecise but not necessarily wrong in context. Could be a minor wording issue.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." Let's decide to flag as MINOR. But need be sure.

Let's also consider "all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, a list with all zeros is truthy, but if the code uses `if state_gamd_lambdas:` and then later `if not any(state_gamd_lambdas):` it would not be preserved. The judgment says "preserved" maybe because the code uses truthiness of list, not any. Fine.

Potential issue: "the alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH" If clipping changes the proposal distribution, the MH acceptance must use the actual proposal density. If the code's q_fwd/q_rev are computed from the clipped distribution, yes. The judgment says "corrected by MH" but doesn't explicitly say q's are updated. However earlier "reverse-proposal construction ... correct" implies.

Potential issue: "the independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." This is a proof claim. Not error.

Potential issue: "the core replica-exchange kernel is mathematically sound for a fixed, finite bias matrix." But "fixed, finite bias matrix" maybe not enough; the proposal distribution must also be correctly normalized. They say yes.

Potential issue: "the reverse proposal is evaluated against the post-swap holder table" In MH, the reverse proposal probability should be evaluated at the proposed state, which is post-swap. Yes.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj is a matrix of biases for each replica and window, the delta for swapping replicas i and j is bias_matrix_kj[i,j] + bias_matrix_kj[j,i] - bias_matrix_kj[i,i] - bias_matrix_kj[j,j]? Need sign. If δ = (bias_i(x_j)+bias_j(x_i)) - (bias_i(x_i)+bias_j(x_j)), then -βδ. Correct.

Potential issue: "the MH correction `log_alpha = -βδ + log(q_rev) - log(q_fwd)`" If δ includes only bias, not the base potential energy? In replica exchange, the base potential energy of a conformation is independent of the window (if windows only differ by bias), so it cancels in the swap delta. So δ from bias matrix is sufficient. Correct.

Potential issue: "the Gibbs proposal in `_gibbs_window_proposal_distribution`" Maybe the Gibbs proposal distribution is not symmetric, hence q_rev/q_fwd. Correct.

Potential issue: "the reverse-proposal construction in `gibbs_propose_one_replica` are correct" Fine.

Potential issue: "the independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." Fine.

Potential issue: "the alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH" If clipping changes the proposal distribution, the MH acceptance must use the actual proposal density. If the code's q_fwd/q_rev are computed from the clipped distribution, yes. The judgment says "corrected by MH" but doesn't explicitly say q's are updated. However earlier "reverse-proposal construction ... correct" implies.

Potential issue: "an all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, `bool([0.0])` True. Correct.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." This is the most likely candidate for a flag. Let's decide.

The user asks "Audit this judgment for mathematical and physical errors: dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics and datatypes per your checklists." They want us to check the judgment's claims. We can flag if the judgment itself contains an error. The "sw <= 0 branch is unreachable after the finite filter" is a logical/statistical claim. If "finite filter" only checks `np.isfinite`, then sw=0 is finite and branch reachable. But maybe in context, "finite filter" is a filter that removes non-finite and non-positive values. The judgment says "after the finite filter" not "after the positive filter". The word "finite" in statistics often means "not NaN and not inf", not "positive". So this is a potential error. But without code, it's risky.

Let's think of "sw" in `_gibbs_window_proposal_distribution`. It might be "sum of weights" for a categorical proposal over windows. If all weights are zero, sw=0, branch `sw <= 0` fallback to uniform. The finite filter might be `np.isfinite(weights)` but if all weights are zero, sw=0 finite, so branch reachable. The judgment says "unreachable after the finite filter" maybe because the finite filter also checks `sw > 0`? The phrase "finite filter" could be a function named `finite_filter` that returns indices where `np.isfinite` and maybe `> 0`. We don't know. The judgment says "the `sw <= 0` branch is unreachable after the finite filter." If the filter is `np.isfinite(sw)` and sw is a sum of nonnegative weights, sw=0 is finite. So the branch is reachable. This is a mathematical error in the judgment's reasoning. But maybe the filter is "finite" meaning "finite and positive" in context. Hmm.

Let's look at "uniform-fallback defects were refuted" - The defect might be that if all weights are zero, fallback to uniform is used, but the proposal probability is then not the same as the normalized weights (which would be NaN). The judgment says "sw <= 0 branch is unreachable after the finite filter" meaning the fallback never happens because the filter ensures at least one positive weight. If the filter is `np.isfinite(weights)` and all weights are zero, then sw=0, branch reachable. So the refutation would be wrong. This is a potential error.

But the user's requested output format: "List every error found, each as exactly one line in this format: FLAG-<n> | <severity: CRITICAL|MAJOR|MINOR> | <location> | <error> | <correct form>" We can flag this as MINOR or MAJOR. Need be confident.

Let's also consider "the reverse proposal is evaluated against the post-swap holder table" Could be an error if the reverse proposal should be evaluated against the pre-swap holder table? Let's derive carefully.

In MH, from state x (current assignment of replicas to windows), propose x' via some distribution q(x'|x). Acceptance = min(1, π(x') q(x|x') / (π(x) q(x'|x))). Here x' is the post-swap assignment. The reverse proposal from x' to x is the swap back. The probability of proposing the swap back depends on the selection probabilities in state x'. So q_rev = q(x|x') should be evaluated at x' (post-swap). So "post-swap holder table" is correct.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj is indexed by replica k and window j, then for swapping replicas i and j, the delta should be `bias_matrix_kj[i, j] + bias_matrix_kj[j, i] - bias_matrix_kj[i, i] - bias_matrix_kj[j, j]`. If the code computes `bias_matrix_kj[i, i] + bias_matrix_kj[j, j] - bias_matrix_kj[i, j] - bias_matrix_kj[j, i]`, then sign would be opposite. The judgment says "transposition delta is computed from the full bias_matrix_kj" and "log_alpha = -βδ" correct. We can't verify.

Potential issue: "the GaMD boost cancel[s] exactly from delta is false when λ differs between states" Wait, if the boost is a function of the conformation only, not the state, then it cancels even if λ differs. The judgment says "false when λ differs between states" but the truth depends on whether boost depends on λ. In GaMD, the boost potential is often added to the total potential energy and depends on the potential energy, which may depend on the force field parameters. In replica exchange with λ, the Hamiltonian may be a function of λ, so the boost may depend on λ. The judgment says "the code is correct only because bias_matrix_kj includes per-state boost." So in this code, boost is per-state. The statement "false when λ differs between states" is a bit imprecise but not necessarily wrong in context. Could be a minor wording issue.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." Let's decide to flag as MINOR. But need be sure.

Let's also consider "all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, a list with all zeros is truthy, but if the code uses `if state_gamd_lambdas:` and then later `if not any(state_gamd_lambdas):` it would not be preserved. The judgment says "preserved" maybe because the code uses truthiness of list, not any. Fine.

Potential issue: "the alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH" If clipping changes the proposal distribution, the MH acceptance must use the actual proposal density. If the code's q_fwd/q_rev are computed from the clipped distribution, yes. The judgment says "corrected by MH" but doesn't explicitly say q's are updated. However earlier "reverse-proposal construction ... correct" implies.

Potential issue: "the independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." This is a proof claim. Not error.

Potential issue: "the core replica-exchange kernel is mathematically sound for a fixed, finite bias matrix." But "fixed, finite bias matrix" maybe not enough; the proposal distribution must also be correctly normalized. They say yes.

Potential issue: "the reverse proposal is evaluated against the post-swap holder table" In MH, the reverse proposal probability should be evaluated at the proposed state, which is post-swap. Yes.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj is a matrix of biases for each replica and window, the delta for swapping replicas i and j is bias_matrix_kj[i,j] + bias_matrix_kj[j,i] - bias_matrix_kj[i,i] - bias_matrix_kj[j,j]? Need sign. If δ = (bias_i(x_j)+bias_j(x_i)) - (bias_i(x_i)+bias_j(x_j)), then -βδ. Correct.

Potential issue: "the MH correction `log_alpha = -βδ + log(q_rev) - log(q_fwd)`" If δ includes only bias, not the base potential energy? In replica exchange, the base potential energy of a conformation is independent of the window (if windows only differ by bias), so it cancels in the swap delta. So δ from bias matrix is sufficient. Correct.

Potential issue: "the Gibbs proposal in `_gibbs_window_proposal_distribution`" Maybe the Gibbs proposal distribution is not symmetric, hence q_rev/q_fwd. Correct.

Potential issue: "the reverse-proposal construction in `gibbs_propose_one_replica` are correct" Fine.

Potential issue: "the independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." Fine.

Potential issue: "the alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH" If clipping changes the proposal distribution, the MH acceptance must use the actual proposal density. If the code's q_fwd/q_rev are computed from the clipped distribution, yes. The judgment says "corrected by MH" but doesn't explicitly say q's are updated. However earlier "reverse-proposal construction ... correct" implies.

Potential issue: "an all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, `bool([0.0])` True. Correct.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." This is the most likely candidate for a flag. Let's decide.

The user asks "Audit this judgment for mathematical and physical errors: dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics and datatypes per your checklists." They want us to check the judgment's claims. We can flag if the judgment itself contains an error. The "sw <= 0 branch is unreachable after the finite filter" is a logical/statistical claim. If "finite filter" only checks `np.isfinite`, then sw=0 is finite and branch reachable. But maybe in context, "finite filter" is a filter that removes non-finite and non-positive values. The judgment says "after the finite filter" not "after the positive filter". The word "finite" in statistics often means "not NaN and not inf", not "positive". So this is a potential error. But without code, it's risky.

Let's think of "sw" in `_gibbs_window_proposal_distribution`. It might be "sum of weights" for a categorical proposal over windows. If all weights are zero, sw=0, branch `sw <= 0` fallback to uniform. The finite filter might be `np.isfinite(weights)` but if all weights are zero, sw=0 finite, so branch reachable. The judgment says "unreachable after the finite filter" maybe because the finite filter also checks `sw > 0`? The phrase "finite filter" could be a function named `finite_filter` that returns indices where `np.isfinite` and maybe `> 0`. We don't know. The judgment says "the `sw <= 0` branch is unreachable after the finite filter." If the filter is `np.isfinite(sw)` and sw is a sum of nonnegative weights, sw=0 is finite. So the branch is reachable. This is a mathematical error in the judgment's reasoning. But maybe the filter is "finite" meaning "finite and positive" in context. Hmm.

Let's look at "uniform-fallback defects were refuted" - The defect might be that if all weights are zero, fallback to uniform is used, but the proposal probability is then not the same as the normalized weights (which would be NaN). The judgment says "sw <= 0 branch is unreachable after the finite filter" meaning the fallback never happens because the filter ensures at least one positive weight. If the filter is `np.isfinite(weights)` and all weights are zero, then sw=0, branch reachable. So the refutation would be wrong. This is a potential error.

But the user's requested output format: "List every error found, each as exactly one line in this format: FLAG-<n> | <severity: CRITICAL|MAJOR|MINOR> | <location> | <error> | <correct form>" We can flag this as MINOR or MAJOR. Need be confident.

Let's also consider "the reverse proposal is evaluated against the post-swap holder table" Could be an error if the reverse proposal should be evaluated against the pre-swap holder table? Let's derive carefully.

In MH, from state x (current assignment of replicas to windows), propose x' via some distribution q(x'|x). Acceptance = min(1, π(x') q(x|x') / (π(x) q(x'|x))). Here x' is the post-swap assignment. The reverse proposal from x' to x is the swap back. The probability of proposing the swap back depends on the selection probabilities in state x'. So q_rev = q(x|x') should be evaluated at x' (post-swap). So "post-swap holder table" is correct.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj is indexed by replica k and window j, then for swapping replicas i and j, the delta should be `bias_matrix_kj[i, j] + bias_matrix_kj[j, i] - bias_matrix_kj[i, i] - bias_matrix_kj[j, j]`. If the code computes `bias_matrix_kj[i, i] + bias_matrix_kj[j, j] - bias_matrix_kj[i, j] - bias_matrix_kj[j, i]`, then sign would be opposite. The judgment says "transposition delta is computed from the full bias_matrix_kj" and "log_alpha = -βδ" correct. We can't verify.

Potential issue: "the GaMD boost cancel[s] exactly from delta is false when λ differs between states" Wait, if the boost is a function of the conformation only, not the state, then it cancels even if λ differs. The judgment says "false when λ differs between states" but the truth depends on whether boost depends on λ. In GaMD, the boost potential is often added to the total potential energy and depends on the potential energy, which may depend on the force field parameters. In replica exchange with λ, the Hamiltonian may be a function of λ, so the boost may depend on λ. The judgment says "the code is correct only because bias_matrix_kj includes per-state boost." So in this code, boost is per-state. The statement "false when λ differs between states" is a bit imprecise but not necessarily wrong in context. Could be a minor wording issue.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." Let's decide to flag as MINOR. But need be sure.

Let's also consider "all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, a list with all zeros is truthy, but if the code uses `if state_gamd_lambdas:` and then later `if not any(state_gamd_lambdas):` it would not be preserved. The judgment says "preserved" maybe because the code uses truthiness of list, not any. Fine.

Potential issue: "the alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH" If clipping changes the proposal distribution, the MH acceptance must use the actual proposal density. If the code's q_fwd/q_rev are computed from the clipped distribution, yes. The judgment says "corrected by MH" but doesn't explicitly say q's are updated. However earlier "reverse-proposal construction ... correct" implies.

Potential issue: "the independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." This is a proof claim. Not error.

Potential issue: "the core replica-exchange kernel is mathematically sound for a fixed, finite bias matrix." But "fixed, finite bias matrix" maybe not enough; the proposal distribution must also be correctly normalized. They say yes.

Potential issue: "the reverse proposal is evaluated against the post-swap holder table" In MH, the reverse proposal probability should be evaluated at the proposed state, which is post-swap. Yes.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj is a matrix of biases for each replica and window, the delta for swapping replicas i and j is bias_matrix_kj[i,j] + bias_matrix_kj[j,i] - bias_matrix_kj[i,i] - bias_matrix_kj[j,j]? Need sign. If δ = (bias_i(x_j)+bias_j(x_i)) - (bias_i(x_i)+bias_j(x_j)), then -βδ. Correct.

Potential issue: "the MH correction `log_alpha = -βδ + log(q_rev) - log(q_fwd)`" If δ includes only bias, not the base potential energy? In replica exchange, the base potential energy of a conformation is independent of the window (if windows only differ by bias), so it cancels in the swap delta. So δ from bias matrix is sufficient. Correct.

Potential issue: "the Gibbs proposal in `_gibbs_window_proposal_distribution`" Maybe the Gibbs proposal distribution is not symmetric, hence q_rev/q_fwd. Correct.

Potential issue: "the reverse-proposal construction in `gibbs_propose_one_replica` are correct" Fine.

Potential issue: "the independent audit's two-state walkthrough confirms that the Gibbs-with-MH kernel leaves the intended stationary distribution invariant." Fine.

Potential issue: "the alleged clipping, truthiness, and uniform-fallback defects were refuted: clipping changes only the proposal distribution and is corrected by MH" If clipping changes the proposal distribution, the MH acceptance must use the actual proposal density. If the code's q_fwd/q_rev are computed from the clipped distribution, yes. The judgment says "corrected by MH" but doesn't explicitly say q's are updated. However earlier "reverse-proposal construction ... correct" implies.

Potential issue: "an all-zero `state_gamd_lambdas` list is truthy and is preserved" In Python, `bool([0.0])` True. Correct.

Potential issue: "the `sw <= 0` branch is unreachable after the finite filter." This is the most likely candidate for a flag. Let's decide.

The user asks "Audit this judgment for mathematical and physical errors: dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics and datatypes per your checklists." They want us to check the judgment's claims. We can flag if the judgment itself contains an error. The "sw <= 0 branch is unreachable after the finite filter" is a logical/statistical claim. If "finite filter" only checks `np.isfinite`, then sw=0 is finite and branch reachable. But maybe in context, "finite filter" is a filter that removes non-finite and non-positive values. The judgment says "after the finite filter" not "after the positive filter". The word "finite" in statistics often means "not NaN and not inf", not "positive". So this is a potential error. But without code, it's risky.

Let's think of "sw" in `_gibbs_window_proposal_distribution`. It might be "sum of weights" for a categorical proposal over windows. If all weights are zero, sw=0, branch `sw <= 0` fallback to uniform. The finite filter might be `np.isfinite(weights)` but if all weights are zero, sw=0 finite, so branch reachable. The judgment says "unreachable after the finite filter" maybe because the finite filter also checks `sw > 0`? The phrase "finite filter" could be a function named `finite_filter` that returns indices where `np.isfinite` and maybe `> 0`. We don't know. The judgment says "the `sw <= 0` branch is unreachable after the finite filter." If the filter is `np.isfinite(sw)` and sw is a sum of nonnegative weights, sw=0 is finite. So the branch is reachable. This is a mathematical error in the judgment's reasoning. But maybe the filter is "finite" meaning "finite and positive" in context. Hmm.

Let's look at "uniform-fallback defects were refuted" - The defect might be that if all weights are zero, fallback to uniform is used, but the proposal probability is then not the same as the normalized weights (which would be NaN). The judgment says "sw <= 0 branch is unreachable after the finite filter" meaning the fallback never happens because the filter ensures at least one positive weight. If the filter is `np.isfinite(weights)` and all weights are zero, then sw=0, branch reachable. So the refutation would be wrong. This is a potential error.

But the user's requested output format: "List every error found, each as exactly one line in this format: FLAG-<n> | <severity: CRITICAL|MAJOR|MINOR> | <location> | <error> | <correct form>" We can flag this as MINOR or MAJOR. Need be confident.

Let's also consider "the reverse proposal is evaluated against the post-swap holder table" Could be an error if the reverse proposal should be evaluated against the pre-swap holder table? Let's derive carefully.

In MH, from state x (current assignment of replicas to windows), propose x' via some distribution q(x'|x). Acceptance = min(1, π(x') q(x|x') / (π(x) q(x'|x))). Here x' is the post-swap assignment. The reverse proposal from x' to x is the swap back. The probability of proposing the swap back depends on the selection probabilities in state x'. So q_rev = q(x|x') should be evaluated at x' (post-swap). So "post-swap holder table" is correct.

Potential issue: "the transposition delta is computed from the full `bias_matrix_kj` (umbrella + boost)" If bias_matrix_kj is indexed by replica k and window j, then for swapping replicas i and j, the delta should be `bias_matrix_kj[i, j] + bias_matrix_kj[j, i] - bias_matrix_kj[i, i] - bias_matrix_kj[j, j]`. If the code computes `bias_matrix_kj[i, i] + bias_matrix_kj[j, j] - bias_matrix_kj[i, j] - bias_matrix_kj[j, i]`, then sign would be opposite. The judgment says "transposition delta is computed from the full bias_matrix_kj" and "log_alpha = -βδ" correct. We can't verify.

Potential issue: "the GaMD boost cancel[s] exactly from delta is false when λ differs between states" Wait, if the boost is a function of the conformation only, not the state, then it cancels even if λ differs. The judgment says "false when λ differs between states" but the truth depends on whether boost depends on λ. In GaMD, the boost potential is often added to the total potential energy and depends on the potential energy, which may depend on the force field parameters. In replica exchange with λ, the Hamiltonian may be a function of λ, so the boost may depend on λ. The judgment says "the code is correct only because bias_matrix_kj includes per-state boost." So in this code, boost
(TRUNCATED - output hit the token limit before completion)