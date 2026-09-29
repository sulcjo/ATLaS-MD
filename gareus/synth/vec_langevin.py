"""Vectorised overdamped Langevin for many independent chains (2D or 3D surfaces).

Same dynamics as :func:`gareus.synth.sampler_langevin.sample_window_langevin`
(Langevin proposals on F + U, finite-difference grad F, analytic grad U), but every chain advances in one numpy step, so a whole window layout
(or a swarm) runs at the cost of one chain. Chains are independent: chain ``i``
carries its own umbrella (``c1[i], k1[i], c2[i], k2[i]``; ``k = 0`` leaves that
axis free). Only cv1 and cv2 are ever restrained; a 3D surface's cv3 is free.

By default each Langevin step is Metropolis-adjusted (MALA): the chain keeps the
exact biased Boltzmann distribution on the box, so an "analytic truth" computed by
quadrature is the right reference (plain Euler-Maruyama inflates a restrained
variance by ~1/(1 - D k dt / 2): 11 % at D k dt = 0.2). ``metropolis=False`` gives
the reflecting Euler-Maruyama scheme of the scalar sampler.

Units are the harness's: F in kBT, k in kBT per CV^2, beta = 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

FD_STEP = 1.0e-4


@dataclass
class ChainRun:
    """Samples ``(n_chains, n_samples, dim)`` and the last position ``(n_chains, dim)``."""

    samples: np.ndarray
    final: np.ndarray
    steps_per_sample: int
    dt: float
    acceptance: float = float("nan")


def _energy_fn(surface) -> Callable[..., np.ndarray]:
    return surface.energy


def _bounds(surface) -> list:
    if hasattr(surface, "bounds"):
        return [tuple(b) for b in surface.bounds]
    return [tuple(surface.cv1_bounds), tuple(surface.cv2_bounds)]


def _reflect(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.where(x < lo, 2.0 * lo - x, x)
    x = np.where(x > hi, 2.0 * hi - x, x)
    return np.clip(x, lo, hi)


def run_chains(surface, x0: np.ndarray, n_samples: int, *, c1=None, k1=None, c2=None, k2=None,
               D: Sequence[float] = (1.0, 0.05, 0.05), dt: float = 2.0e-3, steps_per_sample: int = 20,
               burn_in: int = 0, rng: Optional[np.random.Generator] = None,
               metropolis: bool = True) -> ChainRun:
    """Propagate ``len(x0)`` chains from ``x0`` and record every ``steps_per_sample`` steps.

    ``D`` gives the diffusion constant per coordinate (its first ``dim`` entries are used);
    ``dt`` is a scalar or one value per chain.
    ``burn_in`` steps are run and discarded first.
    """
    rng = rng if rng is not None else np.random.default_rng()
    energy = _energy_fn(surface)
    bounds = _bounds(surface)
    x = np.array(x0, dtype=np.float64, copy=True)
    n, dim = x.shape
    if dim != len(bounds):
        raise ValueError(f"x0 has {dim} coordinates, surface has {len(bounds)}")
    D = np.asarray(list(D)[:dim], dtype=np.float64)
    if D.size != dim:
        raise ValueError("D needs one entry per coordinate")
    zeros = np.zeros(n)
    c1 = zeros if c1 is None else np.broadcast_to(np.asarray(c1, float), (n,))
    k1 = zeros if k1 is None else np.broadcast_to(np.asarray(k1, float), (n,))
    c2 = zeros if c2 is None else np.broadcast_to(np.asarray(c2, float), (n,))
    k2 = zeros if k2 is None else np.broadcast_to(np.asarray(k2, float), (n,))
    dt_arr = np.broadcast_to(np.asarray(dt, dtype=np.float64), (n,))
    Ddt = D[None, :] * dt_arr[:, None]          # per chain, per coordinate
    noise = np.sqrt(2.0 * Ddt)
    for d in range(dim):
        x[:, d] = _reflect(x[:, d], *bounds[d])

    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])

    def total_energy(x):
        cols = [x[:, d] for d in range(dim)]
        e = energy(*cols) + 0.5 * k1 * (x[:, 0] - c1) ** 2 + 0.5 * k2 * (x[:, 1] - c2) ** 2
        inside = np.all((x >= lo) & (x <= hi), axis=1)
        return np.where(inside, e, np.inf)

    def gradient(x):
        cols = [x[:, d] for d in range(dim)]
        grad = np.empty_like(x)
        for d in range(dim):
            up = list(cols); dn = list(cols)
            up[d] = cols[d] + FD_STEP
            dn[d] = cols[d] - FD_STEP
            grad[:, d] = (energy(*up) - energy(*dn)) / (2.0 * FD_STEP)
        grad[:, 0] += k1 * (x[:, 0] - c1)
        grad[:, 1] += k2 * (x[:, 1] - c2)
        return grad

    state = {"e": total_energy(x), "g": gradient(x), "acc": 0, "tries": 0}

    def step(x):
        g = state["g"]
        xn = x - Ddt * g + noise * rng.standard_normal(x.shape)
        if not metropolis:
            gn = gradient(np.column_stack([_reflect(xn[:, d], *bounds[d]) for d in range(dim)]))
            xn = np.column_stack([_reflect(xn[:, d], *bounds[d]) for d in range(dim)])
            state["g"] = gn
            return xn
        # MALA: Metropolis-adjusted Langevin keeps exp(-(F + U)) on the box exact
        # (no Euler-discretisation bias); a proposal outside the box is rejected.
        en = total_energy(xn)
        ok = np.isfinite(en)
        # grad at the clipped proposal (a rejected out-of-box row's value is never used)
        gn = gradient(np.clip(xn, lo, hi))
        fwd = -np.sum((xn - x + Ddt * g) ** 2 / (4.0 * Ddt), axis=1)
        bwd = -np.sum((x - xn + Ddt * gn) ** 2 / (4.0 * Ddt), axis=1)
        with np.errstate(invalid="ignore", over="ignore"):
            log_a = -(en - state["e"]) + bwd - fwd
        accept = ok & (np.log(rng.uniform(size=n)) < log_a)
        state["acc"] += int(accept.sum()); state["tries"] += n
        x = np.where(accept[:, None], xn, x)
        state["e"] = np.where(accept, en, state["e"])
        state["g"] = np.where(accept[:, None], gn, g)
        return x

    for _ in range(int(burn_in)):
        x = step(x)
    out = np.empty((n, int(n_samples), dim), dtype=np.float64)
    for i in range(int(n_samples)):
        for _ in range(int(steps_per_sample)):
            x = step(x)
        out[:, i, :] = x
    acc = state["acc"] / state["tries"] if state["tries"] else float("nan")
    return ChainRun(samples=out, final=x.copy(), steps_per_sample=int(steps_per_sample), dt=float(np.max(dt_arr)),
                    acceptance=float(acc))


def stable_dt(k_max: float, D_max: float, *, dt_max: float = 2.0e-3, courant: float = 0.25) -> float:
    """Largest dt <= dt_max with D k dt <= courant (Euler accuracy on the stiffest restraint)."""
    if not (k_max > 0.0 and D_max > 0.0):
        return float(dt_max)
    return float(min(dt_max, courant / (D_max * k_max)))


def swarm_design_measure(surface, n_members: int, n_samples: int, *, rng: np.random.Generator,
                         D: Sequence[float] = (1.0, 0.05, 0.05), dt: float = 2.0e-3,
                         steps_per_sample: int = 20, discard: int = 0,
                         seed_axes: Sequence[int] = (0, 1), fixed: Optional[dict] = None) -> ChainRun:
    """Unbiased short correlated runs from stratified seeds (a swarm design measure).

    Seeds are a Latin-hypercube stratification of the ``seed_axes`` coordinates over the
    surface bounds; any other coordinate starts at ``fixed[d]`` (default its lower-bound
    quarter point), which is how a swarm whose seed generator does not see a hidden mode
    starts it on one side. The first ``discard`` samples of every member are dropped.
    """
    bounds = _bounds(surface)
    dim = len(bounds)
    x0 = np.empty((int(n_members), dim))
    for d in range(dim):
        lo, hi = bounds[d]
        if d in seed_axes:
            strata = (rng.permutation(int(n_members)) + rng.uniform(0, 1, int(n_members))) / int(n_members)
            x0[:, d] = lo + (hi - lo) * strata
        else:
            x0[:, d] = (fixed or {}).get(d, lo + 0.25 * (hi - lo))
    run = run_chains(surface, x0, int(n_samples) + int(discard), D=D, dt=dt,
                     steps_per_sample=steps_per_sample, rng=rng)
    run.samples = run.samples[:, int(discard):, :]
    return run
