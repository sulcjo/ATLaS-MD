"""Old-vs-new volume move on a production-sized Context, at 1 and 64 threads.

Answers the two questions the unit tests cannot: what fraction of barostat wall
time is the molecule loop (``p``), and whether the single-threaded 182x transform
speedup survives 64 workers contending for the GIL.

Two independent estimates of ``p`` currently disagree by more than a factor of
two -- py-spy puts it near 1.0 (biased upward, because a GIL-blocked thread is
sampled where it last held the lock), cycle arithmetic puts it near 0.42. The
implied end-to-end speedup is anywhere from ~1.5x to ~4.3x. This harness is how
that gets settled without disturbing a running campaign.

Run on a GPU node, NOT the one the campaign occupies:

    python tools/npt_ab_harness.py --threads 1 8 32 64 --gpus 4
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit

import gareus.npt as npt_mod
from gareus.npt import BiasedMCBarostatController, EnergyBreakdown


class RealEnergyAdapter:
    """Forces a REAL potential-energy evaluation, as production's U* does.

    An analytic volume-only adapter would make ``evaluate`` free, and
    ``evaluate`` is one of the phases competing with the loop for the barostat's
    wall time. Measuring p with a free ``evaluate`` biases p upward and could
    make the change look better than it is.

    This does not reproduce Pep-GaMD's boost channel -- that needs the full
    integrator -- so the measured p remains a mild OVER-estimate. Stated rather
    than hidden: true production p is at or below what this reports.
    """

    adapter_id = "ab-harness-real-energy"

    def snapshot(self, context, integrator):
        return {}

    def evaluate(self, context, snapshot):
        st = context.getState(getEnergy=True)
        phys = float(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        return EnergyBreakdown(phys, 0.0, 0.0, 0.0, phys)


def loop_reference_for(molecules):
    """The replaced implementation, iterating the ORIGINAL molecule lists.

    ``np.flatnonzero(mol_ids == m)`` would return ascending atom indices, which
    is not necessarily the order the original loop used; for an unsorted
    molecule that is a different summation order, and the reference would then
    be a third implementation rather than the code being replaced.
    """
    def impl(positions, mol_ids, mol_sizes, scale_minus_one, large=()):
        new_positions = positions.copy()
        for mol in molecules:
            center = positions[mol].mean(axis=0)
            new_positions[mol] = positions[mol] + scale_minus_one * center
        return new_positions
    return impl


def build_context(device_index, platform_name="CUDA"):
    """~19k atoms of water: the production system's size and molecule count.

    ``device_index`` pins the Context to one GPU so the harness reproduces
    production's topology -- 4 GPUs x 16 replicas -- rather than piling 64
    contexts onto one card, which would contend for memory bandwidth in a way
    production does not.
    """
    ff = app.ForceField("amber14/tip3pfb.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(ff, boxSize=openmm.Vec3(6.0, 6.0, 6.0) * unit.nanometer)
    system = ff.createSystem(
        modeller.topology, nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds, rigidWater=True)
    integ = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    props = ({"Precision": "mixed", "DeviceIndex": str(device_index)}
             if platform_name == "CUDA" else {})
    ctx = openmm.Context(system, integ, platform, props)
    ctx.setPositions(modeller.positions)
    return ctx, system


def time_moves(ctrl, n):
    step = ctrl.next_due_step
    for _ in range(n):
        ctrl.attempt_due(step)
        step = ctrl.next_due_step


def phase_breakdown(ctrl):
    t = ctrl.state_dict().get("timings") or {}
    total = sum(v for k, v in t.items() if k.endswith("_s"))
    if total <= 0:
        return None, {}
    shares = {k: v / total for k, v in t.items() if k.endswith("_s")}
    return total, shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--moves", type=int, default=25)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--gpus", type=int, default=4,
                    help="spread contexts over this many GPUs, as production does")
    args = ap.parse_args()

    print(f"building reference context on {args.platform} ...", flush=True)
    ctx0, system = build_context(0, args.platform)
    molecules = [[int(i) for i in mol] for mol in ctx0.getMolecules()]
    print(f"system: {system.getNumParticles()} atoms, {len(molecules)} molecules, "
          f"{args.gpus} GPU(s)\n", flush=True)

    vectorized_impl = npt_mod._scale_about_molecule_centroids
    results = {}

    for n_threads in args.threads:
        contexts = [ctx0] + [build_context(i % max(1, args.gpus), args.platform)[0]
                             for i in range(1, n_threads)]
        for label, impl in (("loop", loop_reference_for(molecules)),
                            ("vectorized", vectorized_impl)):
            npt_mod._scale_about_molecule_centroids = impl
            try:
                ctrls = [
                    BiasedMCBarostatController.initialize(
                        c, RealEnergyAdapter(), pressure_bar=1.0,
                        temperature_k=300.0, frequency_steps=1,
                        volume_step_fraction=0.01, seed=1234 + i)
                    for i, c in enumerate(contexts)
                ]
                threads = [threading.Thread(target=time_moves, args=(c, args.moves))
                           for c in ctrls]
                t0 = time.perf_counter()
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall = time.perf_counter() - t0
                per_move_ms = wall / args.moves * 1000.0
                results[(n_threads, label)] = per_move_ms
                print(f"  threads={n_threads:3d} {label:11s} wall {wall:7.3f}s  "
                      f"{per_move_ms:8.3f} ms per move per replica-set", flush=True)
                if label == "vectorized":
                    total, shares = phase_breakdown(ctrls[0])
                    if shares:
                        print("      phases: " + "  ".join(
                            f"{k}={100 * v:5.1f}%" for k, v in sorted(shares.items())),
                            flush=True)
            finally:
                npt_mod._scale_about_molecule_centroids = vectorized_impl
        print(flush=True)

    print("=" * 64)
    for n_threads in args.threads:
        loop = results.get((n_threads, "loop"))
        vec = results.get((n_threads, "vectorized"))
        if not loop or not vec:
            continue
        ratio = loop / vec
        # Amdahl at the interval-200 baseline: barostat is 77% of production
        # wall, and this ratio tells us how much of the barostat the loop was.
        p = max(0.0, min(1.0, (loop - vec) / loop))
        end_to_end = 1.0 / (0.23 + 0.77 * ((1 - p) + p / max(ratio, 1e-9)))
        print(f"threads={n_threads:3d}  transform ratio {ratio:6.1f}x  "
              f"implied p {p:4.2f}  implied end-to-end {end_to_end:5.2f}x")
    print("=" * 64)
    print("The threads=64 row is the one that predicts campaign speedup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
