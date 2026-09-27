"""P3 (bounded active replicas per GPU) x P4 (MPS active-thread %) benchmark prototype.

All N replicas stay resident (one Context each, round-robin over TEST_DEVICES), exactly as in
production. What changes is admission: per GPU, at most A replica-owner operations advance at once.
Every replica still completes the same CHUNK steps per round before the round ends (the round is
the stand-in for production's common exchange/report boundary), so aggregate ns/day is directly
comparable across A. A=all reproduces today's "submit everything, join everything" loop.

System: the real production force layout -- gareus.production.add_umbrella_cv_forces (shared
contact layout when the residual CV2 carries CV1's contact sum), Pep-GaMD partition, the real
PepGaMDLowerDualIntegrator seeded into stage 5 from the frozen swarm envelope (boost live).

One process builds the contexts once, then sweeps ADMIT (e.g. "all,32,16,8") REPS times,
interleaved, SECONDS each after one warm-up round per arm. The MPS thread percentage is a
process-level setting, so the launcher runs one process per percentage.

Usage: python p3_admission_bench.py N
Env:   ADMIT="all,32,16,8" REPS=2 SECONDS=60 CHUNK=250 QUANTUM=<CHUNK> TEST_DEVICES=0,1,2,3
       PME_DISABLE=true C8_RUN=<run dir> C8_CODE=<code dir>
"""
import json, os, resource, sys, threading, time, types
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.environ.get("C8_CODE", "/home/sulcjo/2026_peptide_sampler"))
import openmm
from openmm import app, unit

N = int(sys.argv[1])
ADMIT = [a.strip() for a in os.environ.get("ADMIT", "all,32,16,8").split(",") if a.strip()]
REPS = int(os.environ.get("REPS", "2"))
SECONDS = float(os.environ.get("SECONDS", "60"))
CHUNK = int(os.environ.get("CHUNK", "250"))
QUANTUM = int(os.environ.get("QUANTUM", str(CHUNK)))
DT_FS = 3.5
RUN = os.environ.get("C8_RUN", "/home/sulcjo/gareus/chignolin/chignolin_8")
SYS = f"{RUN}/swarm/system"
CKPT = f"{RUN}/adaptive_production/epoch_000/checkpoints/production_checkpoint_manifest.json"
ENVELOPE = f"{RUN}/swarm/analysis/shared_gamd_setup/shared_gamd_setup_globals.json"
SOLVENT = {"HOH", "WAT", "NA", "CL", "Na+", "Cl-", "K", "K+", "SOL"}
if CHUNK % QUANTUM:
    raise SystemExit(f"CHUNK {CHUNK} must be a multiple of QUANTUM {QUANTUM}")

from gareus import pep_gamd
from gareus.production import add_umbrella_cv_forces

topology = app.PDBFile(f"{SYS}/topology.pdb").topology
equil = openmm.XmlSerializer.deserialize(open(f"{SYS}/equil_state.xml").read())
system_xml = open(f"{SYS}/base_system.xml").read()
peptide = [a.index for a in topology.atoms() if a.residue.name not in SOLVENT]
primary_cv = json.load(open(CKPT))["primary_cv"]
envelope = json.load(open(ENVELOPE))["all_globals"]


def _cv_args():
    a = types.SimpleNamespace(**json.load(open(f"{RUN}/last_resume_args.json")))
    a.secondary_cv = "residual-torsion-pc"
    a.secondary_cv_model = f"{RUN}/swarm/analysis/cv_pair_model.json"
    a.secondary_cv_candidate_set = f"{RUN}/swarm/analysis/cv_candidate_set.json"
    a.secondary_cv_feature_schema = f"{RUN}/swarm/analysis/cv_feature_schema.json"
    a.cv_force_layout = None  # fresh campaign: let the layout resolve (shared when eligible)
    return a


CV_ARGS = _cv_args()
_layout = {}


def make_system():
    s = openmm.XmlSerializer.deserialize(system_xml)
    for i in reversed(range(s.getNumForces())):
        if isinstance(s.getForce(i), (openmm.MonteCarloBarostat, openmm.MonteCarloAnisotropicBarostat)):
            s.removeForce(i)
    info = add_umbrella_cv_forces(openmm, s, topology, primary_cv, CV_ARGS, secondary_enabled=True)
    _layout.setdefault("cv_force_layout", info.get("cv_force_layout"))
    pep_gamd.ensure_pep_gamd_partition(s, peptide)
    return s


def build(i, devices):
    s = make_system()
    ig = pep_gamd._integrator_class()(
        pep_gamd.DIHEDRAL_GROUP, bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(s),
        dt=DT_FS * unit.femtosecond, ntcmdprep=100, ntcmd=100, ntebprep=100, nteb=100, nstlim=10**9,
        ntave=100, sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
        temperature=300 * unit.kelvin)
    props = {"Precision": "mixed", "DeviceIndex": devices[i % len(devices)],
             "DisablePmeStream": os.environ.get("PME_DISABLE", "true"),
             "UseBlockingSync": "false", "DeterministicForces": "false"}
    sim = app.Simulation(topology, s, ig, openmm.Platform.getPlatformByName("CUDA"), props)
    sim.context.setPeriodicBoxVectors(*equil.getPeriodicBoxVectors())
    sim.context.setPositions(equil.getPositions())
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1000 + i)
    for k in range(ig.getNumGlobalVariables()):
        name = ig.getGlobalVariableName(k)
        if name in envelope:
            ig.setGlobalVariable(k, float(envelope[name]))
    ig.setGlobalVariableByName("stepCount", float(ig.stage_5_start - 1))
    ig.setGlobalVariableByName("stage", 5.0)
    return sim


class AdmissionDispatcher:
    """Per-GPU admission in front of each replica's single-worker executor (P3, spec section 6.2).

    A replica's operation is submitted to its own executor only once admitted, so no thread ever
    blocks on a semaphore; at most one admitted operation per replica; stable round-robin order
    by replica index. run_round(A) returns once every replica has advanced CHUNK steps.
    """

    def __init__(self, sims, pools, gpu_of):
        self.sims, self.pools, self.gpu_of = sims, pools, gpu_of
        self.gpus = sorted(set(gpu_of))
        self.max_seen = {g: 0 for g in self.gpus}

    def run_round(self, limit):
        lock = threading.Lock()
        finished = threading.Event()
        todo = {i: CHUNK // QUANTUM for i in range(len(self.sims))}
        queues = {g: deque(i for i in range(len(self.sims)) if self.gpu_of[i] == g) for g in self.gpus}
        active = {g: 0 for g in self.gpus}
        state = {"left": len(self.sims), "error": None}

        def admit(g):
            """Caller holds lock. Returns the admitted replicas; the caller submits them after
            releasing it (a callback on an already-finished future runs in the submitting
            thread and would re-enter the non-reentrant lock)."""
            out = []
            while active[g] < limit and queues[g] and state["error"] is None:
                i = queues[g].popleft()
                active[g] += 1
                self.max_seen[g] = max(self.max_seen[g], active[g])
                out.append((i, g))
            return out

        def launch(batch):
            for i, g in batch:
                fut = self.pools[i].submit(self.sims[i].step, QUANTUM)
                fut.add_done_callback(lambda f, i=i, g=g: done(f, i, g))

        def done(fut, i, g):
            exc = fut.exception()
            with lock:
                active[g] -= 1
                if exc is not None and state["error"] is None:
                    state["error"] = exc
                todo[i] -= 1
                if todo[i] > 0 and state["error"] is None:
                    queues[g].append(i)  # re-queue behind the others: fair quantum rotation
                else:
                    state["left"] -= 1
                if state["error"] is not None:
                    for q in queues.values():  # cancel not-yet-admitted work
                        state["left"] -= len(q)
                        q.clear()
                batch = admit(g)
                if state["left"] <= 0 and all(v == 0 for v in active.values()):
                    finished.set()
            launch(batch)

        with lock:
            batch = [x for g in self.gpus for x in admit(g)]
        launch(batch)
        finished.wait()
        if state["error"] is not None:
            raise state["error"]


def measure(dispatcher, limit):
    dispatcher.run_round(limit)  # warm-up round for this arm
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    cpu0 = r0.ru_utime + r0.ru_stime
    t1 = time.time()
    rounds = 0
    while time.time() - t1 < SECONDS:
        dispatcher.run_round(limit)
        rounds += 1
    wall = time.time() - t1
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    return rounds, wall, r1.ru_utime + r1.ru_stime - cpu0


devices = [d for d in os.environ.get("TEST_DEVICES", "0,1,2,3").split(",") if d]
gpu_of = [devices[i % len(devices)] for i in range(N)]
resident = {g: gpu_of.count(g) for g in set(gpu_of)}
pools = [ThreadPoolExecutor(max_workers=1) for _ in range(N)]
t0 = time.time()
sims = [pools[i].submit(build, i, devices).result() for i in range(N)]
print(f"built {N} contexts in {time.time()-t0:.0f}s | resident/GPU {resident} | "
      f"cv_force_layout={_layout.get('cv_force_layout')} | bias groups={pep_gamd.pep_gamd_bias_force_groups(sims[0].system)}",
      flush=True)
dispatcher = AdmissionDispatcher(sims, pools, gpu_of)
dispatcher.run_round(10**9)  # first-step quirk + kernel compile, unrestricted
mps_pct = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "inherit")
mps_on = bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY"))
for rep in range(REPS):
    for a in ADMIT:
        limit = 10**9 if a == "all" else int(a)
        if limit <= 0:
            raise SystemExit(f"ADMIT entries must be positive or 'all', got {a!r}")
        dispatcher.max_seen = {g: 0 for g in dispatcher.gpus}
        rounds, wall, cpu = measure(dispatcher, limit)
        sps = rounds * CHUNK / wall
        eff = {g: min(limit, n) for g, n in resident.items()}
        assert all(dispatcher.max_seen[g] <= eff[g] for g in eff), (dispatcher.max_seen, eff)
        e = sims[0].context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ig0 = sims[0].integrator
        print(f"RESULT rep={rep} admit={a} eff/GPU={max(eff.values())} maxseen/GPU={max(dispatcher.max_seen.values())} "
              f"N={N} rounds={rounds} wall={wall:.1f}s | {sps:.2f} steps/s/rep | "
              f"{sps*DT_FS*1e-6*86400:.2f} ns/day/rep | node {N*sps*DT_FS*1e-6*86400:.0f} ns/day | "
              f"cpu {cpu/wall:.1f} cores | E0={e:.0f} finite={e == e} | MPS={mps_on} pct={mps_pct} "
              f"PME_DISABLE={os.environ.get('PME_DISABLE', 'true')} chunk={CHUNK} quantum={QUANTUM} | "
              f"FSF_T {ig0.getGlobalVariableByName('ForceScalingFactor_Total'):.3f}", flush=True)
