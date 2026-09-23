"""Which part of the chignolin_8 step makes 248 time-sliced contexts 5x slower than Langevin?

Arms (all on the real chignolin_8 swarm system, 248 contexts round-robin over 4 GPUs, no MPS,
one dedicated thread per replica, all replicas stepping concurrently in 250-step chunks like
production's report cadence):

  L    LangevinMiddleIntegrator, stock system                    (control: job 2563549 gave 49.7 steps/s)
  L2   LangevinMiddleIntegrator + the Pep-GaMD auxiliary PME      (isolates the cost of the second PME)
  P    the real PepGaMDLowerDualIntegrator, production-stage       (what production runs, minus CV forces
       globals copied from chignolin_8's checkpoint                  and the gareus loop)
  B    branch-free CustomIntegrator: identical force-group reads  (the proposed bypass: same forces and
       and energy globals as P's production step, boost via step()   energies, no beginIfBlock)

Usage: python c8_integ_bench.py ARM N [SECONDS]
"""
import json, os, resource, sys, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/home/sulcjo/2026_peptide_sampler")
import openmm
from openmm import app, unit

ARM, N = sys.argv[1], int(sys.argv[2])
SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
CHUNK = 250
DT_FS = 3.5
RUN = "/home/sulcjo/gareus/chignolin/chignolin_8"
SYS = f"{RUN}/swarm/system"
CKPT = f"{RUN}/adaptive_production/epoch_000/checkpoints/production_checkpoint_manifest.json"
SOLVENT = {"HOH", "WAT", "NA", "CL", "Na+", "Cl-", "K", "K+", "SOL"}

from gareus import pep_gamd

topology = app.PDBFile(f"{SYS}/topology.pdb").topology
equil = openmm.XmlSerializer.deserialize(open(f"{SYS}/equil_state.xml").read())
system_xml = open(f"{SYS}/base_system.xml").read()
peptide = [a.index for a in topology.atoms() if a.residue.name not in SOLVENT]
globals0 = json.load(open(CKPT))["replica_integrator_globals_all"]
globals0 = globals0[0] if isinstance(globals0, list) else globals0[sorted(globals0)[0]]


BIAS = os.environ.get("BIAS", "none")   # none | split (29 + 31, as production) | merged (both in 29)
_cv_args = None
def _add_real_cv_forces(s):
    """The production primary contact umbrella + residual-torsion-pc secondary CV force."""
    global _cv_args
    import types
    from gareus.production import add_primary_umbrella_force, add_secondary_structure_cv_force
    if _cv_args is None:
        ra = json.load(open(f"{RUN}/last_resume_args.json"))
        a = types.SimpleNamespace(**ra)
        a.secondary_cv = "residual-torsion-pc"
        a.secondary_cv_model = f"{RUN}/swarm/analysis/cv_pair_model.json"
        a.secondary_cv_candidate_set = f"{RUN}/swarm/analysis/cv_candidate_set.json"
        a.secondary_cv_feature_schema = f"{RUN}/swarm/analysis/cv_feature_schema.json"
        _cv_args = a
    ck = json.load(open(CKPT))
    g_primary = 29 if BIAS == "merged" else 31
    add_primary_umbrella_force(openmm, s, ck["primary_cv"], _cv_args, g_primary)
    info = add_secondary_structure_cv_force(openmm, s, topology, _cv_args, force_group=29, primary_cv_def=ck["primary_cv"])
    return info

def make_system(partition: bool):
    s = openmm.XmlSerializer.deserialize(system_xml)
    for i in reversed(range(s.getNumForces())):
        if isinstance(s.getForce(i), (openmm.MonteCarloBarostat, openmm.MonteCarloAnisotropicBarostat)):
            s.removeForce(i)
    if BIAS != "none":
        _add_real_cv_forces(s)
    if partition:
        pep_gamd.ensure_pep_gamd_partition(s, peptide)
    return s


def branch_free_integrator(system):
    g = globals0
    dt = DT_FS * unit.femtosecond
    gamma = 1.0 / unit.picosecond
    kT = (unit.MOLAR_GAS_CONSTANT_R * 300 * unit.kelvin).value_in_unit(unit.kilojoule_per_mole)
    ig = openmm.CustomIntegrator(dt)
    vs = float(__import__("math").exp(-(dt * gamma)))
    ig.addGlobalVariable("vscale", vs)
    ig.addGlobalVariable("fscale", (1 - vs) / gamma.value_in_unit(unit.picosecond ** -1))
    ig.addGlobalVariable("noisescale", (kT * (1 - vs * vs)) ** 0.5)
    for ch in ("Total", "Dihedral"):
        k = float(g[f"k0_{ch}"]) / (float(g[f"Vmax_{ch}"]) - float(g[f"Vmin_{ch}"]))
        ig.addGlobalVariable(f"k_{ch}", k)
        ig.addGlobalVariable(f"Eth_{ch}", float(g[f"threshold_energy_{ch}"]))
    for name in ("E0", "E1", "E2", "ED", "ET", "boostD", "FSF_D", "FSF_T"):
        ig.addGlobalVariable(name, 0.0)
    for name in ("newx", "F0", "F1", "F2", "Fb"):
        ig.addPerDofVariable(name, 0.0)
    bias = pep_gamd.pep_gamd_bias_force_groups(system)
    ig.addComputeGlobal("E0", "energy0")
    ig.addComputeGlobal("E1", "energy1")
    ig.addComputeGlobal("E2", "energy2")
    ig.addComputeGlobal("ED", "E2")
    ig.addComputeGlobal("FSF_D", "1 - k_Dihedral*(Eth_Dihedral-ED)*step(Eth_Dihedral-ED)")
    ig.addComputeGlobal("boostD", "0.5*k_Dihedral*(Eth_Dihedral-ED)^2*step(Eth_Dihedral-ED)")
    ig.addComputeGlobal("ET", "E0 - E1 + E2 + boostD")
    ig.addComputeGlobal("FSF_T", "1 - k_Total*(Eth_Total-ET)*step(Eth_Total-ET)")
    ig.addComputePerDof("newx", "x")
    ig.addComputePerDof("v", "vscale*v + noisescale*gaussian/sqrt(m)")
    ig.addComputePerDof("F0", "f0")
    ig.addComputePerDof("F1", "f1")
    ig.addComputePerDof("F2", "f2")
    bexpr = "0"
    if bias:
        ig.addComputePerDof("Fb", "+".join(f"f{b}" for b in bias))
        bexpr = "Fb"
    ig.addComputePerDof("v", f"v + fscale*((F0-F1)*FSF_T + F2*FSF_T*FSF_D + F1 + {bexpr})/m")
    ig.addComputePerDof("x", "x+dt*v")
    ig.addConstrainPositions()
    ig.addComputePerDof("v", "(x-newx)/dt")
    return ig


def build(i, devices):
    if ARM == "L":
        s = make_system(False)
        ig = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, DT_FS * unit.femtosecond)
    elif ARM == "L2":
        s = make_system(True)
        ig = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, DT_FS * unit.femtosecond)
    elif ARM == "P":
        s = make_system(True)
        ig = pep_gamd._integrator_class()(
            pep_gamd.DIHEDRAL_GROUP, bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(s),
            dt=DT_FS * unit.femtosecond, ntcmdprep=100, ntcmd=100, ntebprep=100, nteb=100, nstlim=10**9,
            ntave=100, sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
            temperature=300 * unit.kelvin)
    elif ARM == "B":
        s = make_system(True)
        ig = branch_free_integrator(s)
    else:
        raise SystemExit(f"unknown arm {ARM}")
    props = {"Precision": "mixed", "DeviceIndex": devices[i % len(devices)], "DisablePmeStream": "true",
             "UseBlockingSync": "false", "DeterministicForces": "false"}
    sim = app.Simulation(topology, s, ig, openmm.Platform.getPlatformByName("CUDA"), props)
    sim.context.setPeriodicBoxVectors(*equil.getPeriodicBoxVectors())
    sim.context.setPositions(equil.getPositions())
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1000 + i)
    if ARM == "P":
        n_set = 0
        for k in range(ig.getNumGlobalVariables()):
            name = ig.getGlobalVariableName(k)
            if name in globals0:
                ig.setGlobalVariable(k, float(globals0[name])); n_set += 1
        if i == 0:
            print(f"P: copied {n_set} production-stage globals from the checkpoint (stage={ig.getGlobalVariableByName('stage')})", flush=True)
    return sim


devices = [d for d in os.environ.get("TEST_DEVICES", "0,1,2,3").split(",") if d]
pools = [ThreadPoolExecutor(max_workers=1) for _ in range(N)]
t0 = time.time()
sims = [pools[i].submit(build, i, devices).result() for i in range(N)]
print(f"ARM {ARM} N={N}: {N} contexts built in {time.time()-t0:.0f}s", flush=True)
for f in [pools[i].submit(lambda s=sims[i]: s.step(CHUNK)) for i in range(N)]: f.result()  # warm-up (+ first-step quirk)
r0 = resource.getrusage(resource.RUSAGE_SELF); cpu0 = r0.ru_utime + r0.ru_stime
t1 = time.time(); chunks = 0
while time.time() - t1 < SECONDS:
    for f in [pools[i].submit(lambda s=sims[i]: s.step(CHUNK)) for i in range(N)]: f.result()
    chunks += 1
wall = time.time() - t1
r1 = resource.getrusage(resource.RUSAGE_SELF); cpu = r1.ru_utime + r1.ru_stime - cpu0
sps = chunks * CHUNK / wall
state = sims[0].context.getState(getEnergy=True)
e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
print(f"RESULT arm={ARM} bias={BIAS} groups={pep_gamd.pep_gamd_bias_force_groups(sims[0].system)} N={N} steps/rep={chunks*CHUNK} wall={wall:.1f}s | {sps:.2f} steps/s/rep | "
      f"{sps*DT_FS*1e-6*86400:.2f} ns/day/rep | node {N*sps*DT_FS*1e-6*86400:.0f} ns/day | "
      f"cpu {cpu/wall:.1f} cores ({cpu/wall/N:.2f}/rep) | E0={e:.0f} kJ/mol finite={e==e}", flush=True)
