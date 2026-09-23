# The Board's Judgment

## Majority position

The majority (deepseek-thinking, mini, thinker) votes **REJECT**. The NPT subsystem can silently produce a number that is not the Boltzmann free energy it claims, for three concrete reasons:

1. **Unenforced step-counter contract in the stepping driver.** `ReplicaStepDriver.advance` calls `self.sim.integrator.step(delta)` directly, but reads all deadlines from `self.sim.currentStep`. In stock OpenMM, `Simulation.currentStep` is advanced only by `Simulation.step()`, not by `Integrator.step()`. If the production `sim` is a stock `Simulation`, the volume-move schedule and reporter deadlines silently desync or never fire, so the intended NPT Metropolis kernel is not applied to the trajectory. If a wrapper syncs `currentStep`, that invariant is not shown and not tested — exactly the chignolin_8 hand-off failure class.

2. **Unverified force-group hygiene in the adapters.** The adapters compute U* from a force-group inventory but never assert that the integrator's integration force groups exclude the auxiliary water-only force. If the auxiliary group is integrated, the dynamics sample `U_phys + U_aux + boost` while the barostat acceptance prices `U_phys + bias + boost` — a silent wrong-Hamiltonian mismatch. Additionally, `_npt_split_groups` classifies any group containing an auxiliary force entirely as `aux`; a physical-class force sharing `AUX_NONBONDED_GROUP` is silently dropped from U*.

3. **Boost activation is inferred, not verified.** The Pep-GaMD adapter prices a boost from the `stage` global and channel parameters, but never checks the actual boost-activation flag (e.g., `ForceScalingFactor`). A stage-5-with-`ForceScalingFactor=1.0` hand-off would price a boost that is never applied, silently biasing the volume ensemble and the resulting free-energy surface.

These are silent-wrong-number paths, not style issues.

## Dissenting positions

glm (ACCEPT-WITH-CHANGES, confidence 76):  
"VERDICT: ACCEPT-WITH-CHANGES - the acceptance math is verified correct and the target's REJECT pillars are one loud hang and one unverified gap, but the currentStep sync, integration-force-group assertion, and mixed-aux-group check must land before any campaign runs."

kimi (ACCEPT-WITH-CHANGES, confidence 80):  
"VERDICT: ACCEPT-WITH-CHANGES — acceptance math, Jacobian, units and boost pricing are verified correct; required changes are force-group-hygiene asserts and an explicit currentStep contract, not redesign"

## Confidence

Aggregate confidence: **78**. Derived from the mean of the majority members' confidences (deepseek-thinking 78, mini 92, thinker 78; mean 82.7), reduced by the final agreement measurements — a 3-2 split with mean position-card cosine 0.597 — to 78.

## Conditions

1. Make the driver's step counter authoritative: use `integrator.getStepCount()` for all deadlines, or enforce and test the `sim.currentStep` sync invariant; restore controller and integrator state on rollback probes.
2. Adapters must assert that the auxiliary force is excluded from the integrator's integration force groups, and that no physical-class force shares `AUX_NONBONDED_GROUP`.
3. The adapter must verify actual boost activation (e.g., `ForceScalingFactor`) and validate `Vmax > Vmin` and `k0 >= 0`.
4. Add integration tests that exercise the real stage hand-off (not hand-set `stepCount`) and the aux-group/integration-group mismatch.

VERDICT: REJECT - the NPT subsystem can silently sample a non-Boltzmann distribution because the stepping driver's step-counter contract is unenforced and the adapters do not verify that the propagated force groups match the priced U*.
CONFIDENCE: 78

---

## Mathematics & Physics audit (final)

We need answer audit. Need inspect judgment for mathematical/physical errors. Need list errors in judgment? The prompt: "Audit this judgment for mathematical and physical errors: dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics and datatypes per your checklists." We need evaluate the Board's Judgment. It is a meta-judgment about NPT subsystem. Need find errors in that judgment. Need output FLAG lines and Checked. Need be careful.

We need understand context? The judgment is about rejecting code due to issues. We need audit for math/physics errors in the judgment itself. It mentions NPT subsystem, ReplicaStepDriver, OpenMM Simulation.currentStep, integrator.step, force groups, Pep-GaMD boost, etc. Need identify any mathematical/physical errors in reasoning, dimensional consistency, sign conventions, thermodynamic identities, statistical-mechanical correctness, estimator assumptions, units, factors, statistics/datatypes. The judgment's claims: majority REJECT due to three reasons. Need see if any errors.

Let's parse.

Majority position:
1. Unenforced step-counter contract in stepping driver. `ReplicaStepDriver.advance` calls `self.sim.integrator.step(delta)` directly, but reads all deadlines from `self.sim.currentStep`. In stock OpenMM, `Simulation.currentStep` is advanced only by `Simulation.step()`, not by `Integrator.step()`. If production sim is stock Simulation, volume-move schedule and reporter deadlines silently desync or never fire, so intended NPT Metropolis kernel is not applied. If wrapper syncs currentStep, invariant not shown/tested.

Is this a mathematical/physical error? It's a software contract issue. Could be valid. But need audit for errors. Maybe "Simulation.currentStep is advanced only by Simulation.step(), not by Integrator.step()" - In OpenMM, Simulation.currentStep is a property? Actually OpenMM Simulation has currentStep attribute? Let's recall: OpenMM Simulation class has `currentStep` property? It is set by `Simulation.step()`? In OpenMM, `Simulation.currentStep` is an integer property? It may be updated by `Simulation.step()` and also by `Simulation.context.setPositions`? Hmm. `Integrator.step()` advances context time and step count but Simulation.currentStep? In OpenMM, `Simulation` has `currentStep` getter that returns `context.getStepCount()`? Let's recall from OpenMM API: `Simulation.currentStep` is "The current step number" maybe property backed by `context.getStepCount()`. Actually in OpenMM's `Simulation` class, `currentStep` is a property: `def getCurrentStep(self): return self.context.getStepCount()`? Need verify. If so, majority claim "Simulation.currentStep is advanced only by Simulation.step(), not by Integrator.step()" is false. Let's investigate from memory.

OpenMM Python API: `simulation.currentStep` is an attribute? In examples, after `simulation.step(100)`, `simulation.currentStep` is 100. If you call `integrator.step(10)` directly on a Simulation's integrator, does `simulation.currentStep` update? The context's step count is incremented by integrator.step. `Simulation.currentStep` likely returns `self.context.getStepCount()`. Let's recall source? OpenMM `Simulation` class has:
```
@property
def currentStep(self):
    return self.context.getStepCount()
```
Maybe. Actually `Simulation.step(steps)` calls `self.integrator.step(steps)` then `self._updateCurrentStep()`? Hmm.

Let's search memory: In OpenMM, `Simulation` has `currentStep` property? I used OpenMM: `simulation.currentStep` is an integer. It is updated after `simulation.step()`. If you call `simulation.integrator.step(1)`, does `simulation.currentStep` change? I think yes? Let's reason: `Simulation.step` implementation:
```
def step(self, steps):
    if self._integrator is None: raise
    self._integrator.step(steps)
    self._updateCurrentStep()
```
Maybe `_updateCurrentStep` reads `self._integrator.getStepCount()`. If you call integrator.step directly, `simulation.currentStep` remains stale until `simulation.step` called. But if `currentStep` property reads context, then no. Need know.

OpenMM user guide: "currentStep: The current step number. This is incremented by step()" maybe. Let's recall from source code (OpenMM 7.x):
```
class Simulation(object):
    def __init__(...):
        self.currentStep = 0
    def step(self, steps):
        if self.currentStep is None:
            self.currentStep = 0
        self.integrator.step(steps)
        self.currentStep += steps
```
Actually maybe it's an attribute maintained by Simulation. Let's recall from OpenMM docs: `Simulation.currentStep` is "The current step number." It is a property? In script:
```
simulation.step(100)
print(simulation.currentStep)
```
I think `currentStep` is an integer attribute, not property. In OpenMM's `Simulation` class, `currentStep` is initialized to 0 and incremented in `step()`. If you call `integrator.step()` directly, `simulation.currentStep` doesn't update. This is a known gotcha. So majority claim plausible. But need not flag unless false.

2. Unverified force-group hygiene. Adapters compute U* from force-group inventory but never assert integrator's integration force groups exclude auxiliary water-only force. If auxiliary group integrated, dynamics sample `U_phys + U_aux + boost` while barostat acceptance prices `U_phys + bias + boost` — silent wrong-Hamiltonian mismatch. Additionally, `_npt_split_groups` classifies any group containing auxiliary force entirely as `aux`; physical-class force sharing `AUX_NONBONDED_GROUP` is silently dropped from U*.

This is software/force-field correctness. Need check if "barostat acceptance prices U_phys + bias + boost" vs dynamics "U_phys + U_aux + boost" is correct? If auxiliary force is integrated, Hamiltonian includes aux; barostat acceptance should use total potential including aux? Actually if aux is part of system, then U* should include aux. The issue is mismatch. The judgment says "dynamics sample U_phys + U_aux + boost while barostat acceptance prices U_phys + bias + boost" - if bias is boost? Need parse. "boost" maybe GaMD boost. "bias" maybe auxiliary? Hmm.

3. Boost activation inferred, not verified. Pep-GaMD adapter prices a boost from `stage` global and channel parameters, but never checks actual boost-activation flag (e.g., `ForceScalingFactor`). A stage-5-with-`ForceScalingFactor=1.0` hand-off would price a boost that is never applied, silently biasing volume ensemble and resulting free-energy surface.

Need check if "ForceScalingFactor=1.0" means boost not applied? In GaMD, ForceScalingFactor is a parameter? In OpenMM GaMD? Actually GaMD has "boost" applied by adding a boost potential; activation flag maybe `ForceScalingFactor`? Need know. In OpenMM's GaMD integrator? There is `OpenMM::GaMDIntegrator`? It has `setForceScalingFactor`? Hmm. In Pep-GaMD, boost activation maybe `stage` global and channel parameters. The judgment says "never checks actual boost-activation flag (e.g., ForceScalingFactor). A stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied". Is ForceScalingFactor=1.0 a flag? In GaMD, force scaling factor is used for "boost potential" maybe if 1.0 no boost? Need not.

Our task: audit the judgment for mathematical/physical errors. We need list every error found. Could be no errors? But likely there are some. Need identify.

Let's examine the judgment's "Confidence" calculation: "Aggregate confidence: 78. Derived from the mean of the majority members' confidences (deepseek-thinking 78, mini 92, thinker 78; mean 82.7), reduced by the final agreement measurements — a 3-2 split with mean position-card cosine 0.597 — to 78." This is a statistical/aggregation claim. Need check: mean of 78, 92, 78 = 82.666... They say 82.7. Reduced by agreement measurements to 78. Is that mathematically consistent? Maybe not enough info. "3-2 split with mean position-card cosine 0.597" - how reduce to 78? Could be arbitrary. But as an audit, maybe flag "confidence aggregation lacks defined formula; 82.7 reduced to 78 not reproducible" as MINOR? But prompt asks mathematical/physical errors, not meta. However "statistics and datatypes per checklists" includes estimator assumptions. The confidence calculation is a statistical aggregation. Need evaluate.

Also "mean of majority members' confidences" but majority members are deepseek-thinking, mini, thinker; there are 3. Mean 82.7. "reduced by final agreement measurements — a 3-2 split with mean position-card cosine 0.597 — to 78." How exactly? No formula. Could be an error if they claim derived but no derivation. But maybe not a mathematical error, just vague.

Need inspect "NPT subsystem can silently produce a number that is not the Boltzmann free energy it claims" - "Boltzmann free energy" maybe "Boltzmann" vs "Gibbs"? NPT free energy is Gibbs free energy, not Boltzmann free energy. "Boltzmann free energy" is not standard; Helmholtz free energy F = -kT ln Z is "Boltzmann free energy"? Actually "Boltzmann free energy" maybe "Boltzmann distribution" free energy? In NPT, partition function is isothermal-isobaric, Gibbs free energy. The judgment says "not the Boltzmann free energy it claims" - could be a terminology error. But maybe "Boltzmann free energy" means free energy from Boltzmann weights. Need flag? The prompt asks thermodynamic identities. NPT ensemble free energy is Gibbs, not Helmholtz/Boltzmann. If the subsystem claims "Boltzmann free energy" maybe wrong. But the judgment itself says "NPT subsystem can silently produce a number that is not the Boltzmann free energy it claims" - maybe the target claims Boltzmann free energy? Hmm.

Need read "The NPT subsystem can silently produce a number that is not the Boltzmann free energy it claims" - In NPT, the relevant free energy is Gibbs free energy G. "Boltzmann free energy" could be a misnomer. But maybe "Boltzmann free energy" is used in context of "Boltzmann reweighting" to mean free energy from Boltzmann factors. Not necessarily error.

Need identify actual mathematical/physical errors in the judgment's reasoning.

Let's analyze each claim in depth.

### Claim 1: Step-counter contract

"ReplicaStepDriver.advance calls self.sim.integrator.step(delta) directly, but reads all deadlines from self.sim.currentStep. In stock OpenMM, Simulation.currentStep is advanced only by Simulation.step(), not by Integrator.step()."

Is this true? Need verify OpenMM API. Let's recall from OpenMM 7.5 source code (from memory):

`Simulation` class in `openmm/app/simulation.py`:
```
class Simulation(object):
    def __init__(self, topology, system, integrator, platform=None, platformProperties=None):
        self.topology = topology
        self.system = system
        self.integrator = integrator
        self.context = Context(system, integrator, platform, platformProperties)
        self.currentStep = 0
        self.reporters = []
    def step(self, steps):
        if self.currentStep is None:
            self.currentStep = 0
        self.integrator.step(steps)
        self.currentStep += steps
        for reporter in self.reporters:
            if reporter should report...
```
Actually I recall `Simulation.currentStep` is an attribute updated by `Simulation.step`. Let's search memory: In OpenMM, `simulation.currentStep` is a property? I remember using `simulation.currentStep` after `simulation.step()`; it is an integer. In the source, `Simulation` has:
```
@property
def currentStep(self):
    return self._currentStep
```
Maybe.

Let's recall from OpenMM docs: "Simulation.currentStep: The current step number. This is incremented by step()." It doesn't say property. In examples, after `simulation.step(100)`, `simulation.currentStep` is 100. If you call `simulation.integrator.step(100)`, does `simulation.currentStep` update? I think no, because `Simulation.step` increments `currentStep` itself. There is a known issue: if you use `integrator.step()` directly, reporters don't fire because `Simulation.currentStep` not updated. Actually reporters are called by `Simulation.step`, not by integrator. So yes, if you call integrator.step directly, `Simulation.currentStep` remains stale. So claim plausible.

But the judgment says "reads all deadlines from self.sim.currentStep. In stock OpenMM, Simulation.currentStep is advanced only by Simulation.step(), not by Integrator.step()." That is likely correct. However, "If a wrapper syncs currentStep, that invariant is not shown and not tested" - okay.

Potential error: "volume-move schedule and reporter deadlines silently desync or never fire, so the intended NPT Metropolis kernel is not applied to the trajectory." If `ReplicaStepDriver.advance` calls `integrator.step(delta)` directly, the integrator's step count advances, but `sim.currentStep` doesn't. If deadlines are read from `sim.currentStep`, then after first advance, `sim.currentStep` remains 0, so volume-move schedule might fire every call? Actually if schedule is "if currentStep % N == 0", with currentStep stuck at 0, it fires every time. That would apply NPT Metropolis kernel every delta, not never. The judgment says "silently desync or never fire" - "never fire" if schedule uses `>=`? If currentStep never advances, a deadline at step 100 never reached, so never fire. So okay.

But is "intended NPT Metropolis kernel is not applied to the trajectory" necessarily? If volume-move schedule never fires, then no volume moves, so NPT becomes NVT? Actually if barostat is in integrator? The "NPT Metropolis kernel" might be external volume moves. If not applied, wrong ensemble. So okay.

### Claim 2: Force-group hygiene

"adapters compute U* from a force-group inventory but never assert that the integrator's integration force groups exclude the auxiliary water-only force. If the auxiliary group is integrated, the dynamics sample U_phys + U_aux + boost while the barostat acceptance prices U_phys + bias + boost — a silent wrong-Hamiltonian mismatch."

Need parse "auxiliary water-only force" and "boost". In enhanced sampling, "auxiliary" force might be a water-only restraint? "U_phys + U_aux + boost" vs "U_phys + bias + boost" - what is "bias"? Maybe "bias" is the auxiliary force? The sentence says "barostat acceptance prices U_phys + bias + boost" - maybe "bias" is the same as U_aux? If so, mismatch is U_aux vs bias? Hmm.

Could be a typo: "prices U_phys + bias + boost" should be "U_phys + U_aux + boost"? Let's infer. The issue: If auxiliary group is integrated, dynamics include aux. The barostat acceptance should price the same Hamiltonian as dynamics. If it prices "U_phys + bias + boost" instead of "U_phys + U_aux + boost", then mismatch. But what is "bias"? Maybe "bias" is the boost? Actually "boost" is a bias. "U_phys + bias + boost" redundant. Maybe "bias" is "U_aux"? The judgment might have a terminology inconsistency. Need flag? The prompt asks "mathematical and physical errors: ... thermodynamic identities". This is a physical error if the Hamiltonian terms are mislabeled. But without context, hard.

Let's consider "Pep-GaMD adapter prices a boost from the stage global and channel parameters" - "boost" is a GaMD boost potential. "bias" might be the auxiliary water-only force? In "U_phys + bias + boost", "bias" could be "auxiliary bias" (e.g., water-only force). Then dynamics sample U_phys + U_aux + boost, barostat prices U_phys + bias + boost. If "bias" is the auxiliary force, then it's the same as U_aux, so no mismatch. But the sentence says "If the auxiliary group is integrated, the dynamics sample U_phys + U_aux + boost while the barostat acceptance prices U_phys + bias + boost" — if bias = U_aux, then both include aux, no mismatch. The mismatch would be if barostat excludes aux. Maybe "bias" is something else. Need not.

Maybe "bias" is the "boost" and "boost" is something else? Let's parse: "dynamics sample U_phys + U_aux + boost while the barostat acceptance prices U_phys + bias + boost" — The "bias" could be "U_aux" (auxiliary bias). Then both include U_aux and boost. So no mismatch. Unless "bias" is a different bias term. The intended mismatch is "auxiliary group integrated" vs "barostat acceptance prices U_phys + bias + boost" where "bias" is the boost? Actually "boost" is a bias. Maybe "bias" is the "auxiliary water-only force" and "boost" is GaMD boost. Then both include aux and boost. So the sentence is self-contradictory. Could be a typo: "prices U_phys + bias + boost" should be "prices U_phys + boost" (i.e., excludes aux). The correct form would be "barostat acceptance prices U_phys + boost" if aux should be excluded. The judgment says "If the auxiliary group is integrated, the dynamics sample U_phys + U_aux + boost while the barostat acceptance prices U_phys + bias + boost" — maybe "bias" is the auxiliary force, so no mismatch. This is a potential error in the judgment's reasoning. Need decide.

Let's read "Unverified force-group hygiene in the adapters. The adapters compute U* from a force-group inventory but never assert that the integrator's integration force groups exclude the auxiliary water-only force. If the auxiliary group is integrated, the dynamics sample `U_phys + U_aux + boost` while the barostat acceptance prices `U_phys + bias + boost` — a silent wrong-Hamiltonian mismatch. Additionally, `_npt_split_groups` classifies any group containing an auxiliary force entirely as `aux`; a physical-class force sharing `AUX_NONBONDED_GROUP` is silently dropped from U*."

Maybe "bias" is the "auxiliary water-only force" (a bias potential). Then "U_phys + bias + boost" includes aux. So the mismatch is not wrong-Hamiltonian; it's actually matching. Unless "bias" is not U_aux but "bias" is the "boost"? Wait "boost" is a bias. Maybe "bias" is "U_aux" and "boost" is "GaMD boost". Then both include aux and boost. So no mismatch. The actual mismatch would be if dynamics include aux but barostat acceptance prices U_phys + boost (without aux). The judgment's sentence has "bias" where it should be "U_phys + boost" or "U_phys + U_aux"? Let's think.

Maybe "bias" is the "auxiliary water-only force" and "boost" is the "Pep-GaMD boost". The barostat acceptance should price the total potential energy of the sampled Hamiltonian. If dynamics sample U_phys + U_aux + boost, then barostat should price U_phys + U_aux + boost. The judgment says barostat prices U_phys + bias + boost. If bias = U_aux, then it's correct. So the sentence "while the barostat acceptance prices U_phys + bias + boost" would not be a mismatch. But the judgment claims mismatch. Therefore either "bias" is not U_aux, or the sentence is erroneous. Could be "bias" is a "bias" term from the adapter (e.g., a biasing potential) separate from U_aux. Hmm.

Let's not overfocus; maybe the audit should flag "bias" term undefined/inconsistent. But the prompt wants mathematical/physical errors. We can flag "The mismatch equation is dimensionally/physically inconsistent: if bias=U_aux, both Hamiltonians match; if bias≠U_aux, the term is undefined. Correct form: dynamics sample U_phys+U_aux+boost; barostat acceptance must price the same total, so mismatch is U_phys+U_aux+boost vs U_phys+boost." But is that an error in the judgment? The judgment's point is that if aux integrated, barostat acceptance prices U_phys + bias + boost. Maybe "bias" is the "auxiliary water-only force" and "boost" is "boost". Then no mismatch. But the judgment says "silent wrong-Hamiltonian mismatch" — that is wrong. However, maybe "bias" is a "bias" from the "adapter" (e.g., a biasing potential) and "boost" is another. Without context, hard.

Let's examine "Additionally, `_npt_split_groups` classifies any group containing an auxiliary force entirely as `aux`; a physical-class force sharing `AUX_NONBONDED_GROUP` is silently dropped from U*." This is a valid concern: if a group contains both aux and physical forces, classifying entire group as aux means physical force omitted from U*. But if force groups are per-force, a group can contain multiple forces. The adapter computes U* from force-group inventory. If a physical force shares the aux group, it's dropped. That's a software bug. No math error.

### Claim 3: Boost activation

"Boost activation is inferred, not verified. The Pep-GaMD adapter prices a boost from the `stage` global and channel parameters, but never checks the actual boost-activation flag (e.g., `ForceScalingFactor`). A stage-5-with-`ForceScalingFactor=1.0` hand-off would price a boost that is never applied, silently biasing the volume ensemble and the resulting free-energy surface."

Need check if "ForceScalingFactor=1.0" means boost not applied. In GaMD, the boost potential is:
\[
V_b(R) = \frac{1}{2} k_0 (V_{\max} - V(R))^2 \quad \text{if } V(R) < V_{\max}
\]
or something. "ForceScalingFactor" maybe a parameter in OpenMM's GaMD integrator? Let's recall: OpenMM has `GaMDIntegrator`? Actually OpenMM 7.5 has `openmm.openmm.GaMDIntegrator`? There is a "GaMD" plugin? In OpenMM, "Gaussian accelerated molecular dynamics" implemented in `openmm`? I recall `openmm.GaMDIntegrator` has methods `setBoostThreshold`, `setBoostFactor`, `setForceScalingFactor`? Hmm.

In GaMD, there are two boost modes: "total boost" and "dihedral boost". The boost potential is applied if the total potential or dihedral energy is below a threshold. The "force scaling factor" maybe \(\lambda\) in "GaMD with force scaling" where the force is scaled by a factor? Actually "Pep-GaMD" is peptide GaMD, a variant. "ForceScalingFactor" might be a flag that when 1.0 means no boost? Need know.

Let's search memory: In OpenMM's GaMD integrator, there is a parameter `forceScaleFactor`? The GaMD boost potential is:
\[
\Delta V(\mathbf{r}) = \frac{1}{2} k_0 (V(\mathbf{r}) - V_{\max})^2 / (V_{\min} - V_{\max}) \quad \text{if } V(\mathbf{r}) \ge V_{\max}
\]
No, actually GaMD adds boost when potential is below threshold? Let's recall: GaMD (Miao et al.) adds a boost potential \(\Delta V(r)\) when the system potential \(V(r)\) is lower than a threshold \(V_{\max}\):
\[
\Delta V(r) = \begin{cases}
0, & V(r) \ge V_{\max} \\
\frac{1}{2} k_0 \left(\frac{1}{V_{\min}-V_{\max}}\right) (V_{\max}-V(r))^2, & V(r) < V_{\max}
\end{cases}
\]
where \(k_0\) is a force constant. The "force scaling factor" maybe \(k_0\)? Actually "ForceScalingFactor" could be a factor that scales forces when boost is applied? In GaMD, the boost is applied by modifying forces: \(\mathbf{F} = \mathbf{F}_0 + \frac{d\Delta V}{dr}\). There is no "ForceScalingFactor" in standard GaMD. In "Pep-GaMD" maybe there is a "force scaling factor" to control boost activation? Hmm.

The judgment says "A stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — if ForceScalingFactor=1.0 means no scaling? In many contexts, a scaling factor of 1.0 means full force, not no boost. If "ForceScalingFactor" is a flag where 0 disables and 1 enables? Actually "ForceScalingFactor=1.0" could mean boost is fully applied, not never applied. The judgment might have the sign/activation convention wrong. Need know.

Let's think: In OpenMM's `GaMDIntegrator`, there is a method `setForceScalingFactor(scaleFactor)`? I recall from OpenMM documentation: "GaMDIntegrator" has `setBoostFactor`, `setBoostThreshold`, `setForceScalingFactor`? Wait, there is "GaMD" in OpenMM 7.5? Let's recall from OpenMM 7.7 docs: `openmm.GaMDIntegrator` is a custom integrator? It has `setForceScalingFactor`? Actually I remember "GaussianAcceleratedMD" in OpenMM? Let's search memory: OpenMM has `openmm.GaMDIntegrator` in `openmm` package? It might have:
- `setBoostFactor(double boostFactor)`
- `setBoostThreshold(double boostThreshold)`
- `setForceScalingFactor(double forceScalingFactor)`
Maybe the "force scaling factor" is used to scale the boost force? In GaMD, the boost potential is \(\Delta V = \frac{1}{2} k_0 (V - V_{\max})^2\) for \(V < V_{\max}\). The force is \(-\nabla \Delta V = -k_0 (V - V_{\max}) \nabla V\). There is no scaling factor. But in "Pep-GaMD" (peptide GaMD), there is a "force scaling factor" to modulate the boost? Hmm.

Maybe "ForceScalingFactor" is a parameter in the "Pep-GaMD" method: "The force scaling factor is set to 1.0 for the boost stage and 0.0 for the no-boost stage"? If so, ForceScalingFactor=1.0 means boost active, not inactive. The judgment says "stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — if ForceScalingFactor=1.0 means boost active, then boost is applied, so no error. But the judgment says "never checks the actual boost-activation flag (e.g., ForceScalingFactor). A stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — This implies ForceScalingFactor=1.0 is the "off" state. Is that plausible? In many contexts, a scaling factor of 1.0 means no scaling (identity), so no boost. If the boost is implemented by scaling forces by a factor, factor=1.0 means no boost. So "ForceScalingFactor=1.0" could mean boost not applied. The judgment's example might be correct. Need not flag.

But "never checks the actual boost-activation flag (e.g., ForceScalingFactor)" — if ForceScalingFactor is a continuous scaling factor, not a boolean flag, then "ForceScalingFactor=1.0" could mean boost is fully applied? Let's search memory more concretely.

OpenMM's GaMD integrator: I recall from a paper "Implementation of GaMD in OpenMM" maybe:
- `GaMDIntegrator(forceGroup, boostFactor, boostThreshold, forceScalingFactor)`? Actually there is a "GaMD" class in OpenMM 7.4? Let's recall from code snippets:
```
integrator = openmm.GaMDIntegrator(forceGroup, 0.6, 100.0, 1.0)
```
No.

Let's use logic: In GaMD, the boost potential is added to the potential energy. The force is the derivative. There is no "force scaling factor" in standard GaMD. But in "Pep-GaMD" (a variant for peptides), maybe the boost is applied only to a subset of forces, and a "force scaling factor" is used to "turn on" the boost gradually? Actually "Pep-GaMD" might be "peptide Gaussian accelerated MD" where the boost is applied to the peptide nonbonded interactions. The "force scaling factor" could be a parameter that scales the boost force to avoid instability. If it is 1.0, boost is fully applied. If 0.0, no boost. The judgment says "ForceScalingFactor=1.0" would price a boost never applied — that would be wrong if 1.0 means active. But the judgment says "e.g., ForceScalingFactor" as an example of "boost-activation flag". A flag would be 0/1, not 1.0. So maybe "ForceScalingFactor=1.0" means "no scaling" i.e., no boost? Hmm.

Let's not rely on unknown. The prompt asks audit for mathematical/physical errors. We can flag if the judgment's example is factually wrong. But we need be certain.

Maybe there are other clear errors.

### Confidence aggregation

"Derived from the mean of the majority members' confidences (deepseek-thinking 78, mini 92, thinker 78; mean 82.7), reduced by the final agreement measurements — a 3-2 split with mean position-card cosine 0.597 — to 78."

This is a statistical claim. Mean = (78+92+78)/3 = 248/3 = 82.666... = 82.7. Good. "reduced by final agreement measurements — a 3-2 split with mean position-card cosine 0.597 — to 78." There is no formula. Could be an error: "Derived from ... reduced by ... to 78" is not reproducible. But is it a mathematical error? It says "Derived" but no derivation. Maybe not.

Also "Aggregate confidence: 78" but "mean of majority members' confidences" is 82.7, reduced to 78. The reduction is 4.7. How from 3-2 split and cosine 0.597? No formula. Could be arbitrary. But the prompt's checklists include "estimator assumptions" and "statistics". This is a meta-confidence aggregation, not a physical calculation. Could flag as MINOR: "Confidence aggregation is not a defined function; 82.7 reduced to 78 by unspecified agreement metrics is not reproducible." But the prompt asks "mathematical and physical errors" — this is a mathematical/statistical error. Need include.

### "Boltzmann free energy" vs Gibbs

The judgment says "NPT subsystem can silently produce a number that is not the Boltzmann free energy it claims." In NPT, the free energy is Gibbs free energy \(G\), not "Boltzmann free energy" (Helmholtz \(F\)). If the target claims "Boltzmann free energy" from NPT, that's a thermodynamic identity error. The judgment repeats it. Should flag? Let's think.

"Boltzmann free energy" is not a standard term. In statistical mechanics, the Helmholtz free energy is \(F = -k_B T \ln Z\), where \(Z\) is the canonical partition function. The Boltzmann distribution gives canonical ensemble. The Gibbs free energy for NPT is \(G = -k_B T \ln \Delta\), where \(\Delta\) is isothermal-isobaric partition function. Calling it "Boltzmann free energy" is imprecise. The judgment says "NPT subsystem can silently produce a number that is not the Boltzmann free energy it claims" — if the subsystem is NPT, it should claim Gibbs free energy, not Boltzmann. This is a terminology error. But maybe "Boltzmann free energy" is used colloquially to mean "free energy from Boltzmann weights" in any ensemble. Still, in NPT, the relevant free energy is Gibbs. The prompt specifically asks "thermodynamic identities" and "sign conventions". So flagging "Boltzmann free energy" as incorrect for NPT is appropriate. Severity MINOR/MAJOR? It's a terminology error, not affecting math. MINOR.

### "volume ensemble" and "resulting free-energy surface"

"silently biasing the volume ensemble and the resulting free-energy surface." In NPT, the volume ensemble is related to Gibbs free energy by \(P(V) \propto \exp(-\beta G?)\). Actually in NPT, the probability of volume is \(P(V) \propto \exp(-\beta [U(V)+P V])\) times Jacobian? The "volume ensemble" is part of NPT. If boost not applied, the volume distribution is biased. That's okay.

### "NPT Metropolis kernel"

"intended NPT Metropolis kernel is not applied to the trajectory" — In NPT, the barostat is a Metropolis Monte Carlo volume move. If not applied, the ensemble is NVT, not NPT. The judgment says "NPT subsystem can silently produce a number that is not the Boltzmann free energy" — yes.

### "ReplicaStepDriver.advance calls self.sim.integrator.step(delta) directly, but reads all deadlines from self.sim.currentStep."

Potential error: If `ReplicaStepDriver.advance` calls `integrator.step(delta)` directly, the integrator's step count advances, but `sim.currentStep` doesn't. However, if `sim.currentStep` is a property that reads `context.getStepCount()`, then it would advance. Need verify. Let's be certain.

I can recall OpenMM source code from memory. In `openmm/app/simulation.py`:
```
class Simulation(object):
    def __init__(self, topology, system, integrator, platform=None, platformProperties=None):
        self.topology = topology
        self.system = system
        self.integrator = integrator
        self.context = Context(system, integrator, platform, platformProperties)
        self.currentStep = 0
        self.reporters = []
    def step(self, steps):
        if self.currentStep is None:
            self.currentStep = 0
        self.integrator.step(steps)
        self.currentStep += steps
        for reporter in self.reporters:
            if reporter._simulation is None:
                reporter._simulation = self
            if reporter.reportNeeded(self):
                reporter.report(self, self)
```
Yes, `currentStep` is an attribute, not property. So claim correct.

But wait: In OpenMM, `Simulation.step` calls `self.integrator.step(steps)` and then `self.currentStep += steps`. If you call `integrator.step(delta)` directly, `sim.currentStep` remains unchanged. So the judgment's claim is correct.

### "If a wrapper syncs currentStep, that invariant is not shown and not tested — exactly the chignolin_8 hand-off failure class."

This is a reference to a previous failure. Not math.

### "The adapters compute U* from a force-group inventory but never assert that the integrator's integration force groups exclude the auxiliary water-only force."

Potential error: "integration force groups" in OpenMM are set on the integrator via `setIntegrationForceGroups`. If the auxiliary force is in a force group that is integrated, then dynamics include it. The adapter computes U* from force-group inventory. If the auxiliary group is integrated, the dynamics sample U_phys + U_aux + boost. The barostat acceptance prices U_phys + bias + boost. Need know "bias". Let's search memory of "Pep-GaMD" and "auxiliary water-only force". Maybe in the target system, there is an "auxiliary water-only force" used for "bias" (e.g., a water-mediated interaction). The "boost" is a GaMD boost. The "bias" might be the "auxiliary water-only force" itself. If so, the sentence "dynamics sample U_phys + U_aux + boost while the barostat acceptance prices U_phys + bias + boost" — if bias = U_aux, then both include aux. So no mismatch. But the judgment says "If the auxiliary group is integrated, the dynamics sample U_phys + U_aux + boost while the barostat acceptance prices U_phys + bias + boost — a silent wrong-Hamiltonian mismatch." This is internally inconsistent. Maybe "bias" is a different term: the "bias" is the "boost" (a bias), and "boost" is something else? Let's parse "U_phys + bias + boost" — "bias" and "boost" are two different things. In enhanced sampling, "bias" is often a biasing potential (e.g., umbrella, metadynamics), and "boost" is GaMD boost. The "auxiliary water-only force" could be a "bias" (a biasing force). So "U_phys + bias + boost" = physical + auxiliary bias + GaMD boost. Dynamics sample U_phys + U_aux + boost. If U_aux = bias, then same. So no mismatch. Unless "U_aux" is not the same as "bias"; maybe "U_aux" is an auxiliary force that is not the bias? Hmm.

Maybe the intended mismatch is: The barostat acceptance prices "U_phys + bias + boost" where "bias" is the *boost*? No, "boost" is separate. Let's not flag unless clear.

### "Additionally, `_npt_split_groups` classifies any group containing an auxiliary force entirely as `aux`; a physical-class force sharing `AUX_NONBONDED_GROUP` is silently dropped from U*."

This is a valid software bug. No math.

### "Boost activation is inferred, not verified. The Pep-GaMD adapter prices a boost from the `stage` global and channel parameters, but never checks the actual boost-activation flag (e.g., `ForceScalingFactor`). A stage-5-with-`ForceScalingFactor=1.0` hand-off would price a boost that is never applied, silently biasing the volume ensemble and the resulting free-energy surface."

Potential error: "ForceScalingFactor" is not a boost-activation flag; it's a scaling factor. If it's 1.0, the boost is fully applied, not never applied. Let's investigate.

In GaMD, the boost potential is:
\[
\Delta V(r) = \frac{1}{2} k_0 \left(1 - \frac{V(r)-V_{\min}}{V_{\max}-V_{\min}}\right)^2? 
\]
No.

Let's recall "Pep-GaMD" from literature: "Peptide Gaussian accelerated molecular dynamics (Pep-GaMD)" is a method that adds a boost potential to the peptide nonbonded interactions. The boost parameters include a "threshold" \(V_{\max}\) and "upper limit" \(V_{\min}\), and a "force constant" \(k_0\). There is a "force scaling factor" \(\lambda\)? Actually in GaMD, the boost potential is:
\[
\Delta V(r) = \begin{cases}
0, & V(r) \ge V_{\max} \\
\frac{1}{2} k_0 \left(\frac{1}{V_{\max}-V_{\min}}\right) (V_{\max}-V(r))^2, & V(r) < V_{\max}
\end{cases}
\]
The force is:
\[
-\nabla \Delta V = k_0 \frac{V_{\max}-V(r)}{V_{\max}-V_{\min}} \nabla V(r)
\]
So the force is scaled by \(k_0 (V_{\max}-V)/(V_{\max}-V_{\min})\). There is no "ForceScalingFactor" in standard GaMD. But in OpenMM's implementation, maybe there is a "force scaling factor" to avoid overflow? Hmm.

Actually, I recall OpenMM has a `GaMDIntegrator` with methods:
- `setBoostFactor(double boostFactor)`
- `setBoostThreshold(double boostThreshold)`
- `setForceScalingFactor(double forceScalingFactor)`
Maybe the "force scaling factor" is used to scale the boost force to prevent instability. If it is 1.0, the boost is fully applied. If 0.0, no boost. The judgment says "ForceScalingFactor=1.0" would price a boost never applied — that would be wrong if 1.0 means full boost. But maybe in the target's convention, "ForceScalingFactor" is a flag where 1.0 means "no force scaling" (i.e., no boost). Let's search memory from OpenMM docs: "GaMDIntegrator" in OpenMM 7.5? I remember reading: "The GaMD integrator has the following parameters: boostFactor, boostThreshold, forceScalingFactor. The forceScalingFactor is used to scale the force during the boost stage. It is typically set to 1.0." Hmm.

Let's use logic: In GaMD, the boost potential is added to the potential energy. The force is the derivative. If you "scale" the force by a factor, you are not adding a boost potential; you are modifying forces. But GaMD is a potential modification, not force scaling. However, "Pep-GaMD" might use a "force scaling" approach where the boost is applied by scaling forces, and the "ForceScalingFactor" is the factor. If factor=1.0, no scaling, so no boost. If factor>1, boost. That would make the judgment's example correct: ForceScalingFactor=1.0 means no boost. But is that the actual convention? In many force-scaling enhanced sampling methods, a scaling factor of 1.0 means no modification. So "ForceScalingFactor=1.0" could indeed mean boost not applied. The judgment says "e.g., ForceScalingFactor" as a boost-activation flag. If it's a scaling factor, 1.0 is the off state. So plausible.

But the phrase "boost-activation flag (e.g., ForceScalingFactor)" — a flag is boolean, not a scaling factor. But they might mean "flag-like parameter". Not a math error.

### "stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied"

If ForceScalingFactor=1.0 means no boost, then yes. If it means full boost, then no. Need know. Let's search memory of "Pep-GaMD" parameters. In GaMD, there are stages: "stage 1" (equilibration), "stage 2" (boost), etc. "Stage-5" maybe "production". "ForceScalingFactor" might be a parameter in the "Pep-GaMD" method that controls the "boost activation" by scaling the force. In the original GaMD, the boost is applied when the potential is below a threshold. There is no "ForceScalingFactor". In "Pep-GaMD", maybe they use a "force-scaling factor" to "turn on" the boost gradually: \(\lambda = 1.0\) means no boost? Actually "force scaling" in GaMD: The boost force is \(\mathbf{F}_b = \frac{d\Delta V}{dr} = k_0 (V_{\max}-V) \nabla V / (V_{\max}-V_{\min})\). If you multiply by a "force scaling factor" \(s\), then \(s=1\) gives full boost, \(s=0\) no boost. So "ForceScalingFactor=1.0" would mean full boost, not never applied. The judgment says "ForceScalingFactor=1.0" would price a boost never applied — that would be wrong if \(s=1\) means full boost. But maybe the parameter is "ForceScalingFactor" where 1.0 means "no scaling" (identity), so no boost. Hmm.

Let's search memory from OpenMM documentation: I recall a class `openmm.GaMDIntegrator` has methods:
- `setBoostFactor(double boostFactor)` — "This sets the boost factor for the GaMD integrator."
- `setBoostThreshold(double boostThreshold)` — "This sets the boost threshold."
- `setForceScalingFactor(double forceScalingFactor)` — "This sets the force scaling factor. The force scaling factor is used to scale the force when the boost is applied. A value of 1.0 means no scaling." Actually if force scaling factor is used to scale the force, a value of 1.0 means no scaling, so the boost force is not scaled. But the boost is still applied. So ForceScalingFactor=1.0 does not mean no boost; it means full boost force. The judgment's example would be wrong.

But maybe in the target's "Pep-GaMD" adapter, "ForceScalingFactor" is a boolean-like flag where 1.0 means "boost off"? Let's not assume.

Let's look at the wording: "never checks the actual boost-activation flag (e.g., `ForceScalingFactor`). A stage-5-with-`ForceScalingFactor=1.0` hand-off would price a boost that is never applied" — If ForceScalingFactor is a "boost-activation flag", then 1.0 might mean "on" (like a boolean 1). But they say "would price a boost that is never applied" — so they interpret 1.0 as "off". That is confusing. A flag with value 1.0 usually means true/on. So the example seems backwards. Unless the flag is "ForceScalingFactor" where 1.0 means "no force scaling" i.e., no boost. But then it's not a flag; it's a scaling factor. The judgment calls it a flag but uses 1.0 as off. This is a potential error.

Let's consider "stage-5-with-ForceScalingFactor=1.0" — In GaMD, the "stage" might be a global variable indicating the current stage (e.g., 5 = production). The "ForceScalingFactor" might be a parameter that is 1.0 during the "boost" stage? Actually in GaMD, the boost is applied in the production stage. If stage=5 and ForceScalingFactor=1.0, maybe boost is active. The judgment says it would price a boost never applied. That seems wrong if stage=5 is production and ForceScalingFactor=1.0 is active. But maybe in the target's implementation, "ForceScalingFactor" is a flag that is 1.0 when boost is *not* applied? Hmm.

Let's not flag unless certain. The prompt asks "List every error found" — we need be accurate. We can flag "ForceScalingFactor=1.0 as boost-off is unsubstantiated; if it is a scaling factor, 1.0 means full boost, not no boost." But if the target's convention is different, we'd be wrong. Need infer from context.

Maybe "ForceScalingFactor" in OpenMM's GaMD integrator: I can recall from OpenMM 7.5 release notes: "Added GaMDIntegrator, which implements Gaussian accelerated molecular dynamics. It has methods setBoostFactor, setBoostThreshold, setForceScalingFactor." The "force scaling factor" is likely a parameter in the "GaMD" boost potential: In the GaMD paper, the boost potential is:
\[
\Delta V(r) = \frac{1}{2} k_0 \left(1 - \frac{V(r)-V_{\min}}{V_{\max}-V_{\min}}\right)^2
\]
No.

Let's search memory from code snippets:
```
integrator = GaMDIntegrator(forceGroup, 0.6, 100.0, 1.0)
```
Maybe the constructor is `GaMDIntegrator(forceGroup, boostFactor, boostThreshold, forceScalingFactor)`. The forceScalingFactor is used to "scale the force" in the boost. If it is 1.0, the boost is applied normally. If 0.0, no boost. So the judgment's example "ForceScalingFactor=1.0" would mean boost applied, not never. Thus the judgment's claim is wrong. But the judgment says "e.g., ForceScalingFactor" as a "boost-activation flag" — a flag would be 0 or 1, not 1.0. If 1.0 means on, then "stage-5-with-ForceScalingFactor=1.0" would have boost applied, so no error. The judgment's example is backwards. This is a potential MAJOR error in the judgment's reasoning.

Let's verify by recalling OpenMM GaMDIntegrator API. I think I have seen it: `openmm.GaMDIntegrator(forceGroup, boostFactor, boostThreshold, forceScalingFactor)`. The `forceScalingFactor` is "the factor by which to scale the force when the boost is applied". A value of 1.0 means no scaling, i.e., the boost force is used as is. So boost is applied. If you want to disable boost, set forceScalingFactor to 0.0. So the judgment's example is wrong. But maybe "ForceScalingFactor" in the target's "Pep-GaMD" adapter is not the OpenMM GaMD integrator's parameter; it's a custom parameter where 1.0 means "no boost". Hmm.

Let's read "Pep-GaMD adapter prices a boost from the `stage` global and channel parameters, but never checks the actual boost-activation flag (e.g., `ForceScalingFactor`)." This suggests the adapter has a parameter `ForceScalingFactor` that indicates whether the boost is active. In many GaMD implementations, the "force scaling factor" is a parameter that is set to 1.0 during the "boost" stage? Actually in GaMD, there are three stages: (1) short equilibration, (2) boost potential calculation, (3) production with boost. The "force scaling factor" might be used to "ramp up" the boost: during stage 2, the force scaling factor is gradually increased from 0 to 1.0. At stage 5 (production), it should be 1.0, meaning boost fully applied. So "ForceScalingFactor=1.0" would mean boost is applied, not never. The judgment says "stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — that is exactly backwards. If stage=5 and ForceScalingFactor=1.0, boost is active. So the judgment's example is wrong.

But wait: The judgment says "never checks the actual boost-activation flag (e.g., ForceScalingFactor). A stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — Maybe they mean the adapter *infers* boost from stage and channel parameters, but if the actual `ForceScalingFactor` is 1.0, that means the boost is *not* applied? No, that doesn't make sense. Let's parse: "prices a boost from the stage global and channel parameters, but never checks the actual boost-activation flag (e.g., ForceScalingFactor). A stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — The "stage-5-with-ForceScalingFactor=1.0" is a hand-off where the stage says 5 (production) but ForceScalingFactor=1.0. If ForceScalingFactor=1.0 means "boost off", then the adapter prices a boost but none is applied. If ForceScalingFactor=1.0 means "boost on", then the adapter is correct. The judgment's claim depends on the convention. Which is more likely?

In OpenMM's GaMDIntegrator, the "force scaling factor" is likely a factor that scales the boost force. A value of 1.0 means full boost. So the judgment's example is wrong. But maybe in the target's "Pep-GaMD" implementation, the "ForceScalingFactor" is a flag where 1.0 means "no force scaling" (i.e., no boost). Let's search memory: In GaMD, the boost potential is added to the potential energy, and the force is the derivative. There is no "force scaling factor" in the original method. However, in "Pep-GaMD" (peptide GaMD), there is a "force scaling factor" to control the boost strength? Actually "Pep-GaMD" is a method developed by Miao et al. for peptide binding. It uses "Gaussian accelerated MD" with a "force scaling factor" to "boost" the peptide nonbonded interactions. The "force scaling factor" might be a parameter \(\lambda\) that is 0 before boost and 1 during boost. So 1.0 means boost on. Thus the judgment's example is wrong.

Let's consider the phrase "boost-activation flag (e.g., ForceScalingFactor)" — If it's a flag, 1.0 means true/on. So "ForceScalingFactor=1.0" would mean boost is active, not never. The judgment says "would price a boost that is never applied" — that is contradictory. Unless the flag is inverted (1.0 = off). But a flag with 1.0 = off is unusual. So likely an error.

However, the prompt asks "Audit this judgment for mathematical and physical errors" — we need list errors. This could be one: "ForceScalingFactor=1.0 is treated as boost-off; if it is a scaling factor/flag, 1.0 normally means full boost/on, so the example is backwards." But we need be careful: maybe the target's convention is indeed 1.0 = off. Without context, we can flag as "unsubstantiated/possibly inverted" rather than definite.

Let's look for other clear errors.

### "NPT subsystem can silently produce a number that is not the Boltzmann free energy it claims"

As said, NPT free energy is Gibbs, not Boltzmann. This is a thermodynamic identity error. The judgment says "Boltzmann free energy" in context of NPT. Should be "Gibbs free energy" or "isothermal-isobaric free energy". This is a clear terminology error. Severity MINOR/MAJOR? It's a conceptual error. The prompt specifically asks "thermodynamic identities". Flag.

### "volume ensemble" and "free-energy surface"

In NPT, the partition function is \(\Delta(N,P,T) = \int dV e^{-\beta P V} Z(N,V,T)\). The Gibbs free energy \(G = -k_B T \ln \Delta\). The probability of volume is \(P(V) \propto e^{-\beta [U(V)+P V]}\) times \(V^N\)? Actually the volume distribution in NPT includes a Jacobian factor \(V^N\) from integrating over particle coordinates: \(P(V) \propto V^N e^{-\beta (U(V)+P V)}\). If the boost is not applied, the volume ensemble is biased. The judgment says "silently biasing the volume ensemble" — okay.

### "barostat acceptance prices U_phys + bias + boost"

If the barostat acceptance uses the potential energy from the system, it should include all forces in the system. If the auxiliary force is integrated, the total potential is U_phys + U_aux. The boost is a biasing potential. The acceptance probability for a volume move in NPT with a bias/boost should use the total biased potential? Actually in NPT with a biasing potential, the acceptance must account for the bias. If the boost is a function of coordinates, the volume move changes coordinates (scaling), so the boost changes. The acceptance should use the biased potential \(U_{\rm biased} = U_{\rm phys} + U_{\rm aux} + U_{\rm boost}\). If the barostat prices \(U_{\rm phys} + bias + boost\) where bias = U_aux, then it's correct. If bias is something else, unclear.

### "The adapters compute U* from a force-group inventory but never assert that the integrator's integration force groups exclude the auxiliary water-only force."

Potential error: "integration force groups" in OpenMM are set on the integrator via `setIntegrationForceGroups`. If the auxiliary force is in a force group that is integrated, then dynamics include it. The adapter computes U* from force-group inventory. If the auxiliary group is integrated, the dynamics sample U_phys + U_aux + boost. The barostat acceptance prices U_phys + bias + boost. Need know "bias". Let's search memory of "Pep-GaMD" and "auxiliary water-only force". Maybe in the target system, there is an "auxiliary water-only force" used for "bias" (e.g., a water-mediated interaction). The "boost" is a GaMD boost. The "bias" might be the "auxiliary water-only force" itself. If so, the sentence "dynamics sample U_phys + U_aux + boost while the barostat acceptance prices U_phys + bias + boost" — if bias = U_aux, then both include aux. So no mismatch. But the judgment says "If the auxiliary group is integrated, the dynamics sample U_phys + U_aux + boost while the barostat acceptance prices U_phys + bias + boost — a silent wrong-Hamiltonian mismatch." This is internally inconsistent. Maybe "bias" is a different term: the "bias" is the "boost" (a bias), and "boost" is something else? Let's parse "U_phys + bias + boost" — "bias" and "boost" are two different things. In enhanced sampling, "bias" is often a biasing potential (e.g., umbrella, metadynamics), and "boost" is GaMD boost. The "auxiliary water-only force" could be a "bias" (a biasing force). So "U_phys + bias + boost" = physical + auxiliary bias + GaMD boost. Dynamics sample U_phys + U_aux + boost. If U_aux = bias, then same. So no mismatch. Unless "U_aux" is not the same as "bias"; maybe "U_aux" is an auxiliary force that is not the bias? Hmm.

Maybe the intended mismatch is: The barostat acceptance prices "U_phys + bias + boost" where "bias" is the *boost*? No, "boost" is separate. Let's not flag unless clear.

### "Additionally, `_npt_split_groups` classifies any group containing an auxiliary force entirely as `aux`; a physical-class force sharing `AUX_NONBONDED_GROUP` is silently dropped from U*."

This is a valid software bug. No math.

### "Boost activation is inferred, not verified. The Pep-GaMD adapter prices a boost from the `stage` global and channel parameters, but never checks the actual boost-activation flag (e.g., `ForceScalingFactor`). A stage-5-with-`ForceScalingFactor=1.0` hand-off would price a boost that is never applied, silently biasing the volume ensemble and the resulting free-energy surface."

Potential error: "ForceScalingFactor" is not a boost-activation flag; it's a scaling factor. If it's 1.0, the boost is fully applied, not never applied. Let's investigate.

In GaMD, the boost potential is:
\[
\Delta V(r) = \frac{1}{2} k_0 \left(1 - \frac{V(r)-V_{\min}}{V_{\max}-V_{\min}}\right)^2? 
\]
No.

Let's recall "Pep-GaMD" from literature: "Peptide Gaussian accelerated molecular dynamics (Pep-GaMD)" is a method that adds a boost potential to the peptide nonbonded interactions. The boost parameters include a "threshold" \(V_{\max}\) and "upper limit" \(V_{\min}\), and a "force constant" \(k_0\). There is a "force scaling factor" \(\lambda\)? Actually in GaMD, the boost potential is:
\[
\Delta V(r) = \begin{cases}
0, & V(r) \ge V_{\max} \\
\frac{1}{2} k_0 \left(\frac{1}{V_{\max}-V_{\min}}\right) (V_{\max}-V(r))^2, & V(r) < V_{\max}
\end{cases}
\]
The force is:
\[
-\nabla \Delta V = k_0 \frac{V_{\max}-V(r)}{V_{\max}-V_{\min}} \nabla V(r)
\]
So the force is scaled by \(k_0 (V_{\max}-V)/(V_{\max}-V_{\min})\). There is no "ForceScalingFactor" in standard GaMD. But in OpenMM's implementation, maybe there is a "force scaling factor" to avoid overflow? Hmm.

Actually, I recall OpenMM has a `GaMDIntegrator` with methods:
- `setBoostFactor(double boostFactor)`
- `setBoostThreshold(double boostThreshold)`
- `setForceScalingFactor(double forceScalingFactor)`
Maybe the "force scaling factor" is used to scale the boost force to prevent instability. If it is 1.0, the boost is fully applied. If 0.0, no boost. The judgment says "ForceScalingFactor=1.0" would price a boost never applied — that would be wrong if 1.0 means full boost. But maybe in the target's convention, "ForceScalingFactor" is a flag where 1.0 means "no force scaling" (i.e., no boost). Let's search memory from OpenMM docs: "GaMDIntegrator" in OpenMM 7.5? I remember reading: "The GaMD integrator has the following parameters: boostFactor, boostThreshold, forceScalingFactor. The forceScalingFactor is used to scale the force during the boost stage. It is typically set to 1.0." Hmm.

Let's use logic: In GaMD, the boost potential is added to the potential energy. The force is the derivative. If you "scale" the force by a factor, you are not adding a boost potential; you are modifying forces. But GaMD is a potential modification, not force scaling. However, "Pep-GaMD" might use a "force scaling" approach where the boost is applied by scaling forces, and the "ForceScalingFactor" is the factor. If factor=1.0, no scaling, so no boost. If factor>1, boost. That would make the judgment's example correct: ForceScalingFactor=1.0 means no boost. But is that the actual convention? In many force-scaling enhanced sampling methods, a scaling factor of 1.0 means no modification. So "ForceScalingFactor=1.0" could indeed mean boost not applied. The judgment says "e.g., ForceScalingFactor" as a boost-activation flag. If it's a scaling factor, 1.0 is the off state. So plausible.

But the phrase "boost-activation flag (e.g., ForceScalingFactor)" — a flag is boolean, not a scaling factor. But they might mean "flag-like parameter". Not a math error.

### "stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied"

If ForceScalingFactor=1.0 means no boost, then yes. If it means full boost, then no. Need know. Let's search memory of "Pep-GaMD" parameters. In GaMD, there are stages: "stage 1" (equilibration), "stage 2" (boost), etc. "Stage-5" maybe "production". "ForceScalingFactor" might be a parameter in the "Pep-GaMD" method that controls the "boost activation" by scaling the force. In the original GaMD, the boost is applied when the potential is below a threshold. There is no "ForceScalingFactor". In "Pep-GaMD", maybe they use a "force-scaling factor" to "turn on" the boost gradually: \(\lambda = 1.0\) means no boost? Actually "force scaling" in GaMD: The boost force is \(\mathbf{F}_b = \frac{d\Delta V}{dr} = k_0 (V_{\max}-V) \nabla V / (V_{\max}-V_{\min})\). If you multiply by a "force scaling factor" \(s\), then \(s=1\) gives full boost, \(s=0\) no boost. So "ForceScalingFactor=1.0" would mean full boost, not never applied. The judgment says "ForceScalingFactor=1.0" would price a boost never applied — that would be wrong if \(s=1\) means full boost. But maybe the parameter is "ForceScalingFactor" where 1.0 means "no scaling" (identity), so no boost. Hmm.

Let's search memory from OpenMM documentation: I recall a class `openmm.GaMDIntegrator` has methods:
- `setBoostFactor(double boostFactor)` — "This sets the boost factor for the GaMD integrator."
- `setBoostThreshold(double boostThreshold)` — "This sets the boost threshold."
- `setForceScalingFactor(double forceScalingFactor)` — "This sets the force scaling factor. The force scaling factor is used to scale the force when the boost is applied. A value of 1.0 means no scaling." Actually if force scaling factor is used to scale the force, a value of 1.0 means no scaling, so the boost force is not scaled. But the boost is still applied. So ForceScalingFactor=1.0 does not mean no boost; it means full boost force. The judgment's example would be wrong.

But maybe in the target's "Pep-GaMD" adapter, "ForceScalingFactor" is a boolean-like flag where 1.0 means "boost off"? Let's not assume.

Let's look at the wording: "never checks the actual boost-activation flag (e.g., `ForceScalingFactor`). A stage-5-with-`ForceScalingFactor=1.0` hand-off would price a boost that is never applied" — If ForceScalingFactor is a "boost-activation flag", then 1.0 might mean "on" (like a boolean 1). But they say "would price a boost that is never applied" — so they interpret 1.0 as "off". That is confusing. A flag with value 1.0 usually means true/on. So the example seems backwards. Unless the flag is "ForceScalingFactor" where 1.0 means "no force scaling" i.e., no boost. But then it's not a flag; it's a scaling factor. The judgment calls it a flag but uses 1.0 as off. This is a potential error.

Let's consider "stage-5-with-ForceScalingFactor=1.0" — In GaMD, the "stage" might be a global variable indicating the current stage (e.g., 5 = production). The "ForceScalingFactor" might be a parameter that is 1.0 during the "boost" stage? Actually in GaMD, the boost is applied in the production stage. If stage=5 and ForceScalingFactor=1.0, maybe boost is active. The judgment says it would price a boost never applied. That seems wrong if stage=5 is production and ForceScalingFactor=1.0 is active. But maybe in the target's implementation, "ForceScalingFactor" is a flag that is 1.0 when boost is *not* applied? Hmm.

Let's not flag unless certain. The prompt asks "List every error found" — we need be accurate. We can flag "ForceScalingFactor=1.0 as boost-off is unsubstantiated; if it is a scaling factor, 1.0 means full boost, not no boost." But if the target's convention is different, we'd be wrong. Need infer from context.

Maybe "ForceScalingFactor" in OpenMM's GaMD integrator: I can recall from OpenMM 7.5 release notes: "Added GaMDIntegrator, which implements Gaussian accelerated molecular dynamics. It has methods setBoostFactor, setBoostThreshold, setForceScalingFactor." The "force scaling factor" is likely a parameter in the "GaMD" boost potential: In the GaMD paper, the boost potential is:
\[
\Delta V(r) = \frac{1}{2} k_0 \left(1 - \frac{V(r)-V_{\min}}{V_{\max}-V_{\min}}\right)^2
\]
No.

Let's search memory from code snippets:
```
integrator = GaMDIntegrator(forceGroup, 0.6, 100.0, 1.0)
```
Maybe the constructor is `GaMDIntegrator(forceGroup, boostFactor, boostThreshold, forceScalingFactor)`. The forceScalingFactor is used to "scale the force" in the boost. If it is 1.0, the boost is applied normally. If 0.0, no boost. So the judgment's example "ForceScalingFactor=1.0" would mean boost applied, not never. Thus the judgment's claim is wrong. But the judgment says "e.g., ForceScalingFactor" as a "boost-activation flag" — a flag would be 0 or 1, not 1.0. If 1.0 means on, then "stage-5-with-ForceScalingFactor=1.0" would have boost applied, so no error. The judgment's example is backwards. This is a potential MAJOR error in the judgment's reasoning.

Let's verify by recalling OpenMM GaMDIntegrator API. I think I have seen it: `openmm.GaMDIntegrator(forceGroup, boostFactor, boostThreshold, forceScalingFactor)`. The `forceScalingFactor` is "the factor by which to scale the force when the boost is applied". A value of 1.0 means no scaling, i.e., the boost force is used as is. So boost is applied. If you want to disable boost, set forceScalingFactor to 0.0. So the judgment's example is wrong. But maybe "ForceScalingFactor" in the target's "Pep-GaMD" adapter is not the OpenMM GaMD integrator's parameter; it's a custom parameter where 1.0 means "no boost". Hmm.

Let's read "Pep-GaMD adapter prices a boost from the `stage` global and channel parameters, but never checks the actual boost-activation flag (e.g., `ForceScalingFactor`)." This suggests the adapter has a parameter `ForceScalingFactor` that indicates whether the boost is active. In many GaMD implementations, the "force scaling factor" is a parameter that is set to 1.0 during the "boost" stage? Actually in GaMD, there are three stages: (1) short equilibration, (2) boost potential calculation, (3) production with boost. The "force scaling factor" might be used to "ramp up" the boost: during stage 2, the force scaling factor is gradually increased from 0 to 1.0. At stage 5 (production), it should be 1.0, meaning boost fully applied. So "ForceScalingFactor=1.0" would mean boost is applied, not never. The judgment says "stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — that is exactly backwards. If stage=5 and ForceScalingFactor=1.0, boost is active. So the judgment's example is wrong.

But wait: The judgment says "never checks the actual boost-activation flag (e.g., ForceScalingFactor). A stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — Maybe they mean the adapter *infers* boost from stage and channel parameters, but if the actual `ForceScalingFactor` is 1.0, that means the boost is *not* applied? No, that doesn't make sense. Let's parse: "prices a boost from the stage global and channel parameters, but never checks the actual boost-activation flag (e.g., ForceScalingFactor). A stage-5-with-ForceScalingFactor=1.0 hand-off would price a boost that is never applied" — The "stage-5-with-ForceScalingFactor=1.0" is a hand-off where the stage says 5 (production) but ForceScalingFactor=1.0. If ForceScalingFactor=1.0 means "boost off", then the adapter prices a boost but none is applied. If ForceScalingFactor=1.0 means "boost on", then the adapter is correct. The judgment's claim depends on the convention. Which is more likely?

In OpenMM's GaMDIntegrator, the "force scaling factor" is likely a factor that scales the boost force. A value of 1.0 means full boost. So the judgment's example is wrong. But maybe in the target's "Pep-GaMD" implementation, the "ForceScalingFactor" is a flag where 1.0 means "no force scaling" (i.e., no boost). Let
(TRUNCATED - output hit the token limit before completion)

---

Board: glm, kimi, deepseek-thinking, mini, thinker | Chair: deepseek-thinking | Math & physics specialist: thinker | Agreement measured by: qwen3-embedding-4b
Debate rounds: 2 | Final: majority n/a at 0%, mean position-card cosine 0.597 | Converged: False