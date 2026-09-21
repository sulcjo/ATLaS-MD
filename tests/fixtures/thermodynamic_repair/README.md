# Thermodynamic-repair regression fixtures

Deterministic builders for the fixtures named in the implementation plan (P00). Each
builder documents its seed, units, array ordering and the value it reproduces at the
reviewed baseline. Import as `from fixtures.thermodynamic_repair import <name>` (the test
runner prepends `tests/` to `sys.path`).

| Fixture | Physical / statistical purpose | Baseline behaviour |
|---|---|---|
| `residual_fast_path_case()` | 16-particle synthetic system, degree-1 residual model with an anchor polynomial, two phi + two psi quadruplets, four contact pairs; exercises force ↔ positions ↔ fast-scalar agreement | true z 0.8396162908, faulty z 0.0676117650; umbrella 1.4559287064 vs 0.2700214588 kJ/mol at k = 10, centre 0.3 |
| `separated_support_case()` | 10,000 values in [0.1, 0.3] + 200 in [0.8, 0.82], all seeds | upper cluster lost by endpoint tuning |
| `cap_transition_case()` | 6×4 spatial grid, four rungs, caps 92 and 96 | unrestrained state present at 92, absent at 96 |
| `solvent_collision_case()` | peptide x = 0, 0.5 nm; water O at 0.05, 0.45 nm (TIP3P geometry) | old displacement makes the waters coincident |
| `oscillating_displacement_case()` | peptide x = 0, 0.3 nm; solvent atom at 0.15 nm | old solver returns unresolved without status |
| `dependent_dual_boost_case()` | envelope (20,−20,20,1)/(10,−10,10,1), V = (−10, −5) | B0.5 = 7.4322509766 vs λB1 = 6.5258789063 kJ/mol |
| `quadratic_curvature_case()` | z = −2c², centre 1, k = 1 | curvature 4 at c = 0 (old reported 0) |
| `dependence_diagnostic_case()` | z1 = z2² | forward R² ≈ 0 despite deterministic dependence |
