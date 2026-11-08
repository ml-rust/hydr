# Numerical acceptance coverage

hydr checks synthetic numerical cases separately from application accuracy.

## Reproduce the checks

```sh
cargo test --all-features
cargo test --test analytical_cases -- --nocapture
cargo test --features numr/rayon --test engine_contract --test checkpoint_package --test definition_identity
```

Use the numr source revision recorded in [Contributing](../CONTRIBUTING.md).
Fixtures and analytical evaluations live in `tests/analytical_cases.rs` and `tests/support/analytic.rs`.
They require no external datasets or random inputs.
Expected fields evaluate mathematical formulas independently of the production tensor kernel.
The scalar face oracle checks 48 deterministic face pairs within `1e-9` SI component error.
It establishes arithmetic agreement, not independent scientific acceptance.

## Synthetic results

Results below use CPU `f64` tensors and the declared first-order method.
Depth error is the mean absolute difference between cell-centre values and analytical point samples.
It is no comparison against exact cell averages.

| Case | Resolution | Depth error, m | Enforced criterion |
| --- | --- | --- | --- |
| Ritter dry-bed dam, 4 s | 100 / 200 / 400 cells | 0.00669910 / 0.00459524 / 0.00298978 | Below 0.06 m and decreasing |
| Stoker wet-bed dam, 4 s | 100 / 200 / 400 cells | 0.00828527 / 0.00517757 / 0.00310184 | Below 0.06 m and decreasing |
| Vacuum fans, 1 s | 400 cells | 0.00754073 | Below 0.06 m and nonnegative water |
| Thacker moving shoreline, 0.5 s | 40² / 80² cells | 0.000328837 / 0.000180059 | Below 0.01 m and decreasing |
| Manning equilibrium, 5 s | 40 / 80 cells | Maximum drift 0.000127508 / 0.0000417833 | Depth/discharge drift below 0.01 SI and decreasing |

The dam cases span 200 m. Their outer walls remain outside the disturbed region.
Vacuum fans use exact outward far-field ports because moving states cannot terminate at impermeable walls.
Thacker uses a translating circular shoreline in a parabolic bowl over `[-2,2]² m`.
Manning uses source-complete boundary momentum, including pressure.

Ritter temporal refinement fixes 400 cells and maximum intervals `0.02`, `0.01`, and `0.005 s`.
Successive mean depth differences decrease from `0.000126820 m` to `0.0000624591 m`.
Analytical errors increase slightly as smaller timesteps increase first-order spatial diffusion at fixed spacing.
Combined refinement decreases depth error from `0.00651800` to `0.00458525` to `0.00310221 m`.

## Gate status

| Gate | Checked scope | Remaining evidence |
| --- | --- | --- |
| Contract | SI inputs, shapes, references, masks, sources, boundaries, outputs, and exclusions | Physical suitability of each application domain |
| Determinism | Repeated runs, complete trusted/package restart, bit checks, rollback, and cancellation | Cross-platform bitwise equality remains unsupported |
| Analytical | Selected lake, dam, vacuum, friction, Manning, and moving-shoreline cases | Broader hydraulic jumps, radial waves, and transverse diffusion assessment |
| Conservation | Rain, prescribed losses, shared faces, ports, donor depletion, and barrier isolation | Paired component transfers remain unsupported |
| Wet/dry | Partial dry lakes, dry dam fronts, positive films, rewetting shorelines, and masks | Steep/shallow/coarse-grid accuracy and threshold-sensitivity envelope |
| Resolution | Separate spatial, temporal, and combined refinement on selected cases | Application-specific mesh and time error budgets |
| Independent engine | No established external engine comparison executes yet | Matched HEC-RAS, GeoClaw, or other eligible comparisons |
| Extensions | No soil, structure, network, or coupling capability is admitted | Capability-specific acceptance before admission |
| Field accuracy | No measured-event assessment executes | Independent observations and withheld-event assessment |

Lake tests enforce depth and momentum drift below `1e-11` SI.
The fixed-depth friction substep enforces component error below `1e-12`.
Water checks enforce dimensional absolute/relative tolerances without water clipping.
Film momentum resets remain visible in the adjustment ledger.

These results support native numerical demonstrations. They establish no universal depth, warning-time, velocity, or recession accuracy.
