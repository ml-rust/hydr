# Numerical acceptance coverage

hydr checks synthetic numerical cases separately from application accuracy.

## Reproduce the checks

```sh
cargo test --all-features
cargo test --test analytical_cases -- --nocapture
cargo test --test radial_shoreline --test transverse_transport --test terrain_envelope -- --nocapture
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

## Radial, transverse, and terrain checks

The radial case contracts an affine circular lake over `z=0.1*(x²+y²)` in `[-2,2]² m`.
Its exact scale satisfies `a''+1.962*a=1.962/a³`, starting at `a=1.2` with zero velocity.
At `0.5 s`, relative depth L1 decreases from `0.0667579` to `0.0467074` on `40²/80²` grids.
Both values stay below `0.1`. Mean component discharge errors stay below `0.005 m²/s`.
Initialization and error evaluation use cell-centre point samples, rather than exact cell averages.
Only contraction executes. Expansion, reversal, and complete cycles remain unchecked.

The shear case transports a tangential Gaussian discharge through constant depth and normal velocity.
Its exact peak is `0.1 m²/s`, with depth `1 m` and normal velocity `1 m/s`.
Mean tangential discharge errors decrease through `0.000501248`, `0.000257749`, and `0.000129640 m²/s` at `100/200/400` cells.
Each grid also requires relative discharge L1 below `0.2`, excluding a nearly erased profile.
Sampling uses the centre row beyond transverse walls' numerical domain of dependence during all 32 accepted steps.
Depth and normal discharge remain constant there within `1e-10` SI.
The complete square-grid problem also passes quarter-turn covariance within `1e-11` SI.
These results measure selected first-order diffusion. They establish no general amplitude-preservation guarantee.

Stepped lakes include `1e-9 m` films, dry plateaus, and bed jumps up to several metres.
Coarse and refined grids retain the existing `1e-11` SI still-water criterion.
A `1.5 m` crest blocks an initially stationary `1 m` upstream column.
A `0.2 m` crest permits positive downstream transfer. This checks connectivity, without establishing overtopping discharge accuracy.

Shallow wetting compares momentum reset thresholds `0`, `1e-8`, and `1e-6 m` on identical inputs.
Relative to zero, mean depth differences reach `2.20e-9 m` and `1.46e-7 m` at `0.5 s`.
The initial upstream depth is `2e-6 m`. These differences are material at that scale.
All three runs conserve stored water and record signed removed momentum.
A separate moving-film case checks retained positive water and signed reset diagnostics.
The shallow fixture also covers extreme neighboring depth ratios without increasing the root iteration budget.

## Independent wet-bed reference

Six matched wet-bed dam runs pass against independently executed Clawpack PyClaw v5.12.0 sources.
Both depth and discharge differences decrease through 100, 200, and 400 cells for each downstream depth.
The threshold-defined fronts occupy matching cells. Both engines conserve water within their declared tolerances.
Read [the reference comparison](reference-comparison.md) for pinned sources, reproducible commands, measured norms, and scope limits.
Dry-bed, terrain, and two-dimensional transverse comparisons remain separate gates.

## Gate status

| Gate | Checked scope | Remaining evidence |
| --- | --- | --- |
| Contract | SI inputs, shapes, references, masks, sources, boundaries, outputs, and exclusions | Physical suitability of each application domain |
| Determinism | Repeated runs, complete trusted/package restart, bit checks, rollback, and cancellation | Cross-platform bitwise equality remains unsupported |
| Analytical | Selected lake, dam, vacuum, friction, Manning, contracting radial lake, shear transport, and moving-shoreline cases | Broader hydraulic jumps, radial cycles, and application-specific diffusion budgets |
| Conservation | Rain, prescribed losses, shared faces, ports, donor depletion, and barrier isolation | Paired component transfers remain unsupported |
| Wet/dry | Partial dry lakes, dry fronts, positive films, shorelines, masks, crest invariants, and selected threshold sensitivity | Steep/shallow/coarse-grid dynamic accuracy and application-specific threshold envelope |
| Resolution | Separate spatial, temporal, and combined refinement on selected cases | Application-specific mesh and time error budgets |
| Independent engine | Six matched wet-bed Clawpack runs, with decreasing depth/discharge differences and matched threshold fronts | Independent dry-bed, terrain, transverse-flow, and broader hydraulic comparisons |
| Extensions | No soil, structure, network, or coupling capability is admitted | Capability-specific acceptance before admission |
| Field accuracy | No measured-event assessment executes | Independent observations and withheld-event assessment |

Lake tests enforce depth and momentum drift below `1e-11` SI.
The fixed-depth friction substep enforces component error below `1e-12`.
Water checks enforce dimensional absolute/relative tolerances without water clipping.
Film momentum resets remain visible in the adjustment ledger.

These results support native numerical demonstrations. They establish no universal depth, warning-time, velocity, or recession accuracy.
