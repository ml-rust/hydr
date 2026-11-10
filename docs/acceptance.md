# Numerical acceptance coverage

hydr checks synthetic numerical cases separately from application accuracy.

## Reproduce the checks

```sh
cargo test --all-features
cargo test --test analytical_cases -- --nocapture
cargo test --test radial_shoreline --test transverse_transport --test terrain_envelope -- --nocapture
cargo test --features numr/rayon --test engine_contract --test checkpoint_package --test definition_identity
cargo test --test coupled_level_pools
cargo test --test coupled_level_pool_sources
cargo test --test coupled_pool_inflows
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

## Independent wet-bed and dry-bed references

Six matched wet-bed dam runs pass against independently executed Clawpack PyClaw v5.12.0 sources.
Both depth and discharge differences decrease through 100, 200, and 400 cells for each downstream depth.
The threshold-defined fronts occupy matching cells. Both engines conserve water within their declared tolerances.
Read [the reference comparison](reference-comparison.md) for pinned sources, reproducible commands, measured norms, and scope limits.
Three exactly dry Ritter runs also pass against Clawpack's untouched compiled augmented solver.
Their cross-engine depth and discharge differences decrease under refinement, with matching threshold-defined fronts.
However, support-matched analytical front errors are `2`, `1`, and `2.5 m`. Front accuracy and convergence remain unestablished.
Read [the dry-bed comparison](dry-reference-comparison.md) for the analytical diagnostic, external dry policy, and reproduction commands.
Terrain and genuine two-dimensional transverse comparisons remain separate gates.

## Physical wet-subcritical boundaries

`tests/physical_boundaries.rs` checks open-lake balance, four-face orientation, discharge budgets, reversal, forcing edges, retries, and restart.
Tensor-kernel cases independently check complete uniform flux and constructed shock/rarefaction traces.
Matching open stages preserve a stepped lake within `1e-11` SI.
Constant discharge boundaries agree with their accepted water volume within `1e-12 m³`.
Dry, critical, supercritical, missing-inflow-data, and out-of-branch requests reject without mutating accepted state.
These gates admit only the wet-subcritical boundary contract. They establish no physical-domain or field-accuracy claim.

## Complete exterior and transmissive boundaries

`tests/exterior_boundaries.rs` checks uniform Riemann wiring, rotation, stepped-bed lake balance, dry transitions, reversal, vacuum, and strong incoming shocks.
`tests/exterior_contracts.rs` checks all five boundary policies together, dry transmissive copying, source-dependent retries, input admission, and checkpoint consistency.
Complete exterior states prescribe reservoir data, rather than achieved face discharge.
Transmissive boundaries admit only dry or resolved supercritical-outgoing post-source states.
Accepted face speeds reconstruct the native combined CFL bound during package import.
These tests establish selected boundary behaviour. They establish no finite exterior-compartment coupling or physical-domain accuracy.

## Closed coupled pools

`tests/coupled_level_pools.rs` checks accepted water transfers between separate pools through declared structures. Read [coupled level pools](coupled-level-pools.md) before selecting this component.

| Case | Enforced criterion |
| --- | --- |
| Submerged orifice and reversal | Signed midpoint transfers of `1` and `-1 m³`, within `4e-10 m³` |
| Linear pump and held gate | Independently derived midpoint transfer of `1 m³`, within `4e-10 m³` |
| Competing free weirs | One donor budget produces transfers of `2/3` and `1/3 m³`, independent of connection order |
| Serial free weirs | Incoming water cannot finance another outgoing transfer during the same interval |
| Dry receiver and positive films | Explicit dry support admits wetting. Represented positive water retains its volume and stage. |
| Temporal refinement | Independent discrete recurrence agreement within `1e-8 m³`, quadratic continuous-error bounds, and successive error ratios from `3.8` to `4.2` |
| Rejection and restart | Rejected trials preserve accepted state. Forged checkpoints reject atomically. Trusted and owned restart reproduce complete continuation. |

The refinement case holds the endpoint at `0.5 s` and halves intervals from `0.5` through `0.03125 s`. Its finest continuous transfer error must remain below `1e-4 m³`.

The orifice and gate fixtures explicitly use artificial gravity `0.5 m/s²` for exact algebraic references. These checks cover closed exchange-only pools with static manual operations. They establish no surface-to-network coupling or measured-event accuracy.

`tests/coupled_level_pool_sources.rs` separately checks explicit local sources with static structure operations. Cases check source order, rain-funded withdrawal, capped losses without incoming credit, accepted-start area, and a dry rainfall footprint.

Retry checks compare recalculated sources and transfers against an independent formula at the actual accepted duration. Exact-zero forcing retains common hydraulic bits and zero source counters.

Invalid forcing and forged source packages reject atomically. Trusted and owned restart repeat complete continuation.

These source fixtures establish no runoff attachment or field accuracy.

## Prescribed pool inflows

`tests/coupled_pool_inflows.rs` checks already-selected nonnegative external inputs. It retains the existing head, transfer, mass, and independent-volume tolerances.

| Case | Enforced criterion |
| --- | --- |
| Separate boundaries | Rates `0.25/0.75 m³/s` over `2 s` produce volumes `1.5/2.5 m³` from initial `1/1 m³` |
| Shared receiving pool | Rates `0.25/0.5 m³/s` add `1.5 m³` over `2 s`, independent of boundary order |
| Donor and loss order | External water funds neither same-interval structure withdrawals nor earlier local losses |
| Combined forcing | Local sources, a constant pump, and external rates `0.5/0.25 m³/s` produce final volumes `3.75/3.25 m³` |
| Shortened intervals | Actual-duration products and orifice transfer match an independent formula |
| Arithmetic and continuation | Positive absorption and unrepresentable products reject atomically. Complete restart repeats continuation, and altered current/start ledgers or original operands reject. |

Explicit-zero boundaries retain common closed and source-only numerical bits. Actual CPU/CUDA tests share one inflow and complete-restart body. These synthetic checks establish no measured-replacement selection, generated-runoff attachment, geometric ownership proof, or field accuracy.

Integrated-volume cases check original operand bits, mixed rate/volume inputs, exact spans including zero, and fresh forcing after recalculation requests. A positive receiver film accepts the smallest supplied volume over `2 s`, although dividing that input by duration rounds to zero. Dry receivers retain `3` times that smallest volume exactly, although dividing and multiplying it back changes its bits. Filling a dry receiver with the smallest volume rejects because its midpoint depth is unrepresentable.

Exact/mixed restart checks preserve complete state and original forcing kinds. Forged operands, durations, and retries reject atomically. CPU/CUDA tests share one mixed-input and complete-restart body. Supplied integrated volumes establish no full native-runoff attachment.

## Gate status

| Gate | Checked scope | Remaining evidence |
| --- | --- | --- |
| Contract | SI inputs, shapes, references, masks, sources, boundaries, outputs, and exclusions | Physical suitability of each application domain |
| Determinism | Repeated runs, complete trusted/package restart, bit checks, rollback, and cancellation | Cross-platform bitwise equality remains unsupported |
| Analytical | Selected lake, dam, vacuum, friction, Manning, contracting radial lake, shear transport, and moving-shoreline cases | Broader hydraulic jumps, radial cycles, and application-specific diffusion budgets |
| Conservation | Rain, prescribed losses, shared faces, ports, donor depletion, barrier isolation, and closed pool-to-pool transfers | Surface-to-network transfers remain unsupported |
| Wet/dry | Partial dry lakes, dry fronts, positive films, shorelines, masks, crest invariants, and selected threshold sensitivity | Steep/shallow/coarse-grid dynamic accuracy and application-specific threshold envelope |
| Resolution | Separate spatial, temporal, and combined refinement on selected cases | Application-specific mesh and time error budgets |
| Independent engine | Nine matched wet-bed/dry-bed Clawpack runs, with decreasing depth/discharge differences and matched threshold fronts | Analytical front accuracy, independent terrain/transverse-flow comparisons, and broader hydraulic regimes |
| Extensions | Explicit Modified Green–Ampt land and closed pools with static structure operations have separate checks | Wider networks, dynamic coupled controls, and external coupling require separate acceptance |
| Field accuracy | No measured-event assessment executes | Independent observations and withheld-event assessment |

Lake tests enforce depth and momentum drift below `1e-11` SI.
The fixed-depth friction substep enforces component error below `1e-12`.
Water checks enforce dimensional absolute/relative tolerances without water clipping.
Film momentum resets remain visible in the adjustment ledger.

These results support native numerical demonstrations. They establish no universal depth, warning-time, velocity, or recession accuracy.
