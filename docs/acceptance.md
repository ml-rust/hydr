# Numerical checks and accuracy limits

Use this page to see what Hydr checks before choosing a component. Start with the results for your calculation.

Passing numerical checks establishes selected computational behavior.

**Application limit:** These results establish no measured-event accuracy or physical-domain suitability.

## Run the checks

1. Read [Contributing](../CONTRIBUTING.md) for dependency versions and backend requirements.
2. Run the relevant commands from the repository root.
3. Read the case limits below before interpreting a passing result.

```sh
cargo test --all-features
cargo test --test analytical_cases -- --nocapture
cargo test --test radial_shoreline --test transverse_transport --test terrain_envelope -- --nocapture
cargo test --features numr/rayon --test engine_contract --test checkpoint_package --test definition_identity
cargo test --test coupled_level_pools
cargo test --test coupled_level_pool_sources
cargo test --test coupled_pool_inflows
cargo test --test runoff_pools
cargo test --test cross_sections
cargo test --test channel_flow --test channel_restart -- --nocapture
cargo test --test channel_lateral -- --nocapture
```

The fixtures need no external datasets or random inputs. [Analytical cases](../tests/analytical_cases.rs) use formulas from [the analytical helpers](../tests/support/analytic.rs), independently of the production tensor kernel.

The scalar face reference checks 48 deterministic face pairs within `1e-9` SI component error. This establishes arithmetic agreement only. It supplies no independent scientific acceptance.

## Surface-flow results

These results use CPU `f64` tensors and the declared first-order method.

**Depth error** means the mean absolute difference from analytical values sampled at cell centres. These values are not errors against exact cell averages.

| Case | Resolution | Depth error, m | Enforced criterion |
| --- | --- | --- | --- |
| Ritter dry-bed dam, 4 s | 100 / 200 / 400 cells | 0.00669910 / 0.00459524 / 0.00298978 | Below 0.06 m and decreasing |
| Stoker wet-bed dam, 4 s | 100 / 200 / 400 cells | 0.00828527 / 0.00517757 / 0.00310184 | Below 0.06 m and decreasing |
| Vacuum fans, 1 s | 400 cells | 0.00754073 | Below 0.06 m and nonnegative water |
| Thacker moving shoreline, 0.5 s | 40² / 80² cells | 0.000328837 / 0.000180059 | Below 0.01 m and decreasing |
| Manning equilibrium, 5 s | 40 / 80 cells | Maximum drift 0.000127508 / 0.0000417833 | Depth/discharge drift below 0.01 SI and decreasing |

### Case definitions

| Case | Physical setup |
| --- | --- |
| Dam breaks | A 200 m domain, with outer walls beyond the disturbed region |
| Vacuum fans | Exact outward far-field ports. Moving states cannot terminate at impermeable walls. |
| Thacker shoreline | A translating circular shoreline in a parabolic bowl over `[-2,2]² m` |
| Manning equilibrium | Source-complete boundary momentum, including pressure |

### Refinement in space and time

The Ritter time-refinement case keeps 400 cells. Smaller timesteps do not remove spatial error.

| Check | Result and interpretation |
| --- | --- |
| Maximum intervals | `0.02`, `0.01`, and `0.005 s` |
| Successive mean depth differences | Decrease from `0.000126820 m` to `0.0000624591 m` |
| Error against the analytical solution | Increases slightly as smaller timesteps increase first-order spatial diffusion at fixed spacing |
| Combined space/time refinement | Depth error decreases through `0.00651800`, `0.00458525`, and `0.00310221 m` |

## Flow across two dimensions

### Contracting circular lake

The case contracts an affine circular lake over `z=0.1*(x²+y²)` in `[-2,2]² m`.

| Property | Definition or criterion |
| --- | --- |
| Exact scale | `a''+1.962*a=1.962/a³`, initially `a=1.2` with zero velocity |
| Evaluation time | `0.5 s` |
| Relative depth L1 error | Decreases from `0.0667579` to `0.0467074` on `40²/80²` grids. Both values remain below `0.1`. |
| Mean component discharge error | Below `0.005 m²/s` |
| Sampling | Cell-centre point samples for initialization and error evaluation, without exact cell averaging |

Only contraction executes. Expansion, reversal, and complete cycles remain unchecked.

### Transport of flow across the main direction

The shear case carries a tangential Gaussian discharge through constant depth and normal velocity.

| Property | Definition or criterion |
| --- | --- |
| Exact peak | `0.1 m²/s` |
| Depth and normal velocity | `1 m` and `1 m/s` |
| Mean tangential discharge errors | `0.000501248`, `0.000257749`, and `0.000129640 m²/s` at `100/200/400` cells |
| Profile retention | Relative discharge L1 below `0.2` on each grid, excluding a nearly erased profile |
| Sampling | Centre row beyond the transverse walls' numerical domain of dependence during all 32 accepted steps |
| Constant depth and normal discharge | Within `1e-10` SI on that row |
| Quarter-turn rotation (covariance) | Complete square-grid case agrees within `1e-11` SI |

These results measure selected first-order diffusion. They establish no general guarantee of amplitude preservation.

### Terrain barriers and shallow water

| Check | Enforced behavior |
| --- | --- |
| Stepped lakes | `1e-9 m` films, dry plateaus, and bed jumps of several metres retain the `1e-11` SI still-water criterion on coarse/refined grids |
| High crest | A `1.5 m` crest blocks an initially stationary `1 m` upstream column |
| Low crest | A `0.2 m` crest permits positive downstream transfer. This checks connectivity, without establishing overtopping discharge accuracy. |
| Momentum-reset thresholds | `0`, `1e-8`, and `1e-6 m` on identical shallow-wetting inputs |
| Threshold sensitivity | Mean depth differences from the zero-threshold run reach `2.20e-9 m` and `1.46e-7 m` at `0.5 s` |

The shallow case starts at `2e-6 m` upstream depth. The threshold-dependent differences are material at that scale.

All three threshold runs conserve stored water and record signed removed momentum. A separate moving-film case checks retained positive water and signed reset diagnostics.

The shallow case also covers extreme neighboring depth ratios without increasing the root iteration budget.

## Comparisons with another engine

### Wet-bed dams

The [wet-bed comparison](reference-comparison.md) uses six matched Clawpack PyClaw v5.12.0 runs.

- Depth and discharge: Differences decrease through 100, 200, and 400 cells for each downstream depth.
- Front position: Threshold-defined fronts occupy matching cells.
- Conservation: Both engines conserve water within their declared tolerances.

**Accuracy limit:** Agreement between engines establishes no analytical accuracy.

### Exactly dry Ritter dams

The [dry-bed comparison](dry-reference-comparison.md) uses three runs with Clawpack's untouched compiled augmented solver.

- Depth and discharge: Differences decrease under refinement.
- Front position: Threshold-defined fronts occupy matching cells.
- Conservation: The comparison enforces its declared water-residual criteria.

**Accuracy limit:** Analytical front errors on matching cell-mean support are `2`, `1`, and `2.5 m`. Front accuracy and convergence remain unestablished.

The linked guides contain pinned sources, reproduction commands, measured norms, and external dry policies.

Independent terrain and genuine two-dimensional transverse-flow comparisons remain separate acceptance requirements.

## Open-boundary checks

Read [the boundary guide](boundaries.md) before choosing a boundary policy.

### Prescribed water level (stage) and discharge

Water is present, and flow across the boundary is slower than local gravity waves. These checks admit only this wet-subcritical regime.

| Check | Enforced result |
| --- | --- |
| Matching open stages | Preserve a stepped lake within `1e-11` SI |
| Constant discharge | Agrees with accepted water volume within `1e-12 m³` |
| Uniform flow | Complete boundary flux agrees with independent calculations |
| Waves at the boundary | Calculated traces match constructed shocks (abrupt changes) and rarefactions (spreading waves) |
| Unsupported requests | Dry, critical, supercritical, missing-inflow-data, and out-of-branch requests reject without changing accepted state |

Test details: [wave and flux calculations](../src/numerics/boundary/tests.rs) and [physical boundaries](../tests/physical_boundaries.rs).

The physical cases also cover four-face orientation, reversal, forcing edges, retries, and restart.

### Exterior reservoirs and transmissive outlets

| Policy or check | Meaning |
| --- | --- |
| Complete exterior state | Prescribes reservoir data. It does not prescribe achieved face discharge. |
| Transmissive outlet | Admits only dry or resolved supercritical-outgoing states after source updates |
| Restart | Checkpoint import rebuilds the combined timestep safety limit (CFL) from recorded face wave speeds |

The [exterior-boundary cases](../tests/exterior_boundaries.rs) cover uniform Riemann wiring and rotation. They also cover stepped-bed balance, dry transitions, reversal, vacuum, and strong incoming shocks.

The [exterior-contract cases](../tests/exterior_contracts.rs) combine all five policies. They cover dry transmissive copying, source-dependent retries, input admission, and checkpoint consistency.

These tests establish selected boundary behavior only. Finite exterior-compartment coupling remains unestablished.

## Channel cross-section checks

[Cross-section cases](../tests/cross_sections.rs) compare supplied profiles against closed-form geometry. Read the [geometry guide](cross-sections.md) for inputs and field definitions.

At a water depth of `1 m`, the analytical cases require:

| Profile | Area (m²) | Top width (m) | Wetted perimeter (m) | Pressure integral (m³) |
| --- | --- | --- | --- | --- |
| Rectangle, `2 m` bottom | `2` | `2` | `4` | `1` |
| Trapezoid, `2 m` bottom and `1:1` banks | `3` | `4` | `2 + 2√2` | `4/3` |
| Triangle, `1:1` banks | `1` | `2` | `2√2` | `1/3` |

Normal cases add these allowances:

- Absolute: `1e-12` in the field's SI units.
- Relative: `1e-12` times the expected value.

Scaled profiles compare quantities after removing their expected geometric scale.

Additional cases check vertical walls, exact bench levels, dry results, separated wet portions, profile subdivision, ordering, and admission errors. Unresolved positive geometry must reject instead of returning dry zero.

[Area-to-stage recovery cases](../tests/cross_sections/recovery.rs) check the inverse calculation against independently calculated areas.

| Check | Required result |
| --- | --- |
| Analytical shapes | Recovered stages and areas meet the same dimensional allowances |
| Geometric scales | Normalized geometry meets `1e-12` at scales `2^-100` and `2^100` |
| Bank limits | Capacity returns the original cap. Every area above capacity rejects. |
| Positive water | A positive target never returns dry geometry, even under loose area tolerances |
| Root termination | Height convergence cannot bypass the independent area check |
| Runtime | CPU and CUDA preserve original strided targets and the configured client |

These geometry checks establish no dynamic flow, hydraulic connectivity, or measured-event accuracy.

## Rectangular channel-flow checks

Read [the channel guide](channels.md) for supported geometry and endpoint conditions. [Channel cases](../tests/channel_flow.rs) check first-order flow independently of static cross-section queries.

The dam cases use a 200 m domain, width `2 m`, and distant outer walls. Errors compare cell-centre point samples, not exact cell averages.

| Case at 4 s | Cells | Mean depth error, m | Mean discharge error, m³/s |
| --- | --- | --- | --- |
| Ritter dry-bed dam | 100 / 200 / 400 | 0.00656535 / 0.00457839 / 0.00304661 | 0.0342567 / 0.0220558 / 0.0152417 |
| Stoker wet-bed dam | 100 / 200 / 400 | 0.00919878 / 0.00581262 / 0.00350395 | 0.0451087 / 0.0283505 / 0.0171636 |

Both errors must decrease under spatial refinement. Each depth error stays below `0.06 m`, and discharge error below `0.3 m³/s`.

### Changing widths, dry banks, and friction

| Check | Enforced criterion |
| --- | --- |
| Uniform motion in either direction | Depth drift within `1e-12 m` and discharge drift within `1e-11 m³/s` |
| Wet and partially dry stepped lakes | Depth and discharge drift within `1e-11` SI across changing widths and beds |
| Raised dry bank | No transfer below the crest; positive wetting when upstream water exceeds it |
| Width contrast of one million | Shortened steps respect width-aware wave and donor limits without negative water |
| Smooth varying-width steady flow | Mean depth drift below `0.01 m` and discharge drift below `0.02 m³/s`, both decreasing |
| Isolated Manning resistance | Fixed-area momentum decay agrees with the exact formula within `1e-12 m³/s` |
| Erased numerical products | Report transfer and pressure bounds inside the original water and momentum allowances; reject insufficient allowances |
| Continuation | [Restart cases](../tests/channel_restart.rs) check complete trusted/cold continuation and altered-package rejection |

The smooth-width case runs for `5 s` on 40, 80, and 160 cells. Mean depth drift decreases from `0.000453 m` to `0.000125 m`. Mean discharge drift decreases from `0.00356 m³/s` to `0.000873 m³/s`.

### Separate space and time errors

Ritter timestep refinement keeps 400 cells and halves the maximum interval through `0.02`, `0.01`, and `0.005 s`. Successive mean depth differences decrease from `0.000127 m` to `0.0000625 m`.

Analytical depth error increases slightly at fixed spacing. Smaller timesteps do not remove first-order spatial diffusion. Combined space/time refinement reduces both depth and discharge errors.

### Lateral inflows and withdrawals

[Lateral-source cases](../tests/channel_lateral.rs) check both water and longitudinal momentum. Inputs supply complete named sources over one interval.

The following cases start with `8 m³` of water and `24 m⁴/s` of longitudinal momentum. Matching exterior states cancel native transport changes.

| Input over 0.01 s | Final water, m³ | Final momentum, m⁴/s |
| --- | --- | --- |
| Inflow `2 m³` at `-1 m/s` | 10 | 22 |
| Withdrawal `2 m³` at donor velocity `3 m/s` | 6 | 18 |
| Both inputs | 8 | 16 |

Each result must agree within `1e-12` in its SI units. Separate cases cover dry inflow, complete drainage, grouped inputs, and signed momentum cancellation.

| Admission check | Required behavior |
| --- | --- |
| Donor water | Native face export and withdrawals reserve original water, without incoming credit |
| Rounding near depletion | Grouped withdrawals and face-integral rounding cannot hide an overdraft |
| Integrated inputs | Preserve original operands and variants; request fresh inputs before any shortening, including integrated zero |
| Smallest exact input | Adding the smallest positive `f64` volume to the same stored volume preserves both |
| Strong dry injection | Source-created wave speeds shorten the rate interval; equivalent exact forcing requests recomputation |
| Continuation | Rate, exact, and mixed restart retain source counters, witnesses, original inputs, and configured clients |

### Distributed-source refinement

The manufactured solution has increasing depth and a longitudinal velocity gradient. Its lateral water and momentum inputs satisfy both equations independently.

The case starts dry in a flat, `100 m` channel of width `2 m`. It runs to `0.1 s` without friction. Inputs and exterior states remain fixed during each accepted interval.

| Cells | Mean depth error, m | Mean discharge error, m³/s |
| --- | --- | --- |
| 50 | 0.00000982 | 0.00157 |
| 100 | 0.00000491 | 0.000785 |
| 200 | 0.00000245 | 0.000393 |

Maximum step is `0.0005 s`. Both errors must decrease, with depth below `0.01 m` and discharge below `0.1 m³/s`.

Timestep refinement keeps 200 cells and halves maximum intervals through `0.002`, `0.001`, and `0.0005 s`. Successive normalized differences decrease from `9.37e-7` to `4.66e-7`.

Normalization divides depth differences by `0.1 m` and discharge differences by `8 m³/s` before adding them. This compares unlike units without mixing dimensions.

The source-created wave check covers accepted-start and prefriction endpoint states. It establishes no continuous source-path stability proof.

These checks cover prescribed inputs to rectangular reaches. Separate triangular checks follow below. Arbitrary-profile dynamics, junctions, finite-component exchange, contraction losses, and drainage remain unsupported.

## Prismatic triangular channel checks

[Triangular cases](../tests/channel_triangular.rs) advance actual triangular reaches through the public channel lifecycle. The independent scalar reference integrates conserved cell averages before recovering represented depth.

These cases use gravity `9.81 m/s²`, bank slopes `(1, 1)`, and the existing accounting tolerances. Asymmetry cases also use `(0, 2)`.

Triangular transport uses complete barotropic Godunov sampling. Dry, vacuum, and two-rarefaction cases use analytical samples. Mixed waves and shocks use Solvr brackets. Proved enclosing waves determine CFL limits separately.

The production physical gates select `LimitedLinear`, the default reconstruction. `FirstOrderReference` retains piecewise constant reconstruction for explicit comparisons. Its vacuum comparison requires decreasing depth and discharge errors under the same analytical norm, without claiming the production error ceiling.

| Case | Inputs | Required gate |
| --- | --- | --- |
| Uniform motion | Depth `1 m`, velocity `±1 m/s`, twenty `5 m` cells, matching exteriors, time `5 s` | Depth drift ≤`1e-12 m`, discharge drift ≤`1e-11 m³/s` |
| Closed rest | Unequal cell lengths, depth `1 m`, walls, time `2 s` | Same drift limits, each face water transfer ≤`1e-11 m³` |
| Dry release | Depth-one water released onto dry bed, domain `[-100, 100] m`, time `4 s` | Mean depth error <`0.06 m`, discharge error <`0.3 m³/s`, both strictly decreasing |
| Wet shock | Left/right areas `4/1 m²`, left velocity `sqrt(7g/4)`, right velocity zero, time `4 s` | Same analytical ceilings and strict spatial decrease |
| Two-fan vacuum | Depth `1 m`, separating velocities `±5sqrt(g/2)`, time `1 s` | Independently integrated fans meet the same ceilings and strict spatial decrease |
| Manning decay | Depth `1 m`, roughness `0.03`, velocity `±1 m/s`, interval `0.1 s` | Exact fixed-area decay within `1e-12 m³/s`, unchanged stored volume |
| Bank asymmetry | Slopes `(0, 2)` and `(1, 1)` | Equal inviscid law, distinct independently calculated resistance |
| Reference positive films | `FirstOrderReference`, right depth `10^-j m`, `j = 1..150`, beside depth-one water | CFL within 1% of `0.45 / (5cL - 3cR)`, retained positive water |
| Production positive films | `LimitedLinear`, the same 150 positive depths | Positive advancement respects reported reconstructed half-cell CFL limits |

### Separate spatial and temporal refinement

Spatial checks use 100, 200, and 400 cells with a common maximum interval of `0.002 s`. References split every wave segment and discontinuity before integration. Cell-centre sampling does not define these error norms.

| Refinement | Cells | Maximum intervals, s | Required behavior |
| --- | --- | --- | --- |
| Temporal dry release | 50 throughout | `0.02 / 0.01 / 0.005` | Successive depth and discharge differences strictly decrease |
| Combined dry release | `100 / 200 / 400` | `0.02 / 0.01 / 0.005` | Analytical errors and successive differences strictly decrease |

Both refinement paths retain the analytical ceilings. Combined differences first restrict fine conserved averages to coarse cells, then recover depth.

### Sources, restart, and runtimes

| Check | Required behavior |
| --- | --- |
| Lateral inputs | Incoming momentum follows supplied velocity. Withdrawals use original donor velocity. Incoming water supplies no donor credit. |
| Exact volumes | Shortening requests recomputed forcing. The rejected interval preserves accepted state. |
| Restart | Trusted and cold restart retain flux certificates and reproduce complete sourced continuation. Altered sources or certificates reject. |
| Flux accuracy | Every triangular interval supplies finite face-width certificates within separate water/momentum flux budgets. |
| Geometry identity | Changed slope bits, changed bed bits, and cross-geometry restoration reject. |
| Runtime | Configured CPU and CUDA clients execute production advance, sources, and restart. WGPU rejects F64 without fallback. |

These checks cover prismatic triangular reaches only. They establish no general-profile dynamics, network coupling, SWMM equivalence, or measured-event accuracy.

## Connected-pool checks

Read [coupled level pools](coupled-level-pools.md) before selecting this component.

### Closed pools with static structures

| Case | Enforced criterion |
| --- | --- |
| Submerged orifice and reversal | Signed midpoint transfers of `1` and `-1 m³`, within `4e-10 m³` |
| Linear pump and held gate | Independently derived midpoint transfer of `1 m³`, within `4e-10 m³` |
| Competing free weirs | One donor budget produces transfers of `2/3` and `1/3 m³`, independent of connection order |
| Serial free weirs | Incoming water cannot finance another outgoing transfer during the same interval |
| Dry receiver and positive films | Explicit dry support admits wetting. Represented positive water retains its volume and stage. |
| Temporal refinement | Independent discrete recurrence agreement within `1e-8 m³`, quadratic continuous-error bounds, and successive error ratios from `3.8` to `4.2` |
| Rejection and restart | Rejected trials preserve accepted state. Forged checkpoints reject atomically. Trusted and owned restart reproduce complete continuation. |

The refinement endpoint is `0.5 s`. Intervals halve from `0.5` through `0.03125 s`, with finest continuous transfer error below `1e-4 m³`.

The orifice and gate cases use artificial gravity `0.5 m/s²` for exact algebraic references.

[Closed-pool tests](../tests/coupled_level_pools.rs) cover exchange-only pools with static manual operations. They establish no surface-to-network coupling.

### Local rain and losses

| Check | Required behavior |
| --- | --- |
| Source order | Rain-funded withdrawal and capped losses follow the declared order. Incoming water supplies no withdrawal credit. |
| Area and wetting | Sources use accepted-start area and the declared dry rainfall footprint |
| Retry | Recalculated sources and transfers match an independent formula at the actual accepted duration |
| Explicit zero forcing | Common hydraulic bits remain unchanged, with zero source counters |
| Rejection and continuation | Invalid forcing and forged source packages reject atomically. Trusted and owned restart reproduce complete continuation. |

[Source cases](../tests/coupled_level_pool_sources.rs) retain static structure operations. They establish no runoff attachment.

### Prescribed external inflows

[Inflow cases](../tests/coupled_pool_inflows.rs) use already-selected nonnegative inputs. Existing head, transfer, mass, and independent-volume tolerances remain binding.

| Case | Enforced criterion |
| --- | --- |
| Separate boundaries | Rates `0.25/0.75 m³/s` over `2 s` produce volumes `1.5/2.5 m³` from initial `1/1 m³` |
| Shared receiving pool | Rates `0.25/0.5 m³/s` add `1.5 m³` over `2 s`, independent of boundary order |
| Donor and loss order | External water funds neither same-interval structure withdrawals nor earlier local losses |
| Combined forcing | Local sources, a constant pump, and external rates `0.5/0.25 m³/s` produce final volumes `3.75/3.25 m³` |
| Shortened intervals | Actual-duration products and orifice transfer match an independent formula |
| Arithmetic and continuation | Positive absorption and unrepresentable products reject atomically. Complete restart repeats continuation, and altered current/start ledgers or original operands reject. |

Explicit-zero boundaries retain common closed and source-only numerical bits. CPU/CUDA tests share one inflow and complete-restart calculation.

### Volumes supplied for one interval

These checks use a supplied total volume without converting it to a rate.

| Check | Required behavior |
| --- | --- |
| Original operands | Preserve original floating-point bits for exact and mixed rate/volume inputs |
| Time support | Require the exact supplied interval, including zero-volume inputs. Shortening requests fresh forcing. |
| Smallest positive input | A positive receiver film accepts it over `2 s`, although division by duration rounds the rate to zero |
| Dry receiver | Retains `3` times the smallest volume exactly, although division and multiplication change the bits |
| Unresolved midpoint | A dry receiver rejects the smallest volume because its midpoint depth is unrepresentable |
| Continuation | Exact/mixed restart retains complete state and original input kinds. Forged operands, durations, and retries reject atomically. |

CPU/CUDA tests share one mixed-input and complete-restart calculation.

**Scope limit:** These inflow checks prove no measured-replacement selection or geometric ownership. They establish no generated-runoff attachment or full native-runoff attachment.

### Native runoff into pools

Use the [runoff-to-pool guide](runoff-pools.md) to check whether your outlets drain without receiving-pool feedback.

[Runoff/pool cases](../tests/runoff_pools.rs) check both children over one accepted interval:

| Check | Required behavior |
| --- | --- |
| Original exports | Rain on an instantaneous impervious area exports exactly `1 m³` into a dry pool |
| Outlet mapping | Unequal `0.5/1.5 m³` exports retain their destinations when boundary order reverses |
| Common retries | Pervious soil and internal routing match isolated children at the actual accepted duration |
| Water ownership | Local losses and outgoing transfers cannot consume incoming runoff during the same interval |
| Continuation | Complete restart retains both histories. Altered forcing, timing, runon, or internal pairing rejects atomically. |

Global accounting cancels runoff exports against pool imports. A separate lifetime pairing check uses rounding bounds, independent of the physical mass allowance.

These cases establish no downstream feedback, geometric nonoverlap proof, dynamic network flow, or measured-event accuracy.

## Remaining acceptance work

| Area | Checked scope | Remaining evidence |
| --- | --- | --- |
| Contract | SI inputs, shapes, references, masks, sources, boundaries, outputs, and exclusions | Physical suitability of each application domain |
| Determinism | Repeated runs, complete trusted/package restart, bit checks, rollback, and cancellation | Cross-platform bitwise equality remains unsupported |
| Analytical | Selected lake, dam, vacuum, friction, Manning, contracting radial lake, shear transport, and moving-shoreline cases | Broader hydraulic jumps, radial cycles, and application-specific diffusion budgets |
| Conservation | Rain, prescribed losses, shared faces, ports, donor depletion, barrier isolation, pool transfers, and paired native runoff | Surface-to-network transfers remain unsupported |
| Wet/dry | Partial dry lakes, dry fronts, positive films, shorelines, masks, crest invariants, and selected threshold sensitivity | Steep/shallow/coarse-grid dynamic accuracy and application-specific threshold envelope |
| Resolution | Separate spatial, temporal, and combined refinement on selected cases | Application-specific mesh and time error budgets |
| Independent engine | Nine matched wet-bed/dry-bed Clawpack runs, with decreasing depth/discharge differences and matched threshold fronts | Analytical front accuracy, independent terrain/transverse-flow comparisons, and broader hydraulic regimes |
| Extensions | Land sources, coupled pools, free runoff outlets, and rectangular/triangular reaches with lateral inputs have separate checks | General channel profiles, wider networks, finite-component exchanges, and dynamic coupled controls require separate acceptance |
| Field accuracy | No measured-event assessment executes | Independent observations and withheld-event assessment |

### Numerical limits retained by the checks

| Quantity | Requirement |
| --- | --- |
| Lake depth and momentum drift | Below `1e-11` SI |
| Fixed-depth friction substep | Component error below `1e-12` |
| Water balance | Dimensional absolute/relative tolerances, without water clipping |
| Film momentum resets | Signed changes remain visible in the adjustment ledger |

These results support native numerical demonstrations. They establish no universal depth, warning-time, velocity, or recession accuracy.
