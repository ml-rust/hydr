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
cargo test --test channel_pools -- --nocapture
cargo test --test surface_channels
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

### Restart and forecast continuation

| Check | Enforced result |
| --- | --- |
| Surface continuation | Extending forcing with `continue_with_forcing` or `Solver::resume` matches one run over the spliced schedule bit for bit, with the same intermediate targets. Rewritten frozen forcing, an altered folded prefix, and uncovered or incompatible schedules reject without changing accepted state. See [continuation cases](../tests/engine_contract/continuation.rs). |

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

Test details: [wave and flux calculations](../crates/hydr-solver/src/solver/tensor/boundary_trace/tests.rs) and [physical boundaries](../tests/physical_boundaries.rs).

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

### Certified surveyed-section bounds

Read [the certified-geometry guide](prismatic-sections.md) for depth and conserved-area queries.

[Forward cases](../tests/prismatic_geometry/forward.rs) check 79 original-operand queries. [Inverse cases](../tests/prismatic_geometry/recovery.rs) check 86 conserved-area queries against independent geometry references.

| Check | Required result |
| --- | --- |
| Reference containment | Bounds contain exact original-coordinate geometry and independent inverse brackets |
| Geometry interval width | At most `1e-10 × max(1, abs(exact truth))` in the quantity's SI units |
| Inverse depth width | At most `1e-10 × max(1, cap depth)` metres |
| Coordinate conditioning | Original decimal operands and uncertain band boundaries remain enclosed. Tiny depth remains positive at an elevation offset of `2^40 m`. |
| Capacity | Exact canonical cap areas and their preceding values admit. Their succeeding values reject. |
| Ownership | Repeated owners and strided inputs preserve original operands. Result mutation cannot modify compiled geometry. |

[Admission cases](../tests/prismatic_geometry/admission.rs) cover 90 rejected queries and six rejected definitions. They also check dry results and unresolved positive water. [Allocation checks](../tests/prismatic_geometry/budget.rs) enforce the padded table and query limits.

CPU and CUDA cases use the same references and both input layouts. These checks establish geometry bounds only, without surveyed-profile flow or measured-event accuracy.

### Surveyed wetted-perimeter and hydraulic-radius bounds

Read [the perimeter guide](prismatic-sections.md#add-wetted-perimeter-and-hydraulic-radius-bounds) for opt-in queries and dry-radius handling.

The [perimeter cases](../tests/prismatic_perimeter.rs) check nine profiles against 58 depth references and 27 inverse references. Inverse references retain the original F64 conserved areas. CPU and CUDA check both contiguous and strided layouts.

| Check | Established behavior |
| --- | --- |
| Reference containment | Perimeter and radius enclose independent references. Widths stay within `1e-9 m + 1e-10 * max(abs(lower), abs(upper))` in these cases. |
| Contact geometry | Dry plateau bottoms return zero perimeter. Cap benches add no contact. Matching areas can have different asymmetric bank contact. |
| Radius | Recovery retains conserved area in the numerator. Radius can decrease as depth increases. |
| Ownership and admission | Input snapshots survive source mutation. Result aliases cannot modify compiled geometry. Invalid operands and unsupported native bounds return errors. |
| Arithmetic limits | Unsupported bank-length arithmetic rejects perimeter certification while existing certified geometry remains usable. |

These checks establish geometry bounds for the supplied profiles. They establish neither surveyed Manning resistance nor reach advancement or measured-event accuracy.

### Surveyed interface-flux bounds

Read [the surveyed-flux guide](prismatic-fluxes.md) for supplied states, returned bounds, and rejection actions.

The [interface-flux cases](../tests/surveyed_riemann.rs) contain 37 independent reference cases with these required outcomes:

| Cases | Required outcome |
| --- | --- |
| 34 | Successful flux bounds contain the independent references and meet the default width allowances. |
| One shallow-film uncertainty case | Success meets the same checks, or the query returns a declared numerical rejection. |
| Two | Compression beyond surveyed capacity returns `CapExceeded`. A zero work allowance returns `WorkLimit`. |

Reference pairs supply lower and upper bounds. Equal endpoints supply a point reference. Tests check containment of both endpoints rather than agreement with a midpoint.

The cases cover dry states, spreading waves, shocks, vacuum, and changes between surveyed height bands. Trace bounds contain both states beside a stationary shock.

CPU checks use contiguous and strided inputs. Mixed batches check repeated section owners, independent input snapshots, and cumulative work limits.

These queries calculate interface fluxes from supplied states. They establish neither surveyed reach advancement nor measured-event accuracy.

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
| Reference positive films | `FirstOrderReference`, right depth `10^-j m`, `j = 1..150`, beside depth-one water | CFL within 1% of `2 × 0.45 / (5cL - 3cR)`, retained positive water |
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

## Scalar channel-boundary checks

[Scalar endpoint cases](../tests/channel_boundaries.rs) define Stage and Discharge acceptance for rectangular and triangular reaches. [Surveyed endpoint checks](#surveyed-endpoint-checks) cover surveyed reaches. Scalar inputs require strictly wet-subcritical interior water and traces. These requirements establish no general inlet/outlet suitability.

### Independent traces and incoming waves

| Trace fixture | Outward velocity, m/s | Outward discharge, m³/s | Momentum flux, m⁴/s² |
| --- | --- | --- | --- |
| Rectangle `g=1`, width `3 m`, initial depth `1 m`, boundary depth `0.81 m` | `0.2` | `0.486` | `1.08135` |
| Triangle `g=2`, `k=1`, initial depth `1 m`, boundary depth `0.81 m` | `0.4` | `0.26244` | `0.45927` |
| Rectangle `g=1`, width `1 m`, boundary depth `2 m` | `-sqrt(3)/2` | `-sqrt(3)` | `7/2` |
| Triangle `g=8`, `k=1`, boundary depth `1.5 m` | `-sqrt(95/27)` | `-(9/4)sqrt(95/27)` | `203/12` |

All trace fixtures start at rest. Trace comparisons require relative agreement within `1e-12`. Compression fixtures also exercise the original geometry cap with explicit approximate-compatibility status when needed.

Propagation uses a `20 m` domain and initial depth `1 m` at rest. The right endpoint is at `x=0`. Left-end cases reflect coordinates and longitudinal discharge independently.

| Shape | Gravity | Geometry | Boundary depths | Final time |
| --- | --- | --- | --- | --- |
| Rectangle | `1 m/s²` | Width `3 m` | `0.81 / 1.21 m` | `2 s` |
| Triangle | `2 m/s²` | `k=1` | `0.81 / 1.21 m` | `2 s` |

Stage and Discharge use the same independently derived shock or fan state. Grids contain `50/100/200` cells with common maximum interval `0.002 s`.

The reference splits wave segments before integrating conserved cell averages. Depth derives from average area. Length-weighted mean depth error must stay below `0.02 m`, and full-discharge error below `0.04 m³/s`. Both errors must decrease strictly for every shape, policy, wave, and orientation.

Each finest-grid error must also stay below `25%` of the nonzero analytical change from uniform initial water. This gate rejects unchanged initial fields despite generous coarse ceilings.

### Admission, accounting, and continuation

| Check | Required behavior |
| --- | --- |
| Regime and capacity | Dry, critical, supercritical, unresolved, and certified out-of-cap requests reject atomically |
| Original discharge | Both endpoints preserve longitudinal operands and signed zero, including width-round-trip counterexamples |
| Momentum | One-cell recurrence uses physical endpoint pressure and advection |
| Donors and retries | Endpoint exports compete with lateral withdrawals for original water. Integrated sources require fresh forcing before shortening. |
| Final scalar state | Source-created candidates satisfy fresh scalar compatibility after the final state check |
| Continuation | Trusted/cold restart retain complete targets, traces, config bits, and numerical receipts. Altered packages reject. |
| Runtime | Actual CPU/CUDA clients execute nonuniform scalar-boundary flow and complete restart. WGPU rejects F64 without fallback. |

Discharge inputs remain longitudinal-positive. Trace receipts use outward discharge. Read [the channel guide](channels.md) before selecting signs or compatibility controls.

## Quiescent-reservoir boundary checks

[Reservoir cases](../tests/channel_reservoir.rs) check the entrance-energy and exit-mixing assumptions described in [the channel guide](channels.md#use-a-quiescent-reservoir).

Six independent trace fixtures cover both geometries and flow directions. Incoming fixtures use these reservoir elevations:

| Section | Elevation with `K=0`, m | Elevation with `K=1`, m | Expected trace depth, m |
| --- | --- | --- | --- |
| Rectangle, `g=1`, width `1 m` | `19/8` | `11/4` | `2` |
| Triangle, `g=8`, `k=1` | `743/432` | `419/216` | Approximately `1.5` |

Initial depth is `1 m`, initial velocity is zero, and bed elevation is zero. Outflow fixtures use reservoir elevation `0.81 m`.

The reference uses 150-digit arithmetic on the actual binary64 inputs. Depth, velocity, discharge, and momentum comparisons require absolute-plus-relative agreement within `1e-12`.

Each absolute allowance uses that quantity's SI units. Dissipation comparisons use mechanical power divided by water density, in m⁵/s³.

Reported depth intervals must contain the independently calculated solution. Computed traces must also meet the separate reservoir-head allowance.

| Check | Required behavior |
| --- | --- |
| Passive loss | Dissipation agrees within `1e-10 m⁵/s³ + 1e-12 × reference magnitude` |
| Still water | Equal levels preserve the existing rest-state error limits for both geometries and endpoint orientations |
| Wave propagation | `K=1` reservoir cases meet the same grids, spatial-refinement criteria, and physical-signal limits described above |
| Extreme head arithmetic | Large entrance losses retain amplified tiny heads. Overflow-prone input expressions still produce finite certified heads when representable. |
| Regime and capacity | Unsupported flow and invalid loss coefficients reject without changing state. Uncertain results at the maximum depth also reject. |
| Continuation | Native inputs retain original bits and resist caller-buffer mutation. Trusted/cold replay checks every reservoir receipt bound. |
| Runtime | Actual CPU/CUDA clients execute reservoir roots, mixed endpoint policies, dissipation checks, and complete restart |

Across reservoir grids, maximum mean errors remain below `0.006146 m` for depth and `0.02084 m³/s` for discharge.

These are local boundary checks. Finite-storage feedback has separate checks below.

## Surveyed endpoint checks

Surveyed reaches accept `Stage`, `Discharge`, `QuiescentReservoir`, and `FluxPort` ends, and attach to ports, pool ends, and channel/pool connections. The criteria below were fixed before any run. A case that misses its criterion fails, and no one loosens the criterion. The references use 150-digit arithmetic on the actual binary64 inputs. The profiles are a rectangle, a triangle, a three-band section, and a wide-bottom section.

### Standalone reaches

| Case | Enforced criterion |
| --- | --- |
| [Lake at rest](../tests/channel_surveyed/endpoints_rest.rs) | A rectangle and a three-band reach of 8 cells at rest. Each of `Stage`, `Discharge(0)`, `QuiescentReservoir` with `K = 0` and `1`, and a `FluxPort` with zero water and the reference pressure, at the left end, the right end, and both. Rest depths lie in each band. Over 50 intervals every `\|Q\|` and stage change stays within `1e-12`. |
| [Traces](../tests/channel_surveyed/endpoints_traces.rs) | The reservoir and literal rarefaction and shock fixtures above, rerun on the rectangle and triangle profiles. The three-band and wide-bottom sections run `Stage`, `Discharge`, and `QuiescentReservoir`, entering and leaving, at both ends, with trace depths in every band and within `1e-6 m` of each knot. Each reference lies inside the certified depth enclosure. Depth, velocity, discharge, and momentum flux agree within `1e-11` absolute plus `1e-12` relative. |
| [Triangle identity](../tests/channel_surveyed/endpoints_identity.rs) | A triangle profile and a triangular reach start in the same state with `Wall` and `Exterior` ends. At every interior face of the first interval, the surveyed flux enclosure and the triangular certified bracket intersect. |
| [Propagation](../tests/channel_surveyed/endpoints_propagation.rs) | Three bands, `20 m`, `g = 1`, initial depth `1.5 m` at rest, `25/50/100` cells, `t = 2 s`. `Stage`, `Discharge`, and `QuiescentReservoir` each drive a rarefaction and a shock across a knot. Mean depth and discharge errors fall strictly with refinement. Each finest error stays below `25%` of the physical signal. Right-end errors equal left-end errors within `1e-12` relative. |
| [Admission](../tests/channel_surveyed/endpoints_admission.rs) | Dry, critical, supercritical, and above-cap states reject for each scalar policy. A rejection leaves the reach unchanged bit for bit. |
| [Flux port](../tests/channel_boundaries/flux_port_surveyed.rs) | A rectangle and a three-band reach in a uniform state, with ports that supply the exact physical flux at both ends. Over 50 intervals the relative change of every area and discharge stays within `1e-12`. A port below `\|u\| + sqrt(g A / T)` rejects. Export from a dry end cell rejects, and import into one admits. |
| Mass | Every interval keeps the ledger residual within its bound. A `Discharge` face volume equals the rounded product of rate and duration bit for bit. |
| [Restart](../tests/channel_surveyed/endpoints_restart.rs) | For every policy, snapshot, restore, import, and resume match the uninterrupted run bit for bit. A forged trace or reservoir receipt rejects. |

These cases establish no surveyed convergence acceptance and no measured-event accuracy.

### Coupled reaches

| Case | Enforced criterion |
| --- | --- |
| [Channel/pool references](../tests/channel_pools/surveyed.rs) | The 18 finite-storage fixtures run on the surveyed rectangle and triangle at both ends. Six three-band cases follow. Trace, transfer, head, pool volume, reach volume, and momentum agree within `1e-10` absolute plus `1e-10` relative. |
| [Channel/pool rest and reversal](../tests/channel_pools/surveyed_rest.rs) | A lake keeps depth `1 m` within `1e-12`, and discharge, momentum, and storage change stay within `1e-11` over 16 intervals, at both ends with `K = 0` and `1`. Transfers in both directions exceed `1e-6 m³`. |
| [Two-part inventory](../tests/channel_pools/surveyed_inventory.rs) | On three bands with nonzero low parts, every interval keeps the composite residual within the mass bound. |
| [Channel/pool oscillation](../tests/channel_pools/surveyed_mode.rs) | The rectangular mode of the finite-storage oscillation runs on a surveyed rectangle with 50, 100, and 200 cells. The same refinement and finest-error limits hold. |
| [Channel/pool restart](../tests/channel_pools/surveyed_restoration.rs) | Trusted and cold replay continue bit for bit on a rectangle and on three bands. Forged shared transfer packets reject. |
| [Pool end](../tests/surface_channels/pools_reach_end_surveyed.rs) | The reach-end criteria of the pool-end case above hold on a rectangle, a triangle, and three bands. The pool-end volume equals the reach face volume bit for bit. |
| [Port](../tests/surface_channels/ports_surveyed.rs) | A surveyed rectangle of the face width meets every criterion of the rest, uniform-flow, step-limit, donor, bed-step, and order cases above. Its port receipts equal the rectangular reach's receipts on the first attempt. |
| [Port dam break and seiche](../tests/surface_channels/ports_surveyed_dam_break.rs) | The Ritter, Stoker, sonic-point, and seiche cases above run on a surveyed rectangle. For Ritter and Stoker, at every spacing the L1 depth difference from the rectangular run stays within that run's own L1 error against the exact solution. The seiche checks its period, the port pairing, and the reversal flag. |
| [Port admission](../tests/surface_channels/ports_definition.rs) | A surveyed end that is not a rectangle and a triangular reach end reject with `port-section-mismatch`. A surveyed rectangle off the face width rejects with `port-width-mismatch`. |
| [Structure link](../tests/surface_channels/structures_surface_reach_surveyed.rs) | A one-cell walled surveyed rectangle replaces the rectangular reach in the surface-to-reach orifice case. The same criteria hold. |
| Restart | [Port](../tests/surface_channels/ports_restart_surveyed.rs) and [pool-end](../tests/surface_channels/pools_restart_surveyed.rs) packages restore bit for bit. Forged receipts, ledgers, and pool packages reject. |

These cases establish no measured-event accuracy.

## Finite channel/pool checks

Read [channel-to-pool connections](channel-pools.md) before constructing a connected system. [Coordinator cases](../tests/channel_pools.rs) check common accepted intervals.

### Finite-storage feedback

Eighteen independent feedback fixtures run at both reach ends, producing 36 cases. Comparisons include momentum and the pool's changing level.

Each quantity requires `1e-10` absolute allowance in its SI units, plus `1e-10 × reference magnitude`.

| Check | Required behavior |
| --- | --- |
| Still water and topology | Preserve rest, multiple endpoint connections, and results under declaration-order changes. |
| Reversal | Retain paired transfers when flow changes direction within the supported regime. |
| Donor water and connection beds | Use original donor water and preserve wet support at raised connections. |
| External inputs | Preserve exact integrated zero inputs and resist caller-buffer mutation. |
| Recalculation and restart | Recalculate complete shorter intervals and replay the composite final interval. |

### Coupled oscillation

An independent rectangular channel/pool mode runs on 50, 100, and 200 cells. Samples occur at approximately `7.743 s` and `15.485 s`, one-quarter and one-half period.

Errors use fixed peak amplitudes, including when instantaneous discharge vanishes. The largest normalized errors across both samples on 200 cells are:

| Quantity | Error measure | Largest error |
| --- | --- | --- |
| Channel depth | Root-mean-square field error | 0.81% |
| Channel discharge | Root-mean-square field error | 0.50% |
| Pool level | Absolute scalar error | 1.30% |

Every finest-grid error remains below 5%. Refinement requires at least 20% reduction unless normalized error is below `1e-7`.

| Runtime | Checked scope |
| --- | --- |
| CPU | Feedback, oscillation refinement, donor limits, and complete continuation |
| CUDA | Both shapes at both reach ends. Mixed connections, non-knot cuts, native input ownership, and restart also pass. |
| WGPU | Required F64 precision rejects without CPU fallback |

These cases retain the [connection limits](channel-pools.md#understand-connection-limits), including first-order time integration. They establish no global energy theorem or measured-event accuracy.

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

## Surface/channel checks

Use the [surface/channel guide](surface-channels.md) to declare bank links, ports, pools, structures, pool ends, and controls with their evidence.

[Surface/channel cases](../tests/surface_channels.rs) check the surface and every reach over one accepted interval:

| Check | Required behavior |
| --- | --- |
| Pairing | Each side's net exchange matches the summed link volumes within the paired bound |
| Physics | Free overflow, flood then recession through one link, dry floodplain wetting, and zero transfer from dry or below-crest donors |
| Rest | A lake level with the crest stays still |
| Donors | Competing links share one donor factor. Limited channel withdrawals pass combined depletion. |
| Order | Link and reach declaration order change no bit |
| Recalculation | Nonconvergence halves the interval. A weir volume that would cross heads levels them within one interval. Integrated inputs request recomputation. |
| Continuation | Snapshot, restore, import, resume, and forecast continuation match the uninterrupted run bit for bit. Forged packages reject. A standalone surface restore of a coupled package rejects. |
| Surveyed reaches | Flood then recession through one link pairs and balances each interval, on a rectangle and on three bands. Restart matches the uninterrupted run bit for bit. A crest below the section minimum rejects. Scalar and exterior endpoints admit. CPU and CUDA. |

### Reference and refinement

| Case | Enforced criterion |
| --- | --- |
| [Free-weir drain](../tests/surface_channels/reference.rs) | One `4 m²` cell drains through a free weir into a reach below the crest. Each run matches an independent implicit-midpoint recurrence within `1e-12 m`. Errors against the closed form at `4 s` stay frozen within `1e-12 m`, under a first-order envelope of `3e-4 × Δt`. Intervals halve from `0.25` through `0.015625 s`. Successive error ratios stay from `3.9` to `4.1`. |
| [Split bank](../tests/surface_channels/refinement.rs) | One `2 m` bank becomes 1, 2, 4, 8, and 16 links on refined floodplain cells. Exchanged volumes over `2 s` stay frozen within `1e-9 m³`. Successive level differences shrink by at least `1.4`, then `1.9`, per refinement. The 8-link and 16-link totals agree within `1.1e-2 m³`. |
| [Closed levelling](../tests/surface_channels/levelling.rs) | A floodplain and a reach cell behind a submerged bank reach the exact level within the head allowance, then carry no water. Rectangular, surveyed, and two-links-into-one-cell variants pass. Heads never cross. Storage balances each interval. Each run takes at most 20 accepted intervals to `3 s`. |
| Levelled restart | A package whose last interval binds the cap restores bit for bit. A forged `level_factor` rejects. |
| Bank seiche | A floodplain at `1.6 m` and a reach at `1.75 m` exchange through one bank link over 120 intervals toward `0.25 s`, with and without a reach-end port. Midpoint heads come within `1e-6 m`. Every accepted interval keeps at least half of `0.95` times its smallest child step limit. Mass, link pairing, and port pairing hold each interval. |

The drain converges at second order because an isolated drain makes each interval one implicit-midpoint step. The coordinator still declares first-order time integration.

These cases establish no weir-coefficient accuracy, bank-mapping proof, or measured-event accuracy.

### Reach-end ports

A port joins a rectangular reach end, or a surveyed rectangle end, to a line of surface edge faces. The criteria below were fixed before any run.

| Case | Enforced criterion |
| --- | --- |
| [Lake at rest](../tests/surface_channels/ports_rest.rs) | A reach bed equal to, below, and above the floodplain bed, through a one-face and a four-face port. After 50 intervals every surface `|q|` and reach `|Q|` stays within `1e-12`, and every stage within `1e-12 m`. |
| Water below the higher bed | A port whose stage lies below the higher bed on both sides moves `0.0 m³`, bit for bit, over 50 intervals. Both children keep their start state bit for bit. |
| Mirror symmetry | A reach drains across the four north faces of a symmetric grid. Mirror cells agree within `1e-12` relative each interval. |
| [Uniform flow](../tests/surface_channels/ports_uniform.rs) | Froude `0.5` and `2`, reach to surface and surface to reach. Every `h` and `q` changes by at most `1e-12` relative over 50 intervals. The port carries the state's own discharge and momentum within `1e-12` relative. |
| [Step limits](../tests/surface_channels/ports_limits.rs) | Each interval, the port signal speed covers `|u| + sqrt(g h)` of the reach end and of every surface face cell. The accepted step stays within both children's CFL limits. A fast reach end makes the port speed set the step. |
| Depletion | A near-empty end cell drains onto a dry floodplain. Each port face takes the HLL flux of the start states bit for bit, and no water turns negative. The CFL limit stays at or below the depletion limit, so depletion never scales the flux. |
| [Shared donor](../tests/surface_channels/ports_donors.rs) | A port and a bank weir draw on one reach end cell. The weir takes the cell's water net of port outflow: the gap stays within `1e-12` relative, and the two never exceed the cell's water. Port pairing, link pairing, and mass hold each interval. |
| [Bed step](../tests/surface_channels/ports_step.rs) | Steady flow up and down a `0.2 m` step against a one-grid 2D twin. Steady means every `h` and `q` changes by at most `1e-12` relative over one interval. Inflow and outflow agree within `1e-9` relative in both systems. Reach stages match the twin within 1% of local depth. |
| [Dry dam break](../tests/surface_channels/ports_dam_break.rs) | Ritter: `1 m` of water `1 m` upstream of the port runs onto a dry bed, reach to strip and strip to reach. Each side is `8 m` long and `1 m` wide. Spacings run from `0.5` to `0.0625 m`. At `1 s`, the L1 depth error over every cell stays within `1.25` times that of a one-row 2D twin at the same spacing. Each halving shows an L1 order of at least `0.5`. |
| Sonic point | Ritter's dam sits on the port face, in both directions. Mean port discharge per width over the second half of the run comes within 5% of `(8/27) sqrt(g) h0^(3/2)` at `0.0625 m`. The error falls with each halving. |
| Wet dam break | Stoker: `1 m` against `0.25 m`, with the shock crossing the port, in both directions. The L1 criteria match Ritter. At every spacing the shock lies within `2 Δx` of its exact position. |
| [Seiche](../tests/surface_channels/ports_seiche.rs) | A closed `8 m` reach and `8 m` strip, `1 m` deep, start with a linear tilt of `a/h = 0.01`. Spacings are `2`, `1`, and `0.5 m`, over `2.25` exact periods. The port volume changes sign at least 4 times. The period from sign-change times comes within 3% of `2L/sqrt(g h)` at `0.5 m`, and the error falls with refinement. Port pairing, mass, and composite balance hold each interval. |
| [Admission](../tests/surface_channels/ports_definition.rs) | Each port rejection returns `SurfaceChannelError::Input` and names its capability entry. A caller policy at a ported end and a missing policy at an unported end both reject the trial with `port-endpoint-policy`. |
| [Declaration order](../tests/surface_channels/ports_order.rs) | Two ports, two bank links, and two reaches run 12 intervals. Reversing the port, face, link, and reach lists changes no accepted bit. |
| [Restart](../tests/surface_channels/ports_restart.rs) | A reach drains through a port and a bank link, then refills. Snapshot, restore, import, resume, and forecast continuation match the uninterrupted run bit for bit. A replayed `reversed` flag matches. Forged port receipts, port ledgers, and setup ports reject. A package of an earlier schema or contract rejects by identity. A standalone surface restore rejects. |

These cases establish no measured-event accuracy and no energy guarantee at bed steps.

### Pools, structures, and controls

A pool joins the coordinator as a node. A structure joins two nodes. A pool end joins a reach end to a pool. The criteria below were fixed before any run. A case that misses its criterion fails, and no one loosens the criterion.

Analytical cases use `head_absolute_m = 1e-13`, `transfer_absolute_m3 = 1e-12`, and relative parts of zero. A refinement case runs four durations, each half the last, and compares consecutive errors. Every interval checks the regime, the mass bound, the pairing, and the composite balance.

| Case | Enforced criterion |
| --- | --- |
| [Rest](../tests/surface_channels/pools_rest.rs) | A pond level with a surface cell, through a submerged weir, an orifice, and an open gate. Over 50 intervals every link volume is `0.0` bit for bit, the regime is `EqualHeads`, and every stage changes by at most `1e-12 m`. |
| [Pond drain over a free weir](../tests/surface_channels/pools_weir_drain.rs) | A constant-area pool drains over a free weir onto a reach cell below the crest. Exact `h(t) = (h0^{-1/2} + C_w L t / (2A))^{-2}`. At the end time the finest relative error of `h` is at most `1e-6`, and the order is at least `1.8` on the two finest pairs. |
| [Pond fill through an orifice](../tests/surface_channels/pools_orifice_fill.rs) | A closed one-cell surface tank and a constant-area pool. Exact `Δh(t) = (sqrt(Δh0) - k t / 2)^2`, `k = C_d a sqrt(2g) (1/A_1 + 1/A_2)`. At `t*/2` the finest relative error is at most `1e-6`, and the order is at least `1.8`. After `t*` the heads level within the head allowance, with no crossing rejection. |
| [Surface-to-reach orifice](../tests/surface_channels/structures_surface_reach.rs) | A one-cell surface tank and a one-cell walled reach joined by an orifice. The exact solution and the criteria match the orifice fill. The orifice approximates a culvert. |
| [Pump on a flat curve](../tests/surface_channels/structures_pump.rs) | Plateau `Q0` over the head range, pool to surface tank. `\|ΣV - Q0 T\| <= 8 ε N Q0 T` over `N` intervals, with `ε` the `f64` machine epsilon. |
| Pump on a linear curve | Same file. `Q = Q0 - s Δ`, with `Δ` the lift. Exact `Δ(t) = Q0/s - (Q0/s - Δ0) exp(-s k t)`, `k = 1/A_1 + 1/A_2`. The finest relative error is at most `1e-6`, and the order is at least `1.8`. |
| Suction cap | Same file. A pump drains its pool to the suction bound `z_s`. The cap binds and nothing rejects. The final stage lies in `[z_s, z_s + head_absolute_m]`, and no storage turns negative. Eight further intervals keep the stage in that band. |
| Pump to a dry cell with an outlet | Same file. The volume equals the curve at lift `outlet - donor stage` within `8 ε` relative per interval. |
| [Prescribed gate ramp](../tests/surface_channels/structures_gate.rs) | A linear opening ramp between two ponds, free aperture. Self-refinement order of the final stage is at least `1.8`. Closed intervals move `0.0` bit for bit. |
| Actuated gate on a stage rule | Same file. A `ControlledGateBatch` opens a pond outlet when the pond stage reaches its upper threshold. Every decision stage keeps at least `0.01 m` from each threshold. The gate switches at the first decision with stage at or above the threshold. The self-refinement order of the final stage is at least `1.8`. Receipts hold the midpoint opening and its error. |
| Actuated gate arrival | Same file. An outlet arrives at its target inside the run. The coordinator ends an interval at each bound of the arrival window. The last receipt opening equals the target bit for bit. |
| [Controlled pump](../tests/surface_channels/structures_pump_control.rs) | A relay starts the pump at the upper sump threshold and stops it at the lower threshold. An independent plain-`f64` replay gives the decision stages and switch times. Commands switch at the exact decision times. Each running interval keeps `\|V - Q0 dt\| <= 8 ε Q0 dt`. A stopped interval moves `0.0` bit for bit. |
| [Reversal](../tests/surface_channels/structures_reversal.rs) | A reach with a tidal `Exterior` end and a pool joined through a bidirectional orifice. The orifice volume changes sign at least 4 times. Pairing and composite mass pass every interval. |
| Mirror symmetry | Same file. Swapping endpoints A and B negates every signed volume bit for bit and keeps every child state bit for bit. |
| [Pool end](../tests/surface_channels/pools_reach_end.rs) | A reach end into a pool, filling then reversing. Every `channel_pools` frozen criterion for the same case holds through the coordinator: the 150-digit references at both ends within `1e-10` absolute plus `1e-10` relative, a lake at rest at both ends and both entrance losses, and a reversal. The pool-end volume equals the reach face volume bit for bit. |
| [Competing donors](../tests/surface_channels/pools_donors.rs) | One pool feeds a surface cell, a channel cell, and another pool, with demand above storage. Every withdrawal keeps one share below one. Pool volume stays nonnegative. Rain on the pool funds withdrawals. Incoming water funds none. |
| [Declaration order](../tests/surface_channels/pools_order.rs) | Reversing or rotating the pool, structure, pool-end, link, and port lists changes no receipt, ledger, balance, or child state bit over 8 intervals. Every structure moves water. |
| [Restart](../tests/surface_channels/pools_restart.rs) | A case with a manual, a prescribed, an actuated, and a controlled-pump owner, plus a pool end and a pool inflow. The checkpoint interval ends at a decision. Snapshot, restore, import, and resume match the uninterrupted run bit for bit, with packages, ledgers, and child states. Forged pool, structure ledger, pool-end ledger, inflow ledger, and control packages reject. A package of schema `4` or contract `6` rejects by identity. |
| [Admission](../tests/surface_channels/pools_definition.rs) | One test per rejection row. Each rejection returns `SurfaceChannelError::Input` and names its capability entry. |
| [Capability declaration](../tests/component_capabilities/surface_channel_pools.rs) | Every declared endpoint and boundary kind moves water in a run. Every declared structure and pool-end regime holds. Every `RejectedByAdmission` entry has a rejecting case. |

These rejections run as in-crate tests with frozen criteria: pool capacity, a dry signal node, a dry pool end, and the pump rejections.

These cases establish no measured-event accuracy. An orifice does not satisfy culvert inlet or outlet control. [Culvert coordinator checks](#culvert-coordinator-checks) cover culvert links.

### Native runoff

An impervious subcatchment without depression storage drains into a closed pond, into cells of a walled reach, or into walled surface cells. The criteria below were fixed before any run.

| Case | Enforced criterion |
| --- | --- |
| [Recession](../crates/hydr-coupling/src/coupling/surface_channels/runoff/accounting.rs) | 1 cm of runoff drains without rain into the pond, into a reach cell, and into a surface cell of a coordinator without links. Exact `d(t) = (d0^(1-p) + (p-1) α t)^(1/(1-p))`. The largest relative error over `t = 1` to `4 s` falls at each of five halvings from `1 s`. The finest relative error at `4 s` is at most `2e-3`. |
| Constant rain | Same file. Rain on a dry subcatchment into the pond, against a 50-digit ODE reference. The largest error falls at each refinement. The finest relative error at the final row is at most `2e-2`. |
| Mass identity | Same cases. Receiver gain plus runoff storage equals initial runoff storage plus rain, within the composite bound, every interval. |
| [Shortened attempt](../crates/hydr-coupling/src/coupling/surface_channels/runoff/attempt.rs) | A reach limits the interval by its CFL number. The delivered volume equals the runoff child trialled alone to the accepted end, bit for bit. It differs from the full-interval volume scaled by the duration ratio. |
| Span split | Same file. One outlet spans two reach cells with weights 1 and 3. The parts sum to the export within the split bound and keep the 1:3 ratio within four ulps. |
| Zero momentum | Same file. Runoff into a moving reach equals a caller inflow of the same volume at zero velocity, bit for bit. |
| [Pairing](../crates/hydr-coupling/src/coupling/surface_channels/runoff/delivery.rs) | Each reach and pool binding holds its delivered volume bit for bit. A lane ledger holds the running sum of the transfers bit for bit. |
| [Tailwater](../crates/hydr-coupling/src/coupling/surface_channels/runoff/tailwater.rs) | A receiving stage above the invert at the start rejects as input. A crossing at the end shortens the interval, or without retries returns `Unresolved` naming `runoff-outlet-submerged`. |
| [Surface cells](../crates/hydr-coupling/src/coupling/surface_channels/runoff/cells.rs) | Runoff into one cell of a coordinator without links equals the runoff child trialled alone, bit for bit. The lane ledger holds the running sum bit for bit. The surface exchange totals hold no runoff. A span over two cells with weights 1 and 3 sums to the export within the split bound. Runoff into a moving cell leaves every discharge bit unchanged. Zero runoff on a linked cell keeps every surface, reach, and link bit. Runoff beside bank weir water pairs with the surface exchange within its bound. |
| Surface tailwater and shortening | Same file. A cell above the invert at the start rejects as input. A crossing at the end shortens the interval, or without retries returns `Unresolved` naming `runoff-outlet-submerged`. A surface CFL limit shortens the attempt, and the delivered volume equals the runoff child trialled alone to the accepted end, bit for bit. |
| [Restart](../crates/hydr-coupling/src/coupling/surface_channels/runoff/restart.rs) | After three intervals into the pond, into a reach cell, and into a span over two surface cells, `restore`, `import_snapshot`, and `resume` repeat three more intervals bit for bit. A zero-history package restarts at the epoch. One changed runoff bit in the package, the counters, the start frame, or the transfers rejects and names the field. |
| [Public receipts and restart](../tests/surface_channels/runoff.rs) | Into the pond, a reach cell, a linked surface cell, and a surface cell without links, transfers, lane ledgers, and balance totals agree over four intervals. A package taken after two intervals imports and resumes bit for bit. |
| [Capability declaration](../tests/component_capabilities/surface_channel_runoff.rs) | The outlet endpoint delivers water into every destination. Every runoff `RejectedByAdmission` entry has a rejecting case. No `runoff-attachment` entry remains. |
| [No-runoff regression](../tests/surface_channels/frozen.rs) | The frozen digests of contracts `10` and `9` hold after the text maps back to each format. |

## Culvert barrel checks

[Culvert barrel cases](../tests/culvert_barrels.rs) compare circular and box barrel enclosures against independent 100-digit references. The references are frozen before any kernel runs. Read the [barrel guide](structures.md#culvert-barrel-geometry) for inputs, statuses, and limits.

| Check | Required result |
| --- | --- |
| Geometry | Every reference area, top width, perimeter, hydraulic radius, and moment lies inside its enclosure, within a frozen width |
| Crown and ranges | Full rows carry the full-state constants bit for bit. Range bounds contain every point bound they span. |
| Critical and normal depth | Statuses match the reference class of the exact root. Every promised enclosure contains it. |
| Monotonicity | No certified reversal of geometry with depth, or of critical and normal depth with discharge |
| Layouts | Every lane at widths 1, 31, 32, 33, 64, and 65 equals its single-row result bit for bit |

These checks establish geometry and depth enclosures only. They establish no culvert discharge, inlet or outlet control, or measured-event accuracy.

## Culvert law checks

[Culvert law cases](../tests/culvert_laws.rs) invert independent 100-digit inlet and outlet corpus rows and the kernel's own control headwaters. Tolerances are frozen before any kernel runs. Read the [culvert guide](structures.md#culvert-structure-law) for inputs, regimes, and limits.

| Check | Required result |
| --- | --- |
| Reference rows | Each row's reference headwater returns its reference discharge within `1e-6` relative, under the control the row exercises |
| Round trip | `Q(HW(Q))` returns `Q` within `1e-6` relative on mild, adverse, and level barrels |
| Control selection | Inlet-governed, outlet-governed, and tied lanes report their control |
| Direction and zero flow | Reversal, one-end blocking, equal heads, dry donors, free outfalls, and zero-discharge headwaters give their declared results |
| Transitions | Discharge stays continuous at both inlet anchors and where the outlet turns full |
| Layouts | Every lane at widths 1, 33, and 65 among weirs equals its single-lane result bit for bit |

These checks establish no measured-event accuracy.

## Culvert coordinator checks

These cases run culverts in the surface/channel coordinator and in coupled level pools. Criteria are frozen before the first run.

| Case | Required result |
| --- | --- |
| [Drain](../tests/surface_channels/culverts_drain.rs) | Two ponds drain through a drowned culvert. The upper pond's drop at 8 s matches an independent `mpmath` ODE reference within `1e-6` relative at 32 intervals, with order at least `1.8`. Every interval reports outlet control. |
| [Coupled drain](../tests/coupled_level_pools/culverts.rs) | The same drain in coupled level pools meets the same criteria |
| [Tide](../tests/surface_channels/culverts_reversal.rs) | A tide on a reach cell reverses a two-way culvert to a lagoon. A one-end culvert beside it moves exact zero with `CulvertUndeclaredEnd` while flow runs into the lagoon. |
| [Rising sea](../tests/coupled_level_pools/culverts.rs) | An external inflow raises a sea above a lagoon. The two-way culvert reverses and the one-end culvert moves exact zero. |
| [Control sweep](../tests/surface_channels/culverts_sweep.rs) | A filling basin moves a short steep culvert from inlet to outlet control, never back. Discharge falls and changes by at most 25 % between intervals. A second basin drowns a partly full outlet past the crown with the same continuity. Restore, import, and resume of that outlet match the uninterrupted run bit for bit. |
| [Invert cap and equal heads](../tests/surface_channels/culverts_invert.rs) | A small pond drains to its culvert invert stop and stays there with exact zero. Level ponds rest at `EqualHeads` with exact zero. |
| [Roadway](../tests/surface_channels/culverts_roadway.rs) | A road weir overtops beside a drowned culvert until the upper pond falls to the crest. Restore, import, and resume match the uninterrupted run bit for bit. |
| [Coupled roadway](../tests/coupled_level_pools/culverts.rs) | Both structures carry water. Restore and owned-data restore match the uninterrupted run bit for bit. |

Every surface/channel interval passes the mass, pairing, and composite checks. The [drain fixture](../tests/culverts/drain_rows.rs) holds the reference inputs, end stages, tolerances, and the SHA256 of its source.

These cases establish no field culvert accuracy.

## Runtime support

Every capability declaration claims the same status on each runtime. A named test backs each claim.

| Runtime | Declared status | Evidence |
| --- | --- | --- |
| CPU | Reference. The accuracy checks on this page run on CPU. | One test per component, built without GPU features |
| CUDA | Executes. Each test runs the component on a CUDA client and checks its results and continuation. | `cuda` tests, built with `--features cuda` |
| WGPU | Rejects construction, because numr WGPU lacks F64 arithmetic. No CPU fallback runs. | `wgpu` tests, built with `--features wgpu` |

The CUDA tests of these components also run the same case on CPU and compare complete checkpoint data:

| Components | CPU/CUDA comparison |
| --- | --- |
| Operating controls, controlled gates, controlled pumps | Every field matches bit for bit |
| Structure laws | Free and submerged weir, orifice, and free, mixed and submerged gate outputs match bit for bit over a dense head grid of more than 700 lanes |
| Culvert barrels | Point, range, full-state, critical, and normal outputs match bit for bit over the reference rows and each layout width. Point geometry also matches over the dense sweep. |
| Culvert laws | Regime, status, direction, discharge, and operands match bit for bit over the mixed lanes and each layout width |
| Culvert coordinators | A tide both ways, inlet control into the control sweep's dry basin, one interval of the partly full drowning outlet, and the roadway case of the surface/channel coordinator, and the rising sea and roadway cases of coupled level pools. After every accepted interval the text of every package, ledger, and child state is equal between CPU and CUDA. |
| Level pools, reservoirs, subcatchments | Per-element states match bit for bit. Batch totals and residual diagnostics agree within `volume_absolute + volume_relative * scale`. |
| Surface/channel | Surface and reach cell states match bit for bit after flood, recession, and restart through one bank link. A second case drains a reach through a port and a bank link together, refills it, and drains it again, with restart. Its cell states also match bit for bit. Lifetime counters, ledgers, bounds, and iteration residuals agree within `mass_absolute_m3 + mass_relative * scale`. A third case runs pools with an orifice, a pump, an actuated gate, and a pool end across relay decisions and a gate arrival. After every accepted interval the text of every child state, package, and ledger is equal between CPU and CUDA, signed zero included. A fourth case drains native runoff over three intervals, with resume, into a pond, a reach cell, a linked surface cell, and a surface cell of a coordinator without links. The same equal-text rule applies to it. |
| Surveyed endpoints | Four cases run on both clients: a standalone `Stage` and `FluxPort`, a port on a surveyed rectangle with a bank link, a pool end on three bands, and a channel/pool connection on three bands. Each restarts bit for bit. Cell volumes and momenta equal the CPU run bit for bit every interval. CPU and CUDA ledgers differ by at most the sum of their reported bounds. |

Those totals and diagnostics differ in their last bits between CPU and CUDA. CUDA execution establishes no GPU accuracy qualification.

Each WGPU test also requires `WgpuRuntime::computes_float_dtype(DType::F64)` to return `false`. A numr release with F64 WGPU fails these tests. Then replace the WGPU claims with execution evidence.

[`runtime_claims_name_existing_tests`](../tests/component_capabilities.rs) checks that each claim names one test. CUDA evidence needs `cfg(feature = "cuda")`, and WGPU evidence needs `cfg(feature = "wgpu")`. CPU evidence carries neither gate.

## Remaining acceptance work

| Area | Checked scope | Remaining evidence |
| --- | --- | --- |
| Contract | SI inputs, shapes, references, masks, sources, boundaries, outputs, and exclusions | Physical suitability of each application domain |
| Determinism | Repeated runs, complete trusted/package restart, bit checks, rollback, and cancellation | Cross-platform bitwise equality remains unsupported |
| Analytical | Selected lake, dam, vacuum, friction, Manning, contracting radial lake, shear transport, and moving-shoreline cases | Broader hydraulic jumps, radial cycles, and application-specific diffusion budgets |
| Conservation | Rain, prescribed losses, shared faces, ports, donor depletion, barrier isolation, pool transfers, paired native runoff, paired surface/channel exchange, and paired exchange through pools, structures, and pool ends | Tailwater feedback on runoff outlets and external runon remain unsupported |
| Wet/dry | Partial dry lakes, dry fronts, positive films, shorelines, masks, crest invariants, and selected threshold sensitivity | Steep/shallow/coarse-grid dynamic accuracy and application-specific threshold envelope |
| Resolution | Separate spatial, temporal, and combined refinement on selected cases | Application-specific mesh and time error budgets |
| Independent engine | Nine matched wet-bed/dry-bed Clawpack runs, with decreasing depth/discharge differences and matched threshold fronts | Analytical front accuracy, independent terrain/transverse-flow comparisons, and broader hydraulic regimes |
| Extensions | Land sources, coupled pools, free runoff outlets, finite channel/pool connections, surface/channel bank weirs, and surface/channel pools, structures, pool ends, and sampled controls have separate checks | General channel profiles, wider networks, culverts in coordinators, continuous control feedback, and application-specific blockage scenarios require separate acceptance |
| Field accuracy | No measured-event assessment executes | Independent observations and withheld-event assessment |

### Numerical limits retained by the checks

| Quantity | Requirement |
| --- | --- |
| Lake depth and momentum drift | Below `1e-11` SI |
| Fixed-depth friction substep | Component error below `1e-12` |
| Water balance | Dimensional absolute/relative tolerances, without water clipping |
| Film momentum resets | Signed changes remain visible in the adjustment ledger |

These results support native numerical demonstrations. They establish no universal depth, warning-time, velocity, or recession accuracy.
