# Library integration contract

hydr supplies numerical execution to native Rust application workers through contract `hydr-swe-numr-cpu-1`.

## Ownership

| Layer | Responsibility |
| --- | --- |
| numr | Tensor storage, arithmetic, indexing, conditional selection, reductions, and stable elementwise magnitude |
| solvr | Generic batched bracket expansion/bisection, Newton correction iteration, and retained residuals |
| hydr | Hydraulic/land state, physical equations, soil admission, boundaries, accepted steps, snapshots, and conservation |
| Application | Provider acquisition, GIS preparation, provenance, jobs, persistence, publication, maps, and event assessment |

ARUS integrates through `arus-hydraulics`. `arus-model-api` describes application-facing capabilities and results.
Browser contracts remain free of hydr and numr dependencies.
Native workers call the library directly. No separate solver service is required.

## Lifecycle

| API | Required behaviour |
| --- | --- |
| `capabilities()` | Declare implemented equations, method, execution, sources, boundaries, and exclusions |
| `Domain::new` | Check geometry, array support, active terrain, roughness, and reference metadata |
| `FlowState::new` | Check finite nonnegative depth, momentum, volume, and dry/inactive consistency |
| `ForcingSchedule::new` | Check bounded contiguous intervals, explicit spatial rates, and complete open-face assignments |
| `LandModel::new`, `SoilState::new` | Check explicit physical land and complete antecedent support |
| `Solver::new(setup, initial)` | Check aggregate physical setup and complete flow/soil state, then cache numr CPU tensors |
| `advance_to` | Report accepted time and completion, cancellation, or step-budget status |
| `advance_controlled` | Check cancellation between steps and expose every accepted field to an observer |
| `snapshot`, `restore` | Capture and restore every continuation-relevant value under an identical definition |
| `Snapshot::data`, `restore_data` | Export/import a checked owned package for application-managed persistence |
| `cell`, `cell_at` | Return supported containing-cell output without interpolation |
| `ledger`, `last_step` | Expose cumulative budgets, interval diagnostics, and oriented face fluxes |
| `soil_state` | Expose complete accepted native soil memory |

Numerical errors preserve the last accepted step. Earlier accepted steps within an advancement call remain committed.
Invalid targets reject before advancement. Cancellation and step-budget exhaustion return an incomplete horizon at accepted time.
The caller checks `AdvanceStatus` before treating its requested horizon as complete.

## Boundary and source admission

- Wall constructor: Explicitly declares genuine impermeable exterior boundaries and inactive-cell interfaces.
- Mask: Inactive cells declare impermeable exclusions. Missing influential terrain cannot silently become a wall.
- Port support: An active exterior face with bounded data over each forcing interval.
- Port mass: Outward signed m³/s. Negative values enter the domain.
- Port momentum: Outward x/y momentum flux integrated over face width, in m⁴/s².
- Pressure convention: The port includes normal pressure and all interface/bed corrections exactly once.
- Port eligibility: The provider supplies a compatible regime and conservative normal signal-speed bound with provenance.
- Local rain: Cell-supported m/s, with zero horizontal momentum.
- Soil: Explicit impervious or pinned Modified Green–Ampt physical land with matching complete antecedent memory.
- Potential evaporation: Explicit nonnegative m/s demand capped after infiltration, before prescribed withdrawal.
- Withdrawal: Separate m/s demand with donor momentum. Unavailable water rejects without silently capping demand.

Flux integrals alone cannot establish a wave bound or boundary eligibility.
Stage-only and discharge-only inputs cannot construct a source-complete port.
Wet-subcritical characteristic boundaries use `BoundaryForcing::Stage` or `BoundaryForcing::Discharge` in `ForcingInterval.boundaries`.
`BoundaryConditions::with_open_faces` declares their support. Every interval supplies each open face exactly once.
Stage values use the domain elevation datum. Discharge values impose signed outward numerical water rates in m³/s.
Inflow requires the supplied tangential velocity. Outflow carries the interior tangential velocity.
Reversal rechecks this admission rule during every trial.
Scalar stage/discharge traces reject dry, critical, and supercritical regimes without wall or sonic substitution.
`BoundaryForcing::Exterior` supplies complete wet or explicitly dry reservoir data to the same HLL solver as interior faces.
Its elevation and velocities describe the maintained exterior state. They prescribe no achieved numerical face elevation or discharge.
`BoundaryForcing::Transmissive` copies post-source dry states or resolved supercritical outflow outside the domain.
Transmissive subcritical, critical, or incoming flow rejects and requires complete exterior data.
Read [physical boundaries](boundaries.md) for trace equations, tolerances, and pressure conventions.
Never substitute walls for required open boundaries.

## Results and checkpoints

Dry cells return `CellResult::Dry`. Their physical velocity and water elevation remain undefined.
Positive films return their represented water. Unknown locations return `Unsupported`, never dry zero.
Internal coordinate ties select the positive-coordinate cell. Maximum exterior edges return unsupported.
Face records represent numerical flux over their accepted interval, including its orientation and support.
`StepDiagnostics.boundary_traces` also records accepted physical traces and their compatibility residuals.
Every face records its final numerical `signal_speed` in m/s.
`StepDiagnostics.transmissive_states` records the post-source interior states used by accepted transmissive boundaries.
Boundary ledgers integrate actual oriented exterior face fluxes, including reversals.
The application defines threshold crossings and episode semantics from accepted fields.

Checkpoint compatibility includes numerical contract, geometry, parameters, forcing, boundaries, controls, and complete accepted history.
Checkpoint packages carry `setup: SolverSetup` and `accepted: InitialState`, including full native soil memory.
They retain independent infiltration/evaporation ledgers, signed adjustments, conservative budget charges, and accepted per-cell source receipts.
Surface checkpoints exclude network and operating-control state. Standalone [sampled controls](operating-controls.md) and [gate actuators](gate-actuators.md) expose separate complete snapshots. Coordinated component restoration remains unsupported.
Application persistence preserves every package field and exact floating-point bits.
Positive infinity is valid in unrestricted timestep bounds. Ordinary JSON numbers cannot represent it directly.
Import checks consistency and compatibility. The application owns content hashes and historical authenticity.
Final soil state establishes continuation support. It cannot prove historical infiltration without a separately authenticated history.
Per-axis face-speed maxima reconstruct the native combined CFL bound during import.
The stored bound must agree within 64 epsilon relative rounding, without an absolute unit floor.
Transmissive witnesses must satisfy their regime and physical flux checks. Exterior fluxes remain numerical Riemann fluxes.

Immutable checkpoint identity includes floating-point bits, including signed zero.
Exact restart requires the same build, dependency source, CPU execution envelope, and accepted-step partition.
Different requested intermediate target times can change the step partition and numerical result.
Cross-platform bitwise equality remains unsupported.

The application workspace must supply its own Cargo patches for the checked numr and solvr sources.
Cargo does not inherit dependency-local `.cargo/config.toml` files.
Numerical calls use the configured CPU client even when another dependency enables numr's Rayon feature.

## First supported integration milestone

Run synthetic closed basins or manufactured source-complete-port cases inside an ARUS native worker.
Persist checked packages, report water budgets, and consume accepted fields through the adapter boundary.
Keep these calculations labelled numerical demonstrations.

Real urban-event use remains blocked by unsupported boundary regimes, network/structure coupling, and independent event evidence.
Do not create a competing numerical kernel in ARUS.
