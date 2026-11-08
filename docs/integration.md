# Library integration contract

hydr supplies numerical execution to native Rust application workers through contract `hydr-swe-numr-cpu-1`.

## Ownership

| Layer | Responsibility |
| --- | --- |
| numr | Tensor storage, arithmetic, indexing, conditional selection, reductions, and stable elementwise magnitude |
| solvr | Generic batched bracket expansion/bisection and retained endpoint residuals |
| hydr | Hydraulic domain/state, numerical methods, boundaries, accepted steps, snapshots, and conservation |
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
| `ForcingSchedule::new` | Check bounded contiguous intervals, explicit spatial rates, and complete port assignments |
| `Solver::new` | Recheck compatibility and cache immutable inputs as numr CPU tensors |
| `advance_to` | Report accepted time and completion, cancellation, or step-budget status |
| `advance_controlled` | Check cancellation between steps and expose every accepted field to an observer |
| `snapshot`, `restore` | Capture and restore every continuation-relevant value under an identical definition |
| `Snapshot::data`, `restore_data` | Export/import a checked owned package for application-managed persistence |
| `cell`, `cell_at` | Return supported containing-cell output without interpolation |
| `ledger`, `last_step` | Expose cumulative budgets, interval diagnostics, and oriented face fluxes |

Numerical errors preserve the last accepted step. Earlier accepted steps within an advancement call remain committed.
Invalid targets reject before advancement. Cancellation and step-budget exhaustion return an incomplete horizon at accepted time.
The caller checks `AdvanceStatus` before treating its requested horizon as complete.

## Boundary and source admission

- Wall constructor: Explicitly declares genuine impermeable exterior boundaries and inactive-cell interfaces.
- Mask: Inactive cells declare impermeable exclusions. Missing influential terrain cannot silently become a wall.
- Port support: An active exterior face with fixed rates over each forcing interval.
- Port mass: Outward signed m³/s. Negative values enter the domain.
- Port momentum: Outward x/y momentum flux integrated over face width, in m⁴/s².
- Pressure convention: The port includes normal pressure and all interface/bed corrections exactly once.
- Port eligibility: The provider supplies a compatible regime and conservative normal signal-speed bound with provenance.
- Local rain: Cell-supported m/s, with zero horizontal momentum.
- Land loss: Cell-supported m/s demand with donor momentum. Unavailable water rejects without silently capping demand.

Flux integrals alone cannot establish a wave bound or boundary eligibility.
Stage-only and discharge-only inputs cannot construct a source-complete port.
Characteristic outlets, backwater closure, and flow-reversing open boundaries remain unsupported.
Never substitute walls for required open boundaries.

## Results and checkpoints

Dry cells return `CellResult::Dry`. Their physical velocity and water elevation remain undefined.
Positive films return their represented water. Unknown locations return `Unsupported`, never dry zero.
Internal coordinate ties select the positive-coordinate cell. Maximum exterior edges return unsupported.
Face records represent numerical flux over their accepted interval, including its orientation and support.
The application defines threshold crossings and episode semantics from accepted fields.

Checkpoint compatibility includes numerical contract, geometry, parameters, forcing, boundaries, controls, and complete accepted history.
No soil, network, or control state exists in this capability revision.
Application persistence preserves every package field and exact floating-point bits.
Positive infinity is valid in unrestricted timestep bounds. Ordinary JSON numbers cannot represent it directly.
Import checks consistency and compatibility. The application owns content hashes and historical authenticity.

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

Real urban-event use remains blocked by unsupported characteristic boundaries, soil/network/structure coupling, and independent event evidence.
Do not create a competing numerical kernel in ARUS.
