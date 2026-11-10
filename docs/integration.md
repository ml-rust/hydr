# Library integration contract

Use this guide to construct a Hydr solver and read its results. It also covers complete restart data. The numerical contract is `hydr-swe-numr-cpu-1`.

## Ownership

| Layer | Responsibility |
| --- | --- |
| numr | Tensor storage, arithmetic, indexing, conditional selection, reductions, and stable elementwise magnitude |
| solvr | Generic batched bracket expansion/bisection, Newton correction iteration, and retained residuals |
| hydr | Hydraulic/land state, physical equations, soil admission, boundaries, accepted steps, snapshots, and conservation |
| Application | Provider acquisition, GIS preparation, provenance, jobs, persistence, publication, maps, and event assessment |

Native Rust application workers call the library directly. A separate solver service is unnecessary. Keep browser-facing data contracts free of native solver and tensor dependencies.

## Lifecycle

An accepted step is a completed update that passes Hydr's numerical checks.

### 1. Check physical inputs

| API | What it checks |
| --- | --- |
| `Domain::new` | Geometry, array support, active terrain, roughness, and reference metadata |
| `FlowState::new` | Finite nonnegative depth, momentum, volume, and dry/inactive consistency |
| `ForcingSchedule::new` | Bounded contiguous intervals, spatial rates, and complete assignments for open faces |
| `LandModel::new`, `SoilState::new` | Physical land parameters and complete initial soil memory |

### 2. Construct and advance the solver

| API | What the application does |
| --- | --- |
| `capabilities()` | Read the implemented method, arithmetic requirements, sources, boundaries, and unsupported uses |
| `Solver::<R>::new(client, setup, initial)` | Supply physical inputs and a configured client. Hydr checks them and allocates `Tensor<R>` state. |
| `advance_to` | Advance toward a target time and read the accepted time and completion status |
| `advance_controlled` | Check cancellation between steps and receive accepted fields through an observer |

Invalid targets reject before advancement. Numerical errors preserve the last accepted step. Earlier accepted steps within the same advancement call remain committed.

Cancellation or an exhausted step budget returns the time actually reached. Check `AdvanceStatus` before treating the requested horizon as complete.

For closed storage networks, use [coupled level pools](coupled-level-pools.md). `CoupledLevelPoolBatch` advances declared pool-to-pool transfers with static manual operations. Surface-to-network coupling remains unsupported.

### 3. Read results and retain checkpoints

| API | What it returns |
| --- | --- |
| `snapshot`, `restore` | Complete tensor continuation without copying fields to the host. Restore rejects incompatible definitions or device contexts. |
| `Snapshot::data`, `restore_data` | Checked owned checkpoint data for application-managed persistence. Export and import can return errors. |
| `state`, `cell`, `cell_at` | Accepted fields or the containing cell's output, without interpolation. Device transfers can return errors. |
| `ledger`, `last_step` | Scalar water budgets or detailed interval and face records. Detailed transfers can return errors. |
| `soil_state` | Complete accepted soil memory. Device transfers can return errors. |

## Runtime selection and admission

A runtime selects where calculations run. Its client identifies the device and execution resources.

Tensors store numerical arrays on the selected device. Components use Numr `Tensor<R>` values with `R: Runtime<DType = DType>`. Constructors receive the configured `R::Client` and allocate through `client.device()`. Nested components reuse that client.

Cross-context operations and trusted restoration check `Device::is_same`. Restoration retains the destination client.

- `cuda` and `wgpu` features expose the corresponding Numr and Solvr backend implementations.
- Availability: A backend must also support the required precision and arithmetic. Compilation alone establishes no hydraulic accuracy.
- Construction checks `R::computes_float_dtype(DType::F64)` and the declared basic arithmetic contract before uploading F64 fields.
- Arithmetic probes run on the selected client. Callers preserve nearest-even rounding and gradual underflow throughout continuation.
- Gate actuators additionally require `FusedMulAddContract::Ieee754Fused`. Basic F64 support alone does not admit fused actuator calculations.
- Unsupported precision or contracts return errors. No CPU fallback, host recomputation, or F32 substitution applies.
- CPU F64 remains the checked numerical reference. These documents claim no GPU numerical qualification.

Ordinary `advance_to` keeps accepted flow, soil memory, and recent detailed records as tensors. The host reads scalar control and accounting results.

`advance_controlled` requests copies of accepted fields and diagnostics. State, soil, cell, diagnostic, and owned-checkpoint reads return `Result` and report transfer errors.

## Boundary and source admission

A boundary describes what happens at the edge of a surface domain. Open boundaries represent water entering or leaving the grid.

### Exterior faces

A source-complete port supplies water and momentum fluxes, including pressure, with compatible flow and conservative wave-speed bounds.

- Wall constructor: Explicitly declares genuine impermeable exterior boundaries and inactive-cell interfaces.
- Mask: Inactive cells declare impermeable exclusions. Missing influential terrain cannot silently become a wall.
- Port support: An active exterior face with bounded data over each forcing interval.
- Port mass: Outward signed m³/s. Negative values enter the domain.
- Port momentum: Outward x/y momentum flux integrated over face width, in m⁴/s².
- Pressure convention: The port includes normal pressure and all interface/bed corrections exactly once.
- Port eligibility: The provider supplies a compatible regime and conservative normal signal-speed bound with provenance.

### Local sources

- Local rain: Cell-supported m/s, with zero horizontal momentum.
- Soil: Explicit impervious or pinned Modified Green–Ampt physical land with matching complete antecedent memory.
- Potential evaporation: Explicit nonnegative m/s demand capped after infiltration, before prescribed withdrawal.
- Withdrawal: Separate m/s demand with donor momentum. Unavailable water rejects without silently capping demand.

### Choose the boundary input

Flux integrals alone cannot establish a wave bound or boundary eligibility. Stage-only and discharge-only inputs cannot construct a source-complete port.

Wet-subcritical characteristic boundaries use `BoundaryForcing::Stage` or `BoundaryForcing::Discharge` in `ForcingInterval.boundaries`. `BoundaryConditions::with_open_faces` declares their support. Every interval supplies each open face exactly once.

Stage values use the domain elevation datum. Discharge values impose signed outward numerical water rates in m³/s.

Inflow requires the supplied tangential velocity. Outflow carries the interior tangential velocity. Reversal rechecks this admission rule during every trial.

Scalar stage/discharge traces reject dry, critical, and supercritical regimes without wall or sonic substitution.

`BoundaryForcing::Exterior` supplies complete wet or explicitly dry reservoir data to the same HLL solver as interior faces. Its elevation and velocities describe the maintained exterior state. They prescribe no achieved numerical face elevation or discharge.

`BoundaryForcing::Transmissive` copies post-source dry states or resolved supercritical outflow outside the domain. Transmissive subcritical, critical, or incoming flow rejects and requires complete exterior data.

Read [physical boundaries](boundaries.md) for trace equations, tolerances, and pressure conventions. Never substitute walls for required open boundaries.

## Results and checkpoints

### Interpret cell and face results

| Result | Meaning |
| --- | --- |
| `CellResult::Dry` | Depth is zero. Physical velocity and water elevation remain undefined. |
| Positive film | The cell returns its represented positive water depth. |
| `Unsupported` | The location is unknown or unsupported. It never means dry zero. |

Internal coordinate ties select the positive-coordinate cell. Maximum exterior edges return unsupported.

Face records describe numerical flux over the accepted interval. They retain orientation and spatial support.

- `StepDiagnostics.boundary_traces` records accepted physical traces and their compatibility residuals.
- `signal_speed` records each face's final numerical wave speed in m/s.
- `StepDiagnostics.transmissive_states` records the post-source interior states used by accepted transmissive boundaries.
- Boundary ledgers integrate actual oriented exterior face fluxes, including reversals.

The application defines threshold crossings and episode semantics from accepted fields.

### Save complete restart data

Checkpoint compatibility includes the numerical contract and the physical definition. Checks cover geometry, parameters, forcing, boundaries, controls, and required continuation history.

Checkpoint packages carry `setup: SolverSetup` and `accepted: InitialState`, including full native soil memory. They retain independent infiltration/evaporation ledgers, signed adjustments, conservative budget charges, and accepted per-cell source receipts.

Surface checkpoints exclude network and operating-control state. Standalone [sampled controls](operating-controls.md) and [gate actuators](gate-actuators.md) expose separate complete snapshots. Coordinated component restoration remains unsupported.

[Operating-source bindings](operating-sources.md) are immutable. Reconstruct them from original definitions and restore each actuator before supplying fresh point-query frames. Source bindings retain no independent numerical clock or continuation history.

### Preserve floating-point values

Application persistence preserves every package field and exact floating-point bits. Positive infinity is valid in unrestricted timestep bounds. Ordinary JSON numbers cannot represent it directly.

Import checks consistency and compatibility. The application owns content hashes and historical authenticity. Final soil state establishes continuation support. It cannot prove historical infiltration without a separately authenticated history.

Import reconstructs the combined CFL bound from maximum face speeds along each axis. The stored bound must agree within 64 times `f64::EPSILON` in relative rounding. No fixed absolute tolerance applies.

Transmissive witnesses must satisfy their regime and physical flux checks. Exterior fluxes remain numerical Riemann fluxes.

### Reproduce a run

Immutable checkpoint identity includes floating-point bits, including signed zero. Exact restart requires the same build, dependency source, selected-runtime arithmetic envelope, and accepted-step partition. Different requested intermediate target times can change the step partition and numerical result. Cross-platform bitwise equality remains unsupported.

The application workspace must supply its own Cargo patches for the checked numr and solvr sources. Cargo does not inherit dependency-local `.cargo/config.toml` files. Numerical calls use the caller's configured client. The CPU reference configures one thread explicitly, including under dependency feature unification.

## First supported integration milestone

Run synthetic closed basins or manufactured source-complete-port cases inside a native application worker. Persist checked packages, report water budgets, and consume accepted fields through the adapter boundary. Keep these calculations labelled numerical demonstrations.

Real urban-event use still requires network/structure coupling and independent event evidence. Domains requiring unsupported boundary regimes remain excluded. Applications must reuse Hydr calculations instead of maintaining a competing numerical kernel.
