# Integrate Hydr into an application

Use this guide to supply physical inputs, advance the solver, read results, and save complete restart data.

## Ownership

Native Rust application workers call Hydr directly. A separate solver service is unnecessary.

| Layer | Responsibility |
| --- | --- |
| numr | Tensor storage, arithmetic, indexing, conditional selection, reductions, and stable elementwise magnitude |
| solvr | Generic batched bracket expansion/bisection, Newton correction iteration, and retained residuals |
| hydr | Hydraulic and land state, physical equations, soil admission, boundaries, accepted steps, snapshots, and conservation |
| Application | Provider acquisition, GIS preparation, provenance, jobs, persistence, publication, maps, and event assessment |

Keep browser-facing data contracts free of native solver and tensor dependencies.

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

Read capabilities with `surface_solver_capabilities()`. It returns a `ComponentCapabilities` declaration: precision, arithmetic, runtimes, boundaries, regimes, sources, unsupported uses, and limits.

| API | Application action |
| --- | --- |
| `Solver::<R>::new(client, setup, initial)` | Supply physical inputs and a configured client. Hydr checks them and allocates `Tensor<R>` state. |
| `advance_to` | Request a target time, then read the accepted time and completion status. |
| `advance_controlled` | Check cancellation between steps and receive accepted fields through an observer. |
| `continue_with_forcing(forcing)` | Splice new forcing after the accepted history. Accepted state, ledgers, budget, step count, and last step remain. See [Continue a forecast under new forcing](#continue-a-forecast-under-new-forcing). |
| `Solver::<R>::resume(client, &data, forcing)` | Build an engine from an owned package, then continue it under `forcing`. Pass `data.setup.forcing.clone()` for a plain restart. |

Always read `AdvanceStatus` before treating the requested horizon as complete.

| Outcome | State retained by Hydr |
| --- | --- |
| Invalid target | No advancement occurs. |
| Numerical error | The last accepted step remains. Earlier accepted steps in the same call stay committed. On `Err`, read `time()` and `snapshot()` to find the progress and keep it. |
| Cancellation or exhausted step budget | The result reports the time actually reached. |

### 3. Read results and retain checkpoints

| API | Result |
| --- | --- |
| `snapshot`, `restore` | Complete tensor continuation without copying fields to the host. Incompatible definitions or device contexts reject. |
| `Snapshot::data`, `restore_data` | Checked owned checkpoint data for application-managed persistence. Export and import can return errors. |
| `state`, `cell`, `cell_at` | Accepted fields or the containing cell's output, without interpolation. Device transfers can return errors. |
| `ledger`, `last_step` | Scalar water budgets or detailed interval and face records. Detailed transfers can return errors. |
| `soil_state` | Complete accepted soil memory. Device transfers can return errors. |

### Connected storage pools

Use [coupled level pools](coupled-level-pools.md) for declared pool-to-pool transfers.

`CoupledLevelPoolBatch` supports static manual operations, optional local sources, and prescribed external inflows. Supply already-selected inputs and physical ownership evidence.

Use [runoff-to-pool connections](runoff-pools.md) to generate upstream inflows and accept both engines over one common interval. Every runoff outlet must drain without receiving-pool feedback.

Rectangular, triangular, and surveyed reaches attach to the surface solver through the [surface/channel coordinator](surface-channels.md). Surveyed reaches attach through lateral links only. Pools, structures, and reach-end ports do not attach to the surface solver.

### Channel flow and geometry

Use [cross-section geometry](cross-sections.md) to evaluate supplied channel profiles at declared water levels. This static component provides no dynamic river or drainage flow.

Use [certified section bounds](prismatic-sections.md) for native depth queries or recovery from conserved areas.

Use [channel flow](channels.md) to advance an unbranched rectangular, triangular, or fixed surveyed reach with explicit endpoint conditions.

For surveyed reaches, supply one profile meeting [prismatic certification requirements](prismatic-sections.md#check-whether-the-profile-qualifies). Supply ordered cell lengths and roughness, initial volumes and momenta, and `ChannelSurveyedConfig`. Choose `Wall` or complete `Exterior` endpoints. Scalar endpoints and finite-pool attachments remain unsupported. Surface links attach through the [surface/channel coordinator](surface-channels.md#couple-a-surveyed-reach).

Surveyed advancement supports lateral rates, exact interval volumes, diagnostics, and complete restart through the same channel API. Supply a CPU or CUDA F64 client. Read [surveyed numerical controls and limits](channels.md#set-surveyed-numerical-controls) before selecting tolerances or interpreting accuracy.

Use [channel-to-pool connections](channel-pools.md) for rectangular or triangular reaches when finite pool levels respond to channel transfers. `ChannelPoolBatch` accepts both components together and retains their combined restart data.

Surface state remains separate from these channel/pool calculations.

Use [surface/channel coupling](surface-channels.md) when water crosses a bank between surface cells and rectangular or triangular reach cells. `SurfaceChannelBatch` accepts the surface and every reach over one common interval.

## Read component capabilities

Each component returns a `ComponentCapabilities` declaration. It holds precision, arithmetic requirements, runtimes, admitted boundaries, regimes, and sources. It also holds options, nested components, unsupported uses, and limits.

Ids in a declaration are stable. Persisted decisions can store them.

A test in `tests/component_capabilities` backs every claim. A named test under the matching feature gate backs every runtime claim.

| Component | Function | Doc |
| --- | --- | --- |
| Surface solver | `hydr::solver::surface_solver_capabilities` | [Integrate Hydr](#lifecycle) |
| Rectangular reach | `hydr::channel::rectangular_reach_capabilities` | [Channel flow](channels.md) |
| Triangular reach | `hydr::channel::triangular_reach_capabilities` | [Channel flow](channels.md) |
| Surveyed reach | `hydr::channel::surveyed_reach_capabilities` | [Channel flow](channels.md) |
| Soil | `hydr::soil::soil_capabilities` | [Soil](soil.md) |
| Level pool | `hydr::storage::level_pool_capabilities` | [Level pool](level-pool.md) |
| Subcatchment | `hydr::runoff::subcatchment_capabilities` | [Subcatchment](subcatchment.md) |
| Reservoir | `hydr::runoff::reservoir_capabilities` | [Reservoir](reservoir.md) |
| Operating control | `hydr::structures::operating_control_capabilities` | [Operating controls](operating-controls.md) |
| Gate actuator | `hydr::structures::gate_actuator_capabilities` | [Gate actuators](gate-actuators.md) |
| Controlled gate | `hydr::structures::controlled_gate_capabilities` | [Controlled gates](controlled-gates.md) |
| Controlled pump | `hydr::structures::controlled_pump_capabilities` | [Controlled pumps](controlled-pumps.md) |
| Coupled level pool | `hydr::coupling::coupled_level_pool_capabilities` | [Coupled level pools](coupled-level-pools.md) |
| Channel pool | `hydr::coupling::channel_pool_capabilities` | [Channel-to-pool connections](channel-pools.md) |
| Runoff pool | `hydr::coupling::runoff_pool_capabilities` | [Runoff-to-pool connections](runoff-pools.md) |
| Surface/channel | `hydr::coupling::surface_channel_capabilities` | [Surface/channel coupling](surface-channels.md) |

## Runtime selection and admission

A runtime selects where calculations run. Its client identifies the device and execution resources.

1. Select `R: Runtime<DType = DType>`.
2. Supply the configured `R::Client` to the constructor.
3. Preserve the required arithmetic environment throughout continuation.

Components store arrays as Numr `Tensor<R>`. Allocation uses `client.device()`, and nested components reuse the client.

### Requirements for the selected runtime

| Requirement | Hydr behavior |
| --- | --- |
| F64 support | Construction checks `R::computes_float_dtype(DType::F64)` before uploading fields. |
| Basic arithmetic | Construction checks the declared contract and runs probes on the selected client. |
| Rounding | Callers preserve nearest-even rounding and gradual underflow. |
| Gate actuators | These additionally require `FusedMulAddContract::Ieee754Fused`. Basic F64 support alone is insufficient. |
| Unsupported precision or contract | Hydr returns an error without CPU fallback, host recomputation, or F32 substitution. |
| Context compatibility | Cross-context operations and trusted restore check `Device::is_same`. Restore retains the destination client. |

The `cuda` and `wgpu` features expose the corresponding Numr and Solvr implementations. Backend availability and compilation establish no hydraulic accuracy.

CPU F64 remains the checked numerical reference. These documents establish no general GPU numerical qualification.

### When data leaves the device

Ordinary `advance_to` retains accepted flow, soil memory, and recent detailed records as tensors. The host reads scalar control and accounting results.

`advance_controlled` requests copies of accepted fields and diagnostics.

State, soil, cell, diagnostic, and owned-checkpoint reads return `Result`. Handle transfer errors from these calls.

## Boundary and source admission

A boundary describes behavior at the surface domain's edge. Open boundaries represent water entering or leaving the grid.

### Declare exterior faces

| Support | Required declaration |
| --- | --- |
| Walls | Genuine impermeable exterior boundaries and inactive-cell interfaces |
| Inactive mask | Impermeable exclusions. Missing influential terrain cannot silently become a wall. |
| Open port | An active exterior face with bounded data over every forcing interval |

A source-complete port supplies water and momentum fluxes, including pressure. It also supplies compatible flow and conservative wave-speed bounds.

| Port quantity | Convention |
| --- | --- |
| Water rate | Outward signed m³/s. Negative values enter the domain. |
| Momentum | Outward x/y momentum flux integrated over face width, in m⁴/s² |
| Pressure | Include normal pressure and interface/bed corrections exactly once. |
| Eligibility | Supply a compatible regime and conservative normal signal-speed bound with provenance. |

### Choose boundary data

| Input | Supported meaning |
| --- | --- |
| `BoundaryForcing::Stage` | A compatible wet-subcritical trace at the supplied elevation |
| `BoundaryForcing::Discharge` | A compatible wet-subcritical trace with the supplied signed outward numerical water rate, in m³/s |
| `BoundaryForcing::Exterior` | Complete wet or explicitly dry maintained reservoir data, evaluated through the interior HLL solver |
| `BoundaryForcing::Transmissive` | Copy a post-source dry state or resolved supercritical outflow outside the domain. |

Declare Stage/Discharge support with `BoundaryConditions::with_open_faces`. Supply every declared face exactly once in each `ForcingInterval.boundaries`.

- Elevation: Stage uses the domain's vertical datum.
- Inflow: Supply tangential velocity. Outflow carries the interior tangential velocity.
- Reversal: Every trial checks the incoming/outgoing rule again.
- Scalar rejection: Dry, critical, and supercritical traces reject without wall or sonic substitution.
- Exterior data: Reservoir elevation and velocities prescribe neither achieved face elevation nor achieved face discharge.
- Transmissive rejection: Subcritical, critical, or incoming flow requires complete exterior data.

Flux integrals alone cannot establish a wave bound or boundary eligibility. Stage-only and discharge-only data cannot construct a source-complete port.

Read [physical boundaries](boundaries.md) for equations, tolerances, and pressure conventions. Never substitute walls for required open boundaries.

### Supply local sources

| Process | Input and order |
| --- | --- |
| Rain | Cell-supported m/s with zero horizontal momentum |
| Soil | Explicit impervious or pinned Modified Green–Ampt land with matching complete antecedent memory |
| Potential evaporation | Nonnegative m/s demand, capped after infiltration and before prescribed withdrawal |
| Withdrawal | Separate m/s demand carrying donor momentum. Unavailable water rejects without silently capping demand. |

## Results and checkpoints

### Interpret cell and face results

| Result | Meaning |
| --- | --- |
| `CellResult::Dry` | Zero depth. Physical velocity and water elevation remain undefined. |
| Positive film | The cell retains and reports its represented positive water depth. |
| `Unsupported` | Unknown or unsupported location. This never means dry zero. |

Internal coordinate ties select the positive-coordinate cell. Maximum exterior edges return unsupported.

Face records retain the accepted interval's numerical flux, orientation, and spatial support.

| Record | Contents |
| --- | --- |
| `StepDiagnostics.boundary_traces` | Accepted physical traces and compatibility residuals |
| `signal_speed` | Each face's final numerical wave speed, in m/s |
| `StepDiagnostics.transmissive_states` | Post-source interior states used by accepted transmissive boundaries |
| Boundary ledgers | Actual oriented exterior face fluxes, integrated over time, including reversals |

The application defines threshold crossings and episode semantics from accepted fields.

### Save complete restart data

1. Save the complete package, including `setup: SolverSetup`, `forcing_prefix`, and `accepted: InitialState`.
2. Preserve every package field and floating-point value exactly.
3. Restore into an engine built from the same definition and the same forcing schedule, or call `Solver::resume`.
4. Handle import errors before using the restored state.

The package holds a bounded forcing window, not the complete forcing schedule:

| Field | Contents |
| --- | --- |
| `setup.forcing` | Intervals from the one that contains the last step start. Before the first step, intervals from the one that contains the accepted time. |
| `forcing_prefix` | Count, start, end, and chained SHA-256 digest of every earlier interval. |

The digest checks that the package and the engine agree on earlier forcing. It proves no authenticity.

The destination engine can hold the complete schedule. Its forcing must start at or before the package window.

The package includes full native soil memory and accepted per-cell source records. It retains independent infiltration/evaporation ledgers, signed adjustments, and conservative budget charges.

Compatibility checks cover geometry, parameters, forcing, boundaries, controls, and required continuation history.

### Know each checkpoint's scope

Every snapshot package carries a `schema` and a `contract` identity. The schema names the package layout. The contract names the numerical contract.

Each row names a component that also declares [capabilities](#read-component-capabilities).

| Component | Package type | Schema constant (value) | Contract constant (value) | Import call | Nested packages |
| --- | --- | --- | --- | --- | --- |
| Surface solver | `SnapshotData` | `SNAPSHOT_SCHEMA` (`hydr-checkpoint-3`) | `CONTRACT_VERSION` (`hydr-swe-numr-cpu-1`) | `Solver::restore_data` | none |
| [Channel](channels.md) | `ChannelSnapshotData` | `CHANNEL_SNAPSHOT_SCHEMA` (`hydr-channel-checkpoint-2`) | `CHANNEL_CONTRACT_VERSION` (`hydr-channel-1d-numr-cpu-2`) | `ChannelReach::import_snapshot` | none |
| [Level pool](level-pool.md) | `LevelPoolSnapshotData` | `LEVEL_POOL_SNAPSHOT_SCHEMA` (`hydr-level-pool-checkpoint-1`) | `LEVEL_POOL_CONTRACT_VERSION` (`hydr-level-pool-numr-cpu-1`) | `LevelPoolBatch::restore_data` | none |
| [Subcatchment](subcatchment.md) | `SubcatchmentSnapshotData` | `SUBCATCHMENT_SNAPSHOT_SCHEMA` (`hydr-subcatchment-checkpoint-1`) | `SUBCATCHMENT_CONTRACT_VERSION` (`hydr-subcatchment-numr-cpu-1`) | `SubcatchmentBatch::restore_data` | none |
| [Reservoir](reservoir.md) | `ReservoirSnapshotData` | `RESERVOIR_SNAPSHOT_SCHEMA` (`hydr-reservoir-checkpoint-1`) | `RESERVOIR_CONTRACT_VERSION` (`hydr-reservoir-numr-cpu-1`) | `ReservoirBatch::restore_data` | none |
| [Operating control](operating-controls.md) | `OperatingControlSnapshotData` | `OPERATING_CONTROL_SNAPSHOT_SCHEMA` (`hydr-operating-control-checkpoint-1`) | `OPERATING_CONTROL_CONTRACT_VERSION` (`hydr-operating-control-numr-cpu-1`) | `OperatingControlBatch::restore_data` | none |
| [Gate actuator](gate-actuators.md) | `GateActuatorSnapshotData` | `GATE_ACTUATOR_SNAPSHOT_SCHEMA` (`hydr-gate-actuator-checkpoint-1`) | `GATE_ACTUATOR_CONTRACT_VERSION` (`hydr-gate-actuator-numr-cpu-1`) | `GateActuatorBatch::restore_data` | none |
| [Controlled gate](controlled-gates.md) | `ControlledGateSnapshotData` | `CONTROLLED_GATE_SNAPSHOT_SCHEMA` (`hydr-controlled-gate-checkpoint-1`) | `CONTROLLED_GATE_CONTRACT_VERSION` (`hydr-controlled-gate-numr-cpu-1`) | `ControlledGateBatch::restore_data` | Operating control, gate actuator |
| [Controlled pump](controlled-pumps.md) | `ControlledPumpSnapshotData` | `CONTROLLED_PUMP_SNAPSHOT_SCHEMA` (`hydr-controlled-pump-checkpoint-1`) | `CONTROLLED_PUMP_CONTRACT_VERSION` (`hydr-controlled-pump-numr-cpu-1`) | `ControlledPumpBatch::restore_data` | Operating control |
| [Coupled level pool](coupled-level-pools.md) | `CoupledLevelPoolSnapshotData` | `COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA` (`hydr-coupled-level-pool-checkpoint-1`) | `COUPLED_LEVEL_POOL_CONTRACT_VERSION` (`hydr-coupled-level-pool-numr-cpu-1`) | `CoupledLevelPoolBatch::restore_data` | Level pool |
| [Channel pool](channel-pools.md) | `ChannelPoolSnapshotData` | `CHANNEL_POOL_SNAPSHOT_SCHEMA` (`hydr-channel-pool-checkpoint-1`) | `CHANNEL_POOL_CONTRACT_VERSION` (`hydr-channel-pool-numr-cpu-1`) | `ChannelPoolBatch::import_snapshot` | Channel (per reach), level pool |
| [Runoff pool](runoff-pools.md) | `RunoffPoolSnapshotData` | `RUNOFF_POOL_SNAPSHOT_SCHEMA` (`hydr-runoff-pool-checkpoint-1`) | `RUNOFF_POOL_CONTRACT_VERSION` (`hydr-runoff-pool-numr-cpu-1`) | `RunoffPoolBatch::restore_data` | Subcatchment, coupled level pool |
| [Surface/channel](surface-channels.md) | `SurfaceChannelSnapshotData` | `SURFACE_CHANNEL_SNAPSHOT_SCHEMA` (`hydr-surface-channel-checkpoint-3`) | `SURFACE_CHANNEL_CONTRACT_VERSION` (`hydr-surface-channel-numr-cpu-3`) | `SurfaceChannelBatch::import_snapshot`, `SurfaceChannelBatch::resume` | Surface solver, channel (per reach) |

Import follows these rules:

- Import checks the schema first, then the contract. Both checks run before any other check or state change.
- A nested package carries its own identity. A nested error names its path.
- Schema mismatch: Capture a new package from this build.
- Contract mismatch: Restore with the producing build, or capture a new package.
- Import rejects a package without identity fields. Earlier layouts have none.
- Identities establish the layout and the numerical contract. They prove no authenticity.
- The surface solver identities predate the per-component naming.

| Component | Restart responsibility |
| --- | --- |
| Surface solver | Its checkpoint excludes network and operating-control state. Coordinated restoration with those components remains unsupported. Surface exchange with a channel is coordinator-only: `restore_data`, `restore`, and `resume` reject a package that carries exchange water. |
| Surface/channel | The nested surface package restores only through `SurfaceChannelBatch`. Import recalculates the last interval and names the first differing field. `continue_surface_forcing` splices a forecast under the surface splice rule. |
| [Sampled controls](operating-controls.md) and [gate actuators](gate-actuators.md) | Each exposes a separate complete snapshot. |
| [Operating-source bindings](operating-sources.md) | Reconstruct immutable bindings from original definitions. Restore each actuator before supplying fresh point-query frames. |

Source bindings retain no independent numerical clock or continuation history.

### Encode checkpoints without rounding

Save original numerical values before formatting them for display. Do not round displayed decimals and then persist those rounded values.

Persist `schema` and `contract` verbatim and first. Dispatch on `schema` before decoding the rest.

Choose an encoding that preserves every package field and these special cases:

| Value | Persistence requirement |
| --- | --- |
| Finite floats | Preserve the stored binary value, including the distinct `+0.0` and `-0.0` bit patterns. Signed zero affects immutable identity. |
| Positive infinity | Preserve it in unrestricted timestep bounds. Ordinary JSON numbers cannot represent it directly. |

Import checks internal consistency and compatibility with the model. The application manages content hashes and authenticated history.

Stored soil memory lets the calculation continue. It cannot prove past infiltration without a separately authenticated history.

### Understand restart checks

Hydr combines wave speeds into a timestep safety limit, called the CFL bound.

Import recalculates this limit from the recorded largest face wave speed along each axis.

| Stored data | Import check |
| --- | --- |
| Combined timestep safety limit | Stored and recalculated values must agree within 64 times `f64::EPSILON` in relative rounding. No fixed absolute tolerance applies. |
| Transmissive states | Check the admitted flow regime and physical flux. |
| Exterior fluxes | These remain numerical Riemann fluxes calculated by the boundary solver. |
| Forcing prefix | The prefix end must equal the first window start bit for bit. An empty prefix has equal start and end and the initial digest. |
| Forcing history | The engine folds its own earlier intervals to the package prefix. Count, start, end, and digest must match. Window intervals must match bit for bit. |
| Window coverage | The window must contain the last step's interval. The engine forcing must start at or before the window start. |

### Continue a forecast under new forcing

A forecast cycle extends or replaces future forcing without a new run. Call `continue_with_forcing` on a running engine, or `Solver::resume` on a saved package.

Accepted history stays fixed. With `t` as the accepted time:

| Interval | Rule |
| --- | --- |
| Starts before `t` | Frozen in full, including its end. New forcing that covers it repeats it bit for bit. |
| Contains `t` strictly inside | Frozen to its end. New forcing cannot start at `t` or split the interval there. |
| Ends exactly at `t` | Frozen. The next interval is free to change. |
| Starts at or after `t` | Free to change. |

Align interval boundaries with forecast cycle times. Otherwise the current interval stays frozen to its end.

The new schedule starts at or before `t`. Supply one of these splice forms:

| Form | Requirement |
| --- | --- |
| Tail | Starts at a retained interval start, or at `t` when `t` is an interval boundary. |
| Full schedule | Starts at the first folded interval. The intervals before the retained window fold to the engine's forcing prefix exactly. One interval ends exactly at the window start. |

Frozen comparison covers start, end, every rate, face, boundary variant, and port provenance. Provenance is part of forcing identity. `-0.0` differs from `0.0`.

The spliced schedule passes the `ForcingSchedule::new` checks for the engine's domain and boundaries. It covers `t`. Domain, land, boundaries, and config never change. Every check runs before any change, so an error leaves the engine as it was.

After the splice, the engine keeps only the window a checkpoint retains. Host forcing and device sources stay bounded.

To start a new epoch from an accepted state, build a fresh engine:

```rust
Solver::new(client, SolverSetup { forcing: new_schedule, ..data.setup }, data.accepted)
```

A new epoch drops the forcing prefix and all history. It starts with a fresh ledger, a fresh adjustment budget, zero steps, and no last step. `new_schedule` must start at or before `t` and end after `t`.

### Reproduce a run

1. Use the same build and dependency sources.
2. Use the same runtime and client config.
3. Preserve the same arithmetic guarantees, including rounding and underflow behavior.
4. Use the same sequence of accepted timestep lengths.
5. Preserve the same requested intermediate times when comparing runs.

The accepted timestep sequence is the step partition. Changing intermediate targets can change that sequence and the numerical result. Output cadence sets those targets. An observer or output interval that adds a target splits a step, so bitwise replay needs identical intermediate targets.

A continuation equals a single run over the spliced schedule with the same intermediate targets, bit for bit. Stop the single run at each continuation time to compare.

Cross-platform bitwise equality remains unsupported.

### Configure the application workspace

Supply Cargo patches for the checked numr and solvr sources in the application workspace.

Cargo does not inherit a dependency's local `.cargo/config.toml`. See [local dependency overrides](../CONTRIBUTING.md#local-dependency-overrides).

Numerical calls use the caller's configured client. The CPU reference explicitly selects one thread, including under dependency feature unification.

## First supported integration milestone

Start with an artificial test problem inside a native application worker:

- Closed basin: Keep water inside declared impermeable boundaries.
- Analytical open-boundary case: Supply complete prescribed water and momentum fluxes, including pressure, with compatible wave-speed bounds.

1. Save the checked checkpoint package.
2. Report the water budget.
3. Pass accepted fields to the application's data adapter.
4. Label these calculations numerical demonstrations.

### Run the native worker example

`cargo run --example native_worker` runs this milestone on the CPU reference runtime. It prints each step with a label:

- Capabilities: The worker checks its walls, flux port, rain, and impervious land against `surface_solver_capabilities()`. It refuses to run on a mismatch.
- Cancellation: The observer records accepted steps and sets a cancel flag. The worker resumes to the cycle boundary.
- Checkpoint: The worker prints `schema`, `contract`, and the forcing prefix. Restored and resumed engines match the uninterrupted run bit for bit.
- Forecast cycle: `continue_with_forcing` appends an interval at the boundary. Every continued engine matches a single spliced run bit for bit.
- Water budget: The worker prints storage change, inputs, outputs, residual, residual bound, and adjustment budget.

The shared lifecycle code is in `examples/native_worker/mod.rs`. `tests/native_worker.rs` asserts the same run.

Real urban-event use still requires network/structure coupling and independent event evidence. Domains requiring unsupported boundary regimes remain excluded.

Applications must reuse Hydr calculations instead of maintaining a competing numerical kernel.
