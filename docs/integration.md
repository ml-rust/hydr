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

| API | Application action |
| --- | --- |
| `capabilities()` | Read implemented methods, arithmetic requirements, sources, boundaries, and unsupported uses. |
| `Solver::<R>::new(client, setup, initial)` | Supply physical inputs and a configured client. Hydr checks them and allocates `Tensor<R>` state. |
| `advance_to` | Request a target time, then read the accepted time and completion status. |
| `advance_controlled` | Check cancellation between steps and receive accepted fields through an observer. |

Always read `AdvanceStatus` before treating the requested horizon as complete.

| Outcome | State retained by Hydr |
| --- | --- |
| Invalid target | No advancement occurs. |
| Numerical error | The last accepted step remains. Earlier accepted steps in the same call stay committed. |
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

Surface-to-network coupling remains unsupported.

### Channel geometry

Use [cross-section geometry](cross-sections.md) to evaluate supplied channel profiles at declared water levels. This static component provides no dynamic river or drainage flow.

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

1. Save the complete package, including `setup: SolverSetup` and `accepted: InitialState`.
2. Preserve every package field and floating-point value exactly.
3. Restore into a compatible physical definition and numerical contract.
4. Handle import errors before using the restored state.

The package includes full native soil memory and accepted per-cell source records. It retains independent infiltration/evaporation ledgers, signed adjustments, and conservative budget charges.

Compatibility checks cover geometry, parameters, forcing, boundaries, controls, and required continuation history.

### Know each checkpoint's scope

Checkpoint compatibility includes the numerical contract `hydr-swe-numr-cpu-1`.

| Component | Restart responsibility |
| --- | --- |
| Surface solver | Its checkpoint excludes network and operating-control state. Coordinated restoration with those components remains unsupported. |
| [Sampled controls](operating-controls.md) and [gate actuators](gate-actuators.md) | Each exposes a separate complete snapshot. |
| [Operating-source bindings](operating-sources.md) | Reconstruct immutable bindings from original definitions. Restore each actuator before supplying fresh point-query frames. |

Source bindings retain no independent numerical clock or continuation history.

### Encode checkpoints without rounding

Save original numerical values before formatting them for display. Do not round displayed decimals and then persist those rounded values.

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

### Reproduce a run

1. Use the same build and dependency sources.
2. Use the same runtime and client config.
3. Preserve the same arithmetic guarantees, including rounding and underflow behavior.
4. Use the same sequence of accepted timestep lengths.
5. Preserve the same requested intermediate times when comparing runs.

The accepted timestep sequence is the step partition. Changing intermediate targets can change that sequence and the numerical result.

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

Real urban-event use still requires network/structure coupling and independent event evidence. Domains requiring unsupported boundary regimes remain excluded.

Applications must reuse Hydr calculations instead of maintaining a competing numerical kernel.
