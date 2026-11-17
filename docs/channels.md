# Channel flow

Use `ChannelReach<R>` to advance an unbranched reach with explicit geometry and endpoint conditions.

Read capabilities per geometry with `rectangular_reach_capabilities()`, `triangular_reach_capabilities()`, and `surveyed_reach_capabilities()`. Each returns a `ComponentCapabilities` declaration: runtimes, endpoint policies, regimes, sources, options, unsupported uses, and limits.

## Construct a reach

Import these types from `hydr::channel`.

1. Select one `ChannelGeometry` variant for the whole reach.
2. Order its cells along the channel's positive direction.
3. Create `ChannelReachDefinition` with `geometry`, a common datum, geometry evidence, `lateral_bindings`, and `ChannelConfig`.
4. Supply `ChannelInitialState`, then call `ChannelReach::new(client, definition, initial)` with your configured Numr client.

| Geometry | Reach input | Cell input |
| --- | --- | --- |
| `Rectangular { cells }` | Ordered `RectangularChannelCell` values | Identity, length, width, bed elevation, maximum depth, and Manning roughness |
| `PrismaticTriangular` | Left/right bank slopes, common bed elevation, and ordered `TriangularChannelCell` values | Identity, length, maximum depth, and Manning roughness |
| `PrismaticSurveyed` | One `CrossSectionDefinition`, ordered `SurveyedChannelCell` values, and `ChannelSurveyedConfig` as `numerical` | `id`, positive `length_m` in metres, and nonnegative `manning_n` |

The solver uses the declared physical sections. Geometry remains fixed during advancement.

### Supply one surveyed profile

Choose `PrismaticSurveyed { profile, cells, numerical }` for a horizontal reach. Every cell shares the original surveyed cross-section. Cell lengths and Manning roughness can differ.

Supply the [surveyed-profile inputs](cross-sections.md#supply-the-surveyed-profile), including datum, provenance, and `maximum_stage_m`. The profile must meet [prismatic certification requirements](prismatic-sections.md#check-whether-the-profile-qualifies). Its datum must match the reach datum.

Hydr certifies the profile during construction. Positive roughness additionally requires [certified wetted perimeter](prismatic-sections.md#add-wetted-perimeter-and-hydraulic-radius-bounds). Zero roughness avoids that additional certificate.

Supply your configured `R::Client`. Surveyed calculations run on CPU or CUDA F64 with the required certified arithmetic. Unsupported runtimes reject without fallback.

### Supply triangular bank slopes

Set `left_bank_slope_h_per_v` and `right_bank_slope_h_per_v` in horizontal metres per vertical metre. Both slopes are nonnegative. At least one slope must be positive.

Set `bed_elevation_m` once for the entire triangular reach. Every triangular cell shares this bed and both slopes. Length, maximum depth, and roughness can differ between cells.

For depth `h`, define `k = (left_bank_slope_h_per_v + right_bank_slope_h_per_v) / 2`. The submerged area is `A = k h²`. Slopes `(0, 2)` and `(1, 1)` give equal areas at equal depths. Their wetted perimeters differ, so Manning resistance differs.

A triangular reach requires a runtime that declares correctly rounded F64 square roots. Unsupported arithmetic rejects construction without fallback.

### Supply initial conserved state

Initial arrays contain one value per ordered cell:

| Quantity | Input | From depth and velocity |
| --- | --- | --- |
| Water volume | `volume_m3`, in m³ | Submerged area × cell length |
| Longitudinal momentum divided by water density | `longitudinal_momentum_m4_per_s`, in m⁴/s | Water volume × velocity |

For rectangles, submerged area is width × depth. For triangles, submerged area is `k h²`.

For surveyed cells, obtain area from the shared profile within its supported capacity. Advancement retains `A = volume_m3 / length_m` and `Q = longitudinal_momentum_m4_per_s / length_m` as authoritative area and discharge.

Stored volume and momentum remain authoritative. Depth recovery never replaces stored water with a reconstructed volume.

Velocity is positive along the cell order. A dry cell requires zero volume and zero momentum.

The maximum depth limits the supported geometry. Hydr rejects unresolved positive water and depths above that limit.

## Choose both endpoint conditions

Supply both endpoints through `ChannelForcingInterval`. Keep the supplied values constant throughout the interval.

A boundary trace gives the depth and velocity used to calculate flow across an endpoint.

Scalar policies prescribe one hydraulic quantity and solve for the remaining trace state. They require a runtime that declares correctly rounded F64 square roots. Unsupported arithmetic rejects the interval without fallback.

Surveyed reaches accept only `Wall` or complete `Exterior` data at both endpoints. `Stage`, `Discharge`, and `QuiescentReservoir` remain unsupported for surveyed geometry. The scalar-policy sections below apply to rectangular and triangular reaches.

| Policy | Required input | Meaning |
| --- | --- | --- |
| `Wall` | A genuine impermeable end | Reflects flow and transfers no water across that end |
| `Exterior` | Depth and longitudinal velocity | Describes exterior water using the declared endpoint section and bed |
| `Stage { elevation_m }` | Water-surface elevation in the reach datum | Solves a compatible wet-subcritical trace velocity |
| `Discharge { discharge_m3_per_s }` | Full-section longitudinal discharge in m³/s | Solves compatible wet-subcritical depth and velocity |
| `QuiescentReservoir` | Reservoir elevation and explicit entrance-loss coefficient | Includes entrance velocity head and complete exit mixing |

Exterior depth uses metres above the local bed. Exterior velocity uses m/s in the channel's positive direction.

An exterior state does not prescribe achieved discharge. The numerical interface calculation determines that discharge from both sides.

Missing outlet information requires corrected inputs. A wall cannot replace an unknown outlet.

### Use stage and discharge within their admitted regime

Stage and Discharge each prescribe one boundary quantity. Both require wet interior water and wet boundary traces. Flow speeds must remain below local gravity-wave speeds.

Hydr uses the interior flow to determine the missing boundary value. The compatible wave must travel into the reach.

Stage uses absolute elevation in the declared datum. Depth equals elevation minus the endpoint bed. Discharge uses the same longitudinal sign as cell results and exterior velocity.

| Endpoint | Positive discharge | Negative discharge |
| --- | --- | --- |
| Left | Enters the reach | Leaves the reach |
| Right | Leaves the reach | Enters the reach |

Flow reversal remains supported inside the wet-subcritical branch. States outside that resolved branch reject.

Requests outside the certified range reject. Unresolved calculations reject unless the cap approximation below applies.

Hydr inserts the compatible physical endpoint flux directly. Prescribed discharge retains the original operand instead of reconstructing it through width or area.

Fresh scalar traces participate in every retry and the final state check after friction. Their limits also restrict source-created candidate states. The reported candidate CFL includes both checks when both apply.

Discharge accepts held rates. Integrated endpoint volumes remain unsupported.

### Use a quiescent reservoir

Choose `QuiescentReservoir { elevation_m, entrance_loss_coefficient }` when reservoir velocity is negligible. Supply elevation in the reach datum.

The entrance-loss coefficient `K` must be finite and nonnegative. It refers to channel trace velocity. `K=0` explicitly selects an ideal entrance without entrance loss.

Velocity head, `u²/(2g)`, expresses kinetic energy as an equivalent water height. The relation assumes a uniform section velocity profile (`α=1`).

| Flow direction | Head relation | Physical assumption |
| --- | --- | --- |
| Reservoir into channel | Reservoir elevation equals trace elevation plus `(1+K)` times velocity head | Reservoir head supplies acceleration and entrance loss |
| Channel into reservoir | Reservoir elevation equals trace elevation | Complete mixing dissipates incoming kinetic energy |

Here `u` is outward channel velocity and `g` is gravity. Hydr retains the complete channel momentum and pressure flux in both directions.

This model applies the [energy equation](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.1/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/1d-steady-flow-water-surface-profiles/equations-for-basic-profile-calculations) with [entrance losses and exit mixing](https://www.hec.usace.army.mil/confluence/hmsdocs/hmsum/4.7.1/reservoir-elements/outflow-structures-routing).

Interior water and the boundary trace must remain wet and strictly subcritical. Resolved reversal follows the same continuous head law.

This policy holds the supplied reservoir elevation over its forcing interval. It does not advance finite reservoir storage.

For rectangular or triangular channels whose pool levels respond to transfers, use [channel-to-pool connections](channel-pools.md).

Use this approximation only where rapid reservoir mixing is appropriate. Jets, seiches, dry connections, and critical transitions remain unsupported.

Hydr rejects a result when it cannot establish whether the depth exceeds the declared maximum. Reservoir boundaries do not use the Discharge cap approximation.

### Read scalar compatibility controls

`ChannelConfig.boundary` contains `ChannelBoundaryConfig`. These controls limit mismatches between the computed trace and the interior wave relation. Conservation allowances and flux-width budgets remain separate.

| Field | Default | Quantity |
| --- | --- | --- |
| `discharge_absolute_m3_per_s` | `1e-12 m³/s` | Absolute discharge mismatch allowance |
| `discharge_relative` | `1e-12` | Relative discharge mismatch allowance |
| `momentum_absolute_m4_per_s2` | `1e-12 m⁴/s²` | Absolute momentum-flux mismatch allowance |
| `momentum_relative` | `1e-12` | Relative momentum-flux mismatch allowance |
| `velocity_relative` | `1e-12` | Relative velocity mismatch allowance |
| `critical_relative` | `64 × f64::EPSILON` | Minimum relative gap between flow speed and wave speed |
| `reservoir_head_absolute_m` | `1e-12 m` | Absolute reservoir-head mismatch allowance |
| `reservoir_head_relative` | `1e-12` | Relative reservoir-head mismatch allowance |

Each discharge or momentum budget adds absolute and relative allowances at the corresponding physical scale. The velocity allowance uses the largest flow or gravity-wave speed magnitude at either state. No absolute velocity floor applies.

Controls must be finite and nonnegative. Each discharge/momentum pair requires a positive value. Velocity tolerance remains positive. Critical separation lies strictly between zero and one.

The reservoir-head pair also requires a positive allowance. Its relative scale uses local depths and velocity/loss heads. Absolute datum elevation never increases the allowance.

## Declare lateral water inputs

Lateral inputs add or remove water at named channel cells. Declare each connection through `ChannelReachDefinition.lateral_bindings`.

1. Assign each binding a unique `boundary_id` and an existing `cell_id`. Hydr reserves the `surface-exchange:` prefix for surface exchange and rejects a `boundary_id` or lateral input that starts with it.
2. Select its fixed direction, `Inflow` or `Withdrawal`.
3. Supply an `ownership_assertion` explaining why this water does not duplicate another source.
4. Include exactly one matching `ChannelLateralInput` per binding in every interval's `lateral_inputs`, including zeros.

Hydr rejects missing, duplicate, unknown, and direction-mismatched inputs. The ownership assertion records your evidence without proving geographic coverage.

| Direction | Velocity input | Momentum effect |
| --- | --- | --- |
| `Inflow` | Explicit `longitudinal_velocity_m_per_s` | Adds incoming volume × supplied velocity |
| `Withdrawal` | None | Removes volume at the cell's accepted-start velocity |

Positive incoming velocity follows the channel's cell order. A dry donor cannot supply a positive withdrawal.

Face outflows and lateral withdrawals share the same donor budget. Incoming face water and lateral inflow provide no donor credit during that interval.

An empty `lateral_bindings` vector selects a reach without lateral inputs. Supply an empty `lateral_inputs` vector for that reach.

## Choose a rate or an interval volume

Each lateral input's `amount` uses `hydr::forcing::WaterAmount`. All amounts are nonnegative.

| Variant | Supplied quantity | Shorter interval |
| --- | --- | --- |
| `Rate { rate_m3_per_s }` | Constant flow in m³/s | Hydr recalculates volume from the accepted duration |
| `IntervalVolume { volume_m3 }` | Integrated volume in m³ for the exact span | Caller recomputes every contributing input |

An interval can mix rates and integrated volumes. Any integrated volume, including zero, requires exact start and requested-end time bits.

For integrated inputs, set `start_s` from `reach.time()` and `end_s` to the requested endpoint. Hydr retains volume operands without conversion through rates.

Hydr rejects positive source contributions that disappear during volume preparation or state assembly.

If a shorter interval is necessary, Hydr returns `ChannelError::RecomputeInterval` without accepting state.

1. Read `suggested_end_s` from that error.
2. Recompute every contributing source for the suggested span.
3. Submit fresh complete forcing with that endpoint.

Never scale or slice original integrated inputs. Successful integrated forcing reaches its requested endpoint with zero internal retries.

## Advance to a requested time

1. Read the accepted time with `reach.time()`.
2. Create a fresh interval whose `start_s` equals that returned value exactly.
3. Supply `end_s` through which all forcing remains valid, respecting exact-volume span requirements.
4. Call `reach.step(requested_end_s, &interval)` within that support.
5. Replace the reach with the returned `.next` and repeat until it reaches your target.

Each successful call accepts one interval. Rate-only forcing supports automatic shortening for wave speeds, donor water, and `max_step`.

When endpoint inputs change, end their support at that change. Supply new states for the next interval.

An error leaves the original reach unchanged. Hydr retries complete calculations without clipping negative water or scaling individual fluxes.

### Supply inputs already on the device

Use `step_tensor(requested_end_s, &interval)` when another calculation supplies Numr tensors. Construct `ChannelTensorForcingInterval<R>` on the reach's configured device.

| Input | Tensor type |
| --- | --- |
| Endpoint policy | `ChannelTensorBoundary<R>` |
| Lateral inflow or withdrawal | `ChannelTensorLateralInput<R>` |
| Lateral rate or integrated volume | `ChannelTensorWaterAmount<R>` |

Each numerical parameter requires an F64 tensor with shape `[1]`. Valid offset and strided views are accepted. Wrong devices, shapes, and dtypes reject.

The tensor interface uses the same units, signs, physical limits, and exact-volume span rules as host forcing. Original operands remain on the selected device during advancement.

Hydr copies tensor inputs into private device storage. Reusing caller buffers after the call cannot change accepted receipts or snapshots.

The host `step()` interface uploads its operands once into this shared path. Explicit checkpoint-data calls copy retained operands to host values.

For rectangular or triangular channels, [channel-to-pool connections](channel-pools.md) coordinate transfers with finite storage. Surveyed pool connections remain unsupported.

The [native-forcing checks](../tests/channel_tensor_forcing.rs) compare complete host and tensor continuation.

### How sources affect the timestep

Transport uses the accepted-start state. Lateral inflow changes transport on the following interval.

Nonzero lateral inputs also trigger a wave-speed check on the complete candidate state before friction. Candidate fluxes never supply current transport.

This restriction checks start and candidate endpoint states. It does not prove stability along a continuous source path.

Complete zero lateral inputs preserve the original flow arithmetic and timestep sequence. Integrated zeros still require exact spans and recomputation when shortened.

## Read accepted results

| Method | Result |
| --- | --- |
| `state()` | Cell volume, depth, discharge, and longitudinal momentum, with optional stage and velocity |
| `ledger()` | Storage, separate endpoint and lateral totals, the water residual, and cumulative numerical bounds |
| `lateral_ledgers()` | Cumulative accepted water and counter-rounding bounds for each lateral binding |
| `last_step()` | Shared face transfers, pressure and friction changes, wave limits, and receipts for each lateral binding |

Before the first accepted step, `last_step()` returns `None`.

Dry results contain zero depth and an explicit dry flag. Stage and velocity are absent where no represented water exists.

Face transfers follow increasing cell order. The same water volume enters one adjacent cell and leaves the other.

Lateral receipts retain original inputs and accepted volumes. They report applied velocity, signed momentum change, and cumulative counter additions.

A zero withdrawal from a dry cell has no applied velocity.

### Inspect scalar endpoint receipts

`ChannelStepDiagnostics.boundary_traces` contains left/right optional `ChannelBoundaryTrace` values. Scalar policies produce traces. Wall and Exterior entries remain absent.

These traces describe the accepted interval's flux calculation.

| Receipt group | Recorded quantities |
| --- | --- |
| Physical trace | Depth, area, outward velocity/discharge, physical momentum flux, and signal-speed bound |
| Compatibility | Velocity mismatch and induced water/momentum mismatch bounds |
| Flux accuracy | Water/momentum flux widths and maximum contributing root iterations |
| Cap status | `cap_compatibility_approximation` marks tolerance-based cap admission |

Receipt discharge uses the outward sign. This differs from the longitudinal sign of the supplied Discharge operand. Prescribed discharge has zero water-flux width.

A rounded discharge can fall inside a certified uncertainty band at the original geometry cap. Approximate admission keeps the original discharge and reports bounded compatibility mismatches. Both delivered and compatible velocities must remain strictly subcritical.

An approximate cap trace does not establish an exact cap root. Certified targets outside the cap range reject. Compatibility bounds never enlarge conservation allowances.

Reservoir traces additionally contain `reservoir: Some(ChannelReservoirTrace)`. Its depth bounds contain the solution of the reservoir head relation. The head-mismatch bound measures disagreement with that law.

`dissipation_head_lower_m` and `dissipation_head_upper_m` bound the head lost through entrance resistance or exit mixing. These local bounds establish no energy guarantee for a complete finite-storage calculation.

### Read flux accuracy

Triangular and surveyed intervals report face-flux uncertainty in `ChannelStepDiagnostics.riemann`. Rectangular intervals return `None`.

Surveyed receipts also retain cumulative root and integration work in `surveyed_work`.

To calculate interface fluxes outside a reach, call `riemann_flux_points_tensor` on a certified section batch. See [exact conserved points](prismatic-fluxes.md#calculate-fluxes-from-exact-conserved-points). Reach advancement does not call it.

Scalar boundary receipts report endpoint flux widths for rectangular and triangular reaches. All use the same budgets below.

| Diagnostic | Meaning |
| --- | --- |
| `water_flux_width_m3_per_s` | Water-flux bracket width at each face, in m³/s |
| `momentum_flux_width_m4_per_s2` | Momentum-flux bracket width at each face, in m⁴/s² |
| `max_water_flux_width_m3_per_s` | Largest water-flux width across faces |
| `max_momentum_flux_width_m4_per_s2` | Largest momentum-flux width across faces |
| `root_iterations` | Maximum root iterations across faces, zero for analytical samples |

A bracket encloses the possible physical flux while accounting for numerical uncertainty. Hydr accepts a sample only when both widths meet their configured accuracy budgets.

| `ChannelConfig` field | Default | Purpose |
| --- | --- | --- |
| `riemann_water_flux_absolute_m3_per_s` | `1e-12 m³/s` | Absolute water-flux width allowance |
| `riemann_momentum_flux_absolute_m4_per_s2` | `1e-12 m⁴/s²` | Absolute momentum-flux width allowance |
| `riemann_flux_relative` | `1e-12` | Relative width allowance for both fluxes |

Each budget adds its absolute allowance to the relative allowance times the largest absolute bracket endpoint. Values must be finite and nonnegative. Each absolute/relative pair requires at least one positive value.

These controls are separate from water and momentum conservation tolerances. They do not enlarge conservation allowances or bound total solution error.

### Read water and momentum limits

Water balances include lateral inflow and withdrawal. Lateral totals derive from per-binding counters, without competing copies of physical water.

| Numerical bound | What it covers | Existing allowance |
| --- | --- | --- |
| Computed water transfers | Face products rounded to zero, counting interior faces twice | Interval and lifetime water tolerance |
| Lateral water assembly | Source aggregation and additions to cell water | Interval and lifetime water tolerance |
| Lateral counters | Cumulative volume additions for each binding | Lifetime water tolerance |
| Momentum | Pressure/momentum products rounded to zero and lateral assembly | Separate cell momentum tolerance |

Bounds consume the configured tolerance. They never enlarge the physical water scale or remove stored water.

Hydr rejects intervals whose residuals and bound charges exceed the allowance. Water tolerance cannot admit an excessive momentum error.

These bounds cover recorded arithmetic effects only. They do not bound the full numerical solution error.

State and detailed receipt methods copy arrays from the device and return errors if that copy fails. Advancement keeps arrays on the selected runtime.

## Save and restore

Construct the destination with the same definition, config, and original initial state. Restoration returns a new reach.

| Use | Save | Restore |
| --- | --- | --- |
| In-memory checkpoint on the same device | `snapshot()` | `restore(&snapshot)` |
| Application-owned persistence | `snapshot_data()?` | `import_snapshot(&data)?` |

Both restoration paths retain the destination's configured client. Importing saved data recalculates the complete last interval, including lateral inputs, counters, scalar traces, and triangular flux certificates.

Surveyed restart data also retain the original profile, surveyed numerical controls, flux certificates, work, and friction-geometry receipts. Import recalculates the complete surveyed interval. Modified profile or numerical-control bits reject restoration.

Geometry variants and original geometry bits bind both restart paths. Changed triangular slopes or bed bits reject restoration. Equal area laws do not make different bank declarations interchangeable.

Each stored conserved value is a high part plus a normalized low part. The application must persist every low part with its high part, bit for bit:

| Package type | Fields |
| --- | --- |
| `ChannelSnapshotData` | `volume_low_m3`, `longitudinal_momentum_low_m4_per_s` |
| `ChannelIntervalSnapshotData` | `previous_volume_low_m3`, `previous_longitudinal_momentum_low_m4_per_s` |
| `ChannelWaterLedger` | `storage_selection_abs_bound_m3` |

`ChannelStepDiagnostics` reports the matching accepted-interval records: `face_volume_low_m3`, `face_momentum_low_m4_per_s`, `friction_impulse_low_m4_per_s`, `storage_selection_bound_m3`, `recovery_selection_bound_m2`, and `discharge_recovery_selection_bound_m3_per_s`. The low parts are zero outside surveyed geometry.

Persistence must preserve every floating-point bit. Step bounds can contain positive infinity, which ordinary JSON numbers cannot represent.

Exact continuation requires the same build, dependency sources, runtime arithmetic, and sequence of accepted timesteps. Different intermediate targets can change those steps.

Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `CHANNEL_SNAPSHOT_SCHEMA` and `CHANNEL_CONTRACT_VERSION`. `import_snapshot` compares both before any other check and rejects a foreign value with an input error. The error names the found and expected values. Capture a new package to recover. Packages without the identity fields are rejected. These include earlier layouts and packages with surveyed low parts.

Import checks continuation consistency. The application owns content hashes and the authenticity of earlier history.

## Understand the numerical method

The solver uses finite volumes and forward Euler time integration. It represents one depth-averaged longitudinal velocity.

| Part | Rectangular reach | Prismatic triangular reach |
| --- | --- | --- |
| Pressure | [Roberts–Wilson reconstruction](https://journal.austms.org.au/ojs/index.php/ANZIAMJ/article/view/3948/1542) balances bed and width forces at rest | Hydrostatic pressure is `p = g A h / 3` on the common bed |
| Interior transport | Width-aware HLL with bracketed wave bounds | Complete barotropic Godunov sampling at the stationary interface |
| Step size | Width-weighted inward wave limit | Reconstructed half-cell CFL for `LimitedLinear`, full-cell CFL for `FirstOrderReference` |
| Friction | Wetted perimeter includes the bed and both vertical sides | Wetted perimeter uses both declared bank slopes |

Each face supplies one shared flux, including dry fronts. Every geometry uses a separate donor limit for outgoing water.

The velocity profile is uniform across each section (`β = 1`). Split Manning resistance holds area fixed during decay.

### Bound the channel timestep

Each face carries a signed wave-speed interval `[lower, upper]`. Hydr splits it into two inward speeds.

| Face speed | Enters | Value |
| --- | --- | --- |
| Rightward | The cell right of the face | `max(upper, 0)` |
| Leftward | The cell left of the face | `max(-lower, 0)` |

The interval need not contain zero. A supercritical face gives zero speed toward its upstream cell. A scalar boundary trace supplies one unsigned bound. Hydr uses that bound for both directions.

Each cell frequency is `(s_in_left + s_in_right) / (2 L)`. `s_in_left` is the rightward speed at its left face. `s_in_right` is the leftward speed at its right face. A wave that leaves the cell adds nothing.

Rectangular reaches multiply each speed by its face width. They divide the sum by `2 L` times the cell width.

`LimitedLinear` triangular reaches split each cell at its centre into two half-cells. Each half-cell has length `L / 2` and uses the same rule. Its inner face supplies the internal reconstructed wave interval.

A face with water on exactly one side borders a dry cell. A state is dry when `A = 0`. The face Riemann problem sees only the cell-averaged tip state. Its dry-front speed can fall below the true front speed. The true front carries the Riemann invariant `u + φ(A)` of the water behind it, with `φ(A) = ∫_0^A c(a)/a da`. Godunov-type updates keep every state inside the invariant region of its neighbourhood. Within one step the front therefore comes from the wet run beside the dry cell.

Hydr raises the inward speed into each dry cell to the dry-front bound of that run:

| Dry cell | Run | Bound |
| --- | --- | --- |
| Right of the face | Contiguous wet states that end at the face | Maximum of `u + φ(A)` |
| Left of the face | Contiguous wet states that start at the face | Maximum of `-(u - φ(A))` |

The run includes both face states of each cell and a wet exterior or scalar boundary trace. A scalar trace widens its velocity by its compatibility defect. A dry cell takes the larger of this bound and its face speed. Water beyond the nearest dry cell does not restrict the step.

| Geometry | `φ(A)` | Enclosure |
| --- | --- | --- |
| Rectangular | `2 sqrt(g h)` | Face-bound padding of `16 ε` of the local speed scale |
| Triangular | `4 sqrt(g h / 2)` | Directed rounding of the side signal bounds |
| Surveyed | Certified potential integral | Interval velocity plus the certified potential at the upper area bound |

Surveyed reaches evaluate one certified potential per state. `φ` increases with area, so its value at the upper area bound serves both directions. Hydr evaluates it only when some face has water on exactly one side.

The run maximum uses a segmented scan on the device. Each pass doubles the scan reach, so a reach of `n` faces takes `ceil(log2 n)` passes.

Lateral and friction sources act after transport within a step. They move no water between cells in that step. The accepted-start bound therefore covers the transport. A candidate check reads the fronts again from the post-source state. Without a candidate check, the next accepted start reads the changed state.

The step limit is `cfl / max frequency`. Inward waves then cross at most `2 cfl` of a cell in one step. The `0.45` cap keeps that share at `0.9`. The first-order Godunov non-interaction limit is one cell. The surface solver uses the same share of its own limit.

### Set surveyed numerical controls

Surveyed reaches use piecewise constant states and the [surveyed interface solver](prismatic-fluxes.md) for shared water and momentum fluxes. Complete wave bounds restrict the timestep. `triangular_reconstruction` does not select surveyed reconstruction.

`ChannelConfig` alone supplies gravity and the flux-width budgets listed above. Surveyed advancement retains those defaults, which differ from standalone `PrismaticRiemannConfig` defaults.

Use `ChannelSurveyedConfig` for depth recovery, root solving, integration work limits, and friction-geometry allowances.

See [root controls](prismatic-fluxes.md#choose-bounds-and-work-limits) for search and bisection limits. `normalized_root` supplies the root controls for a reach. Integration panels divide each integration interval. Point ceilings bound native root and integration evaluations.

With positive roughness, split Manning resistance uses the original area and bounded hydraulic radius. `friction_geometry_atol_m4_per_s` and `friction_geometry_rtol` bound momentum uncertainty from radius selection. `ChannelStepDiagnostics.friction_geometry` records that bound.

This bound covers radius selection only. It excludes native power arithmetic and time-splitting errors.

Unresolved geometry, capacity, fluxes, or exhausted work limits return errors without exposing partial face results. Inspect the error before changing inputs or work limits. A shorter interval cannot resolve every geometry or work error.

Surveyed reaches use the same lateral rate, exact-volume, immutable-step, and restart interfaces described above. Functional CPU/CUDA source and restart checks establish no reach convergence or measured-event accuracy. Spatial, temporal, and wet/dry-front acceptance remain incomplete.

### Select triangular reconstruction

Set `ChannelConfig.triangular_reconstruction` with `ChannelTriangularReconstruction`. Rectangular reaches retain their existing method.

| Variant | Spatial representation | Use |
| --- | --- | --- |
| `LimitedLinear` | Slope-limited piecewise linear reconstruction | Production default and triangular physical acceptance gates |
| `FirstOrderReference` | Piecewise constant cell states | Explicit reference calculations and the full-cell film CFL comparison |

Both variants use the same Godunov sampler and flux-width budgets. Forward Euler remains first order in time. Selecting linear reconstruction does not establish second-order accuracy for complete wet/dry calculations.

Reconstruction selection binds restart identity. Restore with the same selection used for the saved run.

### Sample triangular flow at each face

Godunov sampling uses the local Riemann solution: the waves produced when two neighboring constant states meet. Hydr samples that solution at the stationary interface to calculate water and momentum fluxes.

| Local wave pattern | Calculation |
| --- | --- |
| Dry states, vacuum, or two rarefactions | Analytical wave classification and interface sampling |
| Mixed shock/rarefaction or two shocks | Solvr brackets the intermediate depth, then Hydr samples the admitted wave region |
| Uncertain classification | Flux brackets cover every possible region before acceptance |

A vacuum sample supplies zero water and momentum flux together. No depth threshold, stored-state reset, or individual flux scaling occurs.

Numr supplies runtime storage and tensor operations. Solvr supplies generic root solving for triangular shock patterns and rectangular wave bounds. Root convergence alone cannot bypass the separate triangular flux-width budgets.

### Bound the triangular timestep

Gravity-wave speed is `c = sqrt(g h / 2)`. Hydr encloses wave speeds using invariant-region bounds and a lower wave-speed floor. Certified expanding states use their outer rarefaction heads.

These proved wave bounds determine CFL limits only. The Godunov sample determines the triangular flux.

`LimitedLinear` checks actual reconstructed half-cell wave limits under the [channel timestep rule](#bound-the-channel-timestep). Those limits can be stricter than the full-cell reference rule. Read the accepted interval's reported CFL limits when advancing either variant.

The primary basis is [Jiang–Liu](https://faculty.sites.iastate.edu/hliu/files/inline-files/JL_JCP_v373_2018.pdf), Example 1 and section 3.1, for invariant regions. [Chen–Liu](https://faculty.sites.iastate.edu/hliu/files/inline-files/Chen-Liu_SIMA01.pdf), section 2.2, supplies barotropic shock and rarefaction curves. Hydr derives its triangular specialization and directed arithmetic from that basis.

Positive triangular films retain positive water without a dry threshold. Their wave bounds stay finite near dry fronts.

## Understand the physical limits

The solver represents hydrostatic, one-dimensional shallow-water flow. It excludes transverse jets and strong channel curvature.

- Geometry: Rectangular width and bed can vary between cells. Triangles require one common bed and bank slopes. Surveyed reaches require one horizontal profile.
- Profiles: Certified surveyed profiles support reach advancement. Mixed sections, longitudinally varying surveyed or triangular profiles, and changing geometry remain unsupported.
- Boundaries: Scalar policies require resolved wet-subcritical traces. Complete exterior data supports other flow regimes. Use walls only for physical closures. Transmissive policies remain unsupported.
- Connections: [Channel-to-pool connections](channel-pools.md) coordinate rectangular or triangular reaches with finite pool storage. Surveyed attachments, general junctions, and structures remain unsupported.
- Surface: [Surface/channel coupling](surface-channels.md) exchanges water between surface cells and rectangular or triangular reach cells over bank weirs.
- Losses: Separate abrupt contraction and expansion losses remain unsupported.
- Drainage: This solver supplies no SWMM drainage, surcharge, or pressurized-pipe model.
- Accuracy: Read [acceptance coverage](acceptance.md) for numerical cases and limits. Numerical agreement establishes no measured-event accuracy.
