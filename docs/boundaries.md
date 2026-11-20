# Choose and supply surface boundaries

Use this guide to choose boundary data, supply each open face, and understand rejected flow conditions.

## Declare the open faces

1. Declare unique active exterior faces with `BoundaryConditions::with_open_faces`.
2. Supply one `BoundaryForcing` value per declared face in every `ForcingInterval.boundaries`.
3. Declare the remaining exterior faces and inactive-cell interfaces as genuine impermeable walls.

Required open boundaries never become reflected walls when their data reject.

### Faces a coordinator owns

A face with one active cell is a wall unless a coordinator port owns it. Its other side is an inactive cell or lies off the grid. The [surface/channel coordinator](surface-channels.md#couple-a-reach-end-to-the-surface) turns a line of such faces into a reach-end port. Each port face then carries the surface's own HLL flux against the reach end cell. The reach end is a rectangular reach end or a surveyed rectangle end. A surveyed end of any other shape rejects with `port-section-mismatch`.

- Declare a port face as a wall here. Never declare it with `with_open_faces`. A port on a caller open face rejects with `port-face-ownership`.
- Run a ported setup through `SurfaceChannelBatch`. A standalone `Solver` keeps every port face a wall.
- A surface package that holds port water restores only through the coordinator.

## Choose the input

| Variant | Supply | Numerical meaning |
| --- | --- | --- |
| `Flux(FluxPort)` | Complete outward water/momentum rates, conservative speed bound, and compatibility provenance | Provider-attested full flux, including pressure |
| `Stage(StageBoundary)` | Water elevation and incoming tangent velocity when required | Compatible wet-subcritical trace at that elevation |
| `Discharge(DischargeBoundary)` | Signed outward water rate and incoming tangent velocity when required | Compatible wet-subcritical trace with that numerical water flux |
| `Exterior(ExteriorBoundary)` | Complete wet elevation/normal/tangent velocities, or an explicit dry state | Numerical Riemann flux from a maintained exterior reservoir |
| `Transmissive(BoundaryFace)` | Explicit outgoing policy | Copy an admitted dry or supercritical-outgoing interior state outside. |

A trace is the boundary state used to calculate flux. Wet-subcritical traces require positive depth and flow below wave speed.

### Units and directions

| Quantity | Convention |
| --- | --- |
| Stage | Metres on the domain's vertical datum |
| Discharge | m³/s integrated across face width. Negative values enter the domain. |
| Trace bed | The adjacent represented cell bed. No exterior bed step is inferred. |
| Normal velocity | Positive outward, following the exterior face orientation |
| Tangential velocity | Positive y on X faces and positive x on Y faces |

Stage/Discharge inflow requires supplied tangential velocity in m/s. Outflow carries the interior tangential velocity.

Every trial checks these rules again after reversal.

## Stage and Discharge: supported flow

Both the interior and trace must remain wet and strictly subcritical.

| Input | Calculation |
| --- | --- |
| Stage | Set trace depth to `H=elevation-bed`. |
| Discharge | Solve `H*ub(H)=requested_rate/width` on the connected subcritical branch using Solvr's batched bisection. |

The trace stores prescribed conserved discharge. It derives velocity and every momentum flux from that same state.

Hydr checks discharge and velocity compatibility with the wave curve.

### Wave relation and complete flux

In outward coordinates, `hi` and `ui` are the interior depth and normal velocity. The trace uses `ub(H)=ui-fi(H)`:

```text
fi(H) = 2*(sqrt(g*H)-sqrt(g*hi))                 for H<=hi
fi(H) = (H-hi)*sqrt((g/2)*(1/H+1/hi))           for H>hi
```

The complete outward flux is:

```text
[q, q*ub + g*H²/2, q*vb]
```

The selected-runtime tensor path rotates this flux into global face coordinates.

Normal pressure enters once. This same-bed boundary receives no additional bed correction.

The completed trace bypasses HLL. Passing a ghost state through HLL cannot guarantee the prescribed numerical water flux.

The [Clawpack authors](https://www.clawpack.org/riemann_book/html/Shallow_water.html) derive the underlying wave curves and characteristic speeds. Hydr supplies this boundary's implementation and admission rules.

### Numerical controls

`SolverConfig.boundary` records these controls:

| Field | Default | Units or scale |
| --- | --- | --- |
| `discharge_absolute` | `1e-12` | m²/s compatibility residual |
| `discharge_relative` | `1e-12` | Relative discharge compatibility |
| `characteristic_relative` | `1e-12` | Relative local velocity compatibility |
| `critical_relative` | `64*f64::EPSILON` | Relative speed and discharge-endpoint ambiguity margins |

Root refinement uses an absolute depth quantum and relative depth tolerance capped at `1e-14`.

These margins establish neither a physical flood threshold nor a formal interval-arithmetic proof.

### Rejected scalar conditions

| Condition | Why Stage/Discharge rejects |
| --- | --- |
| Dry interior | No wet characteristic trace is inferred. |
| Nonpositive stage depth | No positive film is manufactured. |
| Critical state or rate endpoint | Numerical ambiguity cannot select a branch or sonic substitute. |
| Supercritical state | A scalar elevation or discharge does not supply complete admitted exterior data. |
| Missing incoming tangent | Reversed inflow needs compatible tangential data. |
| Unattainable discharge | The requested rate lies outside the connected admitted branch. |

These conditions return actionable errors. Supply an admitted complete exterior state or transmissive policy when scalar data are insufficient.

## Exterior: a maintained reservoir

| State | Required data and checks |
| --- | --- |
| `ExteriorState::Wet` | Elevation, outward-normal velocity, and tangential velocity. Positive depth equals elevation minus the adjacent bed. |
| `ExteriorState::Dry` | Exact zero depth and discharges, without an invented elevation or velocity |

Wet input checks reject nonfinite depth and overflowing discharge products. Nonzero velocity products that underflow to zero also reject.

### What the reservoir data means

The supplied values describe reservoir state. They do not prescribe the achieved face trace.

- Rewetting: A resting wet reservoir can rewet a dry domain despite zero supplied reservoir discharge.
- Incoming shocks: Strong exterior pressure can oppose initially supercritical interior outflow.
- Reversal and wet/dry interaction: These use the same numerical wave bounds as interior faces.

Hydr does not ignore exterior data because of the old interior Froude number.

### Flux calculation and limits

The tensor path replaces only the missing exterior side before the existing hydrostatic HLL calculation.

The copied bed matches the adjoining represented bed. Normal pressure and same-bed corrections follow the existing Riemann path.

The reservoir remains maintained throughout its forcing interval. It has no finite storage or donor budget.

Finite compartments require an admitted conservative coupling component. Approximate HLL fluxes establish neither an exact boundary solution nor field accuracy.

## Transmissive: admitted outflow only

The boundary copies the post-source interior state into the missing exterior side.

| Interior condition | Admission |
| --- | --- |
| Dry | Both normal and tangential discharges must equal zero. |
| Wet | Velocities must remain finite and satisfy `u_out-c > critical_relative*max(abs(u_out),c)`. |
| Subcritical, critical, or incoming | Reject and require complete exterior data. No reflected wall substitutes for the rejected policy. |

These regime checks repeat after each source update and shortened trial.

## Acceptance and water accounting

1. Restore the saved accepted state after interval reduction.
2. Recalculate sources and boundary traces for the shorter trial.
3. Include boundary outflow in the all-outgoing donor budget without crediting simultaneous inflow.
4. Include boundary signal bounds in the combined directional CFL calculation.

The solver never availability-caps a prescribed discharge. Physical admission errors trigger bounded interval reduction before an unresolved error returns.

Water ledgers integrate actual oriented exterior fluxes. Cached target rates do not replace accepted transfers.

### Read accepted diagnostics

| Record | Contents |
| --- | --- |
| `StepDiagnostics.boundary_traces` | Depth, velocities, achieved outward rate, speed bound, and compatibility residuals |
| `FaceTransfer` | Accepted spatial support, signed interval volume, and final normal `signal_speed` in m/s |
| `StepDiagnostics.transmissive_states` | Post-source depth and discharges. Dry records contain zero discharges without inferred velocity. |

## Save and restore boundary state

Snapshots retain forcing, tolerances, and trace diagnostics.

| Stored data | Restoration check |
| --- | --- |
| Physical definition and inputs | Exact floating-point identity, including signed zero in wet exterior inputs |
| Traces and faces | Physical consistency |
| Global CFL bound | Reconstruct from per-axis maxima of all accepted face speeds. Agreement requires 64 epsilon relative rounding, without an absolute unit floor. |
| Transmissive states | Reproduce the admitted physical flux and regime. |
| Exterior numerical flux | Retain its Riemann meaning. Do not compare it against the reservoir state's physical flux. |

These checks establish consistency. They establish neither historical authenticity nor a replay proof.

Same-build restart still requires the same accepted-step partition.

## Run the checks

```sh
cargo test --test physical_boundaries --test exterior_boundaries --test exterior_contracts
cargo test --lib numerics::boundary::tests
```

Read [acceptance coverage](acceptance.md) for measured errors and the limits of these checks.
