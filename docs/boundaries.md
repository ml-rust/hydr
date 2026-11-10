# Physical boundary contract

Hydraulic applications supply explicit data for every declared open face over each forcing interval.

## Inputs and support

`BoundaryConditions::with_open_faces` checks unique active exterior support.
`ForcingInterval.boundaries` supplies one `BoundaryForcing` value for each declared face.
All remaining exterior faces and inactive-cell interfaces remain explicitly declared genuine walls.

| Variant | Supplied physical data | Numerical meaning |
| --- | --- | --- |
| `Flux(FluxPort)` | Complete outward water/momentum rates, conservative speed bound, and compatibility provenance | Provider-attested full flux including pressure |
| `Stage(StageBoundary)` | Water elevation and optional incoming tangent velocity | Compatible wet-subcritical trace at the prescribed elevation |
| `Discharge(DischargeBoundary)` | Signed outward water rate and optional incoming tangent velocity | Compatible wet-subcritical trace with prescribed numerical water flux |
| `Exterior(ExteriorBoundary)` | Complete wet elevation/normal/tangent velocities, or explicit dry state | Numerical Riemann flux from an externally maintained reservoir |
| `Transmissive(BoundaryFace)` | Explicit outgoing policy | Copies admitted dry or supercritical-outgoing interior state outside |

Stage elevation uses metres on the domain vertical datum.
Discharge uses m³/s integrated across the face width. Negative values enter the domain.
The trace bed equals the adjacent represented cell bed. No exterior bed-step model is inferred.

Outward normal velocity follows the exterior face orientation.
Tangential velocity follows the positive other coordinate: positive y on X faces, positive x on Y faces.
Stage/discharge inflow requires supplied tangential velocity in m/s. Its outflow carries the interior tangential velocity.
Reversal rechecks these rules during every trial.

## Compatible Stage/Discharge trace

The admitted interior and trace states are wet and strictly subcritical.
In outward coordinates, let the interior depth and normal velocity be `hi` and `ui`.
The trace uses the entropy-admissible one-wave relation `ub(H)=ui-fi(H)`:

```text
fi(H) = 2*(sqrt(g*H)-sqrt(g*hi))                 for H<=hi
fi(H) = (H-hi)*sqrt((g/2)*(1/H+1/hi))           for H>hi
```

Stage supplies `H=elevation-bed`.
Discharge solves `H*ub(H)=requested_rate/width` on the connected subcritical branch through solvr's batched bisection.
The trace stores the prescribed conserved discharge and derives velocity and every momentum flux from that consistent state.
It also checks discharge and velocity compatibility with the wave curve.

The complete outward flux is:

```text
[q, q*ub + g*H²/2, q*vb]
```

The selected-runtime tensor path rotates this flux into global face coordinates.
The flux includes normal pressure once and receives no additional bed correction on that same-bed boundary.
It bypasses HLL after trace construction. A ghost state's prescribed discharge does not guarantee HLL's numerical water flux.

The [Clawpack authors](https://www.clawpack.org/riemann_book/html/Shallow_water.html) derive the underlying shallow-water wave curves and characteristic speeds.
This boundary contract uses hydr's implementation and admission rules.

## Tolerances and rejected regimes

`SolverConfig.boundary` records the numerical controls:

| Field | Default | Units or scale |
| --- | --- | --- |
| `discharge_absolute` | `1e-12` | m²/s compatibility residual |
| `discharge_relative` | `1e-12` | Relative discharge compatibility |
| `characteristic_relative` | `1e-12` | Relative local velocity compatibility |
| `critical_relative` | `64*f64::EPSILON` | Relative speed and discharge-endpoint ambiguity margins |

Root refinement uses an absolute depth quantum and relative depth tolerance capped at `1e-14`.
These computational margins establish no physical flood threshold or formal interval-arithmetic proof.

- Dry interior: No wet characteristic trace is inferred.
- Nonpositive stage depth: No positive film is manufactured.
- Critical state or rate endpoint: No branch or sonic substitute is selected through numerical ambiguity.
- Supercritical state: Scalar stage/discharge data do not supply a complete admitted exterior state.
- Missing incoming tangent: Reversed inflow rejects until compatible physical data are supplied.
- Unattainable discharge: The solver rejects requests outside the connected admitted branch.

These cases return actionable errors. Required open boundaries never become reflected walls.
These scalar inputs do not infer missing full exterior data. Use an admitted complete-state or transmissive policy instead.

## Complete exterior states and transmissive outflow

`ExteriorState::Wet` supplies elevation, outward-normal velocity, and tangential velocity.
Its positive depth equals elevation minus the adjacent cell bed.
Input checks reject nonfinite depth, overflowing discharge products, and nonzero velocity products that underflow to zero.
`ExteriorState::Dry` supplies exact zero depth and discharges, without invented elevation or velocity.

The selected-runtime tensor path replaces only the missing exterior side before the existing hydrostatic HLL calculation.
The copied bed matches the adjoining represented bed. Normal pressure and same-bed corrections follow the existing Riemann path.
The exterior state remains maintained throughout its forcing interval. No finite exterior storage or donor budget exists.
Finite compartments require an admitted conservative coupling component.

The input describes reservoir state rather than a face trace.
A resting wet reservoir can rewet a dry domain despite its zero supplied reservoir discharge.
Strong exterior pressure can send a shock inward against initially supercritical interior outflow.
The solver does not ignore exterior data based on the old interior Froude number.
Reversals and wet/dry interactions use the same numerical wave bounds as interior faces.
These approximate HLL fluxes establish no exact boundary solution or field-accuracy claim.

Transmissive boundaries copy the post-source interior state into the missing exterior side.
Dry copying requires zero normal and tangential discharge.
Wet copying requires `u_out-c > critical_relative*max(abs(u_out),c)` and finite velocities.
Subcritical, critical, or incoming states reject and require complete exterior data. They never become reflected walls.
This regime check repeats after every source update and shortened trial.

## Acceptance, accounting, and continuation

Every shortened trial recalculates its sources and boundary trace from the saved accepted state.
Boundary outflow enters the all-outgoing donor budget without crediting simultaneous inflow.
The solver never availability-caps a prescribed discharge.
Boundary signal bounds enter the combined directional CFL calculation.
Physical admission errors trigger bounded interval reduction before returning an unresolved error.

`StepDiagnostics.boundary_traces` records depth, velocities, achieved outward rate, speed bound, and compatibility residuals.
`FaceTransfer` records accepted flux support, signed interval volume, and final normal `signal_speed` in m/s.
`StepDiagnostics.transmissive_states` retains post-source depth and discharges. Dry witnesses contain zero discharges and no inferred velocity.
Water ledgers integrate actual oriented exterior fluxes rather than cached target rates.

Snapshots include boundary forcing, tolerances, and trace diagnostics.
Restoration checks physical trace/face consistency and exact static floating-point identity, including every wet exterior input's signed zero.
It reconstructs the global CFL bound from per-axis maxima of all accepted face speeds.
The stored bound must match within 64 epsilon relative rounding. No absolute unit floor applies.
Transmissive witnesses must reproduce their admitted physical flux and regime.
Exterior numerical fluxes are not compared against the reservoir state's physical flux.
These checks establish consistency, rather than historical authenticity or a replay proof.
Same-build restart still requires the same accepted-step partition.

```sh
cargo test --test physical_boundaries --test exterior_boundaries --test exterior_contracts
cargo test --lib numerics::boundary::tests
```
