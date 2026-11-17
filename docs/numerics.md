# Numerical contract

Use this reference to understand Hydr's surface equations, advancement method, numerical checks, and accuracy limits.

Hydr solves hydrostatic, depth-averaged shallow-water equations.

## Equations and support

| Quantity or assumption | Contract |
| --- | --- |
| Conserved state | Depth `h` in metres and discharge per width `qx=h*u`, `qy=h*v` in m²/s |
| Momentum flux | Full advection in both horizontal directions, with pressure `g*h²/2` |
| Terrain | One bed elevation per Cartesian cell. No subgrid storage or unresolved connections. |
| Coordinates | Projected metres with caller-attested CRS, vertical datum, and grid origin |
| Water levels | Compatible terrain and water elevation references. Hydr performs no datum or CRS transformation. |
| Fluid | Constant density, hydrostatic pressure, and depth-averaged horizontal momentum |

Excluded physics:

- Vertical processes: Nonhydrostatic motion and vertical jets.
- Transported material: Sediment, erosion, and debris.
- Additional forces: Wind, Coriolis, atmospheric pressure gradients, and horizontal eddy viscosity.

## Discretization

Each candidate interval follows this order:

1. Evaluate soil using accepted interval-start ponding and incoming rain once.
2. Add rain, then remove native infiltration with bounded reconciliation.
3. Cap evaporation, then remove prescribed withdrawal.
4. Reconstruct each face against the higher adjoining bed.
5. Calculate shared HLL fluxes and admitted boundary traces using conservative normal wave bounds.
6. Apply side-specific hydrostatic pressure corrections once, without a centred bed source.
7. Bound the interval by combined directional CFL and every proposed outgoing face/port rate.
8. Recalculate from accepted tensors whenever the interval shortens.
9. Advance transport. Retry the interval on negative depth or momentum without water.
10. Apply a coordinator exchange at its linked cells, if one is present.
11. Apply exact fixed-depth Manning vector decay.
12. Check state and ledgers before atomic acceptance.

### Coordinator exchange

A surface-channel coordinator supplies per-linked-cell inflow and withdrawal volumes in m³. Standalone runs apply no exchange and record `+0.0` exchange volumes.

| Quantity | Rule |
| --- | --- |
| Depth | `h + (inflow - withdrawal) / A`, floored at zero |
| Donor momentum | Withdrawal keeps `(h - withdrawal / A) / h` of the post-transport discharge |
| Inflow momentum | Inflow adds `inflow / A` times the supplied inflow velocity |
| Donor limit | Withdrawal above the post-transport volume times `1 + 4ε` rejects the interval |
| Rounding | The local residual compares depth change times area with the exchange volume |
| Ledger | `exchange_in_m3` and `exchange_out_m3` enter the water residual as `+in - out` |

Each linked cell is unique, so every indexed write is order-independent.

The method remains first order. The retained reference uses the same candidate equations and source order.

MUSCL, SSP-RK2, and HLLC remain unimplemented extensions.

### Source and momentum conventions

| Process | Convention |
| --- | --- |
| Modified Green–Ampt land | Preserve the pinned native soil state. |
| Prescribed withdrawal | Treat it as a separate demand without a soil interpretation. |
| Rain and numerical correction | Add zero horizontal momentum. |
| Sinks | Remove proportional donor momentum. |

## Wave bounds

The face calculation chooses its wave bounds from the local wet/dry condition.

| Condition | Calculation |
| --- | --- |
| Dry front | Include twice the wet-state celerity. |
| Separating wet states | Admit vacuum. |
| Two rarefactions | Use analytical outer heads, including identical states and near-vacuum fans. |
| Other faces | Solve the normal Riemann depth relation through Solvr's retained upper bracket. |

### Upper estimates for the root

A compression estimate bounds the search before expansion and bisection. It avoids excessive expansion from thin films.

For `H >= M=max(hL,hR)`, the shock curves satisfy:

```text
fL(H)+fR(H) >= (H-M)*(sqrt(g/(2*hL))+sqrt(g/(2*hR)))
```

Unequal depths also use a one-sided shock estimate:

```text
HR = hR + max(2*sqrt(g*hL)-(uR-uL),0)*sqrt(2*hR/g)
```

For `H>=hR`, these inequalities make `HR` an upper root bound:

```text
fR(H)>=(H-hR)*sqrt(g/(2*hR))
fL(H)>=-2*sqrt(g*hL)
```

The symmetric estimate exchanges left and right depths.

The bracket selects the smallest finite estimate, then applies outward padding and checks the residual. This avoids excessive bisection across extreme depth ratios.

Iteration exhaustion still returns an error.

### Refinement and arithmetic limits

Thin-film shock speeds need relative refinement. An absolute depth error alone cannot control their sensitivity.

| Control | Contract |
| --- | --- |
| Relative depth tolerance | Capped at `1e-12` |
| Absolute depth tolerance | One F64 subnormal quantum |
| Configured ceiling | These stricter tolerances stay within the configured absolute/relative depth ceiling. |
| Wave endpoint padding | `16*f64::EPSILON` times the local speed scale |

Padding is no interval-arithmetic proof. Extreme unrepresentable arithmetic returns an error.

## Physical boundary traces

Wet-subcritical Stage/Discharge inputs connect the interior to a compatible one-wave trace.

| Boundary calculation | Contract |
| --- | --- |
| Completed trace | Contribute the complete physical flux directly, with normal pressure once. |
| Discharge trace | Conserve the requested numerical discharge. Recalculate all momentum components from the same state. |
| Compatibility | Check both discharge and local velocity after root refinement. |
| Wet/dry exterior reservoir | Enter the existing HLL solver before flux calculation. |
| Transmissive boundary | Copy dry or resolved supercritical-outgoing post-source states outside. |
| Accepted face | Expose the final numerical signal speed for interval CFL checks. |

Passing a completed discharge trace through HLL again changes its prescribed mass flux.

Read [the boundary contract](boundaries.md) for admissible regimes, critical margins, units, and pressure conventions.

## Stability, positivity, and accounting

| Check | Enforced behavior |
| --- | --- |
| Combined directional CFL | At most `0.45` |
| Channel CFL | Inward face speeds over twice each cell length, at most `0.45`. See [the channel timestep](channels.md#bound-the-channel-timestep). |
| Depletion | Reserve every outgoing shared-face and outward-port rate without crediting simultaneous inflow. |
| Shortening | Recalculate sources, reconstruction, bounds, and fluxes. |
| Complete flux | No pressure/transport scaling limiter changes the shared numerical flux. |
| Negative depth | Reject without clipping. Proved soil arithmetic excess needs an explicit bounded water adjustment. |
| Positive film | Retain every representable positive depth. |
| Optional momentum reset | Retain film depth, reset momentum, and report signed removal in m⁴/s. |

### Water budgets

Water residual: Storage change minus rain, boundary exchange, infiltration, evaporation, prescribed withdrawal, signed numerical adjustments, and coordinator exchange.

`SolverConfig` records dimensional root, step, film, and water-ledger tolerances.

| Budget | Default and meaning |
| --- | --- |
| Volume tolerance | `1e-9 m³ + 1e-12*scale`, with a `1 m³` scale floor |
| Cumulative adjustment | Independent limit of `1e-9 m³` |

A balanced residual never authorizes unlimited numerical water creation.

These computational tolerances define neither a physical inundation threshold nor an observational accuracy target.

Read [land sources](land-sources.md) for source order, native state, representability, and restart consistency.

## Execution and limitations

### Library responsibilities

| Layer | Work |
| --- | --- |
| Numr `Tensor<R>` | State, geometry, forcing, indexing, fluxes, transport, source updates, and reductions |
| Numr primitives | Scaled Euclidean magnitude |
| Solvr | Generic bracket expansion and bisection |
| Hydr | Physical residuals, compression estimates, wet/dry branches, and conversion to wave speed |

Constructors receive the configured `R::Client`. Allocation uses `client.device()`.

The CPU F64 reference requests one thread at the caller boundary. Explicit dispatch preserves that policy under downstream dependency feature unification.

Other runtimes require [F64 arithmetic admission](integration.md#runtime-selection-and-admission). Backend availability establishes no hydraulic qualification.

### Device data and reads

| Operation | Data behavior |
| --- | --- |
| Ordinary advancement | Retain flow, native soil memory, and bounded detailed records as tensors. |
| Explicit state, soil, cell, observer, diagnostic, or owned-checkpoint request | Copy data through a fallible projection. |
| Rejected-candidate diagnostics | Can project fields. |
| Scalar control and accounting | Reductions remain permitted. |
| Trusted snapshot | Share immutable tensor storage without projection. |

No solver operation mutates accepted tensors in place.

### Accuracy limits

- Hydrostatic reconstruction: Documented artifacts remain on steep, shallow, or coarse grids.
- Two-wave HLL: Transverse transport diffuses.
- Application accuracy: The complete accuracy envelope remains unestablished.

## Mathematical references

- [Clawpack Riemann book](https://www.clawpack.org/riemann_book/html/Shallow_water.html): Shock, rarefaction, dry-front, and vacuum relations.
- [Delestre et al.](https://arxiv.org/html/1206.4986): Hydrostatic corrections and steep/shallow limitations.
- [FullSWOF authors](https://arxiv.org/abs/1401.4125): Shallow-water finite-volume methods and source treatment.

The implementation uses original Apache-2.0 code. It copies no upstream solver source or external benchmark dataset.
