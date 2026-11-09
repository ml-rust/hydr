# Numerical contract

hydr solves hydrostatic, depth-averaged shallow-water equations for hydraulic application developers.

## Equations and support

- Conserved cell state: Depth `h` in metres and discharge per width `qx=h*u`, `qy=h*v` in m²/s.
- Pressure: `g*h²/2`, with full advective momentum flux in both horizontal directions.
- Terrain: One bed elevation per Cartesian cell. No subgrid storage or unresolved connections exist.
- Coordinates: Projected metres with a caller-attested CRS, vertical datum, and grid origin.
- Water levels: Terrain and water inputs require compatible elevation references. hydr performs no datum or CRS transformations.
- Density: Constant, with hydrostatic pressure and depth-averaged horizontal momentum.
- Excluded physics: Nonhydrostatic motion, vertical jets, sediment, erosion, debris, wind, and Coriolis.
- Other excluded forces: Atmospheric pressure gradients and horizontal eddy viscosity.

## Discretization

1. Integrate interval-constant vertical rainfall with zero horizontal momentum.
2. Apply prescribed land withdrawals with proportional donor momentum.
3. Reconstruct each face against the higher adjoining bed.
4. Calculate one complete HLL flux using conservative normal wave bounds.
5. Apply side-specific hydrostatic pressure corrections once, without a centred bed source.
6. Bound the interval by combined directional CFL and every proposed outgoing face/port rate.
7. Recalculate from accepted tensors whenever the interval shortens.
8. Advance transport, then apply exact fixed-depth Manning vector decay.
9. Check state and ledgers before atomic acceptance.

The discretization remains first order. MUSCL, SSP-RK2, and HLLC remain unimplemented extensions.
The retained reference matches the first-order candidate's equations and source order.
Prescribed losses implement no soil state machine and make no Green-Ampt claim.

## Wave bounds

Dry-front bounds include twice the wet-state celerity. Separating wet states admit vacuum.
Two-rarefaction states use their analytical outer heads, including identical states and near-vacuum fans.
Other faces solve the normal Riemann depth relation through solvr's retained upper bracket.
The bracket uses a derived compression upper estimate before bounded expansion and bisection.
For `H >= M=max(hL,hR)`, the shock curves satisfy:

```text
fL(H)+fR(H) >= (H-M)*(sqrt(g/(2*hL))+sqrt(g/(2*hR)))
```

This estimate avoids excessive expansion from thin films.
Unequal depths also use a one-sided shock estimate:

```text
HR = hR + max(2*sqrt(g*hL)-(uR-uL),0)*sqrt(2*hR/g)
```

For `H>=hR`, `fR(H)>=(H-hR)*sqrt(g/(2*hR))`, while `fL(H)>=-2*sqrt(g*hL)`.
Therefore `HR` bounds the root from above. The symmetric estimate exchanges left and right depths.
The bracket selects the smallest finite estimate before outward padding and checked residual evaluation.
This avoids excessive bisection across extreme depth ratios. Iteration exhaustion still returns an error.
Film brackets receive additional relative refinement because absolute depth error alone cannot control shock-speed sensitivity.
Hydr requests a relative tolerance capped at `1e-12` and an absolute tolerance of one F64 subnormal quantum.
These stricter tolerances remain within the configured absolute/relative depth ceiling.
Wave endpoints receive outward padding of `16*f64::EPSILON` times the local speed scale.
This padding is no interval-arithmetic proof. Extreme unrepresentable arithmetic returns an error.

## Stability, positivity, and accounting

- Combined directional CFL: At most `0.45`.
- Depletion: Reserve every outgoing shared face and outward port rate without crediting simultaneous inflow.
- Trial recalculation: Sources, reconstructions, bounds, and fluxes recompute after interval reduction.
- Complete fluxes: No pressure/transport scaling limiter changes the shared numerical flux.
- Negative depth: Reject the trial. No water clipping or water adjustment occurs.
- Positive films: Retain every representable positive depth.
- Optional film reset: Retain depth, reset momentum, and report signed momentum removal in m⁴/s.
- Water residual: Storage change minus rain, boundary inflow, boundary outflow, and prescribed losses.
- Dimensional controls: `SolverConfig` records root, step, film, and water-ledger tolerances.

Default volume tolerance is `1e-9 m³ + 1e-12*scale`, with a `1 m³` scale floor.
These computational tolerances establish no physical inundation threshold or observational accuracy target.

## Execution and limitations

State, geometry, forcing, indexing, flux evaluation, transport, source updates, and reductions execute through numr CPU tensors.
numr owns scaled Euclidean magnitude. solvr owns generic bracket expansion and bisection.
Hydr retains the physical residual, compression estimate, wet/dry branches, and wave-speed conversion.
The cached CPU client requests one thread.
Explicit client dispatch preserves that policy under downstream dependency feature unification.
Accepted arrays project tensors for result access and checks. Numerical continuation uses the retained tensors.
Snapshots share immutable storage. No solver operation mutates accepted tensors in place.

Hydrostatic reconstruction has documented steep/shallow/coarse-grid artifacts.
Two-wave HLL diffuses transverse transport. The complete application accuracy envelope remains unestablished.

## Mathematical references

- [Clawpack Riemann book](https://www.clawpack.org/riemann_book/html/Shallow_water.html): Shock, rarefaction, dry-front, and vacuum relations.
- [Delestre et al.](https://arxiv.org/html/1206.4986): Hydrostatic corrections and steep/shallow limitations.
- [FullSWOF authors](https://arxiv.org/abs/1401.4125): Shallow-water finite-volume methods and source treatment.

The implementation uses original code under Apache-2.0. No upstream solver source or external benchmark dataset is copied.
