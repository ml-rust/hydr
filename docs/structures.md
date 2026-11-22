# Evaluate flow through a structure

Use `hydr::structures::StructureBatch` to calculate flow through declared openings, weirs, pumps, or gates. Choose the physical law before supplying water levels (heads).

## Choose the calculation

| You need | Use |
| --- | --- |
| Flow at one instant | `StructureBatch::propose` or `propose_with_operations` |
| Volume while heads and operation stay fixed | `proposal.held_volumes` |
| Accepted transfers between evolving pools | [Coupled level pools](coupled-level-pools.md) |
| Accepted transfers between pools, surface cells, and channel cells | [Surface/channel coupling](surface-channels.md#join-nodes-through-structures) |
| Commands or moving openings | [Operating controls](operating-controls.md), [actuators](gate-actuators.md), or [operating sources](operating-sources.md) |
| Culvert barrel geometry, critical depth, or normal depth | [`CulvertBarrelBatch`](#culvert-barrel-geometry) |
| Culvert inlet-control headwater | [`CulvertInletBatch`](#culvert-inlet-control) |
| Culvert outlet-control headwater or a jump in the barrel | [`CulvertOutletBatch`](#culvert-outlet-control) |
| Culvert discharge from the heads at both ends | [`StructureLaw::Culvert`](#culvert-structure-law) |

A standalone structure proposal advances no storage. It checks neither donor availability nor acceptance of a paired transfer.

Run the ordinary example:

```sh
cargo run --example structures
```

The [example](../examples/structures.rs) supplies every coefficient and applicability declaration explicitly. Coefficients have no defaults.

## Prepare and evaluate a proposal

1. Choose a supported law and its physical inputs.
2. Declare structure identity, endpoint order, datum, and direction policy.
3. Supply supported heads and an explicit operation where required.
4. Create a proposal, then request its outputs or held-head volumes.

### 1. Choose a law

| Law | Admitted physical setting |
| --- | --- |
| [SubmergedOrifice](#orifices-and-rectangular-weirs) | Known wet stages strictly above the opening crown at both endpoints |
| [SuppressedRectangularWeir](#orifices-and-rectangular-weirs) | Suppressed sharp-crested rectangular free/submerged overflow with the adopted Villemonte correction |
| [Pump](#pumps) | Supplied bounded reference curve with fixed orientation, declared speed policy, and explicit operation |
| [HydrostaticRectangularGate](#hydrostatic-rectangular-gates) | Thin rectangular aperture between hydrostatic quiescent pools, with an explicit held opening |
| [Culvert](#culvert-structure-law) | Circular or box HDS-5 culvert under inlet or outlet control. Build the batch with `StructureBatch::with_culverts`. |


### 2. Bind identities and heads

Every `StructureElement` requires a unique structure ID and distinct ordered endpoint IDs. Parallel structures can share endpoint pairs.

| Declaration | Requirement |
| --- | --- |
| Input order | Match the exact structure order and endpoint identities |
| Datum | Both endpoints use the declared common elevation reference |
| Head convention | `HeadConvention::QuiescentStage` supplies water-surface elevations |
| Energy heads | Explicit energy-head packets reject |
| Conversion | No datum transformation or silent head-convention conversion occurs |

“Head support” records whether an endpoint's water level or dry bed is known.

| Head support | Meaning and admission |
| --- | --- |
| `Wet { stage_m }` | Original finite water-surface elevation |
| `Dry { bed_elevation_m }` | Known finite dry bed, subject to the law's connection restrictions |
| `Unknown` | Influential unknown heads reject. Explicit pump isolation and exact zero gate opening permit uninfluential unknown heads. |

Dry support never receives a fabricated water elevation.

### 3. Supply an operation

`propose_with_operations` requires one ordered, identity-bound `StructureOperation` for every structure.

| Operation | Required use |
| --- | --- |
| `Passive` | Passive laws |
| `PumpStoppedIsolated` | Exact zero with no evaluated hydraulic head operands |
| `PumpRunning { speed_ratio }` | Explicit positive supported speed and checked curve query |
| `GateOpening { opening_m }` | Finite original relative opening within the gate's inclusive range |

`propose(heads)` remains available for passive-only batches. Missing pump or gate operations reject.

Operation checks bind every packet to its identity and law before granting any head-support waiver.

### 4. Read flow or held-head volume

Construct `StructureBatch::<R>::new(client, elements)` with your selected Numr client. Read the [runtime requirements](#select-and-preserve-a-runtime).

`batch.propose(heads)` creates opaque immutable proposal tensors. Repeated calls leave definitions and earlier proposals unchanged.

`proposal.outputs()` explicitly copies results from the selected runtime:

- Original head support and law identity.
- Regime, status, direction, and discharge.
- Hydraulic operands used by active calculations.

A held-head volume assumes unchanged heads and operation throughout the interval.

1. Use a finite nonnegative start and a finite positive represented duration.
2. Call `proposal.held_volumes(HeldHeadInterval { start, end })`.
3. Read the original signed `Q*dt` result and retained instantaneous proposal.

Active product overflow or zero underflow rejects. Physical-zero and blocked proposals retain exact zero volumes.

Held volumes preserve interval and sign. They do not integrate evolving trajectories or limit withdrawals against available water.

No clipping, storage advancement, or fabricated conservation residual occurs in this API.

## Interpret direction and zero flow

Positive discharge means A to B. Reversed heads preserve endpoint identities and produce negative discharge where the law permits reversal.

| Direction policy | Behavior |
| --- | --- |
| `Bidirectional` | Requires explicit symmetric coefficient applicability |
| `AtoBOnly` | Ideal flap permits A-to-B flow |
| `BtoAOnly` | Ideal flap permits B-to-A flow |

Unsupported support or regimes reject before flap blocking. An eligible prohibited direction returns exact zero with `FlapBlocked` status.

`StructureRegime` records physical classification separately from status. Blocked flows retain their nominal proposed direction.

| Zero-flow case | Result |
| --- | --- |
| Equal eligible passive wet heads | `PhysicalZero`, without a proposed direction |
| Weir support below the crest | `PhysicalZero`, without a proposed direction |
| Equal pump heads | Query zero differential head. Positive pump flow remains possible. |

Direction and regime use original supported heads. Rounded depth differences never determine whether unequal original stages produce flow.

Only eligible, permitted branches evaluate derived hydraulic arithmetic. Blocked branches skip unused products and expose no invented operands.

## Orifices and rectangular weirs

### Declare geometry and applicability

| Law | Required parameters |
| --- | --- |
| Orifice | Positive finite area, dimensionless discharge coefficient, and gravity. Finite sill and crown elevations, with crown above sill. |
| Weir | Positive finite crest length and coefficient, plus finite crest elevation. The coefficient uses SI `m^(1/2)/s`. |

`OrificeApplicability` retains coefficient provenance and fully submerged applicability evidence.

`WeirApplicability` separately retains coefficient provenance, free-nappe/approach assumptions, and correction applicability evidence.

Nonempty declarations record the caller's physical claims. They establish no site-specific coefficient accuracy.

### Check endpoint support

| Condition | Result |
| --- | --- |
| Orifice with either endpoint dry | Reject |
| Orifice partially submerged or exactly at crown equality | Reject |
| Orifice equal-head or ideal-flap shortcut | Both original wet stages must first lie strictly above the crown |
| Wet weir donor above crest, eligible dry receiver | Free overflow |
| Both weir endpoints dry, or no wet donor above crest | Below-crest zero |

The rectangular correction excludes contracted, broad-crested, ogee, and other crest geometries.

General partial-orifice models, complete culverts, and inferred blockage remain unsupported by these laws.

### Numerical formulas

Active fully submerged orifice:

```text
gap = highStage-lowStage
logQ = log(Cd)+log(area)+0.5*(log(2)+log(g)+log(gap))
Qabs = exp(logQ)
```

Active free weir overflow:

```text
Hu = highStage-crest
logQ = log(Cw)+log(length)+1.5*log(Hu)
Qabs = exp(logQ)
```

Active submerged weir overflow:

```text
Hu = highStage-crest
Hd = lowStage-crest
gap = highStage-lowStage
r = Hd/Hu
K = 1+r/(1+sqrt(r))
logQ = log(Cw)+log(length)+(1.5-0.385)*log(Hu)
       +0.385*log(gap)+0.385*log(K)
Qabs = exp(logQ)
```

The submerged form equals `(1-(Hd/Hu)^1.5)^0.385` in real arithmetic. It avoids subtracting a rounded power from one.

Every `log` and `exp` above is correctly rounded to nearest through numr `CorrectlyRoundedOps`. Each result is the exact value rounded once, so the discharge bits are identical on every runtime that admits the structures.

| Numerical quantity | Admission rule |
| --- | --- |
| Active depths and direct original-stage gap | Positive finite representation required |
| Represented submergence ratio | `[0,1]`, including rounded one and underflowed zero |
| Distinct stages with equal rounded relative depths | The independently positive direct gap preserves the active case |
| Normalized gap | No normalized-gap calculation or positivity gate applies |
| Unused products | `Cd*area`, `Cw*length`, radicand, and power intermediates need not be representable |
| Final active discharge | Must be finite and strictly positive. Overflow or zero underflow rejects without clipping or manufacturing flow. |

Unused intermediates can overflow or underflow while final discharge remains representable.

## Pumps

`StructureLaw::Pump(PumpParameters)` calculates flow from a supplied curve. It contains no operating-rule, timer, control-state, or hysteresis engine.

### Supply a bounded curve

`PumpCurve::new` uses the lift as its head argument. The lift is receiving stage minus donor stage. A pump with a free outlet lifts to the outlet instead when the receiver lies below it.

| Curve property | Requirement |
| --- | --- |
| Head knots | Finite, strictly increasing, with finite positive represented widths |
| Flow knots | Finite, nonnegative, nonincreasing, and positive at least once |
| Plateaus | Positive plateaus and declared zero-flow plateaus are supported |
| Domain | Closed and bounded. No extrapolation. Negative head requires explicit coverage. |
| Zero-flow endpoint | Creates no implicit high-head check-valve extension |

Curve provenance records source/revision, SI units, reference speed, fluid applicability, and the valid operating envelope.

`PumpApplicability` records how the curve applies to endpoint stages:

- Quiescent endpoint-stage interpretation.
- Connection-loss and velocity-head accounting.
- Stopped-isolation evidence and operating assumptions.

A raw manufacturer total-head curve does not automatically satisfy this endpoint-stage contract.

### Declare orientation and speed

Pump orientation requires `AtoBOnly` or `BtoAOnly`. `Bidirectional` rejects. A reversed gradient never reverses the declared pump direction.

| Speed policy | Contract |
| --- | --- |
| `PumpSpeedPolicy::ReferenceOnly` | Exact ratio one, without an implied affinity claim |
| `DeclaredAffinity { minimum_ratio, maximum_ratio, applicability }` | Positive finite inclusive interval containing one, with supplied affinity evidence |

There is no default speed, suction bound, curve, or operating point.

### Declare a free outlet

A free outlet is the elevation where the delivery pipe discharges to air. `PumpParameters::new` creates a pump without one. Add one with `with_free_outlet`:

```rust
let parameters = PumpParameters::new(curve, minimum_suction_stage_m, speed_policy, applicability)?
    .with_free_outlet(PumpFreeOutlet {
        elevation_m: 4.0,
        evidence: "Delivery pipe discharges to air at 4.0 m on the common datum".into(),
    })?;
```

| Field | Requirement |
| --- | --- |
| `elevation_m` | Finite, on the common datum |
| `evidence` | Nonempty, without control characters. States that the outlet discharges freely there and that the curve's head argument holds there. |

A second call replaces the first outlet. `free_outlet()` returns the declared outlet, or `None`.

With an outlet, the lift is `max(receiving stage, outlet) - donor stage`. A dry receiver counts at its bed elevation.

| Receiver | Lift |
| --- | --- |
| Wet, at or above the outlet | Receiving stage minus donor stage, bit for bit as without an outlet |
| Wet, below the outlet | Outlet minus donor stage. The receiving stage has no effect on flow. |
| Dry, bed at or below the outlet | Outlet minus donor stage |
| Dry, bed above the outlet | Rejects as an unsupported regime. Raise the outlet or supply a wet receiver. |

Two pump definitions match only when both lack an outlet, or both carry the same outlet elevation bits and evidence text.

The [surface/channel coordinator](surface-channels.md#check-structure-admission) uses the outlet head for a pump that delivers to a dry surface cell or pool. A running pump without an outlet that meets a dry receiver rejects with `pump-dry-delivery`.

### Check running and stopped support

| Operation | Required support and result |
| --- | --- |
| Running | Known wet donor, and a known wet receiver or a declared free outlet, including at declared shutoff |
| Suction | Donor stage strictly above its finite common-datum minimum suction stage |
| Dry delivery | Needs a free outlet at or above the receiver's bed. Without one, a running pump into a dry receiver rejects as an unsupported regime. |
| Stopped isolation | Explicitly excludes forward siphoning and reverse flow. Dry or unknown heads can be uninfluential. |

Isolation still requires correct identity, datum, quiescent-stage convention, and valid known values. Energy-head packets reject even while isolated.

No unknown head becomes a fabricated dry elevation. These checks establish no NPSH, priming, or cavitation model.

| Pump outcome | Regime/status pair | Hydraulic operands |
| --- | --- | --- |
| Positive output | `PumpCurve/PumpRunning` | Retained |
| Supported zero-flow query | `PumpCurve/PumpShutoff` | Retained |
| Isolation | `PumpIsolated/PumpStoppedIsolated` | None |

Original operation and support remain attached to projected and held-volume proposals.

### Keep affinity queries inside the supplied curve

Pumps reuse grouped Solvr Linear interpolation on the selected Numr runtime. Hydr copies no interpolation or search algorithm.

Numr TwoSum retains the oriented lift as `headHigh+headLow`. The query uses two ordered divisions:

```text
q1 = headHigh / speed
referenceHead = q1 / speed
Q = speed * Qreference(referenceHead)
```

| Query condition | Behavior |
| --- | --- |
| Exact reference ratio one | Bypass division |
| Exact head zero | Both high and low components must be zero |
| Nonfinite used division or nonzero head collapse | Reject |
| Speed square | Never formed or required to be representable |

The error bound includes the signed-subtraction witness:

```text
errorBound >= abs(headLow)/speed² + b(q1)/speed + b(referenceHead)
b(y) = EPSILON*abs(y) + MIN_SUBNORMAL
```

Each positive bound division or addition rounds outward. Exact zero terms stay zero without padding.

The displayed `speed²` denotes two ordered bound divisions, not a computed square.

Division certificates remove only division-rounding terms:

| Certificate | Condition |
| --- | --- |
| Power-of-two upward scaling | Finite upward scaling applies |
| Power-of-two downward scaling | Final query magnitude is normal |
| Reference or zero | Its exact conditions apply |

Certificates never discard original `headLow` uncertainty. No FMA-zero exactness assumption applies.

### Read curve-domain and shutoff decisions

The complete signed uncertainty interval controls both domain and shutoff checks.

| Complete uncertainty interval | Result |
| --- | --- |
| Entirely outside the curve | `UnsupportedRegime` |
| Overlaps an endpoint without remaining inside | Numerical ambiguity |
| Spans positive and zero-flow regions | Unresolved rejection |
| Exact certified zero knot or wholly supported zero plateau | Exact zero |

The original normalized high query remains unchanged for Solvr. No argument clips to a curve boundary.

Positive support requires finite positive interpolated discharge and finite positive final speed-scaled discharge.

Held products retain overflow/underflow checks. Numerical support establishes no donor availability or accepted-transfer claim.

`PumpHydraulicOperands` exposes the calculation used for the proposal:

- Lift head high/low and speed. With a free outlet the lift uses the outlet where it lies above the receiver.
- Normalized query, uncertainty bound/interval, and division certificate.
- Reference-speed flow.

A certificate alone never implies zero uncertainty. No universal relative-accuracy claim applies near steep or shutoff regions.

## Hydrostatic rectangular gates

`StructureLaw::HydrostaticRectangularGate(GateParameters)` integrates local pressure-head discharge across a thin vertical rectangular aperture.

The model joins quiescent pools with hydrostatic pressure and negligible approach velocity.

### Declare aperture geometry

`GateParameters::new` requires every input below explicitly.

| Input | Requirement |
| --- | --- |
| Width, maximum relative opening, discharge coefficient, gravity | Positive finite values |
| Sill | Finite common-datum elevation |
| `GateApplicability` | Coefficient provenance, hydrostatic/quiescent-pool evidence, complete-aperture connection evidence, and free/submerged coefficient applicability |
| Coefficient envelope | One supplied symmetric coefficient covers the declared opening and head envelope |
| Bidirectional flow | Explicit symmetric applicability |

No width-times-opening or absolute maximum-crown representation requirement applies.

Nonempty evidence establishes no coefficient calibration or field accuracy.

### Supply the held opening

Each proposal requires `GateOpening { opening_m }` within `[0,maximum_opening_m]`.

The original relative opening stays authoritative. A rounded absolute crown minus sill never replaces it.

| Opening/support | Admission |
| --- | --- |
| Exact zero opening | Impermeable closure. Uninfluential unknown heads are allowed after identity, law, datum, convention, and malformed-known-value checks. |
| Positive opening | Known wet or dry support at both endpoints |
| Known dry bed | Must lie at or below the sill, even before ideal flap blocking |

The closed branch exposes no hydraulic operands.

### Interpret the gate regime

Original wet stages determine direction.

| Supported condition | Result |
| --- | --- |
| Wet donor above sill toward eligible known-dry support | Flow |
| Both endpoints dry, or no wet donor above sill | Exact physical zero |
| Equal wet stages | Exact zero |
| Eligible flow blocked by an ideal flap | Retain the nominal direction |

`StructureRegime` distinguishes `GateClosed`, `GateBelowSill`, `EqualHeads`, `FreeGateAperture`, `MixedGateAperture`, and `SubmergedGateAperture`.

Active outputs carry `GateHydraulicOperands`. Physical-zero and flap-blocked outputs carry no evaluated hydraulic operands.

### Pressure integral

| Symbol | Original physical quantity |
| --- | --- |
| `U` | Donor stage |
| `D` | Wet receiver stage |
| `z` | Sill |
| `a` | Opening |

Define `t=min(z+a,U)` and `m=min(t,max(z,D))` in exact geometry.

For an eligible dry receiver, use `m=z` without inventing a stage.

```text
Qabs = Cd*width*sqrt(2*g) *
       [(m-z)*sqrt(U-D) + (2/3)*((U-m)^(3/2)-(U-t)^(3/2))]
```

| Aperture coverage | Integral term |
| --- | --- |
| No wet receiver coverage above sill | Submerged term absent |
| Receiver covers the complete aperture | Free term absent |
| Fully submerged | Original opening times the square root of the direct original-stage gap |

This pressure-integral model does not supply a general channel-sluice or nonhydrostatic-jet model.

It also excludes vena-contracta, hydraulic-jump, energy-head, and air-demand models.

EPA equivalent-weir branches and HEC gated-spillway transitions are different models. Hydr claims no numerical identity with them.

### Check crown and strip geometry

Numr TwoSum retains a finite crown as `crownHigh+crownLow=sill+opening`. Original stages compare against that expansion.

Positive crown overflow places every finite stage below the crown. Invalid overflow residuals never enter active arithmetic.

Fully submerged height uses the original opening and direct stage gap. Unused sill-relative depths can overflow without rejecting this branch.

For a full-height donor and partially submerged receiver, retain `(d,e)=TwoSum(D,-sill)`.

The submerged height is `d`, with independent error bound `abs(e)`. Two witnessed subtractions retain the complementary free height:

```text
(v,r1) = TwoSum(opening,-d)
(L,r2) = TwoSum(v,-e)
freeHeightError >= abs(r1)+abs(r2)
```

The low receiver-depth component preserves a positive free strip when rounded `d` equals the opening.

Partly wetted donors use direct original `U-D` or `U-sill` differences. Independently rounded relative depths never determine that strip.

The same subtraction sequence gives upper free pressure head `B=U-(crownHigh+crownLow)`.

Lower free head uses `A=fl(L+B)`. Its bound includes both input bounds and the addition witness.

Original positive `L` remains a separate factor when absorbed into `A`.

| Geometry condition | Admission |
| --- | --- |
| Strictly positive active height or pressure head | Outward relative error bound at most `64*EPSILON` |
| Exact-zero error | Bypass bound division |
| Exact positive subnormal geometry | Eligible. No absolute metre floor applies. |
| Active overflow, unresolved positive geometry, or excess relative uncertainty | Reject |

Required active geometry must retain finite representation.

### Read arithmetic diagnostics

The represented strip sum provides a closure-discrepancy center and an independent uncertainty bound.

Original length and target witnesses provide a separate physical allowance. Disjoint intervals reject.

These checks accompany the relative operand restriction. They never modify physical lengths or establish universal final-flow accuracy.

The free pressure integral uses this stable positive form:

```text
r = B/A
K = 1+r/(1+sqrt(r))
Jfree = (2/3)*L*sqrt(A)*K
```

The represented ratio permits `[0,1]`. Rounded one retains a positive thin strip through independent `L`.

The evaluator forms submerged and free strip logarithms `a` and `b` separately. A two-input logaddexp combines them:

```text
m = max(a, b)
n = min(a, b)
logJ = m + log1p(exp(n-m))
```

`exp` and `log1p` are correctly rounded. The absolute error of `logJ` is at most `3*2^-54 + ulp(logJ)/2`.

An absent strip uses negative infinity. When both strips are absent, `logJ` is negative infinity. Only the combined final discharge is exponentiated.

`GateHydraulicOperands` exposes:

- Original opening, crown expansion, represented strip heights, active pressure heads, and independent bounds.
- Free ratio/factor, closure receipt, and final log discharge.

An absent crown expansion identifies positive crown overflow. Absent strip pressures remain undefined through optional getters.

Closure bounds are diagnostics, never water adjustments.

A held-head proposal keeps the same original opening throughout its interval. It does not integrate motion or advance storage.

Trajectory, actuator response, motion limits, control state, timers, hysteresis, donor availability, and paired acceptance require separate mechanisms.

## Prescribed opening trajectories

`GateOpeningTrajectory` supplies immutable continuous piecewise-linear openings through original SI knots.

### Declare trajectories

| Input | Contract |
| --- | --- |
| Knots | At least two finite nonnegative times and openings. Original times strictly increase. |
| Motion limits | Explicit finite nonnegative opening/closing speeds. Zero prohibits that direction. |
| Accuracy | Explicit finite nonnegative absolute-metre and relative opening-error limits. No defaults. |
| Provenance | Nonempty source and operating-envelope evidence, without control characters |
| Binding | Every gate appears exactly once in original subsequence order |
| Gate range | Every original opening stays within its actual bound gate maximum |

`StructureBatch::with_gate_trajectories` retains the complete original structure context. Matching IDs cannot substitute different geometry or applicability.

`GateTrajectoryBatch::propose_at` requires complete ordered heads and explicit original-order non-gate operations.

Passive packets remain explicit. Pump operation never defaults to stopped.

### Query within common time support

The closed common support is the intersection of original trajectory domains.

| Query/event rule | Behavior |
| --- | --- |
| Outside common support | No extrapolation or implicit last hold |
| Events | Retain every original knot inside common support, including both boundaries |
| Event merging | Only exactly equal original times merge |
| `next_event_after` | Returns a strictly later event |
| `check_interval` | Reports internal and exact terminal events without integrating or advancing |
| Knot metadata | Original-time equality and right-sided segment. Terminal knot uses the last segment. |

Rounded opening establishes neither arrival nor closed support.

Repeated, shuffled, or rejected queries change no cursor, state, or previous evaluation.

### Understand motion and opening checks

Numr TwoSum retains both duration and displacement lows for motion admission.

| Motion certificate | Requirement |
| --- | --- |
| Power-of-two duration metadata | Advisory until its high witness matches Numr |
| Exact speed | Both lows vanish |
| Exact positive subnormal quotient | Finite upward back-product when required |
| Other segments | Conservative upper-speed acceptance or lower-speed excess certificate |
| Overlapping limit uncertainty | Numerical rejection |

Solvr supplies matching F64 Linear values and independent absolute-error receipts.

| Opening-error case | Admission |
| --- | --- |
| Absolute bound fits declared metre budget | Accept |
| Otherwise, positive opening | Outward relative bound must fit the declared relative limit |
| Both-zero policy | Zero receipts required |

Either the absolute or relative branch suffices. There is no additive budget or unit-scale floor.

Conservative positive receipts reject some mathematically exact interior evaluations.

| Original support | Representation requirement |
| --- | --- |
| Positive opening | Positive represented opening, regardless of accuracy budgets |
| Zero knot or zero plateau | Exact zero opening and receipt |
| Segment and gate range | Value remains inside original bounds, without clipping |

Tolerances govern numerical opening admission. They grant no coefficient accuracy or physical motion-limit exception.

### Read trajectory outputs

`evaluate` and `propose_at` retain derived openings as Numr tensors. Explicit `outputs()` and `operations()` copy them from the runtime.

Diagnostic rejection masks remain separate from numerical state copies.

Held volumes use the sampled opening. They never integrate gate motion across the requested interval.

Actuator response, acceleration, inertia, controls, timers, hysteresis, storage advancement, and paired coupling remain separate mechanisms.

Run the original-bit opening and error-receipt observer:

```sh
cargo run --example gate_trajectories_reference
```

## Culvert barrel geometry

`hydr::structures::CulvertBarrelBatch` encloses the geometry of closed barrels: circles, boxes, ellipses, pipe-arches and flat-floor arches. It also encloses the critical depth and Manning normal depth of circular and box barrels. It computes no culvert discharge.

Read capabilities with `culvert_barrel_capabilities()`. It returns a `ComponentCapabilities` declaration: arithmetic, runtimes, unsupported uses, and limits.

### Declare barrels

| Constructor | Shape |
| --- | --- |
| `CulvertBarrel::circular(diameter_m)` | Circle. Span and rise both equal the diameter. |
| `CulvertBarrel::rectangular_box(span_m, rise_m)` | Box of span `B` and rise `D` |

Each constructor returns `StructureInputErrorKind::Definition` for these inputs:

- A nonfinite or nonpositive span, rise, or diameter.
- A rise below `2^-992`, where `D 2^-30` is no longer a normal F64.
- Dimensions whose `B D`, `2 (B + D)` or `B D^2`, times 64, overflow F64.

The message names the field, its value, and the corrective action.

Four more constructors build barrels from tangent circular arcs. `geometry`, `geometry_over` and `full_geometry` accept them. `critical_depth`, `normal_depth`, `CulvertInletBatch`, `CulvertOutletBatch` and `StructureBatch::with_culverts` reject them with `StructureInputErrorKind::Definition`, naming the row. Use a circular or box barrel for those.

| Constructor | Shape | Rules |
| --- | --- | --- |
| `CulvertBarrel::horizontal_ellipse(span_m, rise_m, side_radius_m)` | Four-arc ellipse, wider than high | Span above rise. Side radius below half the rise. |
| `CulvertBarrel::vertical_ellipse(span_m, rise_m, end_radius_m)` | Four-arc ellipse, higher than wide | Rise above span. End radius below half the span. |
| `CulvertBarrel::pipe_arch(span_m, rise_m, corner_radius_m, bottom_radius_m)` | Bottom arc, two corner arcs, top arc | Corner radius below half the span. Bottom radius above half the span. Rise above the corner-centre height plus the corner radius. Rise above the corner centres at most half the span. |
| `CulvertBarrel::arch(span_m, rise_m)` | One arc on a flat floor as wide as the span | Rise from a quarter of the span to the span. |

The constructor derives the remaining radius from tangency: the invert and crown radius of a horizontal ellipse, the side radius of a vertical ellipse, the top radius of a pipe-arch, and the arc radius of an arch. These constructors also reject a radius whose cube, times 64, overflows F64. `given_radius_m()` returns the radii you supplied, in constructor order, and `None` for a circle, a box or an arch.

`CulvertBarrelBatch::<R>::new(client, barrels)` uploads one row per barrel. Rows of every shape mix in one batch. It rejects an empty list. It also rejects a runtime without these F64 contracts:

- IEEE basic arithmetic with nearest-even rounding and gradual underflow.
- Fused multiply-add.
- Correctly rounded elementary functions.
- Square roots rounded to nearest.

### Run a query

Every query input is an F64 `[m]` tensor on the batch device, one value per barrel. Each value is in SI units.

| Method | Inputs | Result |
| --- | --- | --- |
| `geometry` | `depth_m` | Geometry at each depth |
| `geometry_over` | `lower_m`, `upper_m` | Geometry over every depth in each range |
| `full_geometry` | None | Geometry at the crown `y = D` |
| `critical_depth` | `discharge_m3_per_s` and a host `gravity_m_per_s2` | Depth where `Q^2 T = g A^3` |
| `normal_depth` | `discharge_m3_per_s`, `manning_n`, `slope` | Depth where `Q = (1/n) A R^(2/3) sqrt(S)` |

`discharge_m3_per_s` is the discharge magnitude per barrel. `slope` is positive downstream.

Each result holds lower and upper tensors. The exact value at each represented input lies inside its pair. `project()` copies the rows to the host.

### Read geometry

A geometry row holds `(lower, upper)` pairs of these quantities:

| Field | Quantity |
| --- | --- |
| `area_m2` | Flow area `A` |
| `top_width_m` | Free-surface width `T`. Zero in the full state. |
| `wetted_perimeter_m` | Wetted perimeter `P` |
| `hydraulic_radius_m` | `A / P` |
| `hydrostatic_moment_m3` | Integral of `(y - eta) T(eta)` from the invert to `y`. In the full state it is `A_full (y - D/2)` for a circle or a box, and `M(D) + A_full (y - D)` for an arc barrel, with `M(D)` the moment at the crown. |

Each row also carries a `BarrelState`:

| State | Depth | Meaning |
| --- | --- | --- |
| `Dry` | `y = 0` | Every quantity is an exact zero. A box and a flat-floor arch keep `T = P = B`. |
| `Free` | `0 < y < D` | Free surface below the crown |
| `Full` | `y >= D` | Full-barrel constants. Depth above the crown is pressure head. |

`geometry_over` admits only `0 <= lower_m <= upper_m < D`. A range that reaches the crown rejects. Split it at the crown and use `full_geometry` above it. The hydraulic-radius bound divides the area range by the perimeter range, so it widens with the range.

### Read critical depth

The search domain is `[D 2^-30, top]`. `top` is `D (1 - 2^-53)` for a circle and `D` for a box. A circle has one critical depth below the crown for every `Q > 0`. A box has one only for `Q <= sqrt(g) B D^(3/2)`.

| `CriticalDepthStatus` | Enclosure |
| --- | --- |
| `Dry` | Zero discharge. Exactly `[0, 0]`. |
| `Solved` | Certified root inside the domain |
| `BelowFloor` | The root lies at or below `D 2^-30`. The enclosure starts at 0. |
| `AtCrown` | The residual at the domain top contains zero. The enclosure ends at `D`. A box root lies in it only when `Q` stays within the box limit. |
| `AboveCrown` | A circle root lies in `[D (1 - 2^-53), D]`. A box has no free-surface critical depth and no enclosure. |

### Read normal depth

The normal depth is the root on the rising limb of the conveyance `K = A^(5/3) / P^(2/3)`. The search domain is `[D 2^-30, top]`, with `top = 0.93818 D` for a circle and `D` for a box.

Circle conveyance peaks at `0.9381812 D`, above the full-barrel conveyance, then falls to the crown. Box conveyance rises to its open-section limit at the crown. That limit exceeds the full-barrel conveyance.

| `NormalDepthStatus` | Enclosure |
| --- | --- |
| `Dry` | Zero discharge on a positive slope. Exactly `[0, 0]`. |
| `Solved` | Certified root inside the domain |
| `BelowFloor` | The root lies at or below `D 2^-30`. The enclosure starts at 0. |
| `NearCapacity` | The target lies near the top of the rising limb. A rising-limb root, when one exists, lies in the enclosure. |
| `AboveCapacity` | The target exceeds the peak circle conveyance or the box crown limit. No enclosure. |
| `NonPositiveSlope` | Zero or adverse slope, whatever the discharge. No enclosure. |

A row without an enclosure projects `depth_m: None`. Its tensors hold NaN.

### Handle query rejection

A query rejects the whole call and names the first offending row:

| Input | Kind |
| --- | --- |
| Negative depth, discharge, or range end | `OperatingPoint` |
| Nonfinite depth, discharge, range end, Manning n, or slope | `NonfiniteInput` |
| Nonpositive Manning n | `OperatingPoint` |
| Nonfinite or nonpositive gravity | `OperatingPoint`, with no row |
| Reversed range or a range reaching the crown | `OperatingPoint` |
| A tensor of another shape, dtype, or device | `OperatingPoint`, with no row |

### Check the barrel evidence

Independent 100-digit references are frozen before any kernel runs.

| Check | Criterion |
| --- | --- |
| [Reference rows](../tests/culvert_barrels/geometry.rs) | Three circles and three boxes at frozen depth fractions from dry to `1.5 D`. Every reference lies inside its enclosure, within a frozen width per quantity. |
| [Dense sweep](../tests/culvert_barrels/geometry.rs) | Sorted depths per shape contain every reference. No certified reversal of `A`, `P` or `M`. |
| [Crown](../tests/culvert_barrels/crown.rs) and [ranges](../tests/culvert_barrels/ranges.rs) | Rows at and above the crown carry the full-state constants bit for bit. The free row one F64 step below the crown stays within its continuity gap. Range bounds contain every point bound they span. |
| [Critical depth](../tests/culvert_barrels/critical.rs) and [normal depth](../tests/culvert_barrels/normal.rs) | Statuses match the reference class of the exact root. Every promised enclosure contains it. No depth shows a certified reversal as discharge rises. Normal depth also rises with roughness and as slope falls. |
| [Layouts](../tests/culvert_barrels/layouts.rs) | Widths 1, 31, 32, 33, 64, and 65 mix circle and box rows. Every lane of every query equals its single-row result bit for bit. |
| [Admission](../tests/culvert_barrels/admission.rs) | Each rejection names its row, value, and action |
| [Arc barrels](../tests/culvert_shapes/geometry.rs) | Twelve ellipses, pipe-arches and arches at frozen depths, junctions and seams included. Every reference lies inside its enclosure within a frozen width. [Sweeps](../tests/culvert_shapes/sweeps.rs) show no certified reversal, `dA/dy = T` and `dM/dy = A`, and ranges that hold their ends and references. [Layouts](../tests/culvert_shapes/layouts.rs) of widths 1, 33 and 65 mix arc, circle and box rows bit for bit. |

```sh
cargo test --test culvert_barrels
cargo test --test culvert_shapes
```

| Runtime | Behavior |
| --- | --- |
| CPU | Reference |
| CUDA | [Every query](../tests/runtime_backends/culvert_barrels.rs) over the reference rows and each layout width matches CPU bit for bit. Point geometry also matches over the dense sweep. |
| WGPU | Construction rejects for lack of F64, without CPU fallback |

### Know the barrel limits

- Queries hold no state and advance no storage. No snapshot exists.
- Full-barrel pressure flow and full-barrel friction stay with the caller.
- Critical and normal depth cover circular and box barrels only.
- CUDA parity covers circular and box barrels only.
- No outlet control, barrel profile, or culvert discharge. [`CulvertInletBatch`](#culvert-inlet-control) gives inlet-control headwater.

## Culvert inlet control

`hydr::structures::CulvertInletBatch` encloses the inlet-control headwater `HW_i(Q)` of circular and box culverts. It follows FHWA HDS-5 (2012) Appendix A. It computes no outlet control and no discharge from a head.

Read capabilities with `culvert_inlet_capabilities()`.

### Declare an inlet

| Constructor | Inlet |
| --- | --- |
| `CulvertInlet::hds5(code)` | HDS-5 Table A.1 code, numbered as in SWMM `culvert.c`. Codes 1 to 8 are circular, 9 to 26 concrete box. Code 5 is mitered. |
| `CulvertInlet::explicit(form, [k, m, c, y], mitered)` | Explicit coefficients for any barrel shape |

`CulvertInletForm::One` and `Two` select the unsubmerged equation. `K`, `M` and `c` must be finite and positive, and `Y` finite. Any other input returns `StructureInputErrorKind::Definition` with the value and the action.

### Build a batch

`CulvertInletBatch::<R>::new(client, rows)` takes one `CulvertInletRow` per culvert:

| Field | Meaning |
| --- | --- |
| `barrel` | A `CulvertBarrel` |
| `inlet` | A `CulvertInlet`. A coded inlet must match the barrel shape. |
| `slope` | Signed barrel slope `S` in m/m, positive when the barrel falls away from the inlet |
| `gravity_m_per_s2` | Gravity, for the form-1 critical head |

The runtime needs the same F64 contracts as `CulvertBarrelBatch`.

### Read the headwater

`headwater(discharge_m3_per_s)` takes an F64 `[m]` tensor of per-barrel discharges. It returns lower and upper headwater in metres above the inlet invert. The exact headwater lies inside each pair. `project()` copies the rows to the host.

With `X = Ku Q / (A D^0.5)` and `Ku = 0.3048^(-1/2)`, `X` equals its value in foot units. HDS-5's US coefficients and limits then apply to SI inputs unchanged.

| `InletRegime` | Band | Headwater |
| --- | --- | --- |
| `Unsubmerged` | `X <= 3.5` | Form 1: `Hc + D (K X^M + Ks S)`, with `Hc = dc + A(dc) / (2 T(dc))`. Form 2: `D K X^M`. |
| `Transition` | `3.5 < X < 4` | Linear in `X` from the unsubmerged headwater at `X = 3.5` to the submerged headwater at `X = 4` |
| `Submerged` | `X >= 4` | `D (c X^2 + Y + Ks S)` |
| `UnsubmergedLimit`, `SubmergedLimit` | `X` may lie on either side of 3.5 or 4 | Encloses the values of both bands |

`Ks` is `-0.5`, or `+0.7` for a mitered inlet. HDS-5 draws a hand curve across the transition. Hydr defines it as linear, so the headwater is continuous at both limits.

### Handle admission

The batch admits a row only when its headwater strictly rises with discharge. Each band rises because `K`, `M` and `c` are positive. Construction certifies that the submerged headwater at `X = 4` exceeds the unsubmerged headwater at `X = 3.5`. That check reads the slope and gravity. For example, code 12 rejects at a slope of 0.3.

| Rejection | Kind | Fix |
| --- | --- | --- |
| Empty row list, nonfinite slope, nonpositive gravity, or a code for another barrel shape | `Definition` | Correct the named row |
| Headwater falls across the transition | `Definition` | Supply other coefficients or a smaller slope |
| Form-1 critical depth at `X = 3.5` reaches the crown | `Definition` | Raise the gravity or use a form-2 inlet |
| Negative or nonfinite discharge, or a tensor of another shape, dtype, or device | `OperatingPoint` or `NonfiniteInput` | Correct the named row |

Each message names the row, the code or coefficients, the slope, and the gravity.

### Check the inlet evidence

| Check | Criterion |
| --- | --- |
| [Reference rows](../tests/culvert_inlets/corpus.rs) | Every 100-digit inlet row of codes 1 to 26 at three slopes lies inside its enclosure, within `1e-12` relative. The limit anchors lie within `1e-12` of the rows beside each limit. |
| [SWMM rows](../tests/culvert_inlets/swmm.rs) | SWMM 5.2.4 rows converted to SI agree within their fixed tolerance. Its unsubmerged form-1 rows return the flow of a stale evaluation and miss by more. |
| [Published rows](../tests/culvert_inlets/published.rs) | HDS-5 design-example headwaters lie within their published tolerance. |
| [Sweeps](../tests/culvert_inlets/sweep.rs) | Headwater strictly rises over each band. Enclosures within 16 F64 steps of each limit stay within `1e-12` of each other. |
| [Layouts](../tests/culvert_inlets/layouts.rs) | Widths 1, 31, 32, 33, 64, and 65. Every lane equals its single-row result bit for bit. |
| [Admission](../tests/culvert_inlets/admission.rs) | Each rejection names its row, value, and action |

```sh
cargo test --test culvert_inlets
```

A [CUDA check](../tests/runtime_backends/culvert_inlets.rs) compares every reference row and layout width with CPU bit for bit. WGPU rejects for lack of F64, without CPU fallback.

### Know the inlet limits

- At zero discharge, a form-1 row returns `D Ks S`, the value of its equation, not zero.
- The slope term takes the signed slope as given. HDS-5 drops slope adjustments when the exit invert lies more than `D/2` above the entrance. Keep adverse slopes within that range.
- HDS-5 codes 27 to 57 reject: corrugated metal boxes, ellipses, arches, pipe arches, and tapered inlets. Use explicit coefficients for a box or circle.

## Culvert outlet control

`hydr::structures::CulvertOutletBatch` encloses the outlet-control headwater `HW_o(Q, TW)` of circular and box culverts. It marches the water surface up the barrel from the outlet. It also locates a hydraulic jump in a steep barrel. It computes no inlet control and no discharge from a head.

Read capabilities with `culvert_outlet_capabilities()`.

### Build a batch

`CulvertOutletBatch::<R>::new(client, rows)` takes one `CulvertOutletRow` per culvert:

| Field | Meaning |
| --- | --- |
| `barrel` | A `CulvertBarrel` |
| `length_m` | Barrel length `L` |
| `manning_n` | Manning `n` in s/m^(1/3) |
| `entrance_loss` | Entrance loss coefficient `ke` of the inlet the flow enters |
| `slope` | Signed slope `S` in m/m, positive when the barrel falls toward the outlet. The inlet invert sits `S L` above the outlet invert. |
| `gravity_m_per_s2` | Gravity |

The runtime needs the F64 contracts of `CulvertBarrelBatch`, solvr certified bisection and solvr certified Simpson integration.

### Read the headwater

`headwater(discharge_m3_per_s, tailwater_m)` takes two F64 `[m]` tensors: per-barrel discharge and tailwater depth above the outlet invert. Each projected row holds an `OutletRegime`, the inlet head `h_in`, and the headwater. Both are in metres above the inlet invert. The exact value lies inside each pair.

`HW_o = h_in + (1 + ke) V_in^2 / (2 g)`, with `V_in` the velocity at the inlet head.

| `OutletRegime` | Rows | Inlet head |
| --- | --- | --- |
| `NoFlow` | `Q = 0` | Still level `max(TW - S L, 0)`, which is also the headwater |
| `Full` | `TW >= D`, and the full-barrel head stays at or above the crown up to the inlet | `max(TW, D) + (Sf_full - S) L` |
| `PartlyFull` | A free-surface profile occupies part or all of the barrel | Depth where the profile meets the inlet, or crown plus full-barrel rise |
| `InletCritical` | The subcritical profile reaches critical depth inside the barrel, or none exists | `yc`, at most `D` |
| `Unresolved` | A certified comparison did not resolve | NaN |

Critical depth never exceeds the crown, as in HDS-5. A box above its critical capacity `sqrt(g) B D^(3/2)` takes `yc = D`.

Full-barrel friction is `Sf_full = n^2 V^2 / R^(4/3)`. This is the HDS-5 loss of equation 3.5 with `Ku = 2g`, so full-barrel head is continuous with the free profile at the crown. HDS-5 prints 29 (US) and 19.63 (SI), which round that constant.

### Understand the march

1. The free profile starts at the outlet at `max(TW, yc)`. This is the exit loss with zero tail velocity.
2. It follows the direct-step profile `dx/dy = (1 - Fr^2) / (S - Sf)` upstream. Depth rises upstream toward the normal depth or the crown where `Sf > S`, and falls toward the normal depth or critical depth where `Sf < S`.
3. A profile that reaches the crown continues as full-barrel flow. A full outlet whose head falls to the crown inside the barrel continues as a falling free profile.
4. A steep profile that reaches critical depth before the inlet leaves the inlet at critical depth. Then `HW_o = yc + (1 + ke) Vc^2 / (2 g)`.
5. With `yc = D` and `TW < D`, the profile starts at the crown as free-surface flow. Where free-surface friction at the crown exceeds the slope, the barrel runs full to the inlet. Otherwise the inlet runs at the crown, with `HW_o = D + (1 + ke) V_full^2 / (2 g)`.

Zero and adverse slopes have no normal depth. Their profiles rise to the crown. HDS-5's `ho = max(TW, (dc + D)/2)` approximation is not used.

Solvr certified Simpson integrates the profile in a smooth coordinate: the half angle `phi` of a circle and `y/D` of a box. Order-4 Taylor jets bound the fourth derivative on each panel. The lanes are graded toward both ends of the profile. Solvr certified bisection then finds the depth where the profile covers the barrel.

The flow area of an inlet depth enclosure comes from the area at each end of the enclosure, since the area rises with depth. At a small depth this keeps the area positive, so a tiny discharge still resolves.

Near a circle's conveyance peak capacity, `S - Sf` nearly vanishes at the peak depth `0.9381812 D`. `Sf` falls with depth up to that peak and rises past it. So each jet takes its lower bound of `|S - Sf|` from the panel ends and, where the panel may hold the peak, from `Sf` at the peak. A profile node whose margin still lies within rounding of zero ends the certified distances. A profile that has not covered the barrel by then holds a hull, as the limits below state.

### Read a jump

`jump(discharge_m3_per_s, tailwater_m)` runs the same march. It returns a `CulvertOutletJump`, whose `headwater()` holds the headwater rows. `project()` gives one `CulvertOutletJumpRow` per row:

| `JumpStatus` | Meaning | Values |
| --- | --- | --- |
| `NotSteep` | The row is not inlet-critical | None |
| `Supercritical` | Supercritical flow leaves the outlet | Outlet depth |
| `InBarrel` | A jump sits inside the barrel | Position from the inlet, upstream and downstream depths |
| `Unresolved` | No certified result, the outlet segment flows full, or a crown tie row | None |

The supercritical profile starts at critical depth at the inlet. Along a profile the momentum `M = Q^2 / (g A) + m` changes by `A (S - Sf)` per metre. Where friction falls with depth, the gap between the subcritical and supercritical momentum rises along the barrel. The jump sits at its unique zero.

### Handle admission

| Rejection | Kind | Fix |
| --- | --- | --- |
| Empty row list, nonpositive length, `n` or gravity, negative entrance loss, or nonfinite slope | `Definition` | Correct the named row |
| Outlet invert more than `D/2` above the inlet invert | `Definition` | Supply a smaller adverse fall |
| Negative or nonfinite discharge or tailwater, or a tensor of another shape, dtype, or device | `OperatingPoint` or `NonfiniteInput` | Correct the named row |

### Check the outlet evidence

| Check | Criterion |
| --- | --- |
| [Reference rows](../tests/culvert_outlets/corpus.rs) | All 18 independent full-barrel rows lie inside their enclosures within `2^-40` relative. All 39 profile rows lie within `1e-9` relative in headwater and inlet head. Six have critical depth at or near the crown. Thirteen run a circle through its conveyance peak capacity, on both sides of it and at the F64 discharges next to it. Jump positions lie within `1e-6 L`, jump and outlet depths within `1e-9` relative. |
| [Peak capacity](../tests/culvert_outlets/capacity.rs) | A circle resolves through its conveyance peak capacity at two tailwaters, and at 64 F64 discharges on each side of it. Its headwater never falls, and each enclosure stays within `1e-9` relative. |
| [Published rows](../tests/culvert_outlets/published.rs) | HDS-5 design-example full-barrel losses lie within their published tolerance |
| [Sweeps](../tests/culvert_outlets/sweep.rs) | Headwater never falls as discharge rises. It falls with rising tailwater only where `(1 + ke) Fr^2 > 1` at the inlet. It is continuous where the outlet turns full, the tailwater crosses critical depth, a profile reaches the crown, a full outlet falls to the crown, and a drowned steep profile reaches critical depth. |
| [Crown sweeps](../tests/culvert_outlets/crown.rs) | A steep box resolves through its critical capacity and its headwater never falls. It is continuous at the capacity and where a full outlet head reaches the crown at the inlet. It steps up by `(Sf_full - S) L` where it primes, and every discharge next to that step resolves. |
| [Layouts](../tests/culvert_outlets/layouts.rs) | Widths 1, 33, and 65, one past each of 32 and 64 lanes. Every lane equals its single-row result bit for bit. |
| [Admission](../tests/culvert_outlets/admission.rs) | Each rejection names its row, value, and action |

```sh
cargo test --test culvert_outlets
```

A [CUDA check](../tests/runtime_backends/culvert_outlets.rs) compares every reference row and layout width with CPU bit for bit. WGPU rejects for lack of F64, without CPU fallback.

### Know the outlet limits

- With `ke > 0`, `y + (1 + ke) V^2 / (2 g)` is smallest above critical depth. So as the tailwater drowns a steep inlet, `HW_o` first falls below the critical-inlet value and then rises. It stays continuous.
- Next to a critical-depth limit the inlet depth moves with the square root of the profile length. Enclosures there are wider than elsewhere.
- A box primes where free-surface friction at the crown reaches the slope. Above its critical capacity, `HW_o` steps up there by `(Sf_full - S) L`, because box friction uses the perimeter `B + 2D` below the crown and `2 (B + D)` when full. A circle has no such step.
- A row with `yc = D` whose crown comparison is not certified holds the hull of both outcomes: the inlet at the crown and the barrel full to the inlet.
- A row is `Unresolved` when another comparison of the march is not certified, its critical depth lies below `2^-30 D`, or its start lies within `2^-34` of a normal-depth pole.
- Near a circle's conveyance peak capacity, `S - Sf` can lie within rounding of zero inside a profile. The march holds no distance past that point. A profile that has not covered the barrel before it holds a hull. A rising profile spans from the last certified depth to the crown, with full flow beyond. A profile toward a normal depth spans to the far bound of the normal-depth enclosure.
- A row with `yc = D` and a full outlet segment has no located jump.
- A jump is located only when the subcritical profile is free at the outlet and friction falls with depth across the jump.
- Barrel storage is not represented. Queries hold no state.

## Culvert structure law

`StructureLaw::Culvert(CulvertParameters)` gives the discharge of identical parallel circular or box barrels from the heads at both ends. The headwater is `HW(Q) = max(HW_i(Q), HW_o(Q, TW))` of [inlet control](#culvert-inlet-control) and [outlet control](#culvert-outlet-control). The discharge inverts that curve at the supplied heads.

Read capabilities with `culvert_capabilities()`.

### Declare a culvert

`CulvertParameters::new(barrel, alignment, manning_n, barrel_count, ends, gravity_m_per_s2, applicability)`:

| Argument | Meaning |
| --- | --- |
| `barrel` | A `CulvertBarrel`, shared by every barrel |
| `alignment` | `CulvertAlignment { length_m, invert_a_m, invert_b_m }`, inverts on the structure datum |
| `manning_n` | Manning `n` in s/m^(1/3) |
| `barrel_count` | Number of identical barrels, at least one. The discharge is the count times the discharge of one barrel. |
| `ends` | `CulvertEnds { a, b }`: one `CulvertEnd` per admitted flow direction |
| `gravity_m_per_s2` | Gravity |
| `applicability` | `CulvertApplicability` with coefficient provenance and a barrel-condition assertion |

`CulvertEnd::new(inlet, entrance_loss)` holds the [`CulvertInlet`](#declare-an-inlet) that flow enters and its entrance loss `ke`. The A end serves A-to-B flow with slope `(z_a - z_b) / L`. The B end serves B-to-A flow with the opposite slope. `forward_rows()` and `reverse_rows()` return the inlet and outlet query rows of each direction.

| Direction policy | Ends |
| --- | --- |
| `Bidirectional` | Both ends |
| `AtoBOnly` | The A end only |
| `BtoAOnly` | The B end only |

Each direction keeps its adverse fall within `D/2`, the outlet-control range. So a two-way culvert keeps its inverts within half the rise of each other.

A roadway over the culvert is a separate weir on the same endpoints.

### Build a batch

`StructureBatch::<R>::with_culverts(client, elements)` admits every law, culverts included. The client must implement `CulvertClient`: barrel geometry, solvr certified bisection, and solvr certified Simpson integration. `StructureBatch::new` rejects a culvert and names `with_culverts`.

Construction uploads two oriented rows per culvert and admits each inlet and outlet row. An inlet whose headwater does not certifiably rise across its transition rejects, naming the element.

A culvert takes the `Passive` operation. `propose` and `propose_with_operations` evaluate it with the other laws.

### Read the discharge and regime

The donor is the side that is wet, stands above its invert, and stands higher or faces a dry receiver. The headwater is the donor stage above the inlet invert. The tailwater is the receiver stage above the outlet invert, or zero for a free outfall.

| `StructureRegime` | Status | Meaning |
| --- | --- | --- |
| `CulvertInletControl` | `Active` | `Q_i` lies certainly below `Q_o`. With a headwater at or below `HW_i(0) = D Ks S`, the discharge is exact zero and the status `PhysicalZero`. |
| `CulvertOutletControl` | `Active` | `Q_o` lies certainly below `Q_i`. With the still level `max(TW - S L, 0)` certainly above the headwater, the discharge is exact zero and the status `PhysicalZero`. |
| `CulvertControlTie` | `Active` | The two discharge enclosures overlap, so either control can govern |
| `EqualHeads` | `PhysicalZero` | Equal wet stages with either side above its invert. Also a lane whose still level is not certified below the headwater. |
| `CulvertBelowInvert` | `PhysicalZero` | No wet donor stands above its invert |
| `CulvertUndeclaredEnd` | `FlapBlocked` | The heads drive flow into an end without a declared inlet. The proposed direction stays. |

Active outputs carry `StructureHydraulicOperands::Culvert` with the headwater, the tailwater, and the enclosure of the per-barrel discharge. The signed discharge is the barrel count times the enclosure midpoint.

A receiver that is dry, or wet below its outlet invert, is a free outfall. Each returns the same discharge bits. A known dry end whose bed lies above its invert rejects as `UnsupportedRegime`.

### Understand the inversion

Both control headwaters rise with `Q`. So `HW(Q) <= H` holds exactly when `Q <= Q_i(H)` and `Q <= Q_o(H, TW)`, and the discharge is `min(Q_i, Q_o)`. The inversion runs in `Q` at fixed tailwater. `HW_o` can fall as the tailwater rises, which the inversion never needs.

| Control | Inversion |
| --- | --- |
| Inlet, form 2 unsubmerged | `X = (HW / (D K))^(1/M)` |
| Inlet, form 1 unsubmerged | One certified bisection in the critical-depth fraction `s = dc / D`. The residual rises at least `D` per unit `s`, so it needs no critical-depth solve inside another solve. |
| Inlet, transition | Linear in `X` between the anchors at `X = 3.5` and `4` |
| Inlet, submerged | `X = sqrt((HW / D - Y - Ks S) / c)` |
| Outlet, drowned full barrel | `Q_o = A sqrt(2 g dH / k)` with `k = 1 + ke + 2 g n^2 L / R^(4/3)`, where `TW >= D` and the pressure head stays at or above the crown to the inlet. `dH` is the stage difference, so the result keeps its precision as the heads approach each other. |
| Outlet, other | One outlet march at the upper end of `Q_i`. A headwater below the donor head there means the inlet governs. Otherwise one certified bisection over the outlet march on `[0, 2 Q_i]`. |

A headwater next to an inlet anchor takes the hull of both band inverses. No positive slope floor is proven for the partly full outlet profile. The outlet bisection widens tiny residual bounds so that the solver's slope correction cannot move a bracket end. Its enclosure rests on certified residual signs.

### Read culvert sensitivity

The hidden `head_sensitivity` output of a culvert lane is `1 / E`, with `E = Q dHW/dQ` of the governing control:

| Control | `E` |
| --- | --- |
| Inlet, form 2 unsubmerged | `M HW` |
| Inlet, form 1 unsubmerged | At least `M D K X^M`, at the lower end of the discharge enclosure |
| Inlet, transition | `2 (HW_4 - HW_3.5) X` |
| Inlet, submerged | `2 D c X^2` |
| Outlet, drowned full barrel | The bound is `1 / (2 dH)` |
| Outlet, other | `max(1, |dHW/dTW|) / E` from one coarse march at the reported discharge and tailwater |

The other outlet lanes take certified closed forms at the march's class and depths:

| Outlet class | `E` at least | `|dHW/dTW|` at most |
| --- | --- | --- |
| Critical depth at the inlet | `Q^2 / (g A_c^2)` | `0` |
| Profile reaches the crown, then full | `(1 + ke) Q^2 / (g A^2)` | `(Sf_full - S) (1 - Fr_s^2) / (Sf_s - S)` |
| Full to the inlet | `(1 + ke) Q^2 / (g A^2)` | `1` |
| Profile ends inside the barrel | `Q (d w + ke v / (1 - Fr^2))` at the inlet, `d = 1 - ke Fr^2 / (1 - Fr^2)`, `v = Q / (g A^2)` | `|d| |dEs/dTW|` |

`Es` is the specific energy and `w = dEs/dQ` its gain at the inlet. A rising profile has `w >= v` at the inlet. A falling profile has `w` at least `v` at its deepest point below the circle conveyance peak. `dEs/dTW` is `(1 - Fr_s^2) (Sf - S) / (Sf_s - S)` from an outlet start and `|Sf - S| / (S - Sf_full)` from a crown start. A negative `d` takes an upper bound of `w` in closed form. The proof sits in the `outlet::slope` module.

A tie takes the larger of both controls. The bound is capped at `2^256` per metre. Open, tie and unresolved march rows take the cap, as does a row whose divisors are not certified positive.

### Handle culvert rejection

| Rejection | Kind | Fix |
| --- | --- | --- |
| Nonfinite invert, zero barrel count, no end, negative `ke`, an inlet code for another barrel shape, or an adverse fall above `D/2` | `Definition` | Correct the named field |
| Ends that do not match the direction policy | `Definition` | Declare one end per admitted direction |
| Inlet headwater that falls across its transition | `Definition` | Supply coefficients whose submerged headwater at `X = 4` exceeds the unsubmerged headwater at `X = 3.5` |
| Culvert in `StructureBatch::new` | `Definition` | Build with `StructureBatch::with_culverts` |
| Dry end with its bed above its invert | `UnsupportedRegime` | Supply the bed at or below the invert, or a wet stage |
| Outlet headwater unresolved near the inlet discharge | `Numerical` | A march comparison is not certified below the crown, at a start friction equal to the slope or a full-barrel head equal to the crown at the inlet. Move the heads off that point. |

### Check the culvert evidence

| Check | Criterion |
| --- | --- |
| [Reference rows](../tests/culvert_laws/corpus.rs) | Sampled independent inlet rows where the inlet governs, all 18 full-barrel rows, and eight profile rows return their reference discharge within `1e-6` relative from their reference headwater |
| [Round trip and selection](../tests/culvert_laws/round_trip.rs) | `Q(HW(Q))` returns `Q` within `1e-6` on mild, adverse, and level barrels. The regime names the larger control. A drowned crossing of both controls reports a tie. |
| [Heads](../tests/culvert_laws/heads.rs) | Reversed heads negate the discharge with the B-end inlet. One-end reversal, equal heads, and dry donors give exact zero. Free outfalls agree bit for bit. Zero-discharge headwaters give exact zero. |
| [Sweeps](../tests/culvert_laws/sweep.rs) | Discharge rises across the inlet bands and stays continuous at both anchors. It stays continuous where the outlet turns full. |
| [Peak capacity](../tests/culvert_laws/capacity.rs) | On a circle run through its conveyance peak capacity at two tailwaters, `Q(HW(Q))` returns `Q` within `1e-6` under outlet control. Discharge rises with the donor head across that range. |
| Sensitivity unit tests | The bound holds against a centred difference on inlet and drowned lanes, and over head grids on mild, adverse, zero-slope, steep drowned, and priming barrels. No flowing lane takes the cap. |
| [Layouts](../tests/culvert_laws/layouts.rs) | Widths 1, 33, and 65 with weirs. Every lane equals its single-lane result bit for bit. |

```sh
cargo test --test culvert_laws
```

A [CUDA check](../tests/runtime_backends/culvert_laws.rs) compares every layout with CPU bit for bit. WGPU rejects for lack of F64, without CPU fallback.

### Know the culvert limits

- The law holds no water. Barrel storage is not represented, and proposals keep no state.
- The [surface/channel coordinator](surface-channels.md#attach-a-culvert) and [coupled level pools](coupled-level-pools.md#attach-a-culvert) hold culverts. Each requires the inverts at or above the attachment beds.
- A shallow profile whose depth enclosure gives no certified positive area at a small angle takes the sensitivity cap.
- Every flowing lane runs one outlet march. A lane the inlet does not certainly govern adds one outlet march per bisection round. An outlet-controlled lane that is not a drowned full barrel adds one coarse march for its head sensitivity.
- No blockage, sediment, or debris is represented.

## Read head sensitivity

The tensor proposal holds one hidden output per element: `head_sensitivity`. It bounds `|dQ/dH| / |Q|`, in 1/m, for a move of either endpoint head. It is zero where the element carries no discharge. It is no part of the supported public API. The [surface/channel coordinator](surface-channels.md#resolution-term) reads it to size its convergence allowances.

| Law | Bound |
| --- | --- |
| Suppressed weir | `1.5 / h_u + 0.385 / gap`. `h_u` is the upstream head above the crest. `gap` is the head difference of submerged flow and zero for free flow. |
| Submerged orifice | `0.5 / gap` |
| Gate | `1.5 / h_u + 0.5 / gap`. `h_u` is the donor head above the sill. `gap` is the donor-receiver difference where the receiver stands above the sill, else zero. |
| Pump | `\|dQ_ref/dh\| / (N Q)` on the active curve segment, with `N` the speed ratio. Zero at shutoff and while isolated. |
| Culvert | `1 / E` with `E = Q dHW/dQ` of the governing control, or `1 / (2 dH)` for a drowned full barrel. Other outlet-controlled lanes take `max(1, |dHW/dTW|) / E` from certified closed forms. See [culvert sensitivity](#read-culvert-sensitivity). |

Scope of the output:

- It bounds the change of discharge per metre. It establishes no accuracy of the discharge itself.
- A pump with a free outlet moves its lift one for one with the donor head. A receiver below the outlet moves nothing, and the bound still holds.
- Unit tests check each law's bound against a centred difference over a dense head grid.

## Select and preserve a runtime

`StructureBatch::<R>::new(client, elements)` receives the configured Numr client. Laws retain F64 `Tensor<R>` fields.

| Execution requirement | Contract |
| --- | --- |
| CPU reference | One thread |
| Other runtimes | [F64 arithmetic admission](integration.md#runtime-selection-and-admission) |
| Arithmetic | Round-to-nearest ties-to-even and gradual underflow |
| Elementary functions | Correctly rounded F64 `exp`, `log` and `log1p`, which need IEEE fused multiply-add |
| Context | Checks the shared execution envelope and the elementary-function contracts during construction |
| Caller | Preserves that envelope throughout proposal evaluation |

Coefficient/geometry logarithm sums remain cached. Passive laws and gates use Numr correctly rounded logarithms and exponentials, combined by the two-input logaddexp above. Pumps reuse Solvr interpolation.

## Read the numerical evidence

These results check numerical law evaluation. They establish no empirical correction accuracy, site coefficient accuracy, field accuracy, or complete HEC solver parity.

### Passive laws

Actual Numr execution passes fourteen independently frozen 800-digit reference cases.

| Check | Result or criterion |
| --- | --- |
| Categories | Twelve positive discharges and two rejected unrepresentable final discharges |
| Maximum measured relative error | Approximately `9.53e-14` across admitted cases |
| Ordinary comparison | `1e-12 m³/s + 1e-12*abs(reference)` |
| Tiny/extreme comparison | `max(32*ulp(reference),1e-12*abs(reference))`, with mandatory positive discharge |
| Layouts | Scalar, SIMD, and tail positions. No cross-layout bitwise equality promise. |

Cases include ordinary laws, adjacent stages, rounded depth plateaus, and normalized-gap underflow.

Unused coefficient, power, and radicand products can exceed representable range while final discharge stays finite.

### Pumps

Actual Numr execution passes 26 independently frozen exact-rational cases.

| Check | Result or criterion |
| --- | --- |
| Categories | 13 positive, four declared shutoff, seven numerical ambiguity/arithmetic rejections, and two unsupported transformed domains |
| Maximum measured positive-flow relative error | Approximately `2.13e-16` |
| Ordinary flow tolerance | `64*EPSILON*max(abs(reference),abs(output))`, without a unit floor |
| Exact binary analytical fixtures | Exact expected bits |
| Independent comparisons | TwoSum head witnesses, transformed-head bounds/intervals, zero-region classification, final flow, and held products |

Cases include nonbinary speeds, nonzero head-low contributions, and tiny/huge speeds with unrepresentable unused squares.

Thirteen further rows, frozen before execution, cover the free outlet in both orientations:

| Rows | Criterion |
| --- | --- |
| Wet receiver below, at, and above the outlet. Dry receiver below and at the outlet. Half, reference, and double speed. | Exact expected flow and held-volume bits |
| Inexact stages, wet and dry receiver | `64*EPSILON*max(abs(reference),abs(output))` |
| Shutoff lift at the outlet | Exact zero, `PumpShutoff` |
| Dry bed above the outlet, dry donor, donor at suction | Unsupported-regime rejection |
| All admitted rows | Head high and low bits equal an independent TwoSum of the lifted stage and donor stage |

A wet receiver at or above the outlet returns the same output bits as the pump without an outlet. Moving a receiver below the outlet changes no flow bit.

```sh
cargo test --test pump_laws outlet
```

This evidence establishes no manufacturer-curve accuracy, efficiency, control behavior, or native SWMM parity.

### Gates

Forty original-F64 cases freeze inputs, categories, and tolerances before kernel execution.

Independent references use an 800-digit direct pressure integral and exact rational geometry. They subtract high-precision powers independently of production arithmetic.

| Coverage | Result |
| --- | --- |
| Original cases | 34 successes, four numerical rejections, and two distinct support rejections |
| Direction checks | All 80 forward/reversed rows match frozen categories and applicable gates |
| Ordinary discharge | Maximum relative error `2.2204460375522126e-16`, below `128*EPSILON` |
| Extreme discharge | Maximum relative error `3.706741754709129e-14`, below `1e-12` |
| Positive subnormal | Two-ULP allowance and mandatory positive output |
| Geometry diagnostics | Original-input lengths and pressures lie within independent reported bounds |
| Batch layouts | Widths 1, 31, 32, 33, 64, and 65, including mixed laws and exact rejection indices |
| Additional checks | Sill/crown transitions, opening sweeps, repeated proposals, submerged-orifice overlap, and held-volume representation limits |

```sh
cargo test --test gate_laws
cargo run --example gate_laws_reference
```

The observer includes original input bits, signed discharge, held volume, geometry witnesses, and closure receipts.

Log-discharge error remains diagnostic. Frozen acceptance gates bound discharge and supply no universal pre-exponential log-error bound.

Agreement establishes no calibrated coefficient, field accuracy, control behavior, or accepted coupling.

## Handle rejection and scope limits

`StructureError` separates typed input categories, numerical representation errors, and runtime errors.

| Input category | What differs or is unsupported |
| --- | --- |
| Definition or identity | Physical definition, structure order, or exact endpoint binding |
| Datum or head convention | Elevation reference or supported stage-head interpretation |
| Operating point | Operation or speed/opening policy |
| Unknown support or unsupported regime | Influential unknown heads or an unadmitted physical setting |
| Nonfinite original data | Original values outside finite input requirements |

Ordered element indices remain available without parsing diagnostic strings.

Standalone proposals contain no donor storage, evolving outlet integration, storage snapshots, physical actuator response, network solve, or bed/sill changes.

[Coupled level pools](coupled-level-pools.md) and the [surface/channel coordinator](surface-channels.md#join-nodes-through-structures) separately supply synchronized head estimates and accepted shared transfers for their admitted scope.

### Commands and actuator response

| Separate component | What it supplies |
| --- | --- |
| [Operating controls](operating-controls.md) | Sampled relay commands, qualification delays, immutable trials, and complete snapshots |
| [Ideal constant-rate actuators](gate-actuators.md) | Actual gate opening over time |
| [Mixed operating sources](operating-sources.md) | Manual operations, prescribed gate subsets, and fresh actual-actuated frames |

A gate target differs from actual opening. Standalone commands establish no hydraulic or actuator response.

Point outputs retain source and opening receipts beside discharge evaluated at the admitted center.

They supply no ideal-flow enclosure or integrated transfer.

## Applicability references

- [USBR submerged-orifice relation](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap09_05.html).
- [USBR local orifice pressure-head relation](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap02_08.html).
- [USBR suppressed sharp-crest and approach definitions](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_03.html).
- [HEC rectangular-weir submergence correction](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/7.0/modeling-gated-spillways-weirs-and-drop-structures/uncontrolled-overflow-weirs/submerged-weir-flow).
- [USBR submerged-weir measurement limitations](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_15.html).
- [Hydraulic Institute pump-curve and affinity assumptions](https://datatool.pumps.org/pump-fundamentals/pump-curves).
- [EPA hydraulic addendum pump affinity model](https://www.epa.gov/system/files/documents/2022-02/swmm5-reference-manual-ii-addendum-20220210mas2wr-1.pdf).
