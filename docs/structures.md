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

## Read head sensitivity

The tensor proposal holds one hidden output per element: `head_sensitivity`. It bounds `|dQ/dH| / |Q|`, in 1/m, for a move of either endpoint head. It is zero where the element carries no discharge. It is no part of the supported public API. The [surface/channel coordinator](surface-channels.md#resolution-term) reads it to size its convergence allowances.

| Law | Bound |
| --- | --- |
| Suppressed weir | `1.5 / h_u + 0.385 / gap`. `h_u` is the upstream head above the crest. `gap` is the head difference of submerged flow and zero for free flow. |
| Submerged orifice | `0.5 / gap` |
| Gate | `1.5 / h_u + 0.5 / gap`. `h_u` is the donor head above the sill. `gap` is the donor-receiver difference where the receiver stands above the sill, else zero. |
| Pump | `\|dQ_ref/dh\| / (N Q)` on the active curve segment, with `N` the speed ratio. Zero at shutoff and while isolated. |

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
