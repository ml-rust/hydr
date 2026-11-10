# Typed structure-law proposals

`hydr::structures::StructureBatch` evaluates explicit instantaneous laws and immutable held-head volume proposals. It advances no storage and establishes no donor availability or paired-transfer acceptance.

Run the [ordinary structure example](../examples/structures.rs):

```sh
cargo run --example structures
```

The example supplies every coefficient and applicability declaration explicitly. Coefficients have no defaults.

## Admitted physical laws

| Law | Physical scope |
| --- | --- |
| `SubmergedOrifice` | Known wet stages strictly above the declared opening crown at both endpoints |
| `SuppressedRectangularWeir` | Suppressed sharp-crested rectangular free/submerged overflow with the adopted Villemonte correction |
| `Pump` | Supplied bounded reference curve with explicit fixed orientation, speed policy, and operation |
| `HydrostaticRectangularGate` | Thin rectangular aperture between hydrostatic quiescent pools with an explicit held opening |

Orifice parameters require positive finite area, dimensionless discharge coefficient, and gravity. Sill/crown elevations require finite values with crown above sill. Weir parameters require positive finite crest length and coefficient, plus finite crest elevation. The weir coefficient uses SI `m^(1/2)/s`.

`OrificeApplicability` retains coefficient provenance and fully submerged applicability evidence. `WeirApplicability` separately retains coefficient provenance, free-nappe/approach assumptions, and correction applicability evidence. Nonempty declarations record the caller's physical claims. They establish no site-specific coefficient accuracy.

The rectangular correction does not apply to contracted, broad-crested, ogee, or other crest geometries. General partial-orifice models, culverts, gate trajectories, and inferred blockage remain unsupported. Pump curves and hydrostatic gates follow separate operating contracts below.

## Exact identity and head support

Every `StructureElement` declares a unique structure ID, distinct ordered endpoint IDs, common datum, direction policy, and concrete law. Parallel structures can share endpoint pairs. Input packets must match the exact structure order, endpoint identities, and datum.

`HeadConvention::QuiescentStage` identifies water-surface stage heads. Explicit energy-head packets reject. No datum transformation or silent head-convention conversion occurs.

| Head support | Admission |
| --- | --- |
| `Wet { stage_m }` | Original finite water-surface elevation |
| `Dry { bed_elevation_m }` | Known finite dry bed, subject to the concrete law's connection restrictions |
| `Unknown` | Rejects influential support. Explicit pump isolation and exact zero gate opening admit uninfluential unknown heads |

Dry support never receives a fabricated physical water elevation. Wet-above-crest flow toward eligible dry support uses the free-overflow law. Both dry, or no wet donor above crest, returns below-crest zero.

Orifices reject either dry endpoint, partial submergence, and exact crown equality. Both original wet stages must lie strictly above crown before equal-head or ideal-flap shortcuts.

## Direction, regimes, and zero flow

Positive discharge means endpoint A to B. Reversed heads retain the same endpoint identities and produce negative discharge.

`Bidirectional` requires explicit symmetric coefficient applicability. `AtoBOnly` and `BtoAOnly` declare ideal flap behavior. Unsupported support/regimes reject before flap blocking. A physically eligible prohibited direction returns exact zero with `FlapBlocked` status.

`StructureRegime` records the physical classification separately from status. Blocked flows retain their nominal proposed direction. Equal eligible passive wet heads and below-crest weir support return `PhysicalZero` with no proposed direction. Equal pump heads query zero differential head and can produce positive flow.

Direction and regime use original supported heads. Rounded depth differences never decide whether unequal original stages produce flow. Active-only derived arithmetic follows physical eligibility and flap policy. Blocked branches skip unused constitutive products and expose no invented hydraulic operands.

## Stable numerical evaluation

The laws use persistent F64 tensors on an explicitly configured one-thread numr CPU client. Coefficient/geometry sums of logarithms remain cached. Passive laws and gates use Numr operations directly. Pumps reuse Solvr interpolation.

For an active fully submerged orifice:

```text
gap = highStage-lowStage
logQ = log(Cd)+log(area)+0.5*(log(2)+log(g)+log(gap))
Qabs = exp(logQ)
```

For active free weir overflow:

```text
Hu = highStage-crest
logQ = log(Cw)+log(length)+1.5*log(Hu)
Qabs = exp(logQ)
```

For active submerged weir overflow:

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

The submerged form equals the adopted factor `(1-(Hd/Hu)^1.5)^0.385` in real arithmetic. It avoids subtracting a rounded power from one.

Active head depths and the direct original-stage gap require positive finite representation. The represented submergence ratio admits `[0,1]`, including rounded one and underflowed zero. Its independently positive original gap remains mandatory.

Rounded upstream/downstream depths can coincide despite distinct original stages. The direct gap preserves that active case. No normalized-gap calculation or positivity gate applies.

The evaluator never requires representability of unused `Cd*area`, `Cw*length`, radicand, or power intermediates. Those products can overflow or underflow while final discharge remains representable. Active final discharge must remain finite and strictly positive. Final overflow or zero underflow rejects without clipping or manufacturing flow.

Configured arithmetic requires round-to-nearest ties-to-even and gradual underflow. Context construction checks the shared execution envelope. Callers must preserve that envelope throughout proposal evaluation.

## Immutable proposals and held volumes

`batch.propose(heads)` creates opaque immutable proposal tensors. Repeated calls neither mutate definitions nor alter earlier proposals. `proposal.outputs()` explicitly projects original support, law identity, regime, status, direction, discharge, and active hydraulic operands.

`proposal.held_volumes(HeldHeadInterval { start, end })` requires finite nonnegative start and finite positive represented duration. It calculates the original signed `Q*dt` once. Active product overflow and zero underflow reject. Physical-zero/blocked proposals retain exact zero volumes.

Held volumes preserve the original interval, sign, and instantaneous proposal. They are not integrated evolving trajectories, availability-limited withdrawals, or accepted paired exchanges. No clipping, storage advancement, or fake conservation residual appears.

## Checked numerical scope

Actual passive-law Numr execution passes fourteen independently frozen 800-digit reference cases. Twelve cases retain positive discharge. Two cases reject unrepresentable final discharge. The maximum measured relative discharge error across admitted cases is approximately `9.53e-14`.

These cases cover ordinary laws, adjacent stages, rounded depth plateaus, and normalized-gap underflow. They also cover unused coefficient, power, and radicand products outside representable range while final discharge remains finite.

Ordinary comparison uses `1e-12 m³/s + 1e-12*abs(reference)`. Tiny/extreme comparison uses `max(32*ulp(reference),1e-12*abs(reference))` with mandatory positive discharge. Mixed layouts exercise scalar, SIMD, and tail positions. No cross-layout bitwise equality promise follows.

This evidence checks numerical law evaluation. It establishes no empirical correction accuracy, site coefficient accuracy, field accuracy, or complete HEC solver parity.

## Explicit pump curves and operation

`StructureLaw::Pump(PumpParameters)` supplies a constitutive proposal only. It has no control-state, operating-rule, timer, or hysteresis engine.

`PumpCurve::new` accepts finite head knots and finite nonnegative flow knots. Heads increase strictly with finite positive represented widths. Flow is nonincreasing and includes at least one positive knot. Positive plateaus and declared zero-flow plateaus remain supported.

The head argument is receiving stage minus donor stage. The curve has a closed bounded domain with no extrapolation. Negative head is eligible only when explicitly covered. A zero-flow endpoint never creates an implicit high-head check-valve extension.

Curve provenance records source/revision, SI units, reference speed, fluid applicability, and valid operating envelope. `PumpApplicability` separately records quiescent endpoint-stage interpretation, connection-loss/velocity-head accounting, stopped-isolation evidence, and operating assumptions. A raw manufacturer total-head curve does not automatically meet this endpoint-stage contract.

Pump orientation requires `AtoBOnly` or `BtoAOnly`. `Bidirectional` rejects. Gradient reversal never reverses the declared pump direction.

| Speed policy | Contract |
| --- | --- |
| `PumpSpeedPolicy::ReferenceOnly` | Exact ratio one, with no implied affinity claim |
| `DeclaredAffinity { minimum_ratio, maximum_ratio, applicability }` | Explicit positive finite inclusive interval containing one, with supplied affinity evidence |

There is no default speed, suction bound, curve, or operating point. A running donor must remain strictly above its finite common-datum minimum suction stage. Both running donor and receiver require known wet support, including at declared shutoff. Dry delivery needs a separate outlet-head contract. These checks establish no NPSH, priming, or cavitation model.

`propose_with_operations` requires one ordered identity-bound `StructureOperation` per structure:

| Operating point | Behavior |
| --- | --- |
| `Passive` | Required for passive laws |
| `PumpStoppedIsolated` | Exact zero with no evaluated hydraulic head operands |
| `PumpRunning { speed_ratio }` | Explicit positive supported speed and checked curve query |
| `GateOpening { opening_m }` | Finite original relative opening within the gate's declared inclusive range |

The existing `propose(heads)` remains available for passive-only batches. Missing pump or gate operations reject. Operation checks bind every packet to its exact identity and concrete law before any head-support waiver.

Stopped isolation explicitly excludes forward siphoning and reverse flow. It allows unknown or dry support because those heads are uninfluential. Identity, datum, quiescent-stage convention, and malformed known-value checks remain mandatory. Energy-head packets reject even while isolated. No unknown head becomes a fabricated dry elevation.

Running positive output records `PumpCurve/PumpRunning`. A supported declared zero-flow query records `PumpCurve/PumpShutoff`. Isolation records `PumpIsolated/PumpStoppedIsolated`. Running shutoff retains evaluated operands. Isolation exposes none. Original operation and support remain attached to every projected and held-volume proposal.

## Checked pump affinity normalization

Pump reference/affinity queries reuse grouped Solvr Linear interpolation on configured Numr CPU tensors. No interpolation or search algorithm is copied into hydr.

Numr TwoSum forms the original oriented stage difference as `headHigh+headLow`. The normalized query uses two ordered divisions:

```text
q1 = headHigh / speed
referenceHead = q1 / speed
Q = speed * Qreference(referenceHead)
```

No speed square is formed or required to be representable. Exact reference ratio one bypasses division. Exact head zero requires both high and low components zero. Nonfinite used divisions and nonzero head collapse reject.

The generic uncertainty envelope retains the signed-subtraction witness contribution:

```text
errorBound >= abs(headLow)/speed² + b(q1)/speed + b(referenceHead)
b(y) = EPSILON*abs(y) + MIN_SUBNORMAL
```

Each positive bound division/addition rounds outward. Exact zero terms retain zero without padding. The displayed speed-square notation denotes two ordered bound divisions, not a computed square.

A supplied power-of-two speed has an exact division certificate when finite upward scaling applies. Downward scaling requires normal final query magnitude. Reference/zero certificates also apply. These certificates remove division-rounding terms only. They never discard original head-low uncertainty. No FMA-zero exactness assumption applies.

A complete signed uncertainty interval controls closed-domain admission. An entirely outside interval returns `UnsupportedRegime`. An interval overlapping an endpoint without remaining inside returns numerical ambiguity. No argument clips to the curve boundary.

The same complete interval controls declared shutoff. An interval spanning positive and zero-flow regions rejects as unresolved. An exact certified zero knot or wholly supported zero plateau returns exact zero. The original normalized high query remains unchanged for Solvr.

Certified positive support requires finite positive interpolated discharge and finite positive final speed-scaled discharge. Original signed held-volume products retain the existing overflow/underflow checks. No donor availability or accepted-transfer claim follows.

`PumpHydraulicOperands` exposes original head high/low, speed, normalized query, uncertainty bound/interval, division certificate, and reference-speed flow. The certificate alone never implies zero uncertainty. Near steep or shutoff regions, no universal relative-accuracy claim follows.

## Checked pump numerical scope

Actual Numr execution passes 26 independently frozen exact-rational cases: 13 positive proposals and four declared shutoff proposals. Seven cases reject numerical ambiguity or unrepresentable arithmetic. Two cases reject completely unsupported transformed domains. The maximum measured relative discharge error across positive cases is approximately `2.13e-16`.

The comparison checks original TwoSum head witnesses, transformed-head bounds/intervals, zero-region classification, final flow, and held products independently. Cases include nonbinary speeds, nonzero head-low contributions, and tiny/huge speeds whose unused squares are unrepresentable.

Ordinary flow tolerance is `64*EPSILON*max(abs(reference),abs(output))`, without a unit floor. Exact binary analytical fixtures retain exact expected bits. This numerical evidence establishes no field accuracy, manufacturer-curve accuracy, efficiency, control behavior, or native SWMM parity.

## Hydrostatic rectangular gates

`StructureLaw::HydrostaticRectangularGate(GateParameters)` integrates local pressure-head discharge across a thin vertical rectangular aperture. The connection joins quiescent pools with hydrostatic pressure and negligible approach velocity.

`GateParameters::new` requires explicit width, sill, maximum relative opening, discharge coefficient, gravity, and `GateApplicability`. Width, maximum opening, coefficient, and gravity require positive finite values. The sill requires a finite common-datum elevation. No width-times-opening or absolute maximum-crown representation requirement applies.

`GateApplicability` retains coefficient provenance, hydrostatic/quiescent-pool evidence, complete-aperture connection evidence, and free/submerged coefficient applicability. A single supplied symmetric coefficient covers the declared opening and head envelope. Bidirectional flow requires explicit symmetric applicability. Nonempty evidence establishes no coefficient calibration or field accuracy.

The closure extends the local orifice relation by integrating hydrostatic pressure differences. It establishes no general channel-sluice, vena-contracta, hydraulic-jump, energy-head, air-demand, or nonhydrostatic-jet model. EPA equivalent-weir branches and HEC gated-spillway transitions define different models. No numerical identity with those models is claimed.

Each proposal supplies `GateOpening { opening_m }` within `[0,maximum_opening_m]`. The original opening remains authoritative. A rounded absolute crown minus sill never replaces it.

An exact zero opening declares impermeable closure. Unknown heads become uninfluential only after identity, law, datum, convention, and malformed-known-value checks. No hydraulic operands appear for this closed branch. A positive opening requires known wet or dry support at both endpoints. Every known dry bed must lie at or below the sill, including before ideal flap blocking.

Original wet stages determine direction. A supported wet endpoint above the sill supplies flow toward eligible known-dry support. Both dry endpoints or no wet donor above the sill give exact physical zero. Equal wet stages give exact zero. Positive supported proposals retain nominal direction when blocked by an ideal flap.

Let `U` denote the original donor stage, `D` the wet receiver stage, `z` the sill, and `a` the original opening. Define `t=min(z+a,U)` and `m=min(t,max(z,D))` in exact geometry. For an eligible dry receiver, set `m=z` without inventing a stage.

```text
Qabs = Cd*width*sqrt(2*g) *
       [(m-z)*sqrt(U-D) + (2/3)*((U-m)^(3/2)-(U-t)^(3/2))]
```

The submerged term is absent without wet receiver coverage above the sill. The free term is absent when the receiver covers the complete aperture. Fully submerged geometry reduces to the original opening times the square root of the direct original-stage gap.

`StructureRegime` distinguishes `GateClosed`, `GateBelowSill`, `EqualHeads`, `FreeGateAperture`, `MixedGateAperture`, and `SubmergedGateAperture`. Active outputs carry `GateHydraulicOperands`. Physical-zero and flap-blocked outputs carry no evaluated hydraulic operands.

## Checked gate geometry arithmetic

Numr TwoSum retains the finite crown as `crownHigh+crownLow=sill+opening`. Original stages compare against that expansion. A positive crown overflow places every finite stage below the crown. Invalid overflow residuals never enter active arithmetic.

Fully submerged height uses the original opening and direct stage gap. Unused sill-relative depths can overflow without rejecting this branch. Active required geometry must retain finite representation.

For a full-height donor and partially submerged receiver, retain `(d,e)=TwoSum(D,-sill)`. Represent submerged height by `d` with independent error bound `abs(e)`. Compute the complementary free height through two witnessed subtractions:

```text
(v,r1) = TwoSum(opening,-d)
(L,r2) = TwoSum(v,-e)
freeHeightError >= abs(r1)+abs(r2)
```

The low receiver-depth component preserves a positive free strip when rounded `d` equals the opening. Partly wetted donors use the direct original `U-D` or `U-sill` difference. No subtraction of independently rounded relative depths decides this strip.

The same witnessed subtraction sequence supplies upper free pressure head `B=U-(crownHigh+crownLow)`. Lower free head uses `A=fl(L+B)`. Its bound includes both input bounds and the addition witness. Original positive `L` remains a separate factor when absorbed into `A`.

Each strictly positive active height or pressure head requires an outward relative error bound at most `64*EPSILON`. Exact-zero error bypasses bound division. This numerical admission has no absolute metre floor. Exact positive subnormal geometry remains eligible. Active overflow, unresolved positive geometry, and excess relative uncertainty reject.

The represented strip sum supplies a closure discrepancy center and an independent uncertainty bound. Original length and target witnesses supply a separate physical allowance. Disjoint intervals reject. These receipts check arithmetic consistency alongside the relative operand restriction. They establish no universal final-flow accuracy bound and never modify physical lengths.

The free pressure integral uses a stable positive form:

```text
r = B/A
K = 1+r/(1+sqrt(r))
Jfree = (2/3)*L*sqrt(A)*K
```

The represented ratio admits `[0,1]`. Rounded one preserves a positive thin strip through its independent `L` factor. The evaluator forms submerged and free strip logarithms separately. Numr `stack` and `logsumexp` combine them. An absent strip uses negative infinity. Only the final combined discharge is exponentiated.

`GateHydraulicOperands` exposes the original opening, crown expansion, represented strip heights, active pressure heads, and independent bounds. An absent crown expansion identifies positive crown overflow. It also exposes the free ratio/factor, closure receipt, and final log discharge. Absent strip pressures remain undefined through optional getters. Closure bounds are diagnostic evidence, never water adjustments.

Held-head volume proposals retain the same original opening throughout the interval. They establish no motion trajectory, actuator response, motion limit, control state, timer, or hysteresis behavior. Storage advancement, donor availability, and paired-volume acceptance remain separate mechanisms.

## Gate numerical evidence

Forty original-F64 cases freeze their inputs, categories, and tolerances before kernel execution. Independent references use an 800-digit direct pressure integral and exact rational geometry. The reference integral subtracts high-precision powers independently of the production formula.

| Coverage | Result |
| --- | --- |
| Original cases | 34 successes, four numerical rejections, and two distinct support rejections |
| Direction checks | All 80 forward/reversed rows match their frozen categories and applicable numerical gates |
| Ordinary discharge | Maximum relative error `2.2204460375522126e-16`, below `128*EPSILON` |
| Extreme discharge | Maximum relative error `3.706741754709129e-14`, below `1e-12` |
| Positive subnormal | Two-ULP allowance with mandatory positive output |
| Geometry receipts | Original-input lengths and pressures lie within their independent reported bounds |
| Batch layouts | Widths 1, 31, 32, 33, 64, and 65, including mixed laws and exact rejection indices |
| Additional checks | Sill/crown transitions, opening sweeps, repeated proposals, submerged-orifice overlap, and held-volume representation limits |

```sh
cargo test --test gate_laws
cargo run --example gate_laws_reference
```

The observer includes original input bits, signed discharge, held volume, geometry witnesses, and closure receipts. Log-discharge error remains diagnostic because the frozen acceptance gates bound discharge. No universal pre-exponential log-error bound follows from those gates. Numerical agreement establishes no calibrated gate coefficient, field accuracy, control behavior, or accepted coupling.

## Prescribed gate trajectories

`GateOpeningTrajectory` supplies immutable continuous piecewise-linear openings through explicit original SI knots.

| Input | Contract |
| --- | --- |
| Knots | At least two finite nonnegative times and openings. Original times strictly increase |
| Motion limits | Explicit finite nonnegative opening/closing speeds. Zero prohibits motion in that direction |
| Accuracy | Explicit finite nonnegative absolute-metre and relative opening-error limits. No defaults apply |
| Provenance | Nonempty source and operating-envelope evidence without control characters |
| Binding | Every gate appears exactly once in its original subsequence order |
| Gate range | Every original opening lies within its actual bound gate maximum |

`StructureBatch::with_gate_trajectories` retains the complete original structure context. Same IDs never substitute different geometry or applicability. `GateTrajectoryBatch::propose_at` requires complete ordered heads and explicit original-order non-gate operations. Passive packets remain explicit. Pump operation never defaults to stopped.

Numr TwoSum retains both duration and displacement lows for motion admission. Power-of-two duration metadata remains advisory until its high witness matches Numr. Exact speed certificates require both lows to vanish. Exact positive subnormal quotients use a finite upward back-product when required. Other segments require conservative upper-speed acceptance or lower-speed excess certificates. Unresolved limit overlap rejects numerically.

Solvr supplies matching F64 Linear values and independent absolute-error receipts. Hydr admits a receipt when its absolute bound fits the declared metre budget. Otherwise, positive lanes require an outward relative bound within the declared relative limit. Either branch suffices. No additive budget or unit-scale floor applies. Both-zero policies require zero receipts. Conservative positive receipts reject some mathematically exact interior evaluations.

Original positive support always requires positive represented opening, regardless accuracy budgets. An original zero knot or zero plateau requires exact zero opening and receipt. Values remain inside original segment bounds and the actual gate maximum. No clipping applies. Tolerances establish numerical opening admission, never coefficient accuracy or physical motion-limit exceptions.

The closed common support is the intersection of original trajectory domains. No extrapolation or implicit last hold applies. Events retain every original knot inside common support, including both boundaries. Only exactly equal original times merge. `next_event_after` returns strictly later events. `check_interval` reports internal events and exact terminal events without integrating or advancing.

Point outputs retain original-time knot equality and right-sided segment metadata. The terminal knot uses the last segment. Rounded opening never establishes arrival or closed support. Repeated, shuffled, or rejected queries mutate no trajectory cursor, state, or previous evaluation.

`evaluate` and `propose_at` retain derived openings as Numr tensors. Explicit `outputs()` and `operations()` project them. Existing diagnostic rejection masks remain distinct from numerical state projection. Proposal held volumes retain the sampled opening and never integrate motion over the requested interval.

Actuator response, acceleration, inertia, controls, timers, hysteresis, storage advancement, and paired coupling remain separate mechanisms.

Run `cargo run --example gate_trajectories_reference` for original-bit opening and error-receipt observations.

## Errors and remaining mechanisms

`StructureError` separates typed input categories, numerical representation errors, and runtime errors. Input categories distinguish definition, identity, datum, head convention, operating point, unknown support, unsupported regime, and nonfinite original data. Ordered element indices remain available without parsing diagnostic strings.

The component supplies no donor storage, outlet integration, storage snapshots, controls, network solve, bed or sill changes, or hydraulic coupling. A future coordinator owns synchronized head provenance and paired shared-volume admission.

## Applicability references

- [USBR submerged-orifice relation](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap09_05.html).
- [USBR local orifice pressure-head relation](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap02_08.html).
- [USBR suppressed sharp-crest and approach definitions](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_03.html).
- [HEC rectangular-weir submergence correction](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/7.0/modeling-gated-spillways-weirs-and-drop-structures/uncontrolled-overflow-weirs/submerged-weir-flow).
- [USBR submerged-weir measurement limitations](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_15.html).
- [Hydraulic Institute pump-curve and affinity assumptions](https://datatool.pumps.org/pump-fundamentals/pump-curves).
- [EPA hydraulic addendum pump affinity model](https://www.epa.gov/system/files/documents/2022-02/swmm5-reference-manual-ii-addendum-20220210mas2wr-1.pdf).
