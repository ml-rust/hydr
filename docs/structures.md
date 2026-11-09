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

Orifice parameters require positive finite area, dimensionless discharge coefficient, and gravity. Sill/crown elevations require finite values with crown above sill. Weir parameters require positive finite crest length and coefficient, plus finite crest elevation. The weir coefficient uses SI `m^(1/2)/s`.

`OrificeApplicability` retains coefficient provenance and fully submerged applicability evidence. `WeirApplicability` separately retains coefficient provenance, free-nappe/approach assumptions, and correction applicability evidence. Nonempty declarations record the caller's physical claims. They establish no site-specific coefficient accuracy.

The rectangular correction does not apply to contracted, broad-crested, ogee, or other crest geometries. Partial orifices, culverts, moving gates, and inferred blockage remain unsupported. Pump curves follow the separate operating contract below.

## Exact identity and head support

Every `StructureElement` declares a unique structure ID, distinct ordered endpoint IDs, common datum, direction policy, and concrete law. Parallel structures can share endpoint pairs. Input packets must match the exact structure order, endpoint identities, and datum.

`HeadConvention::QuiescentStage` identifies water-surface stage heads. Explicit energy-head packets reject. No datum transformation or silent head-convention conversion occurs.

| Head support | Admission |
| --- | --- |
| `Wet { stage_m }` | Original finite water-surface elevation |
| `Dry { bed_elevation_m }` | Known dry weir connection with finite bed at or below crest |
| `Unknown` | Rejects influential support. Explicit stopped-isolated pumps retain unknown uninfluential heads |

Dry support never receives a fabricated physical water elevation. Wet-above-crest flow toward eligible dry support uses the free-overflow law. Both dry, or no wet donor above crest, returns below-crest zero.

Orifices reject either dry endpoint, partial submergence, and exact crown equality. Both original wet stages must lie strictly above crown before equal-head or ideal-flap shortcuts.

## Direction, regimes, and zero flow

Positive discharge means endpoint A to B. Reversed heads retain the same endpoint identities and produce negative discharge.

`Bidirectional` requires explicit symmetric coefficient applicability. `AtoBOnly` and `BtoAOnly` declare ideal flap behavior. Unsupported support/regimes reject before flap blocking. A physically eligible prohibited direction returns exact zero with `FlapBlocked` status.

`StructureRegime` records the physical classification separately from status. Blocked flows retain their nominal proposed direction. Equal eligible passive wet heads and below-crest weir support return `PhysicalZero` with no proposed direction. Equal pump heads query zero differential head and can produce positive flow.

Direction and regime use original supported heads. Rounded depth differences never decide whether unequal original stages produce flow. Active-only derived arithmetic follows physical eligibility and flap policy. Blocked branches skip unused constitutive products and expose no invented hydraulic operands.

## Stable numerical evaluation

The laws use persistent F64 tensors on an explicitly configured one-thread numr CPU client. Coefficient/geometry sums of logarithms remain cached. No Solvr algorithm or copied generic numerical implementation enters these explicit laws.

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

The existing `propose(heads)` remains available for passive-only batches. Missing pump operations reject instead of assuming stopped or running. An operation cannot silently apply to a different structure or passive law.

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

## Errors and remaining mechanisms

`StructureError` separates typed input categories, numerical representation errors, and runtime errors. Input categories distinguish definition, identity, datum, head convention, operating point, unknown support, unsupported regime, and nonfinite original data. Ordered element indices remain available without parsing diagnostic strings.

The component supplies no donor storage, outlet integration, storage snapshots, controls, network solve, moving geometry, or hydraulic coupling. A future coordinator owns synchronized head provenance and paired shared-volume admission.

## Applicability references

- [USBR submerged-orifice relation](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap09_05.html).
- [USBR suppressed sharp-crest and approach definitions](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_03.html).
- [HEC rectangular-weir submergence correction](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/7.0/modeling-gated-spillways-weirs-and-drop-structures/uncontrolled-overflow-weirs/submerged-weir-flow).
- [USBR submerged-weir measurement limitations](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_15.html).
- [Hydraulic Institute pump-curve and affinity assumptions](https://datatool.pumps.org/pump-fundamentals/pump-curves).
- [EPA hydraulic addendum pump affinity model](https://www.epa.gov/system/files/documents/2022-02/swmm5-reference-manual-ii-addendum-20220210mas2wr-1.pdf).
