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

Orifice parameters require positive finite area, dimensionless discharge coefficient, and gravity. Sill/crown elevations require finite values with crown above sill. Weir parameters require positive finite crest length and coefficient, plus finite crest elevation. The weir coefficient uses SI `m^(1/2)/s`.

`OrificeApplicability` retains coefficient provenance and fully submerged applicability evidence. `WeirApplicability` separately retains coefficient provenance, free-nappe/approach assumptions, and correction applicability evidence. Nonempty declarations record the caller's physical claims. They establish no site-specific coefficient accuracy.

The rectangular correction does not apply to contracted, broad-crested, ogee, or other crest geometries. Partial orifices, culverts, pumps, moving gates, and inferred blockage remain unsupported.

## Exact identity and head support

Every `StructureElement` declares a unique structure ID, distinct ordered endpoint IDs, common datum, direction policy, and concrete law. Parallel structures can share endpoint pairs. Input packets must match the exact structure order, endpoint identities, and datum.

`HeadConvention::QuiescentStage` identifies water-surface stage heads. Explicit energy-head packets reject. No datum transformation or silent head-convention conversion occurs.

| Head support | Admission |
| --- | --- |
| `Wet { stage_m }` | Original finite water-surface elevation |
| `Dry { bed_elevation_m }` | Known dry weir connection with finite bed at or below crest |
| `Unknown` | Always rejects before blocking or zero-flow shortcuts |

Dry support never receives a fabricated physical water elevation. Wet-above-crest flow toward eligible dry support uses the free-overflow law. Both dry, or no wet donor above crest, returns below-crest zero.

Orifices reject either dry endpoint, partial submergence, and exact crown equality. Both original wet stages must lie strictly above crown before equal-head or ideal-flap shortcuts.

## Direction, regimes, and zero flow

Positive discharge means endpoint A to B. Reversed heads retain the same endpoint identities and produce negative discharge.

`Bidirectional` requires explicit symmetric coefficient applicability. `AtoBOnly` and `BtoAOnly` declare ideal flap behavior. Unsupported support/regimes reject before flap blocking. A physically eligible prohibited direction returns exact zero with `FlapBlocked` status.

`StructureRegime` records the physical classification separately from status. Blocked flows retain their nominal proposed direction. Equal eligible wet heads and below-crest weir support return `PhysicalZero` with no proposed direction.

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

Actual Numr execution passes fourteen independently frozen 800-digit reference cases. Twelve cases retain positive discharge. Two cases reject unrepresentable final discharge. The maximum measured relative discharge error across admitted cases is approximately `9.53e-14`.

These cases cover ordinary laws, adjacent stages, rounded depth plateaus, and normalized-gap underflow. They also cover unused coefficient, power, and radicand products outside representable range while final discharge remains finite.

Ordinary comparison uses `1e-12 m³/s + 1e-12*abs(reference)`. Tiny/extreme comparison uses `max(32*ulp(reference),1e-12*abs(reference))` with mandatory positive discharge. Mixed layouts exercise scalar, SIMD, and tail positions. No cross-layout bitwise equality promise follows.

This evidence checks numerical law evaluation. It establishes no empirical correction accuracy, site coefficient accuracy, field accuracy, or complete HEC solver parity.

## Errors and remaining mechanisms

`StructureError` separates typed input categories, numerical representation errors, and runtime errors. Input categories distinguish definition, identity, datum, head convention, unknown support, unsupported regime, and nonfinite original data. Ordered element indices remain available without parsing diagnostic strings.

The component supplies no donor storage, outlet integration, storage snapshots, controls, network solve, moving geometry, or hydraulic coupling. A future coordinator owns synchronized head provenance and paired shared-volume admission.

## Applicability references

- [USBR submerged-orifice relation](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap09_05.html).
- [USBR suppressed sharp-crest and approach definitions](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_03.html).
- [HEC rectangular-weir submergence correction](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/7.0/modeling-gated-spillways-weirs-and-drop-structures/uncontrolled-overflow-weirs/submerged-weir-flow).
- [USBR submerged-weir measurement limitations](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap07_15.html).
