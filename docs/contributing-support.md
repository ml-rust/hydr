# Checked contributing support

`hydr::runoff::SupportPlan` checks declared rainfall/source ownership and constructs real numerical input consumers.
ARUS supplies GIS overlays, source acquisition, and partition evidence. Hydr checks declared metadata without proving geometric disjointness.
This unit adds no hydraulic runoff attachment or physical source composition.

Run the [support example](../examples/contributing_support.rs):

```sh
cargo run --example contributing_support
```

The example suppresses observed upstream rain on one active cell. The selected standalone model receives its original forcing separately.
The surface cell remains hydraulically active. Its terrain and boundary geometry never change.

## Immutable declaration

Construct `SupportPlan::new(input, domain, boundaries, definitions)` with explicit checked numerical targets.
`PartitionAssertion` requires named complete/disjoint support and provenance evidence. Distinct IDs alone establish no geometric proof.

| Declaration | Required contract |
| --- | --- |
| `SupportAtom` | Unique identity and finite positive represented area |
| `SurfaceCellSupport` | One complete disjoint footprint per active cell, including suppressed cells |
| `ModelledRunoff` | Exactly one source per supplied definition, with a unique future outlet ID |
| `ImportedRunoff` or `MeasuredFlow` | Complete already-composed input at exactly one matching bound face |
| `Baseflow` or `Release` | Empty-support leaf included exactly once by a selected imported/measured root |
| `BoundaryBinding` | Exactly one identity/provenance binding per declared open face |

Every partition atom has exactly one selected primary owner. Direct local rainfall owns its complete active-cell footprint.
Suppressed cells name their selected upstream owner explicitly. Their complete footprint must lie within that owner's support.
Cell footprints cannot overlap even when both cells suppress rainfall.

Modelled footprints independently match actual definition area. Every active-cell footprint independently matches actual uniform grid cell area.
Static area checks retain fixed-order addition bounds and the target area's rounding allowance.
Absorbed positive additions and nonfinite arithmetic reject admission. These arithmetic bounds supply no GIS error tolerance.

`WholeReplacement` selects a measured root over the exact union of explicitly replaced complete sources.
Replaced sources can be modelled runoff or imported runoff. Replacing imported runoff preserves its boundary identity.
Partial selection and residual-area scaling remain unsupported. Missing selected data never reactivates a replaced model.

Additive leaves declare composition already performed upstream. Hydr never sums their rates or momentum.
Modelled runoff cannot include additive leaves. Orphan, duplicate, shared, cyclic, or non-leaf inclusion references reject admission.
Replacing a leaf's former parent never activates that leaf independently.

## Exact interval admission

`admit_interval` accepts one `SupportIntervalInput` with exact original elapsed start/end values.
Surface observations, active modelled requests, and bound physical boundary inputs require that identical interval.
Coverage containment is insufficient. Shortened intervals require freshly supplied input packets.

Only active modelled sources receive requests. Suppressed requests reject even when their values equal zero.
`active_definitions()` and `active_definition_indices()` preserve original numerical definition order.
The empty active set requires no fabricated numerical batch or zero forcing entries.

This first ownership unit requires all modelled external runon volumes to equal zero.
Positive runon needs donor provenance and a future paired coordinator. A receiving model ID cannot establish donor ownership.
The lower-level standalone subcatchment API retains explicit destination-volume support without acquiring this ownership claim.

Raw observed rain remains visible even where upstream ownership suppresses injection.
Each `RainSuppressionReceipt` retains cell support, observed rate, injected zero, and selected owner identity.
Direct cells preserve observed rain. Evaporation and withdrawal pass unchanged through existing numerical checks.
Inactive cells require explicit zero observations and forcing.

## Concrete consumers

Admitted packets retain a private immutable plan-instance token. Public getters expose immutable values only.
Cloning a plan retains its token. Independently constructed plans cannot exchange packets, even with equal declarations.

| Consumer | Admission and result |
| --- | --- |
| `surface_schedule` | Constructs the actual `ForcingSchedule`, retaining complete physical and contiguous-interval checks |
| `trial_subcatchments` | Checks exact definition IDs/order/parameter bits and accepted start time, then invokes the actual immutable numerical trial |

The numerical batch owns its independently checked config. The support plan binds physical definitions and exact input time.
No active models require `batch=None`, returning `None`. Active models require their exact numerical batch.
Repeated trials against unchanged accepted state remain legitimate. Reusing a packet after advancement rejects its old start time.

Hydrological boundary roots accept complete `Flux` or `Discharge` data only.
Unassociated boundaries accept Stage, Exterior, and Transmissive data, plus explicitly outward or zero Flux/Discharge.
Incoming prescribed rates require a selected root. Identity/provenance bindings remain mandatory for every open face.
Caller-supplied physical values pass unchanged. Hydr performs no rate, momentum, stage, or tangential-velocity composition.
Existing numerical boundary regime limits remain binding.

## Errors and persistence

`SupportErrorKind` separates partition, area, ownership, replacement, inclusion, coverage, identity, destination, and composition errors.
Concrete consumer errors retain typed `EngineError` or `SubcatchmentError` causes.
Rejected admission preserves immutable plans and every numerical state.

The component snapshots acquire no unused support metadata. A future coordinator must persist its complete immutable plan and accepted receipts.
Durable token serialization, paired runoff exchange, provider frameworks, source aggregation, and hydraulic attachment remain outside this unit.
