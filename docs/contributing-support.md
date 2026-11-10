# Assign rainfall and source ownership

Use `hydr::runoff::SupportPlan` to declare who supplies rainfall/runoff and construct checked numerical inputs.

The application supplies GIS overlays, acquired data, and partition evidence. Hydr checks declared metadata without proving geometric disjointness.

This component establishes no hydraulic runoff attachment or physical source composition.

## Try the example

```sh
cargo run --example contributing_support
```

The [example](../examples/contributing_support.rs) assigns one active cell's observed rain to an upstream owner. Its standalone model receives the original forcing separately.

The cell remains hydraulically active. Its terrain and boundary geometry stay unchanged.

## 1. Declare the physical targets

Construct `SupportPlan::new(input, domain, boundaries, definitions)` with explicit numerical targets.

`PartitionAssertion` requires named complete/disjoint support and provenance evidence. Distinct IDs alone establish no geometric proof.

| Declaration | Required data |
| --- | --- |
| `SupportAtom` | Unique identity and finite positive represented area |
| `SurfaceCellSupport` | One complete disjoint footprint per active cell, including suppressed cells |
| `ModelledRunoff` | Exactly one source per supplied definition, with a unique future outlet ID |
| `ImportedRunoff` or `MeasuredFlow` | Complete already-composed input at exactly one matching bound face |
| `Baseflow` or `Release` | Empty-support leaf included exactly once by a selected imported/measured root |
| `BoundaryBinding` | Exactly one identity/provenance binding per declared open face |

### Rainfall ownership rules

| Rule | Required declaration |
| --- | --- |
| Primary owner | Every partition atom has exactly one selected primary owner. |
| Direct rain | Own the complete active-cell footprint. |
| Suppressed rain | Name the selected upstream owner. The complete cell footprint must lie within that owner's support. |
| Cell overlap | Footprints cannot overlap, even when both cells suppress rainfall. |

### Area checks

Modelled footprints independently match their actual definition areas. Every active-cell footprint independently matches the actual uniform grid cell area.

Area checks retain fixed-order addition bounds and the target area's rounding allowance.

Absorbed positive additions and nonfinite arithmetic reject admission. These arithmetic bounds supply no GIS error tolerance.

## 2. Select replacements and composed sources

### Whole measured replacements

`WholeReplacement` selects a measured root over the exact union of explicitly replaced complete sources.

| Rule | Behavior |
| --- | --- |
| Eligible sources | Modelled runoff or imported runoff |
| Replaced imported runoff | Preserve its boundary identity. |
| Partial selection or residual-area scaling | Unsupported |
| Missing selected data | Never reactivate a replaced model. |

### Additive leaves

Leaves declare composition already performed upstream. Hydr never sums their rates or momentum.

- Modelled runoff: Cannot include additive leaves.
- Inclusion graph: Orphan, duplicate, shared, cyclic, and non-leaf references reject.
- Replacement: Replacing a leaf's former parent never activates that leaf independently.

## 3. Supply the exact interval

Call `admit_interval` with one `SupportIntervalInput` carrying the original elapsed start/end values.

All three inputs require that identical interval:

1. Surface observations.
2. Active modelled requests.
3. Bound physical boundary inputs.

Coverage containment is insufficient. Supply fresh input packets after interval shortening.

### Active model requests

| Condition | Required behavior |
| --- | --- |
| Active modelled source | Supply its request. |
| Suppressed source | Supply no request, including zero-valued requests. |
| Empty active set | Construct no fabricated numerical batch or zero forcing entries. |

`active_definitions()` and `active_definition_indices()` preserve the original numerical definition order.

### External runon limit

This ownership component requires every modelled external runon volume to equal zero.

Positive runon needs donor provenance and a future paired coordinator. A receiving model ID cannot establish donor ownership.

The lower-level standalone subcatchment API still accepts explicit destination volumes. That API does not acquire this ownership claim.

### Read rainfall suppression records

Raw observed rain remains visible where an upstream owner suppresses injection.

| Record or process | Behavior |
| --- | --- |
| `RainSuppressionReceipt` | Retain cell support, observed rate, injected zero, and selected owner identity. |
| Direct cell | Preserve observed rain. |
| Evaporation and withdrawal | Pass unchanged through existing numerical checks. |
| Inactive cell | Require explicit zero observations and forcing. |

## 4. Use the admitted inputs

Admitted packets identify their specific plan through a private immutable token. Public getters expose immutable values only.

Cloning a plan retains its token. Separately constructed plans cannot exchange packets, even when their declarations match.

| Consumer | Checks and result |
| --- | --- |
| `surface_schedule` | Construct the actual `ForcingSchedule`, including complete physical and contiguous-interval checks. |
| `trial_subcatchments` | Check definition IDs/order/parameter bits and accepted start time. Then invoke the actual immutable numerical trial. |

The numerical batch owns its independently checked config. The support plan binds physical definitions and exact input time.

| Active models | Batch requirement |
| --- | --- |
| None | Require `batch=None` and return `None`. |
| Present | Require their exact numerical batch. |

Repeated trials against unchanged accepted state remain legitimate. Reusing a packet after advancement rejects its old start time.

### Boundary consumers

| Boundary association | Accepted input |
| --- | --- |
| Hydrological root | Complete `Flux` or `Discharge` data only |
| Unassociated | Stage, Exterior, Transmissive, or explicitly outward/zero Flux or Discharge |
| Incoming prescribed rate | Require a selected root. |

Every open face still requires identity/provenance bindings. Existing numerical boundary-regime limits remain binding.

Physical values pass unchanged. Hydr composes no rate, momentum, stage, or tangential velocity here.

## Handle errors and persistence

| Error source | Detail |
| --- | --- |
| Declaration rules | `SupportErrorKind` distinguishes partition, area, ownership, replacement, and inclusion errors. |
| Interval/input rules | The same type distinguishes coverage, identity, destination, and composition errors. |
| Numerical consumer | Retain typed `EngineError` or `SubcatchmentError` causes. |

Rejected admission preserves immutable plans and every numerical state.

Component snapshots acquire no unused support metadata. A future coordinator must persist its complete immutable plan and accepted records.

## Unsupported uses

- Persistence: Durable token serialization remains outside this component.
- Coupling: Paired runoff exchange and hydraulic attachment remain unsupported.
- Data services: Provider frameworks and source aggregation remain outside this component.
