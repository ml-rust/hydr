# Choose one operating source per structure

Use `OperatingSourceBatch` to query structures whose operations come from different owners. Declare one writer for every structure first.

Point queries combine manual operations, prescribed openings, actuator positions, and sampled pump operations. They advance no source or water.

## Register every structure

`StructureBatch::with_operating_sources` covers the complete actual `StructureBatch`. It requires unique nonempty source identities and nonempty ordered groups.

| Binding | Assigned elements | Input at query time |
| --- | --- | --- |
| `Manual` | Explicit passive laws, pumps, and gates | Actual operations |
| `Prescribed` | Actual gate subset from an immutable trajectory batch | Original trajectory support |
| `Actuated` | Actual gate subset from a compatible actuator context | Fresh accepted frame |
| `ControlledPump` | Actual pump subset with an explicit instantaneous response | Fresh accepted control frame |

Every original structure must belong to exactly one group. Overlap, omission, duplicates, unknown structures, and incompatible laws reject before evaluation.

Multiple groups of each kind are supported. Each follows original structure order. Registration order determines dynamic frame order and event origins.

### Keep the physical context unchanged

Every bounded source retains the complete actual structure context. Equal IDs cannot substitute changed geometry, coefficients, datum, applicability, direction, or order.

`with_gate_trajectory_subset` admits a nonempty ordered gate subset. `with_gate_trajectories` retains its exhaustive meaning.

## Supply a point query

1. Choose a finite query time inside all bounded sources' common support.
2. Supply heads and complete `manual_operations` in original subsequence order.
3. Supply one fresh `OperatingSourceFrame` per dynamic group, in registration order.
4. Call `propose_at`.
5. Read the proposal's explicit values when needed.

Manual ownership has no inferred complement. Passive operations remain explicit. Pumps require running or stopped-isolated operations, and manual gates require actual `GateOpening` values.

Head packets have no independent acquisition timestamp. A query establishes no historical head provenance.

### Supply fresh dynamic frames

| Group | Frame | Required agreement |
| --- | --- | --- |
| Actuated | `ActuatedGate` | Source identity, response kind, complete immutable context, and accepted time |
| Controlled pump | `ControlledPump` | Source identity, response kind, complete immutable context, and accepted time |

Accepted time must numerically equal query time without tolerance. Receipts retain each timestamp's original bits separately.

Advance or restore response batches explicitly before supplying frames. Registration retains immutable definitions, not accepted position or controller continuation.

An independently reconstructed compatible actuator is eligible. Changed initial states, policies, or calendars reject even when current centers match.

### How actual values reach the law

Actuated frames supply checked persistent position tensors. Arbitrary public position arrays cannot bypass actuator checks.

Controlled pump frames select original stopped/running speed literals through Numr from the accepted latch. Their ideal instantaneous response makes the command actual on the decision's right side.

The registry retains response assumptions and control definitions. It infers neither motor dynamics nor native controller authority.

Read [controlled pump response](controlled-pumps.md) for the complete response contract.

## Check time support

| Registry | Admitted time support |
| --- | --- |
| All manual | Any finite signed point time. `support()==None` means no registered calendar |
| Bounded sources | Their original closed-domain intersection |
| Coincident endpoints | A single supported point |
| Disjoint bounded domains | Construction rejects |

All-manual queries still need complete operations. They imply no origin, duration, or future constant state.

Negative time rejects through bounded-source support when applicable. There is no unrelated blanket restriction on point time.

Schedule coverage does not provide actual state at every covered time. Each dynamic query still needs a fresh accepted frame.

The registry performs no interpolation, advancement, extrapolation, gap filling, or last-value hold.

## Find the next event

`next_event_after` returns the earliest strictly later supported event from:

- Prescribed knots.
- Actuator events.
- Pump control decisions.
- The common terminal bound.

Exactly equal represented times combine flags. Event origins retain registration order. Different original times never merge through an epsilon.

| Situation | Result |
| --- | --- |
| Event beyond common support | Omitted |
| Common terminal or point-only support | No later event |
| All-manual registry | No event because no calendar exists |

Returned events describe current frames, not a complete predicted sequence. Actuator target changes or completion can change later events.

Advance sources explicitly before querying again.

## Handle unknown heads

Identity, datum, head convention, and malformed known values are checked before any isolation waiver.

Unknown support stays unknown. It never becomes a dry bed or fabricated stage.

| Source | Evidence required for isolation with unknown heads |
| --- | --- |
| Manual | Exactly zero gate opening, or explicit stopped-isolated pump operation |
| Prescribed | Original-time zero knot/hold, zero represented opening, and zero error |
| Actuated | Canonical zero center, singleton zero bounds, and zero error from admitted actual state |
| Controlled pump | Selected original stopped-isolated command under the declared response |

An interval containing positive opening cannot establish closure from an approximate center. Positive center with a zero-containing interval remains nonisolated.

A running pump with zero curve discharge grants no isolation authority. All remaining law-specific support checks still apply.

## Read proposals and uncertainty

`propose_at` returns an immutable `OperatingSourceProposal`.

| Request | Result |
| --- | --- |
| `time_s()` | Query time |
| `operations()` | Actual selected operations |
| `outputs()` | Hydraulic results attached to typed source receipts |

Each `OperatingSourceOutput` retains its hydraulic result and typed source receipt.

Earlier proposals stay unchanged after later source acceptance. Repeated value requests neither reevaluate nor advance sources.

### What each receipt retains

| Receipt | Retained evidence |
| --- | --- |
| Manual | Source and structure identity, query time, explicit operation |
| Prescribed gate | Absolute opening error, segment/exact-knot metadata, original closure support |
| Actuated gate | Actuator identity, accepted time/steps, center, interval, error, phase, and support |
| Controlled pump | Controller identity, accepted time/steps, latch, response model, actual operation |

Gate discharge uses the represented center opening and supplied heads. Opening uncertainty does not provide a universal discharge-error bound across nonlinear laws or regime changes.

The receipt therefore stays attached to the result. The result does not enclose every ideal discharge inside that opening interval.

Pump results retain wet-support, suction, speed-policy, and numerical checks from the existing pump evaluator.

Successful assembly retains derived positions and pump operations as tensors. Explicit outputs and error diagnostics are the data-copy boundaries.

## Reconstruct sources after restart

The registry has no evolving state or accepted clock. It needs no numerical snapshot.

1. Reconstruct it from identical immutable sources.
2. Restore actuators or controlled pumps through their complete snapshot contracts.
3. Supply fresh frames before querying.

## Scope

This point-only registry exposes no `held_volumes` conversion. It establishes no integrated trajectory, hydraulic consumption, donor availability, or accepted paired exchange.

Native execution and continuous hydraulic feedback remain unsupported. Gate commands cannot bypass actual response, and pump response requires its explicit ideal model.

## Checks and example

The 52 source-specific tests cover ownership, frame identity, events, support, tensor routing, and center-law equivalence.

| Coverage | Cases |
| --- | --- |
| Isolation | 25 cases distinguish actual zero, zero targets, and unresolved closure |
| Assembly | 129 reversed-registration groups with distinct values, signed zeros, and retained uncertainty |
| Malformed heads | Every known-head category under manual, prescribed, and actuated closure |
| Layout | Scalar, vector, and tail cases retain original indices and receipts |

Stable Rust and Rust 1.95 pass the same cases. The public example emits identical output on both compilers.

These checks establish point composition and source admission. They establish no ideal-flow enclosure, integrated-volume accuracy, or measured-event accuracy.

Run the [mixed-source example](../examples/operating_sources.rs):

```sh
cargo run --example operating_sources
```
