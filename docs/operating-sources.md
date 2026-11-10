# Exclusive operating sources at a point

`OperatingSourceBatch` binds one writer per element in the complete actual `StructureBatch`. Point queries combine manual operations, prescribed openings, accepted actuator positions, and explicitly modelled sampled pump operations. Queries advance no source.

Run the [mixed-source example](../examples/operating_sources.rs):

```sh
cargo run --example operating_sources
```

## Exhaustive source ownership

`StructureBatch::with_operating_sources` requires unique nonempty source identities and nonempty ordered groups. Every original structure belongs to exactly one group. Overlap, omission, duplicates, unknown structures, and incompatible laws reject before numerical evaluation.

| Binding | Admitted elements and query input |
| --- | --- |
| `Manual` | Explicitly named passive laws, pumps, and gates. Every query supplies their actual operations |
| `Prescribed` | Actual assigned gate subset from an immutable trajectory batch |
| `Actuated` | Actual assigned gate subset from a compatible actuator context, with fresh accepted frames |
| `ControlledPump` | Actual assigned pump subset with explicit instantaneous response assumptions and fresh accepted control frames |

Multiple groups of each kind are supported. Each group follows original structure order. Registration order determines dynamic frame order and event origins.

Manual ownership has no inferred complement. `manual_operations` covers all manual-owned elements in complete original subsequence order. Passive operations remain explicit. Pumps supply explicit running or stopped-isolated operations. Manual gates supply actual `GateOpening` values.

Every bounded source retains the complete actual structure context. Equal IDs never substitute changed geometry, coefficients, datum, applicability, direction, or ordering. `with_gate_trajectory_subset` explicitly admits a nonempty ordered gate subset. Existing `with_gate_trajectories` retains its exhaustive meaning.

## Fresh accepted response frames

Each point query supplies one `OperatingSourceFrame` per dynamic group in registration order. Actuated groups use `ActuatedGate`, and controlled pump groups use `ControlledPump`. Source identity, response kind, and complete immutable context must match. Accepted response time numerically equals query time without tolerance. Receipts retain each time's original bits separately.

The registry retains immutable actuator definitions rather than capturing accepted position at construction. Callers advance or restore actuator batches explicitly and then supply fresh frames. An independently reconstructed compatible actuator is eligible. Changed initial states, policies, or calendars reject even when current centers match.

Actuated frames supply admitted persistent position tensors. No public arbitrary position array bypasses actuator checks. Earlier proposals remain immutable when callers later accept different actuator state.

Controlled pump frames select original stopped/running speed literals through Numr from the accepted relay latch. Their declared ideal instantaneous model makes commanded operation actual on the decision's right side. The registry retains immutable response assumptions and control definitions, without capturing control continuation during registration. It infers no motor dynamics or native controller authority. Read [controlled pump response](controlled-pumps.md) for its complete contract.

## Point time and closed support

All queries require finite time. All-manual registries admit signed point times without inventing an origin, duration, or future constant state. `support()==None` identifies the absence of a registered calendar. Every query still requires complete explicit manual operations.

Bounded sources retain their original domains. Common support is their closed intersection, including a single point when endpoints coincide. Disjoint domains reject. Negative time rejects through bounded-source support rather than an unrelated blanket point-time restriction.

Declared schedule coverage never supplies actual state at every supported time. A fresh accepted frame remains mandatory at each queried time. No interpolation, source advancement, extrapolation, gap filling, or last-value hold occurs inside the registry.

Heads and operations describe the query point. Head packets carry no independent acquisition timestamp, so the registry establishes no historical head provenance.

## Original events and origins

`next_event_after` combines prescribed knots, actuator events, pump control decisions, and the common terminal bound. It returns the earliest strictly later supported event. Exactly equal represented times merge their flags and retain source origins in registration order. No epsilon merges different original times.

Events beyond common support are omitted. Common terminal and point-only support have no later event. An all-manual registry returns no event because it declares no calendar.

Actuator target changes or completion can alter later events. The returned event describes current frames, not a complete predicted future sequence. Callers advance sources explicitly before querying again.

## Certified closure and unknown heads

Identity, datum, convention, and malformed known-value checks apply before any isolation waiver. Unknown support remains unknown rather than becoming a dry bed or fabricated stage.

| Source | Isolation required for unknown heads |
| --- | --- |
| Manual | Gate opening is exactly zero, or the pump operation explicitly declares stopped isolation |
| Prescribed | Original-time zero knot/hold support, zero represented opening, and zero error |
| Actuated | Canonical zero center, zero lower/upper bounds, and zero error from admitted actual state |
| Controlled pump | Selected original stopped-isolated command under the declared response assumption |

An interval containing positive opening cannot establish closure through an approximate center. Positive center with an interval containing zero remains nonisolated. Pump isolation requires its explicit stopped-isolated operation. Running at a curve's zero discharge never supplies isolation authority. Existing law-specific support checks remain binding afterward.

## Immutable point outputs and uncertainty

`propose_at` returns an `OperatingSourceProposal` with explicit `time_s()`, `operations()`, and `outputs()` projections. Each `OperatingSourceOutput` keeps its hydraulic output attached to a typed source receipt.

| Receipt | Retained meaning |
| --- | --- |
| Manual | Original source identity, structure identity, query time, and explicit operation |
| Prescribed gate | Absolute opening-error receipt, segment/exact-knot metadata, and original closure support |
| Actuated gate | Actuator identity, accepted time/steps, full center/interval/error, phase, and support |
| Controlled pump | Controller identity, accepted time/steps, latch, response model, and actual stopped/running operation |

Gate discharge uses the admitted represented center opening and supplied heads. It is not an enclosure of discharge for every ideal opening inside the position receipt. Nonlinear laws and regime changes prevent a universal flow-error claim from opening accuracy alone. Source receipts therefore remain attached to the result.

Pump laws consume the selected actual operation through the existing pump evaluator. Their existing wet support, suction, speed-policy, and numerical admission remain binding.

Repeated projections neither reevaluate nor advance sources. Successful assembly retains derived positions and pump operations as tensors. Explicit outputs and existing error diagnostics provide projection boundaries.

## Numerical checks

The original 52 source-specific tests check ownership, frame identity, events, support, tensor routing, and center-law equivalence. They include 25 isolation cases and a 129-group case with reversed registration, distinct values, signed zeros, and retained uncertainty.

The isolation cases distinguish actual zero from zero targets and unresolved closure. Every malformed known-head category remains checked under manual, prescribed, and actuated closure. Scalar, vector, and tail layouts retain original element indices and source receipts.

Stable Rust and Rust 1.95 pass the same cases. The public example emits identical output on both compilers. These checks establish point composition and source admission. They establish no ideal-flow enclosure, integrated-volume accuracy, or measured-event accuracy.

## Continuation and scope

The registry has no accepted clock or evolving state and needs no numerical snapshot. Reconstruct it from identical immutable sources. Restore actuator or controlled pump continuation through its complete snapshot contract before supplying frames.

This point-only wrapper exposes no `held_volumes` conversion. It establishes no integrated trajectory, hydraulic consumption, donor availability, accepted paired exchange, native execution, or continuous hydraulic feedback. A sampled gate command is not an actual opening and cannot bypass actuator response. Pump response requires its explicit ideal model. Registry composition infers no physical motor dynamics.
