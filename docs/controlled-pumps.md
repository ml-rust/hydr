# Sampled instantaneous pump response

`ControlledPumpBatch` supplies hydraulic application developers an explicitly assumed instantaneous response to sampled pump commands.

## Response assumptions

`ControlledPumpBatch::new` consumes a checked pump-only `OperatingControlBatch` and complete ordered response definitions.

| Input | Contract |
| --- | --- |
| `PumpResponseDefinition.structure_id` | Matches the corresponding actual controlled pump |
| `PumpResponseModel::IdealInstantaneous` | Actual operation equals the accepted command on the decision's right side |
| `response_provenance` | Nonempty evidence identifying the response approximation |
| `ideal_response_applicability` | Nonempty applicability evidence and excluded motor dynamics |
| Original pump law | Retains curve, speed policy, orientation, suction bound, and stopped-isolation evidence |

The response adds no motor state, counter, or calendar. Construction accepts advanced checked controls because this response retains no independent memory. Adoption preserves qualification timers, accepted time, original decisions, and control receipts.

The model excludes acceleration, startup delay, coast-down, and native control authority. Select it only when its instantaneous approximation fits the supplied operating envelope.

## Decisions and results

`trial_to(end_s, samples)` reuses the existing sampled hysteresis and qualification model. A candidate command supplies the endpoint's right-side operation. Nondecision intervals require empty sample slices. Queries never advance controls.

Dropping a trial preserves accepted continuation. `into_next()` accepts the complete candidate by ownership. Any rejection preserves accepted time, commands, pending qualification, counters, and receipts.

`controls()` exposes the accepted controller immutably. `outputs()` explicitly projects actual operation beside controller identity, accepted time/count, latch, and response model. `operations()` explicitly projects original-order operation packets.

Stopped output remains `StructureOperatingPoint::PumpStoppedIsolated`. Internal safe arithmetic operands never become a reported running speed. Running output retains the selected original speed bits.

## Source authority and hydraulic admission

Register the owner through `OperatingSourceBinding::ControlledPump`. Supply a fresh `OperatingSourceFrame::ControlledPump` for every point or event query. [Operating sources](operating-sources.md) require correctly typed dynamic frames in registration order.

The frame's accepted time numerically equals the query time. Source receipts retain accepted-time bits separately from query-time bits. Complete original control, pump, and response assumptions govern compatibility.

Numr selects original running masks, speed literals, and power-of-two metadata from accepted latch tensors. Point evaluation uses the existing pump curve and shared structure evaluator without derived-operation array readback.

| Selected operation | Admission |
| --- | --- |
| Stopped isolation | Declared isolation waives influential Unknown support |
| Running | Requires supported wet suction, wet receiving support, admitted speed, and closed curve coverage |
| Running curve shutoff | Retains running admission and supplies no isolation authority |
| Malformed known input | Rejects before isolation, including nonfinite values and incompatible identities, datum, or head convention |

Source receipts retain actual operation, controller identity, accepted time/count, latch, and response model. Center-based hydraulic results remain instantaneous proposals. They establish no donor availability, integrated transfer, or accepted water movement.

## Complete restoration

Trusted snapshots retain original response assumptions and complete control continuation without numerical projection. Restoration retains compatible destination contexts.

Caller-owned packages include complete response definitions and the existing controller package. Preserve every original F64 bit during encoding. Import checks response compatibility before atomically replacing accepted continuation. Rejected imports preserve the destination.

The underlying bounded receipt checks establish no complete historical authenticity. Bitwise continuation requires the same build, runtime, layout, accepted partition, and future causal samples.

## Example and limits

Run the deterministic qualification, operation, and point-law example:

```sh
cargo run --example controlled_pumps
```

The example declares synthetic response assumptions and pump inputs. It establishes no motor dynamics, native parity, field calibration, donor budget, or accepted transfer.
