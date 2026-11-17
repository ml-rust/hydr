# Apply sampled commands to pumps

Use `ControlledPumpBatch` when actual pump operation follows sampled commands instantaneously. Declare that response assumption before constructing the batch.

## Declare the response

`ControlledPumpBatch::new` consumes a checked pump-only `OperatingControlBatch` and complete ordered response definitions.

| Input | Requirement |
| --- | --- |
| `PumpResponseDefinition.structure_id` | Match the corresponding actual controlled pump |
| `PumpResponseModel::IdealInstantaneous` | Actual operation equals the accepted command on the decision's right side |
| `response_provenance` | Nonempty evidence identifying the approximation |
| `ideal_response_applicability` | Nonempty applicability evidence and excluded motor dynamics |
| Original pump law | Retain curve, speed policy, orientation, suction bound, and stopped-isolation evidence |

The response adds no motor state, counter, or calendar. It can adopt advanced checked controls because it retains no independent memory.

Adoption preserves qualification timers, accepted time, original decisions, and control receipts.

### Check the approximation

Select this model only where its instantaneous response fits the supplied operating envelope. It excludes acceleration, startup delay, coast-down, and native control authority.

## Advance a decision

1. At a decision, prepare the existing sampled-control model's causal observations. Otherwise, prepare an empty sample slice.
2. Call `trial_to(end_s, samples)` with that slice.
3. Consume `into_next()` to accept the complete candidate.

The candidate command supplies the endpoint's right-side operation. Queries never advance controls.

Dropping or rejecting a trial preserves accepted time, commands, pending qualification, counters, and receipts.

## Read actual operation

| Accessor | Result |
| --- | --- |
| `controls()` | Immutable accepted controller |
| `outputs()` | Actual operation with controller identity, accepted time/count, latch, and response model |
| `operations()` | Original-order operation packets |

`outputs()` and `operations()` explicitly copy values from the runtime. Running results retain the selected original speed bits.

Stopped output remains `StructureOperatingPoint::PumpStoppedIsolated`. Internal operands used for safe arithmetic never become a reported running speed.

## Register the operating source

1. Register `OperatingSourceBinding::ControlledPump`.
2. Supply a fresh `OperatingSourceFrame::ControlledPump` for each point or event query.
3. Keep dynamic frames in registration order, as required by [operating sources](operating-sources.md).

Accepted frame time must numerically equal query time. Receipts retain both timestamps' original bits separately.

Compatibility covers complete original control, pump, and response assumptions.

### What a selected operation permits

| Operation | Admission |
| --- | --- |
| Stopped isolation | Declared isolation waives influential Unknown support |
| Running | Requires supported wet suction and receiving support, admitted speed, and closed curve coverage |
| Running curve shutoff | Keeps running admission and grants no isolation authority |
| Malformed known input | Rejects before isolation, including nonfinite values or incompatible identities, datum, or head convention |

Numr selects original running masks, speed literals, and power-of-two metadata from accepted latch tensors. Point evaluation reuses the pump curve and shared structure evaluator without copying derived operation arrays.

Source receipts retain actual operation, controller identity, time/count, latch, and response model.

Flow results describe instantaneous proposals. They establish no donor availability, integrated transfer, or accepted water movement.

## Save and restore

Trusted snapshots retain response assumptions and complete control continuation without numerical projection. Restore retains compatible destination contexts.

For caller-owned packages:

1. Preserve complete response definitions and the controller package.
2. Encode every original F64 bit.
3. Import through the response compatibility checks.

Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `CONTROLLED_PUMP_SNAPSHOT_SCHEMA` and `CONTROLLED_PUMP_CONTRACT_VERSION`. `restore_data` compares both before any other check and rejects a foreign value with a definition input error. The nested `controls` package carries its own identity and is checked too, before the controller changes. The error names the found and expected values. Capture a new package to recover. Packages without the identity fields are rejected.

Import replaces accepted continuation atomically. Rejection preserves the destination.

Bounded receipt checks do not authenticate complete history. Bitwise continuation requires the same build, runtime, layout, accepted partition, and future causal samples.

## Run the example

```sh
cargo run --example controlled_pumps
```

This example covers qualification, operation, and point-law queries using synthetic response assumptions and pump inputs.

It establishes no motor dynamics, native parity, field calibration, donor budget, or accepted transfer.
