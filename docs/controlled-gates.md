# Advance gate controls and response together

Use `ControlledGateBatch` to accept sampled control and ideal gate response as one state. Construct fresh child batches first.

The batch owns both children. Requested target and actual opening remain separate values.

## Construct the owner

`ControlledGateBatch::new` consumes an `OperatingControlBatch` and a `GateActuatorBatch`.

| Check | Requirement |
| --- | --- |
| Fresh state | Both children retain their initial state with zero accepted intervals |
| Physical model | Complete original structure definitions match, including geometry, applicability, orientation, and numerical bits |
| Ownership | Every controller binds exactly one supplied actuator through the same actual gate |
| Commands | Both commands are `ControlCommand::GateTarget` within the gate range |
| Order | Preserve the actual-structure order without reordering or expanding the child layout |
| Calendar | Initial time, terminal time, and all decision/target times match by F64 bits |
| Initial target | Selected initial command numerically equals the actuator's initial target |

Numerical equality permits differently signed zeros. Each child retains its original bits.

The owner supplies the only target writer for its actuators. It accepts no manual target packets. Other structures can remain outside this subset.

## Advance to an event

1. Read `next_event()`.
2. Choose an endpoint no later than that next-event bound.
3. Supply complete ordered causal samples at a decision endpoint, or an empty slice otherwise.
4. Call `trial_to(end_s, samples)`.
5. Consume `into_next()` to accept both candidate children.

The next-event bound already includes the terminal horizon.

Dropping or rejecting the candidate preserves both children. Control, response, and numerical errors cannot partially accept a step.

Exactly coincident represented events combine flags. Hydr never merges different original times through an epsilon.

### What happens at a decision

| Order | Action |
| --- | --- |
| 1 | Every controller reads its complete original sample and determines a candidate latch |
| 2 | The actuator follows the old target through the endpoint's left state |
| 3 | A coincident arrival completes the old episode |
| 4 | The selected command applies to the right interval |

A new command cannot repair a rejected old-target endpoint.

Numr selects the latch's original low or high target from persistent F64 tensors. Normal trials create no host target packets or derived command arrays.

Computed issue and availability times equal the original decision time. Earlier sensor availability does not make the command available earlier.

### Intervals without decisions

Both children advance once over the same original interval. The controller retains its command while the actuator advances actual opening.

Repeated numerically equal commands preserve active target bits and the motion anchor. The actuator separately retains the latest raw command bits.

Child accuracy policies remain binding. Composition does not relax qualification timing, opening uncertainty, arrival windows, or positive-opening support.

## Read results and query flow

| Accessor | Result |
| --- | --- |
| `controls()` | Immutable controller reference with its existing projections |
| `actuators()` | Immutable actuator reference with position, support, and diagnostics |
| `time()` | Common accepted time |
| `accepted_steps()` | Common accepted interval count |

Use the actuator with `OperatingSourceBinding::Actuated` and a fresh `OperatingSourceFrame::ActuatedGate`.

The [source registry](operating-sources.md) keeps opening uncertainty attached to center-based flow proposals. It never substitutes a controlled target for actual opening.

These proposals establish no enclosure of ideal discharge or integrated transfer volume.

## Save and restore

### Trusted snapshots

Trusted snapshots retain both complete child continuations without copying derived arrays. Restore checks compatible original inputs and retains destination contexts.

### Caller-owned packages

Preserve both complete child packages and every original F64 bit during encoding.

Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `CONTROLLED_GATE_SNAPSHOT_SCHEMA` and `CONTROLLED_GATE_CONTRACT_VERSION`. `restore_data` compares both before any other check and rejects a foreign value with a definition input error. The nested `controls` and `actuators` packages carry their own identity and are checked too, before either child changes. The error names the found and expected values. Capture a new package to recover. Packages without the identity fields are rejected.

Import checks each child's receipt arithmetic before checking their relationship:

| Cross-child check | Required agreement |
| --- | --- |
| Progress | Clocks, counters, calendar progress, and retained interval boundaries |
| Authority | Current command/target correspondence and retained command provenance |
| Generations | Control transitions match target generations when low/high targets differ numerically |
| Equal targets | Generation stays zero despite latch transitions |
| Retained evidence | Immediate before states and latest transition/change receipts agree |

Latest raw commands retain their original bits. Rejected imports change neither child.

Packages retain no recursive prior-state chain. Bounded consistency checks do not authenticate complete history.

Bitwise continuation requires the same build, runtime, layout, accepted step partition, and future causal samples.

## Scope and example

The response uses the supplied ideal constant directional rates. It excludes mechanical dynamics and native actuator parity.

This component does not establish:

- Continuous trigger completeness or endogenous hydraulic feedback.
- Pump response.
- Accepted hydraulic transfers or ideal-discharge enclosures.
- Measured-event accuracy.

Run the opening/reversal example:

```sh
cargo run --example controlled_gates
```

The example prints requested targets beside actual openings. Its synthetic rates and policies establish no field calibration.
