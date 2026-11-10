# Atomic controlled gate response

`ControlledGateBatch` gives hydraulic application developers one accepted state for sampled controls and ideal gate response.

The batch owns both child batches. Each immutable trial advances their complete candidate states together. Actual opening remains distinct from the requested target.

## Construction and authority

`ControlledGateBatch::new` consumes a fresh `OperatingControlBatch` and a fresh `GateActuatorBatch`.

| Requirement | Contract |
| --- | --- |
| Fresh children | Both children remain at their original initial state with zero accepted intervals |
| Actual model | Complete original structure definitions match, including geometry, applicability, orientation, and numerical bits |
| Gate ownership | Every supplied controller binds exactly one supplied actuator through the same actual gate identity |
| Commands | Both original commands are `ControlCommand::GateTarget` inside the actual gate range |
| Order | Existing child admission retains the same actual-structure order. Construction adds no reordered or expanded child layout |
| Calendar | Original initial time, terminal time, and every decision/target time match by F64 bits |
| Initial target | The initial selected command numerically equals the actuator's original target |

Initial numerical equality permits differently signed zeros. Each child retains its own original bits. Repeated numerically equal commands preserve the actuator's active target bits and motion anchor. The actuator separately retains the latest raw command bits.

The composite supplies one target writer for its owned actuators. It accepts no manual target packets. Other structures can remain outside the controlled subset.

## Decision and response ordering

At a decision endpoint, every controller consumes its complete ordered causal sample. The existing sampled-control model determines the candidate latch. Numr selects that latch's original low or high target directly from persistent F64 tensors. Normal trials create no host target packets and project no derived command arrays.

The actuator evaluates the old target through the endpoint's left state. The selected endpoint command applies to the right interval. A coincident arrival completes the old episode before a new target starts. A new command never repairs a rejected old-target endpoint retroactively.

The computed command's issue and availability times equal the original decision time. Sensor availability remains separate. A causal observation available earlier does not make the computed command available before its decision.

Non-decision intervals require an empty sample slice. Both children advance once through the same original interval. The controller retains its command while the actuator advances actual opening.

## Events and atomic trials

`next_event()` combines original decisions, actuator arrival boundaries, and the terminal horizon. Exactly coincident times combine their flags. No epsilon merges distinct original events. A trial cannot cross the next event or extend terminal support.

`trial_to(end_s, samples)` borrows the accepted batch immutably. Dropping a candidate preserves both children. Any control, response, or numerical rejection preserves both children. `into_next()` accepts the complete candidate pair by ownership.

The accepted pair preserves the children's dimensional policies. No composite tolerance weakens qualification timing, opening uncertainty, arrival windows, or positive-opening support.

## Results and hydraulic proposals

`controls()` and `actuators()` expose immutable child references. Their explicit projections retain the existing command, position, support, and diagnostic contracts. `time()` and `accepted_steps()` describe the common accepted continuation.

The actuator reference can supply an `OperatingSourceBinding::Actuated` and each fresh `OperatingSourceFrame::ActuatedGate`. The [operating-source registry](operating-sources.md) retains actual opening uncertainty beside center-based hydraulic proposals. The controlled target never substitutes for actual opening. Source selection establishes no enclosure for ideal discharge or integrated transfer volume.

## Complete restoration

Trusted snapshots clone both complete child continuations without derived array projection. Restoration checks compatible original inputs and retains destination contexts.

Caller-owned snapshot packages include both complete child packages. Preserve every original F64 bit during encoding. Import first checks each child through its existing receipt arithmetic. Cross-child checks then require matching clocks, counters, calendar progress, and retained interval boundaries.

The composition also checks current command/target correspondence and retained command provenance. Latest raw commands retain their original bits. Target generations agree with control transitions when low and high targets differ numerically. Numerically equal targets retain generation zero despite latch transitions. Retained immediate before states and latest transition/change receipts must agree across children.

A rejected import changes neither child. These bounded consistency checks establish no complete historical authenticity. Snapshots retain no recursive prior-state chain. Bitwise continuation requires the same build, runtime, layout, accepted partition, and future causal samples.

## Scope and example

The response inherits the explicitly supplied ideal constant directional rates. It excludes mechanical dynamics and native actuator parity. Sampled qualification establishes no continuous trigger completeness or endogenous hydraulic feedback.

Pump response, accepted hydraulic transfers, ideal-discharge enclosures, and measured-event accuracy remain unsupported by this component.

Run the deterministic opening/reversal example:

```sh
cargo run --example controlled_gates
```

The example prints requested targets beside actual openings. Its synthetic rates and policies establish no field calibration.
