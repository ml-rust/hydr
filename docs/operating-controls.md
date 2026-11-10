# Generate sampled operating commands

Use `OperatingControlBatch` to generate pump commands or gate targets from sampled stage observations. Start by declaring the thresholds and decision times.

A command requests an operation. It does not establish actual device response, flow, donor availability, or water transfer.

## Declare the controller

| Input | Required value |
| --- | --- |
| Controlled devices | A nonempty unique subset in original structure order, bound to the complete actual `StructureBatch` context |
| Signal | Controller, signal, support, and common-datum identities |
| Thresholds | Finite lower stage strictly below finite upper stage |
| Delays | Explicit finite nonnegative high/low qualification durations |
| Commands | Low/high mappings admitted by the actual pump speed policy or gate opening range |
| Initial state | An explicit latch with complete antecedent memory, or an explicit fresh-memory assumption |
| Schedule | Original finite, strictly increasing decisions inside a finite nonnegative support horizon |
| Timing | Maximum decision gap, maximum timer lateness, and evidence supporting sampled control |

The signal's physical support can differ from both structure endpoints. Declare its identity and elevation reference explicitly.

## Supply observations at decisions

1. Supply one sample per controller, in controller order.
2. Use the exact original decision time as the observation time.
3. Supply an availability time no later than the observation time.
4. Keep the original finite timestamps, including negative availability and antecedent times.

Unknown, dry, stale, future, or incompatible samples reject. Hydr supplies no replacement or last-value hold.

## Understand qualification

The latch records whether the controller currently selects its low or high command. Qualification measures persistence across supplied samples.

| Current latch | Observation | Result |
| --- | --- | --- |
| Low | Stage reaches or exceeds the upper threshold | Start or continue high qualification |
| High | Stage reaches or falls below the lower threshold | Start or continue low qualification |
| Either | Opposing condition is false | Retain the latch and cancel incomplete qualification |
| Either | Qualification reaches its delay within the lateness allowance | Switch once and clear pending state |

A zero delay switches at the qualifying decision. Inside the hysteresis band, the latch stays unchanged and incomplete opposing qualification cancels.

All controllers read the same original sample batch. They update simultaneously.

### What qualification does not prove

Persistence applies only to the supplied samples. The component does not establish continuous threshold history or rule out excursions between samples.

Unsupported control models include:

- Minimum on/off dwell.
- Continuous event localization.
- Generic rule grammar or PID.
- Native SWMM rule execution.

## Advance and accept a trial

1. Choose a positive interval ending no later than the next decision or terminal horizon.
2. Call `trial_to` with the required samples at a decision endpoint.
3. Supply no samples for a non-decision interval.
4. Consume `into_next()` to accept the complete candidate.

A non-decision interval changes accepted time only.

A decision endpoint retains both commands. The old command applies on the left of that endpoint, and the transition applies on its right.

Dropping or rejecting a trial preserves accepted state, timers, counters, and diagnostics. Acceptance covers the controller only.

## Read results

Normal advancement keeps derived state and commands in runtime tensors. These calls copy data into caller-owned values and can return errors:

| Request | Result |
| --- | --- |
| `commands()` | Requested device operations |
| `states()` | Latches and qualification state |
| `diagnostics()` | Retained control evidence |
| Snapshot value package | Complete values for persistence |

The final step retains immediate before/after tensors and original inputs. It retains no previous batch or recursive step chain.

## Save and restore

### Save a complete state

Trusted snapshots retain tensor state without copying derived arrays to the host. Restore checks original definition bits and retains the destination's bound structure context.

Caller-owned packages must preserve:

| Definitions and policy | Continuation |
| --- | --- |
| Complete actual structure and control definitions | Current state and accepted counters |
| Initial antecedents and future decisions | Bounded final diagnostics |
| Timing policy | Every original F64 bit, including signed zero |

### Check an imported package

Import checks schedule position, sample causality, timer chronology, transition-count parity, and complete retained receipts.

Numr recomputes the retained transition witnesses and immediate final trial. Import compares explicit receipt values with those results. These copies occur only during caller-owned import checks.

Rejected imports change no accepted value. Consistency does not authenticate historical observations.

Repeatable continuation requires the same future samples and declared build, runtime, and layout.

## Timing arithmetic

Controls retain F64 `Tensor<R>` state on their bound `StructureBatch<R>` client.

Runtime tensors retain latches, qualification predicates, timer anchors, transition counters, and receipts. Read [runtime admission](integration.md#runtime-selection-and-admission) before selecting a provider.

| Timing rule | Numerical requirement |
| --- | --- |
| Timer age | Use original timestamps, never repeated interval additions |
| Readiness | Compare TwoSum elapsed high/low components with the original delay |
| Lateness | Compare elapsed with the exact finite delay-plus-allowance expansion |
| Equal high components | Compare their signed low components |
| Used elapsed overflow | Reject before residual arithmetic |

An overflowing positive allowed threshold uses an explicit tag. It creates no nonfinite package scalar, and its residual arithmetic stays masked.

A finite elapsed high admits that branch through monotone rounding. This includes positive residuals near maximum finite values.

Schedule-gap checks cover initial, consecutive, terminal, and declared antecedent-to-first-future-observation spacing. They establish known spacing, not historical authenticity.

Counter and index tensors stay within exact F64 integer capacity. Accepted interval counters use checked integer metadata.

## Connect commands to devices

Only admitted actual position can supply `StructureOperatingPoint::GateOpening`. A requested `GateTarget` cannot substitute for that position.

Use [controlled gates](controlled-gates.md) or [controlled pumps](controlled-pumps.md) for the supported response models. Exclusive device authority must apply before consuming commands.

Hydraulic coupling owns common-time trials and paired acceptance. Standalone control continuation establishes no native actuator behavior, endogenous hydraulic feedback, or coupled accuracy.
