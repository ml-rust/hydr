# Advance actual gate opening

Use `GateActuatorBatch` to calculate opening under a declared ideal response. Supply directional rates and causal targets before advancing.

This model binds to existing hydrostatic rectangular gates. It retains complete state for continuation, but does not consume hydraulic water.

## Check that the model applies

The response uses constant, independently supplied opening and closing rates.

| Included | Excluded |
| --- | --- |
| Ideal directional motion within the declared gate range | Acceleration, inertia, backlash, and deadband |
| Explicit zero rates that stall one direction | Load-dependent rates and mechanical stops beyond the declared range |
| Numerical opening and arrival bounds | Native actuator parity, coefficient calibration, or field accuracy |

Continuous feedback remains outside this model.

## Construct the actuator batch

Call `StructureBatch::with_gate_actuators` with definitions, initial states, and a finite target schedule.

Each actuator binds one unique gate. The nonempty subset follows the complete original structure order.

| Input | Requirement |
| --- | --- |
| `GateActuatorDefinition` | Unique actuator/structure identities, rates, numerical policy, rate provenance, and ideal-response applicability |
| Directional rates | Explicit finite nonnegative metres/second. Zero stalls that direction |
| `GateActuatorInitialState` | Actual opening and target inside the bound gate range |
| `GateActuatorSchedule` | Finite nonnegative initial time strictly before the finite terminal horizon |
| Target calendar | Original finite times, strictly increasing within `(initial_time_s,end_s]`. An empty calendar retains the initial target |
| `GateActuatorAccuracy` | Explicit finite nonnegative opening and arrival-window allowances |

There are no defaults for rates, position, target, numerical policy, or applicability.

Non-gates and duplicate authority reject. Matching IDs cannot substitute changed geometry, applicability, or structure context.

## Supply targets and advance

`next_event` selects the next strictly later target, arrival boundary, or terminal event.

1. Read `next_event` and choose an endpoint no later than that event.
2. At an original target event, supply one complete ordered `GateTargetUpdate` per actuator.
3. Set `issued_at_s` to the exact event time.
4. Supply finite `available_at_s` no later than `issued_at_s`. Negative availability times remain valid.
5. Call `trial_to` for a positive represented interval inside the declared support.
6. Consume `into_next()` to accept the complete candidate.

Missing packets, stale issue times, future availability, or packets outside target events reject. Hydr supplies no implicit hold or replacement.

### Endpoint ordering

| At the endpoint | Result |
| --- | --- |
| Old target | Determines the left state |
| Changed target | Applies on the right interval |
| Coincident completion | Completes the old episode before installing the new target |
| Numerically repeated target | Retains the motion anchor and does not restart motion |
| Terminal horizon | Returns no next event and grants no extrapolation or implicit hold |

Exactly coincident represented event times combine flags. Different original times never merge through an epsilon.

Scalar reductions select event time and flags. Requests for affected identities copy those values explicitly.

Dropping or rejecting a trial preserves positions, anchors, targets, counters, certificates, and time. Acceptance covers actuator state only.

## Read opening and arrival status

`outputs()`, `states()`, `last_step()`, and trial diagnostics explicitly copy values and can return errors.

Ordinary successful advancement keeps derived position/state tensors on the selected runtime without copying full arrays.

### Opening support

Every position satisfies finite `0 <= lower <= opening <= upper <= gate_maximum`.

| Support | Meaning |
| --- | --- |
| `ExactClosed` | Canonical zero center, singleton zero interval, and zero error |
| `CertainlyPositive` | Strictly positive lower bound |
| `ClosureUnresolved` | Bounds include both zero and positive opening |

An upper bound above zero with a zero center rejects, regardless of the accuracy allowance. Rounded center equality never proves arrival or closure.

An exact target singleton retains original active target bits for center and endpoints, with zero error.

### Arrival phases

The arrival window covers every position admitted by the anchor bounds. It retains prior uncertainty and is not truncated at the horizon.

| Phase | Meaning |
| --- | --- |
| `BeforeWindow` | Accepted time precedes the certified earliest boundary |
| `InsideWindow` | Accepted time lies inside the unresolved finite window |
| `Complete` | A retained analytic upper or forward target singleton proves all-position completion |
| `BeyondHorizon` | A valid lower certificate proves no arrival within remaining support |
| `Stalled` | Every admitted position lies strictly on the same zero-rate side |
| `UnresolvedWithStalledSupport` | Some positions cannot reach the target, while others can arrive or already equal it |

Public clocks remain finite. Missing endpoints use optional values or status tags, never infinite event times.

A finite upper beyond terminal support remains a censored certificate. It neither extends support nor marks terminal completion.

An earliest bound overlapping support with an unavailable upper rejects numerically. It cannot become an unsupported beyond-horizon claim.

## Save and restore

### Choose a snapshot form

| Form | What it retains |
| --- | --- |
| Trusted snapshot | Complete persistent tensor state, without derived array copies |
| Caller-owned package | Definitions, original inputs, calendar, policies, accepted state, counters, and bounded final receipts |

Trusted restore checks the full bound context. Caller-owned encoding must preserve every original F64 bit, including signed zeros and repeated raw target packets.

### What import checks

Import recomputes arrival and forward proofs before accepting the package.

| Evidence | Import check |
| --- | --- |
| Position | Canonical support and effective completion bounds |
| Targets | Generations and retained causal receipts |
| Previous completion | One bounded receipt preserves old-target authority across later target changes |

Invalid imports leave accepted state unchanged. Receipts contain no recursive history chain.

Consistency checks do not authenticate history or replay arbitrary command history. Equivalent continuation requires compatible build, runtime, and layout with the same future causal targets inside support.

## Numerical response contract

### Motion and anchors

For anchor position `x`, target `g`, and original elapsed time `h`, the ideal response is:

```text
x < g: min(g, x + opening_rate*h)
x > g: max(g, x - closing_rate*h)
x = g: g
```

An anchor stores a represented center and certified lower/upper opening bounds. Ordinary substeps keep that original anchor.

Hydr never repeatedly accumulates timestep displacements or resets earlier uncertainty. A changed target carries the entire admitted left-position interval into its new anchor.

Numerically equal targets, including differently signed zeros, retain:

- Active target bits and anchor.
- Episode and arrival certificate.
- Counters.

The latest raw packet retains its own original bits and causal metadata separately.

### Opening and timing allowances

Opening admission uses an absolute allowance **or** a relative allowance. The allowances never add or change physical rates and support.

| Policy | Admission |
| --- | --- |
| `maximum_absolute_opening_error_m` | Finite error bound lies within the metre allowance |
| `maximum_relative_opening_error` | If absolute admission does not apply, positive opening needs an outward relative bound within this allowance |
| Both opening allowances zero | Only zero-error receipts pass |
| `maximum_arrival_window_s` | The entire certified arrival window fits the independent timing allowance |

Arrival-window width uses its exact TwoSum expansion, including the signed low component. Rounded width alone cannot establish admission.

Neither policy repairs an invalid center, false closure, or unsupported completion.

The absolute opening error covers both distances from center to interval endpoints. The physical capped response maps endpoint bounds into range. An unexplained center clamp cannot repair containment.

### Required arithmetic

State uses F64 `Tensor<R>` on the bound `StructureBatch<R>` client. The CPU reference explicitly selects one thread.

Construction requires:

- The basic arithmetic contract.
- `FusedMulAddContract::Ieee754Fused`.
- Nearest-even rounding with gradual underflow.

Keep that arithmetic environment during continuation and import. Runtime availability alone is insufficient. No host or unfused fallback substitutes for the provider contract.

TwoSum retains original elapsed time as high `h` and signed low `l`. For signed rate `k` and anchor operand `x`:

```text
q0 = fused_mul_add(k,h,x)
if l != 0:
    q1 = fused_mul_add(k,l,q0)
```

The second stage includes the signed elapsed low and adds uncertainty outward to the first-stage allowance. Exact zero low skips that operation and allowance.

A separate `rate*abs(l)` charge never duplicates the evaluated correction. Unused standalone products can overflow without rejecting a finite fused result.

Used nonfinite results or TwoSum residuals reject.

### Exact zero and completion

Fused zero does not alone prove mathematical zero. Local zero-grid certificates or checked exact-binary witnesses can prove exact local results.

Uncertified zero retains its rounding allowance. Missing optional witness arithmetic does not invalidate an otherwise admitted fused path.

Arrival bounds use the entire anchor interval, original rates, target, and anchor epoch.

| Completion proof | Result |
| --- | --- |
| Retained analytic upper | Set canonical target state without unused displacement arithmetic |
| Earlier forward endpoint singleton | Retain the original lower and tighten only the effective upper |

Both paths retain one bounded completion receipt. They increment arrival count once per accepted active episode.

Initial holds, repeated targets, discarded trials, and repeated proofs add no arrivals.

## Checks and reference output

The CPU F64 reference covers 16 original-bit histories, two supplemental closure families, and two separate policy companions.

Six layouts cover scalar, vector, and tail widths. Exact rational replay uses no enclosure slack.

[Runtime checks](../tests/runtime_backends/gate_actuators.rs) run analytical opening, reversal, and nonbinary closure on CPU and CUDA. Both runtimes check discarded trials and complete restart at batch widths 1 and 33. These cases check runtime execution separately from the larger CPU reference below.

| Checked quantity | Coverage |
| --- | --- |
| Physical response | Position enclosures and fused-stage errors |
| Timing | Complete arrival windows and causal targets |
| Continuation | Retained receipts |
| Motion histories | Asymmetric reversals, repeated targets, stalls, and ordinary nonbinary closure |
| Arithmetic cases | Subnormal support, large clocks, and signed-low overflow cancellation |

| Observation | Passing rows |
| --- | ---: |
| Initial states | 22,826 |
| Accepted intervals | 35,934 |
| Rejected trials preserving state | 1,130 |
| Successful requested results | 21,696 |
| Required numerical rejections | 1,808 |

All 83,394 rows pass, including 23,504 requested results.

Stable Rust and Rust 1.95 emit identical reference bytes. Thirty-five integration tests cover admission, events, continuation, and forged imports. A separate tensor test checks exact counter capacity.

Run the original-bit CSV observer with stdout redirected to a file. Output is approximately 381 MB.

```sh
cargo run --quiet --example gate_actuators_reference > gate-actuators.csv
```

These checks establish behavior for the declared ideal model. They establish no mechanical calibration, native parity, or measured-event accuracy.

## Connect to controls and hydraulic laws

[Sampled controls](operating-controls.md) generate targets. This actuator supplies the separately admitted ideal actual response.

The [source registry](operating-sources.md) resolves manual, prescribed, and actual-actuated authority before [structure proposals](structures.md).

Only certified actual closure grants the waiver for unknown heads. Native SWMM actuator behavior, continuous feedback, accepted hydraulic consumption, and coupled accuracy remain unsupported.
