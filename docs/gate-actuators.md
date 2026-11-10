# Ideal constant-rate gate actuators

`GateActuatorBatch` models actual opening under an explicitly declared ideal response with constant, independently supplied opening and closing rates. The model binds to existing hydrostatic rectangular gates. It consumes causal targets and retains complete immutable continuation.

This model excludes acceleration, inertia, backlash, deadband, mechanical stops beyond the declared range, and load-dependent rate changes. It supplies no native actuator parity, coefficient calibration, field accuracy, continuous feedback, or hydraulic consumption claim.

## Definitions and applicability

`StructureBatch::with_gate_actuators` accepts explicit definitions, initial states, and a finite target schedule. Every actuator binds one unique gate inside a nonempty subset ordered by the complete original structure context. Matching IDs alone never substitute different geometry, applicability, or context. Non-gates and duplicate authority reject.

| Input | Contract |
| --- | --- |
| `GateActuatorDefinition` | Unique actuator/structure identities, directional rates, numerical policy, rate provenance, and ideal-response applicability |
| Directional rates | Explicit finite nonnegative metres/second. Zero stalls motion in that direction |
| `GateActuatorInitialState` | Explicit actual opening and target inside the bound gate range |
| `GateActuatorSchedule` | Finite nonnegative initial time strictly before the finite terminal horizon |
| Target calendar | Original finite times strictly increasing within `(initial_time_s,end_s]`. An empty calendar retains the initial target |
| `GateActuatorAccuracy` | Explicit finite nonnegative opening and arrival-window allowances |

No rate, initial position, target, numerical policy, or applicability default exists.

## Ideal response and anchors

For an exact anchor position `x`, target `g`, and original elapsed time `h`, the declared response is:

```text
x < g: min(g, x + opening_rate*h)
x > g: max(g, x - closing_rate*h)
x = g: g
```

Each accepted anchor retains a represented center and a certified lower/upper opening interval. Ordinary accepted substeps retain that original anchor. They never repeatedly accumulate timestep displacements or reset prior uncertainty. An actual changed target propagates the complete admitted left-position interval into its new anchor.

Repeated numerically equal targets preserve the active target bits, anchor, episode, arrival certificate, and counters. This equality includes differently signed zeros. The latest raw packet retains its original bits and causal metadata separately.

## Explicit numerical policies

Opening accuracy uses `maximum_absolute_opening_error_m` OR `maximum_relative_opening_error`. The absolute branch admits a finite error bound within the explicit metre allowance. Otherwise, positive opening requires an outward relative bound within the explicit relative allowance. The allowances never add. Both-zero opening policies admit only zero error receipts. A numerical allowance never changes the physical rate or support.

`maximum_arrival_window_s` independently bounds the complete certified arrival window. The comparison uses its exact TwoSum width expansion, including the signed low. Rounded width alone never establishes admission. Position and timing policies remain independent. Neither allowance repairs an invalid center, false closure, or unsupported completion.

## Strict fused execution

The runtime retains state on an explicitly configured one-thread numr CPU F64 client. Construction requires the basic arithmetic contract and `FusedMulAddContract::Ieee754Fused` under nearest-even rounding with gradual underflow. Callers preserve that execution envelope throughout continuation and import. No host or unfused fallback substitutes for the provider contract.

Numr TwoSum retains the exact original elapsed difference as a high component `h` and signed low component `l`. For signed directional rate `k` and anchor operand `x`, the forward path uses:

```text
q0 = fused_mul_add(k,h,x)
if l != 0:
    q1 = fused_mul_add(k,l,q0)
```

The second stage retains the signed elapsed low. Its uncertainty adds outward to the first-stage allowance. Exact zero low skips the second operation and allowance. A separate `rate*abs(l)` charge never duplicates the evaluated correction. Unused standalone products can overflow without rejecting a finite fused result. Used nonfinite results or TwoSum residuals reject.

Fused zero alone never proves mathematical zero. Local zero-grid certificates or checked exact-binary witnesses can establish exact local results. Uncertified zero retains its rounding allowance. Unavailable optional witness arithmetic never invalidates an otherwise admitted fused path.

## Position support and closure

Every admitted position satisfies `0 <= lower <= opening <= upper <= gate_maximum` with finite scalars. The reported absolute error covers both distances from the center to its interval endpoints. The physical capped response maps certified endpoint bounds into the physical range. No unexplained center clamp repairs containment.

| Support | Meaning |
| --- | --- |
| `ExactClosed` | Canonical zero center, zero singleton interval, and zero error |
| `CertainlyPositive` | Strictly positive lower opening bound |
| `ClosureUnresolved` | The interval includes both zero and positive support |

If the upper bound remains positive, represented center zero rejects regardless accuracy allowance. An exact target singleton uses original active target bits for center and endpoints, with zero error. Rounded center equality never proves arrival or closure. Actual hydraulic isolation still requires a separate exclusive operating-source adapter.

## Arrival windows and completion

Arrival bounds derive from the complete anchor interval, original directional rates, target, and anchor epoch. Bounds never discard prior uncertainty or truncate their width at the terminal horizon. Public clocks remain finite. Unavailable endpoints use explicit optional values or status tags instead of infinite event times.

| Phase | Contract |
| --- | --- |
| `BeforeWindow` | Accepted time precedes the certified earliest boundary |
| `InsideWindow` | Accepted time lies inside the unresolved finite arrival window |
| `Complete` | A retained analytic upper or forward target singleton proves all-position completion |
| `BeyondHorizon` | A valid lower certificate proves no arrival within remaining support |
| `Stalled` | Every admitted position lies strictly on the same zero-rate side |
| `UnresolvedWithStalledSupport` | Some admitted positions cannot reach the target while others can arrive or already equal it |

A finite upper beyond terminal support remains a censored certificate. It never extends support or marks terminal completion. An overlapping earliest bound with an unavailable upper rejects numerically. It never becomes an unsupported beyond-horizon claim.

At the analytic upper, canonical target state follows from the retained completion proof without unused displacement arithmetic. A forward endpoint singleton can prove completion earlier. It preserves the original lower and tightens only the effective upper. Both paths retain one bounded completion receipt and increment arrival count once per accepted active episode. Initial holds, repeated targets, discarded trials, and repeated proofs add no arrivals.

## Causal targets and events

Every original target event requires one complete ordered `GateTargetUpdate` per actuator. `issued_at_s` equals that exact event time. Finite `available_at_s` cannot exceed issue time. Negative availability timestamps remain valid. Missing packets, stale issue times, future availability, and packets outside target events reject without implicit holds or replacements.

`next_event` reports the next strictly later target, arrival boundary, or terminal event. Exactly coincident represented times combine flags. Different original times never merge through an epsilon. Scalar reductions select event time/flags. Dynamic affected identities appear only in explicit projections.

`trial_to` requires a positive represented interval inside support and cannot cross the next event. The old target determines the endpoint's left state. A changed endpoint target applies to the right interval. Coincident completion finishes the old episode before installing the new target. A repeated target never restarts motion. Terminal state returns no next event. No extrapolation or implicit terminal hold extends the declared horizon.

## Trials, projections, and restoration

Trials borrow accepted state immutably. Dropping or rejecting a trial preserves positions, anchors, targets, counters, certificates, and time. `into_next()` accepts the complete candidate by ownership. This acceptance establishes no hydraulic state or shared-transfer acceptance. `outputs()`, `states()`, `last_step()`, and trial diagnostics explicitly project positions, support, phases, and retained numerical evidence. Successful ordinary advancement retains derived position/state tensors without array readback.

Trusted snapshots clone complete persistent state and check the full bound context before restoration. Caller-owned packages retain definitions, original inputs, calendar, policies, accepted state, counters, and bounded final receipts. Preserve every original F64 bit during encoding, including signed zeros and raw repeated-target packets.

Owned import recomputes current arrival/forward proofs, effective completion bounds, canonical support, target generations, and retained causal receipts. One retained previous-completion receipt preserves old-target authority across later target changes. Receipts contain no recursive history chain. Invalid imports change no accepted state. Consistency checks establish no historical authenticity or arbitrary command-history replay. Equivalent continuation requires compatible build/runtime/layout and the same future causal targets within the declared support.

## Numerical checks

The CPU F64 reference checks 16 original-bit physical histories, two supplemental closure families, and two separate policy companions. Six layouts cover scalar, vector, and tail widths. Exact rational replay checks physical position enclosures, fused-stage errors, complete arrival windows, causal targets, and retained receipts without enclosure slack.

| Observations | Passing rows |
| --- | ---: |
| Initial states | 22,826 |
| Accepted intervals | 35,934 |
| Rejected trials with unchanged state | 1,130 |
| Successful requested results | 21,696 |
| Required numerical rejections | 1,808 |

All 83,394 rows pass, including all 23,504 requested results. Cases include asymmetric reversals, repeated targets, stalls, ordinary nonbinary closure, subnormal support, large clocks, and signed-low overflow cancellation. Stable Rust and Rust 1.95 emit identical reference bytes.

Thirty-five integration tests check admission, events, continuation, and forged imports. A separate tensor-kernel test checks exact counter capacity. These checks establish numerical behavior for the declared ideal model. They establish no mechanical calibration, native actuator parity, or measured-event accuracy.

Run `cargo run --quiet --example gate_actuators_reference` for the original-bit CSV observations. The observer emits approximately 381 MB. Redirect stdout to a file.

## Integration boundary

[Sampled controls](operating-controls.md) emit requested targets. This actuator supplies its separately admitted ideal actual response. An exclusive operating-source adapter must resolve manual, prescribed, and actuator authority before feeding [structure proposals](structures.md). No actuator output alone grants unknown-head isolation, native SWMM actuator behavior, continuous feedback, hydraulic consumption, or coupled accuracy.
