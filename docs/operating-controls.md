# Sampled operating controls

`OperatingControlBatch` emits deterministic sampled device commands with explicit hysteresis, qualification delays, and complete transactional continuation.

Commands describe requested operations. `GateTarget` never asserts an actual gate opening. Pump commands establish no response dynamics, flow, donor availability, or hydraulic transfer.

## Definitions and applicability

| Input | Contract |
| --- | --- |
| Controlled devices | Nonempty unique subset in original structure order, bound to the complete actual `StructureBatch` context |
| Signal | Explicit controller, signal, support, and common-datum identities |
| Thresholds | Finite lower stage strictly below finite upper stage |
| Delays | Explicit finite nonnegative high/low sampled qualification durations |
| Commands | Explicit low/high mappings admitted by the actual pump speed policy or gate opening range |
| Initial state | Explicit latch and complete supplied antecedent memory, or an explicit fresh-memory assumption |
| Schedule | Original finite strictly increasing decisions inside a finite nonnegative support horizon |
| Timing | Explicit maximum decision gap, maximum timer lateness, and sampled applicability evidence |

External signal support need not equal either structure endpoint. The caller supplies causal stage samples at the exact decision time. Availability must precede or equal observation. Finite negative availability and antecedent times remain valid.

Every decision requires one ordered sample per controller. Unknown, dry, stale, future, and incompatible sample support receives no implicit replacement or last-value hold.

## Sampled qualification

| Current latch | Qualifying sample | Result |
| --- | --- | --- |
| Low | Stage at or above the upper threshold | Start or continue high qualification |
| High | Stage at or below the lower threshold | Start or continue low qualification |
| Either | Opposing predicate is false | Retain latch and cancel incomplete qualification |
| Either | Qualification reaches its explicit delay within lateness allowance | Switch once and clear pending state |

Zero delay switches at the qualifying decision. The hysteresis band retains the latch and cancels incomplete opposing qualification. All controllers read the same original pre-decision sample batch and update simultaneously.

Qualification establishes persistence across the supplied samples only. It establishes no continuous threshold history or absence of between-sample excursions. No minimum on/off dwell, continuous event localization, generic rule grammar, PID, or native SWMM rule execution applies.

## Numerical timing

Numr CPU F64 tensors retain the relay state, predicates, timer anchors, transition counters, and receipts. Original timestamps anchor timers. Repeated interval additions never accumulate timer age.

TwoSum retains exact finite elapsed high and low components. Readiness compares that expansion against the original delay. Lateness compares elapsed against the exact finite delay-plus-allowance expansion. Equal highs compare through their lows. Used elapsed overflow rejects numerically before residual arithmetic.

A positive overflowing allowed threshold uses an explicit overflow tag and no nonfinite DTO scalar. Its residual arithmetic remains masked. A finite elapsed high admits this branch through monotone rounding, including positive residuals near maximum finite values.

Exact schedule-gap checks cover initial, consecutive, terminal, and declared antecedent-to-first-future-observation spacing. These checks establish known spacing, never historical authenticity. Counter and index tensors remain within exact F64 integer capacity. Accepted interval counters use checked integer metadata.

## Trials and projections

`trial_to` requires a positive original interval ending no later than the next decision or terminal horizon. A non-decision interval requires no samples and changes accepted time only. A decision endpoint retains both left and right commands. The endpoint transition applies to the right interval.

Dropping a trial preserves accepted state, timers, counters, and retained diagnostics. `into_next()` accepts the complete candidate by ownership. This component does not atomically accept hydraulic state or conserved transfers.

Normal advancement retains derived state and commands as tensors. `commands()`, `states()`, `diagnostics()`, and snapshot value packages are explicit projections. Retained last-step storage contains immediate before/after tensors and original inputs. It contains no previous batch or recursive step chain.

## Complete restoration

Trusted snapshots clone complete tensor continuation without projection. Restore checks complete original definition bits and retains the destination's immutable bound structure context.

Caller-owned snapshot packages retain:

- Complete actual structure definitions.
- Control definitions.
- Initial antecedents.
- Future decision schedule.
- Timing policy.
- Accepted counters.
- Current state.
- Bounded final diagnostics.

Preserve every F64 bit when encoding, including signed zero.

Import checks current schedule position, sample causality, timer chronology, transition-count parity, and complete retained receipts. Numr recomputes retained transition witnesses and the immediate final trial. Explicit receipt projections compare the imported package against these recomputed values. Those projections occur only during caller-owned import checks.

A rejected import changes no accepted value. Receipt consistency establishes no historical authenticity. Equivalent continuation requires the same future supplied samples and declared build/runtime/layout envelope.

## Integration boundary

Actuator response remains separate from command generation. Only admitted actual gate position can feed `StructureOperatingPoint::GateOpening`. The controller supplies no native SWMM control grammar or native actuator behavior. A later adapter must establish exclusive device authority before consuming commands.

Hydraulic coupling owns common-time trials and paired acceptance. Standalone sampled command continuation establishes no endogenous hydraulic feedback or coupled accuracy claim.
