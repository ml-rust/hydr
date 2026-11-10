# Atomic land sources

The engine advances surface flow and explicit soil memory together. Every shortened trial restarts from the last accepted flow, soil, and ledgers.

## Setup and antecedent state

`Solver::<R>::new(client, setup, initial)` receives a configured Numr client and two aggregate values:

| Value | Fields |
| --- | --- |
| `SolverSetup` | `domain`, `land`, `boundaries`, `forcing`, `config` |
| `InitialState` | `time`, `flow`, `soil` |

Both types live under `hydr::solver`. Land and antecedent types live under `hydr::soil`.

`LandCell` selects `Impervious` or `ModifiedGreenAmpt(parameters)` per cell. `LandModel::new` checks full-grid support. `LandModel::impervious` explicitly selects no soil infiltration.

`SoilCellState` matches the land variant. Pervious cells carry complete `NativeSoilState`. `SoilState::new` checks correspondence without inferring history. `SoilState::initialized` explicitly selects original EPA initialization.

Inactive hydraulic cells require impervious bookkeeping and zero forcing. Their hydraulic output remains `Unsupported`.

## Source order and momentum

1. Evaluate Modified Green–Ampt using interval-start ponding and incoming rain once.
2. Add rain with zero horizontal momentum.
3. Remove returned native infiltration, with an explicitly admitted numerical correction when required.
4. Remove availability-limited potential evaporation.
5. Remove prescribed withdrawal, rejecting unmet demand.
6. Calculate hydraulic transport and its interval bounds.
7. Check all state and accounting, then accept together.

`ForcingInterval.potential_evaporation` supplies an explicit full-grid nonnegative m/s array. Zero evaporation requires an explicit zero array. Rain and withdrawal remain separate physical inputs.

Every sink removes proportional donor momentum. Rain and numerical correction introduce zero horizontal momentum. Sources never reinterpret native event memory as a lifetime infiltration ledger.

## Arithmetic admission

The [standalone soil contract](soil.md) preserves EPA Modified Green–Ampt branches and complete native state. Surface integration retains its returned rate and advanced state. It never caps native infiltration afterward.

Native interval reconstruction can exceed native available supply through arithmetic rounding. SI conversion can add a separate discrepancy. The engine retains separate native and bridge envelopes.

An excess requires all these gates:

- Native excess fits the selected branch's outward rounding envelope.
- SI excess fits the combined outward envelope.
- Adding the represented correction to SI available water reproduces infiltration exactly.
- Positive correction times area remains positive, finite, and bounded outward.
- Step and cumulative conservative adjustment charges fit the configured budget.

An admitted correction creates numerical water explicitly. The engine removes full infiltration and sets post-infiltration depth and momentum to zero. It retains positive residual water when infiltration fits available water.

`SolverConfig.max_cumulative_water_adjustment_m3` defaults to `1e-9 m³`. It is finite and nonnegative. Zero corrections preserve the cumulative charge exactly. Positive charges use outward accumulation, including additions too small for ordinary signed totals.

The budget is independent of water-residual tolerances. Conservation balance alone cannot authorize a correction. Failed proof, reconciliation, representation, or budget checks reject the trial.

This admission requires round-to-nearest F64 arithmetic and gradual underflow. Flush-to-zero execution is unsupported. Initial/accepted storage and actual source volumes require representable positive per-cell products. Rain and withdrawal increments cannot silently underflow. Potential evaporation increments must remain finite. Positive potential underflow rejects on a wet post-soil donor. A dry donor permits zero actual evaporation.

The configured runtime reduction cache bounds full-grid correction-volume sums. Its count includes zero entries. The conservative charge covers product, reduction, and cumulative addition rounding separately from reported signed totals.

## Atomic retries and diagnostics

Timestep-dependent numerical source and budget errors use bounded retries. Invalid static support and runtime operation errors remain fatal. `max_retries=0` permits the initial trial and no retries. Smaller intervals cannot guarantee admission against fixed event-memory spacing.

Rejected trials preserve all accepted surface, soil, counters, and diagnostics. Typed soil errors map compact pervious lanes back to global hydraulic cell identities. Successful continuation uses cached tensors without native-state projection during trials.

`Solver::soil_state()` fallibly projects accepted complete native memory. `WaterLedger` independently records infiltration, evaporation, prescribed withdrawal, signed water adjustments, and conservative absolute adjustment consumption.

Explicit diagnostic projection produces `StepDiagnostics.sources` with one ordered receipt per full-grid cell. It records original ponding and rain, infiltration, evaporation, withdrawal, and correction depths. Pervious receipts also retain native rate, native available/infiltration depths, native excess, and separate native/bridge/total bounds.

Source corrections remain separate from momentum-film resets. Film resets remove momentum without removing water.

## Restart consistency and limits

Snapshots preserve immutable land, forcing, reference identity, complete accepted native soil state, and adjustment budgets. Import constructs candidate tensors before replacing accepted state.

Import checks source receipt algebra, potential evaporation priority, native/SI evidence, independent sink totals, charge bounds, and extended residuals. Final native state proves continuation support and consistency. It cannot authenticate historical infiltration or recovery.

Exact continuation requires identical definitions, floating-point bits, build/runtime envelope, and accepted-step partition. Forcing discontinuities split intervals before evaluation.

This capability excludes groundwater coupling, soil evaporation, finite subsurface reservoirs, structures, networks, and measured-event accuracy guarantees.
