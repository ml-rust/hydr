# Apply rain and soil losses to surface flow

Use land sources to advance surface water and soil memory together. Start with an explicit land model and antecedent state.

A rejected or shortened trial restarts from the last accepted flow, soil, and ledgers. The engine accepts these states together.

## Construct the surface solver

1. Declare the land mechanism for every grid cell.
2. Supply soil state matching each land mechanism.
3. Supply the configured Numr client, setup, and initial state.

`Solver::<R>::new(client, setup, initial)` receives these aggregate values:

| Value | Required fields |
| --- | --- |
| `SolverSetup` | `domain`, `land`, `boundaries`, `forcing`, `config` |
| `InitialState` | `time`, `flow`, `soil` |

Both types live under `hydr::solver`. Land and antecedent types live under `hydr::soil`.

### Choose land and initial soil state

| API | Meaning |
| --- | --- |
| `LandCell::Impervious` | No soil infiltration in that cell |
| `LandCell::ModifiedGreenAmpt(parameters)` | Native Modified Green–Ampt infiltration |
| `LandModel::new` | Check complete grid support |
| `LandModel::impervious` | Explicitly select no soil infiltration |
| `SoilState::new` | Check supplied state without inferring history |
| `SoilState::initialized` | Explicitly select original EPA initialization |

Each `SoilCellState` must match its land variant. Pervious cells carry complete `NativeSoilState`.

Inactive hydraulic cells require impervious bookkeeping and zero forcing. Their hydraulic output remains `Unsupported`.

## Supply forcing and preserve process order

`ForcingInterval.potential_evaporation` requires an explicit full-grid nonnegative array in m/s. Supply an explicit zero array for zero evaporation. Rain and withdrawal remain separate physical inputs.

The engine applies sources in this order:

1. Evaluate Modified Green–Ampt using interval-start ponding and incoming rain once.
2. Add rain with zero horizontal momentum.
3. Remove returned native infiltration, with an admitted numerical correction when required.
4. Remove availability-limited potential evaporation.
5. Remove prescribed withdrawal, rejecting unmet demand.
6. Calculate hydraulic transport and its interval bounds.
7. Check state and accounting, then accept everything together.

Every sink removes proportional donor momentum. Rain and numerical correction introduce zero horizontal momentum.

The engine keeps native event memory separate from lifetime infiltration. Read [the soil contract](soil.md) for native branches and complete state.

## Read accepted results

Numerical continuation uses cached tensors. It does not copy native state to the host during trials.

| API or record | What you can inspect |
| --- | --- |
| `Solver::soil_state()` | Copy complete accepted native memory. The call returns an error if copying fails. |
| `WaterLedger` | Infiltration, evaporation, prescribed withdrawal, signed water adjustments, and conservative absolute adjustment consumption |
| `StepDiagnostics.sources` | One ordered receipt per full-grid cell, copied only on explicit request |

Each source receipt records original ponding and rain, then infiltration, evaporation, withdrawal, and correction depths.

Pervious receipts also retain evidence for checking arithmetic:

- Native rate and native available/infiltration depths.
- Native excess above available supply.
- Separate native, bridge, and total bounds.

Source corrections remain separate from momentum-film resets. Film resets remove momentum without removing water.

## Handle rejected intervals

| Rejection | Engine behavior |
| --- | --- |
| Timestep-dependent numerical source or budget error | Retry within configured limits |
| Invalid static support | Stop with an error |
| Runtime operation error | Stop with an error |

`max_retries=0` permits the initial trial and no retries. The stored event-memory values have finite numerical spacing. A smaller interval does not guarantee admission.

Rejected trials preserve accepted surface water, soil, counters, and diagnostics. Typed soil errors identify the corresponding global hydraulic cell.

## Understand numerical water corrections

Native interval arithmetic can reconstruct infiltration above available supply. SI conversion can introduce a separate discrepancy. The engine keeps separate native and conversion bounds.

The engine retains the soil component's returned rate and advanced state. It never caps native infiltration afterward.

### Required correction checks

An outward bound rounds uncertainty upward to avoid understating it.

Every condition below must pass before the engine admits an excess:

- Native excess fits the selected branch's outward rounding envelope.
- SI excess fits the combined outward envelope.
- Adding the represented correction to available SI water reproduces infiltration exactly.
- Positive correction times area stays positive, finite, and bounded outward.
- Step and cumulative conservative adjustment charges fit the configured budget.

An admitted correction explicitly creates numerical water. The engine then removes full infiltration and sets post-infiltration depth and momentum to zero.

When infiltration fits available water, the engine retains positive residual water.

### Independent adjustment budget

`SolverConfig.max_cumulative_water_adjustment_m3` defaults to `1e-9 m³`. The value must be finite and nonnegative.

| Charge | Accumulation rule |
| --- | --- |
| Zero correction | Preserve the cumulative charge exactly |
| Positive correction | Accumulate outward, including additions too small for ordinary signed totals |

This budget is independent of water-residual tolerances. Conservation balance alone cannot authorize a correction. Failed proof, reconciliation, representation, or budget checks reject the trial.

The runtime caches a bound for full-grid correction sums, including zero entries. The conservative charge covers product, reduction, and cumulative addition rounding separately from reported signed totals.

### Required arithmetic

The calculation requires round-to-nearest F64 arithmetic and gradual underflow. Flush-to-zero execution is unsupported.

| Quantity | Required representation |
| --- | --- |
| Initial/accepted storage and actual source volumes | Positive per-cell products remain represented |
| Rain and withdrawal increments | No silent positive underflow |
| Potential evaporation increments | Finite values |
| Wet post-soil donor | Positive potential underflow rejects |
| Dry donor | Zero actual evaporation is permitted |

## Save and restore

1. Retain a complete snapshot after acceptance.
2. Preserve definitions, floating-point bits, runtime/build conditions, and the accepted-step partition.
3. Import into a candidate before replacing accepted state.

Snapshots include immutable land, forcing, reference identity, complete native soil state, and adjustment budgets.

Import checks the following separately:

- Source receipt algebra and potential evaporation priority.
- Native/SI evidence and independent sink totals.
- Charge bounds and extended residuals.

Final native state establishes continuation support and consistency. It cannot authenticate historical infiltration or recovery.

Forcing discontinuities split intervals before evaluation. Exact continuation requires the same accepted-step partition.

## Current limits

| Area | Excluded from this surface-source capability |
| --- | --- |
| Soil physics | Groundwater coupling, soil evaporation, and finite subsurface reservoirs |
| Hydraulic connections | Structures and networks |
| Accuracy | Measured-event accuracy guarantees |
