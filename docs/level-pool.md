# Store water in level pools

Use `hydr::storage::LevelPoolBatch` to advance supplied pool volumes and read checked water levels. Define the storage curve and physical ownership before constructing the batch.

This component represents connected, approximately level water. Disconnected depressions require separate elements. It supplies no outlet law or hydraulic attachment.

For accepted transfers through structures, use [coupled level pools](coupled-level-pools.md).

## Construct a pool

1. Create ordered `LevelPoolKnot` values for volume and stage.
2. Construct `LevelPoolCurve::new` with a vertical reference and curve provenance.
3. Supply a unique element ID and storage-support identity.
4. Declare physical evidence in `connected_pool_assertion`.
5. Supply an explicit initial volume through `LevelPoolState`.
6. Construct the batch with your configured Numr client.

Volumes use m³. Stages use metres. Time uses seconds.

### Check the curve

- Knots: Volumes and stages must be finite and strictly increasing.
- Origin: The first volume equals zero.
- Segments: Adjacent volume differences, stage differences, and secant areas must remain positive and finite.
- Range: Queries must stay within the closed volume range. Extrapolation is unsupported.

A secant area is the volume change divided by the stage change between adjacent knots.

`LevelPoolCurve` uses existing Solvr Linear interpolation for forward and inverse queries. Exact knot queries retain supplied stages, including the maximum-volume endpoint.

### Check physical ownership

The storage-support identity establishes no geometric proof of separation from hydraulic cells. Exclude overlapping storage explicitly before any attachment.

Your application supplies the evidence for a connected, approximately level pool. Hydr performs no datum transformation.

### Interpret a dry or shallow pool

| Result | Meaning |
| --- | --- |
| `stage_m=None` at zero volume | The pool is dry. The first curve elevation remains an internal origin. |
| Positive volume with a checked stage | The stage lies above the origin and inverse interpolation reconstructs positive volume. |
| Unresolved wet height | Large datum offsets or extreme slopes prevent admission. Hydr invents no positive film. |

Authoritative volume remains the state. Inverse interpolation never replaces it.

A positive volume change can retain an unchanged rounded stage when inverse admission still passes.

## Choose the water inputs

`LevelPoolInput` supplies original incoming and outgoing volumes for the requested interval. There is no silent empty initialization.

| Calculation | Construction and trial |
| --- | --- |
| Exchanges only | Use the ordinary constructor and `trial_to(end, inputs)`. |
| Local sources and exchanges | Use `new_with_sources` and `trial_with_sources_to`. |

### Enable local rain and losses

1. Use `new_with_sources` with the original construction arguments.
2. Add one ordered `LevelPoolSourcePolicy` per element.
3. Declare the first segment's rain-receiving basal footprint, including dry storage.
4. Supply separate provenance for potential bottom demand.
5. Supply one `LevelPoolSources` record per element to `trial_with_sources_to`.

| Rate | Units |
| --- | --- |
| Rain | m/s |
| Potential evaporation | m/s |
| Potential bottom demand | m³/s |

Rates remain constant over the exact original interval. Split forcing discontinuities outside the component.

Bottom demand represents a supplied assumption or external model. It establishes no inferred soil or groundwater process.

### Choose the source area

Rain and potential evaporation use one area frozen at the accepted start.

| Accepted position | Selected area |
| --- | --- |
| Inside a segment | That segment's secant area |
| Interior knot | Right segment |
| Maximum volume | Last segment |
| Dry storage | Declared first-segment footprint |

This convention supplies no unique derivative at knots. Hydr performs no internal knot-event splitting.

Surrounding catchment rainfall remains separately owned external runoff.

## Advance and accept a trial

1. Choose a finite endpoint strictly after the accepted time.
2. Supply original exchange volumes for that exact interval.
3. Call the selected trial method.
4. Read the candidate outputs or diagnostics.
5. Consume `into_next` to accept the complete candidate.

The duration is `dt=end-accepted_time`. Hydr neither rescales transfer volumes nor converts them to rates.

Trials borrow accepted state immutably. Rejection or discarding a candidate preserves:

- Volumes and accepted time.
- Cumulative counters and bounds.
- Last accepted receipts.

`into_next` transfers ownership of persistent state without projection.

### Understand the application order

1. Add local rain when sources are supplied.
2. Apply capped bottom loss.
3. Apply capped evaporation.
4. Withdraw outgoing exchange from the available donor water.
5. Add incoming exchange.

The exchange-only path uses initial storage as its donor budget. The source-aware path uses checked post-source storage.

Already applied rain can supply outgoing exchange after losses. Incoming exchange cannot supply earlier losses or outgoing removal.

Outgoing volume never clips to available storage. Positive absorbed exchanges reject atomically. Zero exchanges preserve their high operands and explicit zero witnesses exactly.

### Use source-free intervals on an enabled batch

`trial_to` remains available and performs no source arithmetic. All-zero source records follow the same exchange-only path.

These intervals preserve numerical fields and zero-operation bits. Source counters and bounds remain unchanged. Enabling evidence remains attached to continuation state.

## Read outputs

| API | Returned information |
| --- | --- |
| Batch `states()` | Authoritative element volumes |
| Batch or trial `outputs()` | Volume and checked optional stage |
| Batch `element_ledgers()` | Per-element initial/current storage, lifetime counters, bounds, and residual |
| Batch `totals()` | Independently reduced cumulative batch accounting |
| Batch `last_step()` or trial `diagnostics()` | Original operands, witnesses, bounds, stage evidence, and interval accounting |

Array output methods return fallible projections. They never advance numerical storage state.

Numerical fields remain persistent F64 `Tensor<R>` values on the configured client. Grouped tensors reuse immutable curve data. Advancement never evaluates a scalar interpolation copy.

The checked CPU reference uses one thread. Other runtimes require [F64 arithmetic admission](integration.md#runtime-selection-and-admission).

## Save and restore

| Task | API |
| --- | --- |
| Retain complete persistent state without projection | `snapshot()` |
| Create application-owned complete data | Fallible `snapshot_data()` |
| Restore trusted state | `restore` |
| Import owned state | `restore_data` |

Restore compares immutable definition and config bits before replacing accepted state. Compatibility covers:

- Every curve knot and vertical datum.
- Provenance and physical assertions.
- Storage-support identity.
- Source-policy presence and exact evidence.

Source-enabled packages retain all source counters, bounds, and final nested receipts.

### Understand import checks

Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `LEVEL_POOL_SNAPSHOT_SCHEMA` and `LEVEL_POOL_CONTRACT_VERSION`. `restore_data` compares both before any other check and rejects a foreign value with an input error. The error names the found and expected values. Capture a new package to recover.

1. Check initial/current storage and per-element accounting before aggregate cancellation.
2. Recompute final-interval TwoSum operations from original operands.
3. Check exact continuity of final counters and bounds into accepted ledgers.
4. Recompute stage/inverse evidence and aggregate diagnostics.

Source-aware import also recalculates selected area and original products. It replays capped sources and exchanges before checking exact final continuity.

Caps and cap reasons must match recomputation. Source-policy presence cannot change silently during restoration.

Zero-history packages require exact initial/current identity and zero counters/bounds. They contain no final receipt.

Exact continuation requires the same build/runtime, ordered layout, and interval partition. Consistency checks do not authenticate arbitrary historical data.

## Check the arithmetic requirements

Numr TwoSum records the signed rounding error of an addition. Its witnesses diagnose arithmetic and never modify stored water.

Continuation and import require:

- Round-to-nearest ties-to-even.
- Gradual underflow.
- No FTZ/DAZ execution.

Unsupported arithmetic rejects context construction. Callers must preserve the admitted arithmetic throughout continuation and import.

### Exchange calculation

```text
Require outgoing <= initialVolume
(remaining, outgoingWitness) = TwoSum(initialVolume, -outgoing)
(volume, incomingWitness) = TwoSum(remaining, incoming)
```

On the source-aware path, the donor operand is checked post-source storage.

Withdrawal before incoming water also avoids unnecessary overflow. Positive exchanges must change their respective rounded storage values.

### Independent physical and arithmetic gates

The physical interval residual uses this fixed order:

```text
a = volume - initialVolume
b = a - incoming
residual = b + outgoing
```

The dimensional gate checks this residual before adding witnesses. A separate bound checks consistency with the signed witnesses.

Small arithmetic inconsistency cannot excuse material physical drift.

| Config field | Default |
| --- | --- |
| `volume_absolute` | `1e-9 m³` |
| `volume_relative` | `1e-12` |

Both values must be finite and nonnegative. At least one allowance must be positive.

Physical tolerance is `absolute+relative*scale`. Scale uses the maximum relevant actual storage or exchange magnitude. There is no `1 m³` floor.

Rounding diagnostics never enlarge physical tolerance scales. Inverse-volume error passes its own dimensional gate.

### Cumulative accounting

Every element retains initial/current volume and cumulative original incoming/outgoing volumes. Counter additions use TwoSum with conservative absolute error bounds.

Zero increments preserve counters and bounds exactly. Every element passes its own cumulative physical residual gate before batch aggregation. Opposite element errors cannot cancel admission.

Batch accounting passes a separate gate. Aggregate reduction uncertainty remains an independent diagnostic.

- Zero and single-element reductions require no uncertainty padding.
- Aggregate addition-error bounds use outward reductions of element bounds.
- Reduction padding changes no physical volume, residual, or tolerance allowance.

Stored volume stays within its curve range. Lifetime throughput can exceed capacity through repeated valid intervals.

Capacity-and-step-count ceilings reject impossible counters before relative residual checks. Theoretical ceiling overflow selects an unbounded consistency ceiling.

Actual nonfinite storage or counters still reject. Hydr never serializes theoretical infinity as physical state.

## Check source calculations and limits

### Source equations

```text
rain = (rainRate * frozenArea) * dt
potentialEvaporation = (evaporationRate * frozenArea) * dt
bottomDemand = bottomRate * dt
afterRain = initialVolume + rain
actualBottom = min(bottomDemand, afterRain)
afterBottom = afterRain - actualBottom
actualEvaporation = min(potentialEvaporation, afterBottom)
postSource = afterBottom - actualEvaporation
```

Each storage operation uses Numr TwoSum. Original process volumes remain separate from rounded storage differences.

Receipts retain the selected area and original rate-area-duration products. Only used products receive representability checks.

An unused `rate*dt` depth product cannot reject a representable area-volume product. Positive potential products require finite positive representation, even without remaining donor water.

### Interpret capped losses

Potential demands cap against preceding available water. Receipts retain actual/unmet volumes, cap reasons, and independent reconciliation bounds.

Equality records unrestricted demand, including zero demand. A cap never becomes a numerical water adjustment.

Potential and unmet demand never enlarge physical residual scales. Unmet-demand witnesses remain separate diagnostics.

### Check intermediate and final storage

Intermediate rain and bottom-loss budgets receive neither stage queries nor curve-range caps.

Separate curve/stage admission applies to:

- Accepted-start storage.
- Post-all-source storage.
- Final storage after exchanges.

Out-of-range post-source storage rejects even if a later exchange restores capacity.

Each actual interval process volume must remain within that element's maximum curve volume. This applies independently to rain, evaporation, and bottom loss.

This source-step resolution gate requests a shorter interval. It never clips an actual process volume.

Balanced positive rain and evaporation at maximum storage remain supported when both meet that gate. Overflowing budgets or absorbed positive source changes reject atomically.

### Source-aware residual and counters

The physical residual extends the original order:

```text
r = finalVolume - initialVolume
r = r - incoming
r = r + outgoing
r = r - rain
r = r + actualBottom
r = r + actualEvaporation
```

The dimensional gate precedes witness reconciliation. Five storage witnesses enter the consistency check in source/exchange order.

Source ledgers retain cumulative actual rain, evaporation, and bottom loss. They also retain potential bottom demand and addition-error bounds.

- Actual process counters use capacity/count ceilings.
- Potential bottom demand remains finite diagnostic accounting without a capacity ceiling.
- Cumulative actual bottom loss cannot exceed cumulative supplied demand.
- Per-element physical gates still precede aggregate cancellation.

## Check timestep accuracy

The frozen-area method is first order. Changing-area competition requires refinement evidence for the intended forcing and geometry.

It reconstructs no continuous within-interval storage trajectory.

The analytical comparison uses volume knots `[0,10,30] m³` and stage knots `[100,101,102] m` over `2 s`. Inputs below use exact decimal reference specifications.

Evaporation and bottom rates below are potential demands.

| Case | Initial volume, m³ | Rain, m/s | Evaporation, m/s | Bottom, m³/s |
| --- | --- | --- | --- | --- |
| Filling | `9.1` | `0.3` | `0.1` | `0.5` |
| Draining | `12.1` | `0` | `0.1` | `0.5` |
| Rain only | `9.1` | `0.2` | `0` | `0` |
| Evaporation only | `12.1` | `0` | `0.1` | `0` |

| Case | Reference volume, m³ | Reference stage, m |
| --- | --- | --- |
| Filling | `14.9` | `101.245` |
| Draining | `8.26` | `100.826` |
| Rain only | `16.2` | `101.31` |
| Evaporation only | `9.05` | `100.905` |

All four cases pass the independently frozen first-order error envelopes across seven timestep grids. Timesteps are `0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125 s`. These 28 rows retain process-volume differences and knot-phase plateaus. Strict decrease at every grid is unsupported.

At the finest `0.0078125 s` timestep:

| Case | Stage error, m | Storage error, m³ | Maximum process-volume error, m³ |
| --- | --- | --- | --- |
| Filling | `0.00015625` | `0.003125` | `0.0046875` |
| Draining | `0.000375` | `0.00375` | `0.00375` |
| Rain only | `0.0003125` | `0.00625` | `0.00625` |
| Evaporation only | `0.00046875` | `0.0046875` | `0.0046875` |

The finest rows meet the frozen `0.0005 m` stage and `0.01 m³` storage/process-volume targets. Numerical stage allowances divide volume tolerance by the relevant positive segment area. Cubic-metre tolerances never add directly to metre errors.

Run the tracked observer to inspect every grid and process counter:

```sh
cargo run --example level_pool_sources_reference
```

These selected analytical cases establish no general timestep requirement or field-accuracy guarantee. Source ownership, variable geometry, outlet laws, and synchronized hydraulic exchange remain separate mechanisms.

## Try the existing example

```sh
cargo run --example level_pool
```

The [closed tank example](../examples/level_pool.rs) uses `H=10+V/2` and a two-square-metre area.

Its interval starts with `1 m³`, withdraws `0.1 m³`, then receives `0.2 m³`. Receipts retain both original exchanges and their signed witnesses.

## Unsupported uses

This standalone component supplies no automatic hydraulic attachment. Further mechanisms require separate checks:

- Outlet structures and controls.
- Geometry events.
- Soil and groundwater processes.
- Rainfall ownership attachment and synchronized hydraulic coupling.

Complete system definitions and paired-transfer provenance belong to the coupling layer. Use [coupled level pools](coupled-level-pools.md) for its admitted scope.
