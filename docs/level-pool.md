# Closed level-pool storage

`hydr::storage::LevelPoolBatch` advances authoritative pool volumes and checks stage through immutable storage curves. An explicit optional policy enables frozen-area rain, capped potential bottom demand, and capped potential evaporation. The component supplies no outlet law or hydraulic attachment. ARUS supplies evidence for a connected, approximately level pool. Disconnected depressions require separate physical elements.

Run the [closed tank example](../examples/level_pool.rs):

```sh
cargo run --example level_pool
```

The example declares a two-square-metre tank with `H=10+V/2`. Its original interval exchanges withdraw `0.1 m³` from initial `1 m³`, then admit incoming `0.2 m³`. The component retains both original exchanges and their signed addition witnesses.

## Explicit curves and state

`LevelPoolCurve::new` accepts ordered `LevelPoolKnot` values, a vertical reference, and curve provenance. Volumes and stages require finite strictly increasing knots. The first volume equals zero. Every adjacent volume/stage difference and secant area must remain positive and finite. Queries stay within the closed volume range. Extrapolation is unsupported.

The component uses existing Solvr Linear interpolation for forward and inverse queries. Exact knot queries retain supplied stage values, including the maximum-volume endpoint. Grouped persistent tensors reuse immutable curve data. Numerical advancement never evaluates a scalar interpolation copy.

Exact zero volume is dry and returns `stage_m=None`. The first elevation remains the internal storage origin. Positive volume requires a stage above that origin and a positive inverse-reconstructed volume. Inverse-volume error passes an independent dimensional gate. Reconstructed volume never replaces authoritative state. Large datum offsets or extreme slopes can make wet height unresolved. Those scales reject without inventing a positive film. Positive volume changes can retain an unchanged rounded stage when inverse admission remains satisfied.

Each `LevelPoolElement` requires a unique ID and storage-support identity. Its nonempty `connected_pool_assertion` records the application's physical evidence. The support identity establishes no geometric proof of disjointness from hydraulic cells. Exclude overlapping storage explicitly before any future attachment. This component performs no datum transformation.

`LevelPoolState` requires explicit initial volume. There is no silent empty initialization. `LevelPoolInput` supplies original incoming and outgoing volumes over the exact requested interval.

## Authoritative exchange order

`trial_to(end, inputs)` uses `dt=end-accepted_time`, with finite strictly increasing time. Transfer volumes retain their original interval meaning. The component supplies no automatic duration rescaling or rate conversion.

```text
Require outgoing <= initialVolume
(remaining, outgoingWitness) = TwoSum(initialVolume, -outgoing)
(volume, incomingWitness) = TwoSum(remaining, incoming)
```

Incoming exchange cannot finance the donor budget. The exchange-only path uses initial storage. The source-aware path uses checked post-source storage. Outgoing volume never clips to available storage. This order also avoids unnecessary overflow from adding incoming water before withdrawal. Positive exchanges must change their respective rounded storage values. Absorbed positive exchanges reject atomically. Zero exchanges preserve their high operands exactly and retain explicit zero witnesses.

Numr TwoSum retains signed rounding witnesses under round-to-nearest ties-to-even with gradual underflow. Unsupported FTZ/DAZ execution rejects context construction. Callers must preserve this arithmetic envelope throughout continuation and import. Witnesses remain diagnostic quantities. They never modify storage or become numerical water adjustments.

## Independent numerical gates

The component reports the physical interval residual in this fixed order:

```text
a = volume - initialVolume
b = a - incoming
residual = b + outgoing
```

The dimensional gate checks the original residual before adding either witness. A separate arithmetic bound checks consistency between that residual and the signed TwoSum witnesses. Small witness inconsistency cannot excuse material physical drift.

| Config field | Default |
| --- | --- |
| `volume_absolute` | `1e-9 m³` |
| `volume_relative` | `1e-12` |

Both values require finite nonnegative inputs, with at least one positive allowance. Physical tolerance uses `absolute+relative*scale`. Scale is the maximum relevant actual storage or exchange magnitude. There is no `1 m³` floor. Roundoff diagnostics never enlarge physical tolerance scales.

Persistent per-element accounting retains initial/current volume and cumulative original incoming/outgoing volumes. Cumulative additions use TwoSum and conservative absolute addition-error bounds. Zero increments preserve counters and bounds exactly. Every element passes its own cumulative physical residual gate. Opposite element errors cannot cancel admission. Batch aggregate accounting passes a separate gate. Aggregate reduction uncertainty remains an independent diagnostic. Zero and single-element reductions require no uncertainty padding. Returned aggregate cumulative addition-error bounds use outward reductions of the element bounds. Their reduction padding never modifies physical volumes, residuals, or tolerance allowances.

Stored volume remains within its own curve range. Lifetime throughput can exceed capacity through repeated valid intervals. Capacity-and-step-count ceilings reject impossible cumulative counters before relative residual checks. Theoretical ceiling overflow selects an unbounded consistency ceiling. Actual nonfinite storage or counters still reject. The component never serializes theoretical infinity as physical state.

## Optional frozen-area sources

`new_with_sources` retains original construction arguments and adds one ordered `LevelPoolSourcePolicy` per element. Immutable evidence acknowledges the first segment's dry rain-receiving basal footprint. Separate bottom-loss provenance identifies the supplied potential-demand assumption or external model. No inferred soil or groundwater process follows from that demand.

`trial_with_sources_to` accepts original exchanges and one `LevelPoolSources` record per element. Rain and potential evaporation use rates in metres/second. Bottom demand uses cubic metres/second. All rates remain constant over the exact original interval. Callers split forcing discontinuities externally.

The operator freezes the accepted-state segment's secant area for rain and potential evaporation. Interior knots select the right segment. Maximum volume selects the last segment. Dry storage uses the acknowledged first-segment footprint. Surrounding catchment rainfall remains separately owned external runoff. This convention supplies no unique derivative at knots and performs no internal knot-event splitting.

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

Each storage operation uses Numr TwoSum. Original source volumes remain separate from rounded storage differences. Selected area and original rate-area-duration products remain visible in source receipts. Only used products receive representability checks. No unused `rate*dt` depth check rejects a representable area-volume product. Positive potential products require finite positive representation even when no donor water remains.

Potential demands cap physically against preceding available water. Actual/unmet volumes retain explicit cap reasons and independent arithmetic reconciliation bounds. Equality records an unrestricted demand, including zero demand. This cap never becomes a numerical water adjustment. Potential and unmet demand never enlarge physical residual scales.

Intermediate rain and bottom-loss budgets are algebraic quantities. They receive no stage query or curve-range cap. Accepted-start, post-all-source, and final storage receive independent curve/stage admission. Out-of-range post-source storage rejects even when a later exchange restores capacity.

Every actual interval rain, evaporation, and bottom-loss volume must remain within that element's maximum curve volume. This independent source-step resolution gate requests a shorter interval. It never clips an actual process volume. Balanced positive rain and evaporation at maximum storage remain supported when each process meets that gate. Overflowing budgets and absorbed positive source changes reject atomically.

Outgoing exchange uses post-source donor water. Already applied rain can therefore finance that exchange after capped losses. Incoming exchange arrives afterward and cannot finance preceding losses or outgoing removal.

The source-aware physical residual extends the original order without modifying authoritative quantities:

```text
r = finalVolume - initialVolume
r = r - incoming
r = r + outgoing
r = r - rain
r = r + actualBottom
r = r + actualEvaporation
```

The dimensional gate applies before witness reconciliation. Five storage witnesses enter the consistency check in source/exchange order. Unmet-demand witnesses remain separate diagnostics.

Source-enabled ledgers retain actual rain, evaporation, bottom loss, and potential bottom-demand counters with addition-error bounds. Capacity/count ceilings apply to actual process counters. Potential bottom demand remains finite diagnostic accounting without a capacity ceiling. Cumulative actual bottom loss cannot exceed cumulative supplied demand. Per-element physical gates still precede aggregate cancellation.

Existing `trial_to` remains available and performs no source arithmetic. All-zero source records follow that exchange-only path, preserving numerical fields and zero-operation bits. Source counters/bounds remain unchanged across those intervals. Immutable enabling evidence remains attached to continuation state.

The frozen-area method is first order. Changing-area source competition requires timestep-refinement evidence for the intended forcing and geometry. It does not reconstruct a continuous within-interval storage trajectory.

## Measured changing-area refinement

The independent analytical comparison uses volume knots `[0,10,30] m³` and stage knots `[100,101,102] m` over `2 s`. Inputs below use exact decimal reference specifications.

| Case | Initial volume, m³ | Rain, m/s | Potential E, m/s | Bottom demand, m³/s | Reference volume, m³ | Reference stage, m |
| --- | --- | --- | --- | --- | --- | --- |
| Filling | `9.1` | `0.3` | `0.1` | `0.5` | `14.9` | `101.245` |
| Draining | `12.1` | `0` | `0.1` | `0.5` | `8.26` | `100.826` |
| Rain only | `9.1` | `0.2` | `0` | `0` | `16.2` | `101.31` |
| Evaporation only | `12.1` | `0` | `0.1` | `0` | `9.05` | `100.905` |

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

## Atomic lifecycle and projections

Trials borrow accepted state immutably. Rejection or discarding a candidate preserves all volumes, time, counters, and last receipts. `into_next` accepts complete persistent state by ownership without projection.

| API | Explicit output boundary |
| --- | --- |
| Batch `states()` | Authoritative current element volumes |
| Batch or trial `outputs()` | Current volume and checked optional physical stage |
| Batch `element_ledgers()` | Initial/current storage, lifetime counters, error bounds, and residual per element |
| Batch `totals()` | Independently reduced cumulative batch accounting |
| Batch `last_step()` or trial `diagnostics()` | Complete original operands, TwoSum witnesses, bounds, stage evidence, and interval accounting |

The batch uses persistent F64 `Tensor<R>` fields on a caller-configured Numr client. The checked CPU reference uses one thread. Other runtimes require [F64 arithmetic admission](integration.md#runtime-selection-and-admission). Arrays serve construction, explicit output, and error diagnosis. Array output methods return fallible projections. They never advance numerical storage state.

## Complete restart

`snapshot()` clones complete persistent state without projection. `snapshot_data()` produces fallible caller-owned complete values. Trusted and owned restore paths compare immutable definition/config bits before replacing accepted state. Compatibility includes every curve knot, datum, provenance, physical assertion, and storage-support identity. Source-enabled packages also retain exact policy evidence and all source counters, bounds, and final nested receipts.

Owned import checks initial/current storage and per-element accounting before aggregate cancellation. It recomputes final-interval TwoSum operations from their original operands. The reconstructed final counters and bounds must continue exactly into accepted element ledgers. Reported stage/inverse evidence and aggregate diagnostics must match complete recomputation. Zero-history packages require exact initial/current identity and zero counters/bounds, with no final receipt.

Source-aware import recomputes selected area and original products before replaying capped source substeps and exchanges. Caps, cap reasons, stage evidence, and cumulative source additions require exact final continuity. Source-policy presence cannot silently change during restoration.

Exact continuation requires the same build/runtime, ordered layout, and interval partition. Consistency checks bound numerical claims without authenticating arbitrary historical data. Future coupling owns complete system definitions and paired-transfer provenance.

The optional source operator supplies an explicit one-sided area convention at curve knots. Outlet structures, controls, geometry events, soil/groundwater processes, rainfall ownership attachment, and hydraulic coupling remain unsupported.
