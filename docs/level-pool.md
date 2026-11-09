# Closed level-pool storage

`hydr::storage::LevelPoolBatch` advances authoritative pool volumes and checks stage through immutable storage curves. The component supplies no outlet law, atmospheric source calculation, or hydraulic attachment. ARUS supplies evidence for a connected, approximately level pool. Disconnected depressions require separate physical elements.

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

Incoming water cannot finance the initial donor budget. Outgoing volume never clips to available storage. This order also avoids unnecessary overflow from adding incoming water before withdrawal. Positive exchanges must change their respective rounded storage values. Absorbed positive exchanges reject atomically. Zero exchanges preserve their high operands exactly and retain explicit zero witnesses.

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

## Atomic lifecycle and projections

Trials borrow accepted state immutably. Rejection or discarding a candidate preserves all volumes, time, counters, and last receipts. `into_next` accepts complete persistent state by ownership without projection.

| API | Explicit output boundary |
| --- | --- |
| Batch `states()` | Authoritative current element volumes |
| Batch or trial `outputs()` | Current volume and checked optional physical stage |
| Batch `element_ledgers()` | Initial/current storage, lifetime counters, error bounds, and residual per element |
| Batch `totals()` | Independently reduced cumulative batch accounting |
| Batch `last_step()` or trial `diagnostics()` | Complete original operands, TwoSum witnesses, bounds, stage evidence, and interval accounting |

The runtime uses persistent CPU F64 tensors on an explicitly configured one-thread numr client. Arrays serve construction, explicit output, and error diagnosis. They never advance numerical storage state.

## Complete restart

`snapshot()` clones complete persistent state without projection. `snapshot_data()` produces caller-owned complete values. Trusted and owned restore paths compare immutable definition/config bits before replacing accepted state. Compatibility includes every curve knot, datum, provenance, physical assertion, and storage-support identity.

Owned import checks initial/current storage and per-element accounting before aggregate cancellation. It recomputes final-interval TwoSum operations from their original operands. The reconstructed final counters and bounds must continue exactly into accepted element ledgers. Reported stage/inverse evidence and aggregate diagnostics must match complete recomputation. Zero-history packages require exact initial/current identity and zero counters/bounds, with no final receipt.

Exact continuation requires the same build/runtime, ordered layout, and interval partition. Consistency checks bound numerical claims without authenticating arbitrary historical data. Future coupling owns complete system definitions and paired-transfer provenance.

This closed-storage unit implements no surface-area convention at curve knots. Atmospheric sources, outlet structures, controls, geometry events, and hydraulic attachment require separately checked mechanisms.
