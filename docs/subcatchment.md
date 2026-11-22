# Calculate runoff from a subcatchment

Use `hydr::runoff::SubcatchmentBatch` to calculate storage, soil losses, and runoff from three declared subareas. Start by checking the physical support.

This component infers no GIS ownership. On its own it attaches runoff to no hydraulic model. Run the [example](../examples/subcatchment.rs) to inspect its inputs and outputs:

```sh
cargo run --example subcatchment
```

To send free outlet volumes into storage pools, use the [runoff-to-pool coordinator](runoff-pools.md). To send them into surface cells, reach cells, and coordinator pools, [attach native runoff](surface-channels.md#attach-native-runoff) to the surface/channel coordinator.

Read capabilities with `subcatchment_capabilities()`. It returns a `ComponentCapabilities` declaration: runtimes, the outlet endpoint, sources, the routing option, nested and parent components, unsupported uses, and limits.

## Define the contributing area

`SubcatchmentGeometry` supplies explicit SI geometry and area fractions. `SubcatchmentParameters::new` checks physical support and uses the checked [reservoir relations](reservoir.md).

The area formulas use these quantities:

- `A`: Total physical area, `total_area_m2`, in m².
- `fi`: Impervious fraction of total area, `impervious_fraction`, dimensionless.
- `f0`: Fraction within impervious area having zero depression storage, `zero_storage_impervious_fraction`, dimensionless.

| Ordered subarea | Actual storage area | Area used for runoff coefficient class |
| --- | --- | --- |
| `ImperviousZeroStorage` | `A*fi*f0` | `A*fi` |
| `ImperviousStorage` | `A*fi*(1-f0)` | `A*fi` |
| `Pervious` | `A*(1-fi)` | `A*(1-fi)` |

### Check active and absent subareas

- Subarea zero has zero depression storage.
- Positive pervious support requires soil parameters and complete native antecedent memory.
- Absent subareas require zero depth and receive no positive external or internal volume.
- Absent subareas perform no soil call.
- Mathematically positive areas cannot underflow into absent support.

Element IDs must be unique and nonempty. They identify definitions without proving disjoint contributing areas.

External runon is water entering from outside this modeled subcatchment. The caller owns rainfall support and runon allocation. Local hydraulic rainfall cannot cover the same contributing support twice.

## Supply initial state and forcing

`SubcatchmentState` retains three depths and optional complete `NativeSoilState`. Use `initialized` only to explicitly select dry storage and original initialized soil memory.

Imported native state preserves signed event infiltration `F` and recovery time `T`. Saturation remains an independent flag.

| Input | Meaning |
| --- | --- |
| Rain | Constant nonnegative rate in metres/second |
| Potential evaporation | Constant nonnegative rate in metres/second |
| Three external runon quantities | Authoritative destination volumes in cubic metres over the original interval |

Each `SubcatchmentInput` supplies these values. Positive inputs require represented interval depths and physical volumes. Unresolved positive transfers return errors without clipping.

The component stores numerical state in Numr `Tensor<R>` fields. Supply the configured Numr client when constructing it. The checked CPU F64 reference uses one thread. Other runtimes require [F64 arithmetic admission](integration.md#runtime-selection-and-admission).

## Choose internal routing

`SubcatchmentRouting` supports these choices:

| Choice | Destination |
| --- | --- |
| Direct outlet | Export leaves the subcatchment |
| Impervious-to-pervious | Impervious export enters the pervious subarea |
| Pervious-to-impervious-storage | Pervious export enters the impervious subarea with storage |

Routing fractions lie within `[0,1]`. Positive fractions require supported receivers.

Donors advance before receivers in two fixed topological phases. Each pervious soil state advances once per trial. Accepted boundaries retain no pending routing queue.

### How transferred water is represented

The receiver converts declared volume into a supply rate. Its separately represented storage increment retains a visible conversion residual.

The original declared volume remains authoritative. The component never changes that volume to conceal conversion error. Positive split portions and merged additions must remain represented. Absorbed or underflowed transfers reject the complete trial.

## Advance and accept a trial

1. Call `trial_to(end_time, inputs)` with a finite endpoint after the accepted time.
2. Read trial outputs or diagnostics when needed.

Then choose one outcome:

| Decision | Action |
| --- | --- |
| Accept the candidate | Consume `into_next()` to obtain the complete next batch. |
| Keep the original state | Discard the trial without calling `into_next()`. All accepted states and counters remain unchanged. |

The duration is `dt=end_time-accepted_time`. This component supplies no automatic timestep controller.

### Source order

1. Apply rain once over represented active areas.
2. Merge original external and routed volumes in fixed interface order.
3. Evaluate Modified Green–Ampt using original ponding and the constructed total supply rate.
4. Apply infiltration and proved bounded soil correction, then capped evaporation.
5. Advance reservoir drainage using remaining available water.
6. Partition donor export into identical routed debit/credit and complementary outlet volume.

Drainage retains the [reservoir contract](reservoir.md), including its branches and reconstructed-state checks.

Parameter construction caches these reservoir relations for reuse during trials.

| Path | Timing and output rule |
| --- | --- |
| No-loss direct path | Preserve the reservoir's timing expression |
| Loss or transfer path | Derive event timing from post-source available depth |
| Integrated export | Use represented storage depletion |
| Instantaneous overflow | Endpoint discharge is undefined and appears as `None` |

Endpoint discharge multiplied by the full interval never replaces integrated export.

### What happens on rejection

Any donor, receiver, soil, root, or accounting error rejects the complete candidate.

Discarding a successful trial preserves storage, both phase-role soil batches, accepted time, and every ledger counter. `into_next()` accepts by ownership without copying numerical arrays to the host.

## Read results

Numerical continuation uses persistent tensors and never advances copied output arrays. Array output methods return an error if copying fails.

| API | What you can inspect |
| --- | --- |
| Batch `states()` | Current depths and complete native soil memory |
| Batch `last_step()` | Last accepted source, drainage, and paired-interface diagnostics |
| Trial `outputs()` | Supported storage, losses, integrated exports, endpoint discharge, and residuals |
| Trial `diagnostics()` | Original interval, phase work, budgets, source receipts, and transfer identities |
| Batch `ledger()` | Complete cumulative physical and uncertainty counters |

## Check the water balance

The physical ledger counts each process once:

| Stored water | Inputs | Outputs | Explicit numerical term |
| --- | --- | --- | --- |
| Initial and current subarea storage | Rain and external runon | Infiltration, evaporation, and outlet export | Soil correction |

Identical internal debit/credit volumes cancel physically. Their traffic remains visible in a separate counter.

Water checks apply independently to every subarea, catchment, aggregate interval, and cumulative history.

### Keep uncertainty budgets separate

An outward bound rounds uncertainty upward to avoid understating it.

| Control or diagnostic | Meaning |
| --- | --- |
| `config.reservoir` | Existing root and independent physical volume controls |
| `max_cumulative_water_adjustment_m3` | Separate absolute soil correction budget, default `1e-9 m³` |
| Transfer uncertainty charge | Outward accumulated partition, merge, and receiver-conversion envelopes |
| Physical scale lower bound | Conservative cumulative initial storage, rain, external runon, and internal traffic scale |
| Rain-area difference | Represented applied rain minus total-area rain, with a separate arithmetic bound |

Transfer uncertainty uses a conservatively rounded `volume_absolute+volume_relative*physical_scale` allowance. It cannot replace a physical residual or consume the soil correction budget.

Rain-area bounds remain separate diagnostics. They authorize no additional rain or water correction.

Owned imports check the scale lower bound against an independent count-dependent counter ceiling. Cumulative nominal soil correction must fit its conservative absolute charge, including reduction and cumulative-addition rounding allowances.

### Preserve the required arithmetic

Construction probes the configured client for round-to-nearest ties-to-even and gradual underflow. Preserve those conditions throughout continuation and import.

The component does not recheck FTZ/DAZ or rounding mode before every operation. FTZ/DAZ modes replace subnormal values with zero and violate this arithmetic contract. Subnormal values fall below the normal floating-point range.

Positive storage and transfers require finite represented volumes. Bounds and counters cannot silently overflow.

## Save and restore

| Task | API and behavior |
| --- | --- |
| Retain an in-process checkpoint | `snapshot()` retains complete tensor continuation without copying arrays to the host |
| Save application-owned data | `snapshot_data()` copies values on request and returns an error if copying fails |
| Restore either form | Check exact definition/config bits before atomic replacement |

Owned import checks accepted depths, final diagnostics, cumulative counters, and independent budgets. It checks complete source algebra and numerical receipts without replaying soil history or root-finding iterations.

Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `SUBCATCHMENT_SNAPSHOT_SCHEMA` and `SUBCATCHMENT_CONTRACT_VERSION`. `restore_data` compares both before any other check and rejects a foreign value with an input error. The error names the found and expected values. Capture a new package to recover. Packages without the identity fields are rejected.

Exact restart requires the same build/runtime, ordered batch layout, and accepted interval partition. Receipt consistency establishes numerical admission. It does not authenticate history.

## Understand differences from native SWMM

The method preserves pinned soil-call semantics and native-derived runoff coefficients.

Its first-order source priority, integrated export, and same-interval routing differ from native SWMM bookkeeping. Coarse-step native identity and field accuracy remain unestablished. Use refinement evidence for your forcing and geometry.

The [native runoff oracle](runoff-reference.md) independently exposes discrepancies in endpoint volume, competing losses, and previous-rate routing.

## Check temporal accuracy

### Smooth routing cases

Two finite-Manning cases use constant-conductivity, zero-deficit soil. Mobile depths stay positive throughout a 64-second horizon.

Both route half the donor export, use area fractions `(1/4,1/4,1/2)`, and use native-derived coefficients.

An independent SciPy DOP853 calculation integrates depths, gross exports, outlets, and routed volumes separately. Primary and tighter references agree within `2.22e-7` of the production tolerance. The reference acceptance limit is `0.01`.

| Quantity | Physical tolerance |
| --- | --- |
| Depth | `1e-9 m + 0.005*initial_mobile_depth` |
| Volume | `1e-9 m³ + 0.005*max(abs(reference), relevant_initial_mobile_volume)` |

Each result reports the largest ratio of error to its allowed physical tolerance. It covers every endpoint and checked quantity. Values above one exceed the physical target.

| Timestep, seconds | Impervious to pervious | Pervious to impervious storage |
| --- | --- | --- |
| 1 | 0.2349832944 | 1.2540406319 |
| 0.5 | 0.1185892238 | 0.6404202830 |
| 0.25 | 0.0595743354 | 0.3236490576 |
| 0.125 | 0.0298577846 | 0.1626852984 |
| 0.0625 | 0.0149466582 | 0.0815597663 |
| 0.03125 | 0.0074777801 | 0.0408344056 |

Both errors decrease strictly, and every finest-grid error meets its frozen tolerance.

The one-second pervious-routing case exceeds the physical target. Conservation alone does not establish temporal accuracy.

Separate branch-crossing soil histories retain all four timestep observations and native differences. They establish no general accuracy claim.

### Native soil comparisons

Matching-input EPA comparisons check 8,244 constructed soil calls across 24 complete sequences.

All 16,488 full-sequence and isolated-call comparisons meet the unchanged [soil tolerances](soil.md#comparison-criteria). Checks cover returned rate, complete native memory, exact zero/nonzero branches, and saturation flags.

Restricted native soil restarts reproduce 24 midpoint suffixes exactly. This establishes no complete SWMM-driver rewind capability.

### Reproduce the comparison

The comparison uses CPU F64, SciPy `1.16.3`, and NumPy `2.4.0`. Use fresh output paths:

```sh
python3 tools/reference/subcatchment_temporal.py freeze --output /tmp/hydr-subcatchment/spec.json
cargo run --example subcatchment_reference -- smooth /tmp/hydr-subcatchment/observer.csv
python3 tools/reference/subcatchment_temporal.py run --frozen-spec /tmp/hydr-subcatchment/spec.json --observer /tmp/hydr-subcatchment/observer.csv --output-dir /tmp/hydr-subcatchment/comparison
```

The tools freeze physical inputs, norms, reference controls, and source hashes before comparison. Generated fields remain outside public source history.

## Current limits

This standalone component excludes:

- Inter-catchment graphs.
- Groundwater, LID, snow, and water quality.
- Tailwater feedback.
