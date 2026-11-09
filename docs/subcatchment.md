# Conservative standalone subcatchment

`hydr::runoff::SubcatchmentBatch` advances explicit three-subarea storage, native soil memory, losses, and paired internal transfers.
The component uses persistent F64 tensors on an explicitly configured one-thread numr CPU client.
It supplies no hydraulic attachment or GIS ownership inference.

Run the [standalone example](../examples/subcatchment.rs):

```sh
cargo run --example subcatchment
```

## Physical support

`SubcatchmentGeometry` supplies explicit SI geometry and fractions. `SubcatchmentParameters::new` checks support and caches existing reservoir closures.

| Ordered subarea | Actual storage area | Runoff coefficient class area |
| --- | --- | --- |
| `ImperviousZeroStorage` | `A*fi*f0` | `A*fi` |
| `ImperviousStorage` | `A*fi*(1-f0)` | `A*fi` |
| `Pervious` | `A*(1-fi)` | `A*(1-fi)` |

Subarea zero has zero depression storage. Positive pervious support requires soil parameters and complete native antecedent memory.
Absent subareas require zero depth and receive no positive external or internal volume. They perform no soil call.
Mathematically positive areas cannot silently underflow into absent support.

Element IDs are unique and nonempty. They establish definition identity without asserting disjoint contributing areas.
The caller owns rainfall support and external runon allocation. Local hydraulic rainfall cannot cover the same contributing support twice.

## Explicit state and forcing

`SubcatchmentState` retains three depths and optional complete `NativeSoilState`. `initialized` explicitly selects dry storage and original initialized soil memory.
Imported native state preserves signed event infiltration `F` and recovery time `T`. Saturation remains an independent flag.

Each `SubcatchmentInput` supplies constant nonnegative rain and potential evaporation rates in metres/second.
Its three external runon quantities are authoritative destination volumes in cubic metres over the original interval.
Positive inputs require representable interval depths and physical volumes. Unresolved positive transfers return errors without clipping.

`trial_to(end_time, inputs)` uses `dt=end_time-accepted_time`. The endpoint must be finite and strictly increasing.
The component supplies no automatic timestep controller.

## Source order and routing

1. Rain applies once over represented active areas.
2. Original external and routed volumes merge in fixed interface order.
3. The existing Modified Green–Ampt trial receives original ponding and the constructed total supply rate.
4. Infiltration and proved bounded soil correction precede capped evaporation.
5. Existing reservoir drainage advances the remaining available water.
6. Donor export partitions into identical routed debit/credit and complementary outlet volume.

`SubcatchmentRouting` supports direct outlets, impervious-to-pervious routing, and pervious-to-impervious-storage routing.
Routing fractions lie within `[0,1]`. Positive fractions require supported receivers.
Donors advance before receivers in two fixed topological phases. Each pervious soil state advances exactly once per trial.
Accepted boundaries retain no pending routing queue.

The receiver converts declared volume into a supply rate. Its independently represented storage increment retains a visible bridge residual.
The original declared volume remains authoritative. The component never adjusts that volume to conceal conversion error.
Positive split portions and merged additions must remain represented. Absorbed or underflowed transfers reject the complete trial.

Drainage reuses the [reservoir contract](reservoir.md), including its checked branches and reconstructed-state admission.
No-loss direct paths preserve the primitive's timing expression. Loss or transfer paths derive event timing from post-source available depth.
Integrated export follows represented storage depletion. Endpoint discharge multiplied by the full interval never replaces that export.
Instantaneous-overflow endpoint discharge remains undefined and appears as `None`.

## Independent accounting

The physical water ledger includes initial/current storage, rain, external runon, infiltration, evaporation, outlet export, and explicit soil correction.
Identical internal debit/credit volumes cancel physically. Internal traffic remains visible in its own counter.
Every subarea, catchment, aggregate interval, and cumulative history retains independent water checks.

| Control or diagnostic | Meaning |
| --- | --- |
| `config.reservoir` | Existing root and independent physical volume controls |
| `max_cumulative_water_adjustment_m3` | Separate absolute soil correction budget, default `1e-9 m³` |
| Transfer uncertainty charge | Outward accumulated partition, merge, and receiver-bridge envelopes |
| Physical scale lower bound | Conservative cumulative initial storage, rain, external runon, and internal traffic scale |
| Rain-area difference | Represented applied rain minus total-area rain, with a separate arithmetic bound |

Transfer uncertainty uses a conservatively rounded `volume_absolute+volume_relative*physical_scale` allowance.
It cannot replace a physical residual or consume the soil correction budget.
Rain-area bounds remain separate diagnostics. They authorize no additional rain or water correction.
Owned imports check the scale lower bound against an independent count-dependent counter ceiling.
Cumulative nominal soil correction must fit its conservative absolute charge, including reduction and cumulative-addition rounding allowances.

The arithmetic contract requires round-to-nearest ties-to-even and gradual underflow. Construction probes the configured client for these properties. The caller must preserve this arithmetic envelope throughout continuation and import. The component does not recheck FTZ/DAZ or rounding mode before every operation.
Positive storage and transfers require finite represented volumes. Bounds and counters cannot silently overflow.

## Atomic continuation and output

Trials borrow accepted state immutably. Any donor, receiver, soil, root, or accounting error rejects the complete candidate.
Discarding a successful trial preserves accepted storage, both phase-role soil batches, time, and every ledger counter.
`into_next` accepts the complete candidate by ownership without numerical projection.

| API | Explicit output boundary |
| --- | --- |
| Batch `states()` | Current depths and complete native soil memory |
| Batch `last_step()` | Complete last accepted source, drainage, and paired-interface diagnostics |
| Trial `outputs()` | Supported subarea storage, losses, integrated exports, endpoint discharge, and residuals |
| Trial `diagnostics()` | Original interval, phase work, budgets, source receipts, and transfer identities |
| Batch `ledger()` | Complete cumulative physical and uncertainty counters |

Numerical continuation consumes persistent tensors. It never advances projected arrays.

## Restart and scope

`snapshot()` clones complete persistent continuation state. `snapshot_data()` projects caller-owned values on request.
Both restore paths check exact definition/config bits before atomic replacement.
Owned import checks complete source algebra and numerical receipts without replaying soil history or roots.
It also checks accepted depths, final diagnostics, cumulative counters, and independent budgets.

Exact restart requires the same build/runtime, ordered batch layout, and accepted interval partition.
Receipt consistency establishes numeric admission, not historical authenticity.

The method preserves pinned soil-call semantics and native-derived runoff coefficients.
Its first-order source priority, integrated export, and same-interval routing depart from native SWMM bookkeeping.
Coarse-step native identity and field accuracy remain unestablished. Use refinement evidence for the intended forcing and geometry.

The [native runoff oracle](runoff-reference.md) exposes native endpoint-volume, competing-loss, and previous-rate routing discrepancies independently.
This component supports no inter-catchment graph, hydraulic attachment, groundwater, LID, snow, quality, or tailwater feedback.

## Checked temporal accuracy

Two smooth finite-Manning cases use constant-conductivity, zero-deficit soil and positive mobile depths throughout a 64-second horizon.
Both route half the donor export. The cases use area fractions `(1/4,1/4,1/2)` and native-derived coefficients.
An independent SciPy DOP853 calculation integrates depths, gross exports, outlets, and routed volumes separately.
Primary and tighter references agree within `2.22e-7` of the production tolerance. The reference acceptance limit is `0.01`.

Depth tolerance is `1e-9 m + 0.005*initial_mobile_depth`.
Volume tolerance is `1e-9 m³ + 0.005*max(abs(reference), relevant_initial_mobile_volume)`.
Each row reports the maximum normalized dynamic error across every endpoint and checked quantity. A value above one exceeds that physical tolerance.

| Timestep, seconds | Impervious to pervious | Pervious to impervious storage |
| --- | --- | --- |
| 1 | 0.2349832944 | 1.2540406319 |
| 0.5 | 0.1185892238 | 0.6404202830 |
| 0.25 | 0.0595743354 | 0.3236490576 |
| 0.125 | 0.0298577846 | 0.1626852984 |
| 0.0625 | 0.0149466582 | 0.0815597663 |
| 0.03125 | 0.0074777801 | 0.0408344056 |

Both errors decrease strictly. Every finest-grid error meets its frozen tolerance.
The one-second pervious-routing case exceeds the physical target. Conservation alone does not establish temporal accuracy.
Separate branch-crossing soil histories retain all four timestep observations and native differences without a general accuracy claim.

Matching-input EPA comparisons check 8,244 constructed soil calls across 24 complete sequences.
All 16,488 full-sequence and isolated-call comparisons meet the unchanged soil component tolerances.
Checks include returned rate, complete native memory, exact zero/nonzero branches, and saturation flags.
Restricted native soil restarts reproduce 24 midpoint suffixes exactly. This establishes no complete SWMM-driver rewind capability.

The comparison uses CPU F64, SciPy `1.16.3`, and NumPy `2.4.0`. Use fresh output paths:

```sh
python3 tools/reference/subcatchment_temporal.py freeze --output /tmp/hydr-subcatchment/spec.json
cargo run --example subcatchment_reference -- smooth /tmp/hydr-subcatchment/observer.csv
python3 tools/reference/subcatchment_temporal.py run --frozen-spec /tmp/hydr-subcatchment/spec.json --observer /tmp/hydr-subcatchment/observer.csv --output-dir /tmp/hydr-subcatchment/comparison
```

The tools freeze physical inputs, norms, reference controls, and source hashes before comparison. Generated fields remain outside public source history.
