# Connect level pools

Use `CoupledLevelPoolBatch` to advance water between separate pools through declared structures. Start by defining the physical storage and connections.

This guide covers static manual operations with optional local sources and prescribed external inflows. Read [acceptance coverage](acceptance.md#connected-pool-checks) for checked cases and accuracy limits.

For generated upstream inflows, use the [runoff-to-pool coordinator](runoff-pools.md). It owns both children and their common interval.

Read capabilities with `coupled_level_pool_capabilities()`. It returns a `ComponentCapabilities` declaration: runtimes, structure and inflow endpoints, regimes, sources, nested components, unsupported uses, and limits.

## Define the physical model

1. Give each pool its own [stage-volume curve](level-pool.md).
2. Declare separate physical storage without counting the same water twice.
3. Use one vertical reference for each structure and its connected pools.
4. Declare each pool's physical zero-volume elevation and each connection's local bed.
5. Supply evidence supporting the storage and physical connections.

Each pool represents connected, approximately level water. Supply volumes in m³, elevations in metres, and time in seconds.

### Describe dry pools and raised outlets

Zero volume has no reported water elevation. Midpoint estimates use the declared zero-volume limit when a pool wets or dries.

The declared limit must match the storage curve origin. Connection beds must lie at or above that origin. Water below a raised connection cannot supply its outlet.

### Bind structures and manual operations

1. Give pools and structures stable identities.
2. Bind each structure endpoint to its declared pool.
3. Supply one explicit static operation for every structure.
4. Choose timestep limits and separate water-level (head), transfer-volume, and water-balance tolerances.
5. Construct the batch with `new(client, setup, initial)`.

Choose the matching operation from [structure laws](structures.md):

- Passive operation for passive structures.
- Explicit isolated-stop or running operation for pumps.
- Fixed opening for gates.

Hydr supplies no implicit pump operation. The configured client selects the runtime used by the batch.

## Supply local rain and losses

1. Set `CoupledLevelPoolSetup.source_policies` before constructing the batch.
2. Supply one `LevelPoolSourcePolicy` per pool.
3. Declare evidence supporting rainfall area and supplied bottom demand.
4. Supply rates through `CoupledLevelPoolForcingInterval.sources`.

When policies are enabled, `sources` must contain one record per pool. Supply explicit zero rates where a process does not apply. When local sources are disabled, use `None`.

| Source field | Units |
| --- | --- |
| Rain | m/s |
| Potential evaporation | m/s |
| Potential bottom loss | m³/s |

Rates remain constant over the declared forcing interval. Bottom demand supplies a potential loss. It establishes no inferred soil process.

### Choose the area used by sources

Rain and evaporation use the area at the accepted interval start. At an exact storage knot, Hydr selects the right segment.

Dry pools use the basal footprint: their declared rain-receiving area. Source calculations retain this area throughout the attempted interval.

## Declare external inflows

1. Add each `CoupledPoolInflowBinding` to `CoupledLevelPoolSetup.external_inflows`.
2. Identify its receiving pool, boundary, selected input, and physical location or area.
3. Supply boundary evidence and independent ownership evidence.
4. Supply one `CoupledPoolInflow` per boundary in `CoupledLevelPoolForcingInterval.inflows`.

`physical_support_id` names the actual location or area represented by the input.

Boundary IDs and selected input IDs must each be unique. Distinct, independently owned inputs can share a declared physical support.

### Check ownership before supplying water

Your application supplies already-selected flows and physical evidence. Ownership must establish:

- External origin, outside represented pool storage.
- Exclusion of water already represented by local rainfall.
- No duplicated supplied input.

Identity checks establish no geometry. Hydr neither selects measured replacements nor proves GIS overlap or contributing-area ownership.

### Choose a rate or a supplied volume

| Value | Use it for | Units |
| --- | --- | --- |
| `CoupledPoolInflowValue::Rate` | A constant inflow rate over the declared interval | m³/s |
| `IntervalVolume` | A volume already calculated for a specific time interval | m³ |

Values must be finite and nonnegative. Supply explicit zeros for declared boundaries without incoming water.

Record order can differ from binding order. Missing, duplicate, unknown, or mismatched inputs reject. Rate and volume inputs can share one forcing interval.

## Set the time interval

### Rate-only forcing

Set forcing coverage to include the accepted time and requested endpoint.

Endpoints must be finite and the requested duration must be positive. Coverage satisfies `start_s <= batch.time() < requested_end <= end_s`.

Rate-only nonconvergence triggers automatic shortening. Maximum-step limits can also cap the accepted endpoint.

Each shortened attempt recalculates source effects and boundary volumes from the original accepted state.

### A volume already calculated for an interval

1. Set forcing `start_s` from `batch.time()`.
2. Set forcing `end_s` from the requested endpoint.
3. Calculate every supplied volume for that complete span.
4. Submit this forcing at the same requested endpoint.

Hydr checks exact floating-point bit equality for both endpoints. This rule also applies to zero supplied volumes.

Hydr retains the original volume operand. It never divides a supplied volume by duration and multiplies it back.

## Advance and handle the outcome

Use `trial_to` only when local sources and external inflows are absent. Otherwise, use `trial_with_forcing_to`.

Both methods calculate a candidate without changing the original accepted state.

| Outcome | Next action |
| --- | --- |
| Successful trial | Read the actual endpoint. Consume `into_next` to retain the candidate. |
| Shortened rate-only interval | Continue from the actual accepted time. The requested endpoint remains incomplete. |
| `RecomputeInterval` | Accepted state stays unchanged. Follow the [fresh-forcing procedure](#respond-to-a-recalculation-request). |
| Rejected interval | Read the error. The original accepted state remains unchanged. |

### Respond to a recalculation request

Integrated inputs require `RecomputeInterval` when a maximum step or retryable calculation requires a smaller interval.

1. Read the suggested endpoint.
2. Recompute every contributing input from the accepted state for that span.
3. Submit fresh forcing with exact start and end bits.

Never scale, slice, or reuse the old integrated inputs for a shorter span.

Suggestions lie strictly between the accepted time and requested endpoint. Their durations meet the minimum-step limit. An impossible smaller interval returns `Unresolved`.

A successful integrated interval reaches its requested endpoint with zero internal retries. Terminal input and runtime errors remain errors.

### Understand the numerical checks

Hydr estimates structure flow from midpoint heads and repeats the calculation until separate head and transfer-volume tolerances pass.

A support flag records whether water reaches a pool or connection. These flags must agree exactly between iterations.

Iteration, retry, and minimum-step limits stop unresolved calculations.

## Understand when water becomes available

Hydr applies each attempted interval in this order:

1. Add local rain.
2. Apply bottom loss, capped by available water.
3. Apply evaporation, capped by remaining water.
4. Allocate structure withdrawals against post-source storage.
5. Credit incoming structure transfers and external water.

Local rain can supply a withdrawal during the interval. Incoming structure transfers and external water cannot fund earlier withdrawals or local losses.

Original accepted heads remain the starting anchor for midpoint estimates.

### Interpret transfer and water-balance results

Each structure retains one signed transfer volume. Positive volume moves from endpoint A to endpoint B. Both endpoints use that same volume.

Competing outlets share their donor's post-source available water.

- Positive films: Represented positive water remains positive. Conservative arithmetic can leave a small film in a mathematically exhausted donor.
- Unresolved transfers: Positive transfers return an error if aggregation cannot represent them.
- Rounding bounds: Bounds describe arithmetic uncertainty. Hydr adds no water to close a balance.

`external_inflows()` returns cumulative external volumes and rounding bounds by boundary. Native incoming counters also contain structure transfers. Never infer external water by subtracting native incoming and outgoing counters.

## Save and restore a complete checkpoint

| Task | API |
| --- | --- |
| Continue within a running process | `snapshot()` and `restore()` |
| Store application-owned checkpoint data | `snapshot_data()` and `restore_data()` |

Trusted snapshots retain numerical arrays on the selected runtime. Owned checkpoint data copies those arrays into application-owned values. Copying data can return errors.

Restore checks the physical definition and numerical controls. It retains the destination's configured client.

Preserve every checkpoint field and floating-point bit when encoding data. Complete forcing checkpoints retain:

- Physical definitions, source policies, and external bindings.
- Cumulative process counters and boundary counters.
- Original forcing and one bounded start frame with pool state and external ledgers.

Import recalculates the final interval's forcing. It checks transfers and accepted state before restoration.

Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA` and `COUPLED_LEVEL_POOL_CONTRACT_VERSION`. `restore_data` compares both before any other check and rejects a foreign value with an input error. Next it checks the nested `pools` and `start.pools` level-pool packages. A nested error names its path, such as `start.pools`. The error names the found and expected values. Capture a new package to recover. Packages without the identity fields are rejected.

## Unsupported uses and support limits

### Hydraulic support still applies

- Running pumps require a wet donor. The receiver must be wet, or the pump must declare a free outlet at or above the dry receiver's bed.
- Submerged orifices require water above the opening's top elevation at both endpoints.
- Positive water requires a represented positive midpoint depth.

A dry pool receiving the smallest positive floating-point volume cannot represent that midpoint and rejects. Exact volume retention does not bypass this check.

A hydraulic cutoff is an elevation where the permitted flow regime changes.

Bed or hydraulic-cutoff transitions require bounded shortening or fresh integrated inputs. Unresolved transitions return errors.

Connections can become unavailable at different elevations. Availability scaling does not locate those separate closure events. Those events require further event localization.

### These capabilities remain excluded

- External withdrawals and runoff feedback from receiving-pool levels.
- Measured-replacement selection.
- Control schedules, trajectories, and actuator frames.
- Surface-to-network coupling.

Read [acceptance coverage](acceptance.md) for numerical evidence. Synthetic checks establish neither measured-event accuracy nor universal GPU qualification.
