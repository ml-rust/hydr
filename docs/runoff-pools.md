# Connect runoff to storage pools

Use `RunoffPoolBatch<R>` to advance subcatchment runoff and connected pools together. First check that receiving water cannot affect runoff generation.

Read capabilities with `runoff_pool_capabilities()`. It returns a `ComponentCapabilities` declaration: runtimes, the outlet endpoint, sources, nested components, unsupported uses, and limits.

## Check the physical connection

A free runoff outlet exports water without feedback from the receiving pool's water level. Declare this condition for every connection.

If pool levels can slow or reverse runoff, this component does not represent that interaction. It also supplies no dynamic river, pipe, or surface-flow connection.

| Required condition | What you must supply |
| --- | --- |
| Independent runoff outlets | Evidence supporting free drainage without downstream hydraulic feedback |
| Separate water ownership | Evidence that runoff areas do not overlap represented pool storage or local pool rainfall |
| Complete destinations | One pool boundary for every potential runoff outlet |
| No other incoming water | A physical model without external runon or additional prescribed pool inflows |

IDs and written assertions do not prove geometry or hydraulic independence. Do not treat missing incoming-water data as evidence of absence.

## Define the children and outlet bindings

`RunoffPoolSetup` holds the existing child definitions and their numerical controls. Read [subcatchments](subcatchment.md) and [coupled pools](coupled-level-pools.md) before constructing it.

1. Supply subcatchment definitions in a fixed order through `subcatchments`.
2. Supply their native numerical controls through `runoff_config`.
3. Supply the connected-pool setup through `pools`.
4. Map potential runoff outlets through `outlets`.
5. Supply common step and water-balance controls through `config`.

### Identify every potential outlet

Each outlet identifies one of a subcatchment's three subareas. It needs a binding when both conditions hold:

- Its represented area is positive.
- Its internal routing leaves some export for the component outlet.

Absent subareas and fully routed donors need no invented physical outlets. Their native outlet values must remain zero.

Current dryness, depression storage, or infiltration does not remove a potential outlet. The model retains all existing native routing modes.

Every potential outlet maps exactly once to one declared pool boundary. Every pool boundary requires one runoff route. Duplicate, missing, and unknown mappings reject.

### Supply the physical evidence

Use `RunoffPoolOutletBinding` for each connection:

| Field | Meaning |
| --- | --- |
| `subcatchment_id` | Exact source definition identity |
| `subarea` | Native subarea supplying the outlet |
| `boundary_id` | Declared boundary in the connected-pool setup |
| `free_outlet_assertion` | Evidence supporting drainage without receiving-pool feedback |
| `nonoverlap_assertion` | Evidence separating runoff support from pool storage and local rainfall |

Existing pool boundary definitions retain their receiving pool, selected input identity, and ownership evidence.

## Supply an explicit initial state

`RunoffPoolInitialState` contains one common `time_s`, ordered subcatchment states, and ordered pool states.

Supply complete native soil memory where the runoff model requires it. Do not infer dry storage or initialized soil from missing observations.

Construct `RunoffPoolBatch::new(client, setup, initial)` using the configured Numr client. Both child engines inherit that client and runtime. Each child keeps its own arithmetic-admission requirements.

The public types live under `hydr::coupling::runoff_pools` and are also re-exported by `hydr::coupling`.

## Prepare complete forcing

`RunoffPoolForcingInterval` covers one declared interval through `start_s` and `end_s`. Supply coverage for the accepted time and requested endpoint.

| Field | Required input |
| --- | --- |
| `subcatchments` | One `RunoffPoolMeteorology` record per subcatchment, in definition order |
| `pool_sources` | Complete `LevelPoolSources` records when pool source policies are enabled, otherwise `None` |

Each meteorology record supplies finite nonnegative values:

| Field | Units |
| --- | --- |
| `rain_rate_m_per_s` | Metres/second |
| `potential_evaporation_m_per_s` | Metres/second |

Supply explicit zeros for known zero forcing. Split forcing at changes in the supplied rates.

This API excludes external runon. It supplies explicit zero runon to the native child. Additional prescribed pool inflows and disposal routes are also outside this coordinator's scope.

## Trial one common interval

1. Call `trial_with_forcing_to(end_s, &forcing)` on the accepted batch.
2. Read the candidate's actual `time()` and `diagnostics()`.
3. Check whether the actual endpoint reaches the requested endpoint.

Then choose one outcome:

| Decision | Action |
| --- | --- |
| Accept both children | Consume the trial's `into_next()` to obtain the complete next batch |
| Keep both originals | Discard the trial without accepting it |

A trial can return an earlier endpoint after step limits or retries. Continue from that actual accepted time when the requested horizon remains incomplete.

### How common retries work

The coordinator calculates runoff from accepted hydrology, soil, and routing state. It passes the original outlet tensors into pool forcing for the identical interval.

If the pool trial requires a shorter interval, the coordinator discards both candidates. It recalculates every contributing process from both accepted children.

No intermediate child is accepted. The children do not advance independently through smaller intervals.

Incoming runoff cannot fund earlier pool losses or same-interval pool withdrawals. Existing donor limits remain in force.

### Choose common limits

`RunoffPoolConfig` controls common advancement:

| Field | Meaning |
| --- | --- |
| `minimum_step_s` | Parent minimum trial duration, in seconds |
| `maximum_step_s` | Parent maximum trial duration, in seconds |
| `retry_limit` | Maximum common retries |
| `mass_absolute_m3` | Absolute global water-residual allowance, in m³ |
| `mass_relative` | Relative global water-residual allowance |

The effective minimum is the larger parent/pool minimum. The effective maximum is the smaller parent/pool maximum. An empty range rejects.

Child iteration, conservation, and soil-adjustment budgets remain independent. Common retries cannot enlarge them.

Exhausted common retries return `Unresolved` with the attempted duration, retry count, and reason. Both accepted children remain unchanged.

## Inspect water and continuation state

| API | What you can inspect |
| --- | --- |
| `runoff()` | Read-only native subcatchment batch |
| `pools()` | Read-only connected-pool batch |
| `ledger()` | Derived global water balance |
| `last_step()` | Last common interval, forcing, outlet pairing, and balance |

The trial also exposes read-only `runoff()` and `pools()` views. Child access does not permit separate acceptance.

Detailed outputs and owned checkpoint export copy numerical values on explicit request. Those calls can return errors. Advancement and trusted snapshots avoid copying full numerical arrays to the host.

### Read the global water balance

Global storage includes represented runoff storage and pool storage. The model derives totals from checked child ledgers.

| Process | Global treatment |
| --- | --- |
| Runoff and pool rain | External addition |
| Infiltration, evaporation, and pool bottom loss | External removal |
| Runoff soil adjustment | Explicit numerical water, subject to its independent budget |
| Runoff exported into pools | Paired internal transfer, canceled from external terms |

The same original outlet volume supplies the runoff debit and pool credit. The coordinator never divides these volumes by duration and multiplies them back.

Component ledgers retain their own exports and inflows for inspection. Interval checks use original process volumes, not differences between cumulative counters.

Water-balance checks include rounding errors from sums, transfers, and cumulative counters. Internal transfers and soil adjustments cannot enlarge the relative physical tolerance. Each child retains its own numerical budgets.

## Save and restore both children

| Task | API |
| --- | --- |
| Retain a trusted in-process checkpoint | `snapshot()` |
| Export complete application-owned data | `snapshot_data()` |
| Restore a trusted checkpoint | `restore(&trusted)` |
| Restore owned checkpoint data | `restore_data(&owned)` |

Preserve the immutable setup, both complete child packages, and the last common receipt. No previous composite snapshot chain is required.

Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `RUNOFF_POOL_SNAPSHOT_SCHEMA` and `RUNOFF_POOL_CONTRACT_VERSION`. `restore_data` compares both before any other check and rejects a foreign value with an input error. Next it checks the nested `runoff` subcatchment package and the `pools` coupled level-pool package, with its own nested packages. A nested error names its path, such as `pools: coupled level pool checkpoint pools`. The error names the found and expected values. Capture a new package to recover. Packages without the identity fields are rejected.

Store every field and floating-point bit unchanged. Exact continuation also requires the build, runtime conditions, and accepted-step partition described in [integration](integration.md).

The initial time identifies the ledger epoch. Restoration requires the same setup and exact original initial epoch. Matching the current time alone is insufficient.

Import stages both children using destination clients. It checks common timing, original outlet/input pairing, complete child continuation, and global accounting before replacing either accepted child.

An import error leaves both originals unchanged. Trusted and owned restoration retain the destination clients.

Numerical consistency checks do not authenticate historical observations or provenance. The application owns that evidence.

## Current limits

This coordinator excludes:

- Downstream backwater, tailwater feedback, and reversed runoff through an outlet.
- External runon, additional prescribed pool inflows, and disposal routes.
- Inter-catchment routing graphs and geometric nonoverlap proofs.
- New measured-event accuracy claims or complete SWMM-driver rewind guarantees.

Read [acceptance coverage](acceptance.md) for checked numerical cases and their limits.
