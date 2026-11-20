# Connect channels to finite pools

Use `ChannelPoolBatch<R>` to advance channel reaches and storage pools together. Start by declaring each physical connection.

Pool levels respond to the channel's actual water transfers. Multiple channel endpoints can share one pool.

Read [channel flow](channels.md) and [level pools](level-pool.md) before constructing their inputs.

To add a surface, structures, or sampled controls to channel and pool connections, use the [surface/channel coordinator](surface-channels.md#couple-a-reach-end-to-a-pool). It joins a reach end to a pool through a pool end and accepts every child over one common interval. This page covers reach-to-pool connections without a surface.

Read capabilities with `channel_pool_capabilities()`. It returns a `ComponentCapabilities` declaration: runtimes, connected endpoints, regimes, sources, nested components, unsupported uses, and limits.

## Construct the connected system

Import the coordinator types from `hydr::coupling`.

1. Create named `ChannelPoolReach` entries using supported channel geometry.
2. Create `LevelPoolElement` entries with storage curves in the same vertical datum.
3. Declare each endpoint-to-pool connection in `ChannelPoolConnection`.
4. Supply physical connection evidence and the explicit entrance-loss coefficient `K` for each connection.
5. Create `ChannelPoolSetup` with explicit controls and evidence excluding duplicated channel and pool storage.
6. Supply `ChannelPoolInitialState` in setup order, with one common initial time.
7. Call `ChannelPoolBatch::new(client, setup, initial)` with your configured Numr client.

Each endpoint has one owner. Both endpoints of a reach can connect to pools.

Connection beds come from the channel geometry. A surveyed reach takes the minimum elevation of its section as the bed. The pool curve must represent water storage at each connection bed.

All reaches use the same gravity value. Their initial times match the common initial time exactly.

### Set explicit controls

`ChannelPoolConfig` has no default. Choose allowances in physical units before running the calculation.

| Controls | Purpose |
| --- | --- |
| `head_absolute_m`, `head_relative` | Limit changes in the iterated pool head. The relative scale uses local water depths. |
| `transfer_absolute_m3`, `transfer_relative` | Limit changes in the iterated signed water transfers. |
| `mass_absolute_m3`, `mass_relative` | Limit water-accounting residuals, including bounded arithmetic errors. |
| `minimum_step_s`, `maximum_step_s` | Bound the common interval length. |
| `iteration_limit`, `retry_limit` | Bound iteration and complete recalculations with shorter intervals. |
| `storage` | Supply the underlying level-pool controls. |

Each absolute/relative pair requires a positive combined allowance. Channel controls still govern each reach's numerical checks.

## Supply external forcing

Create one `ChannelPoolReachForcing` record per reach inside `ChannelPoolForcingInterval`. Reach IDs identify records, so their order can differ.

| Input | What you supply |
| --- | --- |
| Connected endpoint | `None`. The coordinator supplies the pool head. |
| Unconnected endpoint | `Some(ChannelBoundary)`, with an explicit physical policy. |
| Declared lateral binding | One matching `ChannelLateralInput`, including explicit zero inputs. |
| Interval start | `start_s = batch.time()` |
| Rate-only interval end | `end_s >= requested_end_s` |
| Integrated interval end | `end_s = requested_end_s`, including when every integrated volume is zero |

Never supply an external condition at a connected endpoint. Never use a wall to replace missing outlet data.

For inputs already on the device, use `ChannelPoolTensorForcingInterval<R>` and `trial_tensor_to`. Native numerical parameters follow [channel tensor forcing](channels.md#supply-inputs-already-on-the-device).

The coordinator copies caller operands into private device storage. Later caller-buffer changes cannot modify accepted history.

## Advance one common interval

1. Read `batch.time()` and create forcing starting at that exact value.
2. Call `trial_to(requested_end_s, &forcing)` or `trial_tensor_to`.
3. Read the returned trial's `time()` and `diagnostics()`.
4. Replace the batch with `trial.into_next()` when accepting the candidate.
5. Repeat from the returned time until reaching your target.

One successful call accepts one common interval. Every reach and pool reaches the same accepted endpoint.

Rate-only forcing permits automatic shortening. Hydr recalculates the complete system when a child or coupling check requires a smaller interval.

An error leaves the original batch unchanged. Hydr never scales individual connection transfers to force acceptance.

### Recompute integrated inputs after shortening

An integrated lateral volume belongs to its exact supplied interval, including when that volume is zero.

If a smaller interval is required, Hydr returns `ChannelPoolError::RecomputeInterval`.

1. Read `suggested_end_s` from the error.
2. Recompute every contributing input over the suggested interval.
3. Submit fresh complete forcing from the unchanged accepted start.

Never slice or proportionally scale old integrated inputs. A successful integrated-input trial reaches its requested endpoint.

End forcing support at every known input discontinuity. Submit the changed inputs for the following interval.

## Read transfers and water accounting

Use `last_step()?` for the final common interval and `ledger()?` for lifetime water accounting.

| Result | Meaning |
| --- | --- |
| `transfers[].signed_volume_m3` | Positive channel-to-pool, negative pool-to-channel. Each side uses the same original transfer. |
| `transfers[].midpoint_pool_elevation_m` | Pool head actually supplied to the retained channel trial. |
| `transfers[].longitudinal_momentum_impulse_m4_per_s` | Connection contribution to channel momentum, including pressure, in the reach's positive direction. |
| `connection_ledgers()?` | Separate cumulative volumes for both directions at each connection. |
| `ledger()?` | Combined channel/pool storage and external water. Internal connection traffic contributes no external inflow. |
| `donor_cuts` | Original-water availability and export demand checked at each connection elevation. |

Pools store water volume, without a directed momentum counter.

A surveyed reach stores each cell volume as a high part and a low part. The combined storage adds both parts. The rounding bound of that sum counts against the mass allowance. Rectangular and triangular reaches have a zero low part and keep the single-part total.

Read child outputs through `reach(id)` and `pools()`. Detailed output calls can return device-transfer errors.

Iteration stages are numerical values. Check `iteration.pool_wet` before interpreting them as physical levels.

## Understand connection limits

The connection treats each pool as approximately level, with negligible approach velocity. Pool head means its water elevation.

| Direction | Physical assumption |
| --- | --- |
| Pool into channel | Pool head supplies channel velocity head and the explicit entrance loss `K`. |
| Channel into pool | Complete exit mixing dissipates channel kinetic energy. |

`K` must be finite and nonnegative. `K=0` explicitly selects an ideal entrance.

Interior and connection flow must remain wet and strictly subcritical. Subcritical flow travels slower than its shallow-water wave speed.

### Reserve original donor water

Each connection can export only eligible water present at the accepted start. Incoming transfers provide no same-interval donor credit.

At each connection bed, Hydr checks combined exports through connections at that elevation or higher.

Lower connections retain access to eligible deeper water. Hydr does not restrict every export to water above the highest connection.

Active connections must remain wet throughout the accepted interval. A connection-bed crossing requires shortening or rejection.

### Stay within the supported scope

| Supported | Excluded |
| --- | --- |
| Rectangular, prismatic triangular, and surveyed reaches | General channel junctions |
| Multiple declared endpoints per pool | Inferred geographic connections or overlapping represented storage |
| Exchange-only pools and channel lateral inputs | Pool structures, local pool sources, and surface coordination. The [surface/channel coordinator](surface-channels.md) provides them. |
| Reversing flow inside the wet-subcritical branch | Dry connections, critical transitions, jets, and oscillating pool surfaces |

Time integration remains first order. The local connection law establishes no energy guarantee for the complete numerical calculation.

Numerical agreement establishes no measured-event accuracy.

Read [finite channel/pool checks](acceptance.md#finite-channelpool-checks) for feedback comparisons, oscillation refinement, and current runtime acceptance.

## Save the whole connected state

Construct the destination with the same setup and original initial state.

| Use | Save | Restore |
| --- | --- | --- |
| In-memory continuation | `snapshot()` | `restore(&snapshot)` |
| Application-owned persistence | `snapshot_data()?` | `import_snapshot(&data)?` |

Both restoration methods return a new batch and retain the destination client. Handle an error before replacing your accepted batch.

Import recalculates the final common interval to check shared heads and transfers. Separate child checkpoints cannot establish this connected consistency.

Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `CHANNEL_POOL_SNAPSHOT_SCHEMA` and `CHANNEL_POOL_CONTRACT_VERSION`. `import_snapshot` compares both before any other check and rejects a foreign value with an input error. Next it checks every nested package before any replay:

- `reaches[i]` channel packages.
- The `pools` level-pool package.
- `last_interval.previous.reaches[i]` and `last_interval.previous.pools`.

A nested error names its path and, for a reach, the reach ID. The error names the found and expected values. Capture a new package to recover. Packages without the identity fields are rejected.

Preserve every package field and floating-point value exactly. Follow the [persistence and reproducibility requirements](integration.md#save-complete-restart-data).
