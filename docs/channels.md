# Rectangular channel flow

Use `ChannelReach<R>` to advance flow along an unbranched rectangular channel. Start with explicit cell geometry and endpoint conditions.

## Construct a reach

Import these types from `hydr::channel`.

1. Order `RectangularChannelCell` values along the channel's positive direction.
2. Supply each cell's length, width, bed elevation, maximum supported depth, and Manning roughness.
3. Create `ChannelReachDefinition` with a common datum, geometry evidence, `lateral_bindings`, and `ChannelConfig`.
4. Supply `ChannelInitialState`, then call `ChannelReach::new(client, definition, initial)` with your configured Numr client.

The geometry describes physical rectangles. The solver does not convert surveyed profiles into rectangles.

Initial arrays contain one value per ordered cell:

| Quantity | Input | From depth and velocity |
| --- | --- | --- |
| Water volume | `volume_m3`, in m³ | Width × length × depth |
| Longitudinal momentum divided by water density | `longitudinal_momentum_m4_per_s`, in m⁴/s | Water volume × velocity |

Velocity is positive along the cell order. A dry cell requires zero volume and zero momentum.

The maximum depth limits the supported geometry. Hydr rejects unresolved positive water and depths above that limit.

## Choose both endpoint conditions

Supply both ends through `ChannelForcingInterval`. Its endpoint states remain constant over the supplied time interval.

| Policy | Required input | Meaning |
| --- | --- | --- |
| `Wall` | A genuine impermeable end | Reflects flow and transfers no water across that end |
| `Exterior` | Depth and longitudinal velocity | Describes exterior water using the endpoint cell's width and bed |

Exterior depth uses metres above the local bed. Exterior velocity uses m/s in the channel's positive direction.

An exterior state does not prescribe achieved discharge. The numerical interface calculation determines that discharge from both sides.

Missing outlet information requires corrected inputs. A wall cannot replace an unknown outlet.

## Declare lateral water inputs

Lateral inputs add or remove water at named channel cells. Declare each connection through `ChannelReachDefinition.lateral_bindings`.

1. Assign each binding a unique `boundary_id` and an existing `cell_id`.
2. Select its fixed direction, `Inflow` or `Withdrawal`.
3. Supply an `ownership_assertion` explaining why this water does not duplicate another source.
4. Include exactly one matching `ChannelLateralInput` per binding in every interval's `lateral_inputs`, including zeros.

Hydr rejects missing, duplicate, unknown, and direction-mismatched inputs. The ownership assertion records your evidence without proving geographic coverage.

| Direction | Velocity input | Momentum effect |
| --- | --- | --- |
| `Inflow` | Explicit `longitudinal_velocity_m_per_s` | Adds incoming volume × supplied velocity |
| `Withdrawal` | None | Removes volume at the cell's accepted-start velocity |

Positive incoming velocity follows the channel's cell order. A dry donor cannot supply a positive withdrawal.

Face outflows and lateral withdrawals share the same donor budget. Incoming face water and lateral inflow provide no donor credit during that interval.

An empty `lateral_bindings` vector selects a reach without lateral inputs. Supply an empty `lateral_inputs` vector for that reach.

## Choose a rate or an interval volume

Each lateral input's `amount` uses `hydr::forcing::WaterAmount`. All amounts are nonnegative.

| Variant | Supplied quantity | Shorter interval |
| --- | --- | --- |
| `Rate { rate_m3_per_s }` | Constant flow in m³/s | Hydr recalculates volume from the accepted duration |
| `IntervalVolume { volume_m3 }` | Integrated volume in m³ for the exact span | Caller recomputes every contributing input |

An interval can mix rates and integrated volumes. Any integrated volume, including zero, requires exact start and requested-end time bits.

For integrated inputs, set `start_s` from `reach.time()` and `end_s` to the requested endpoint. Hydr retains volume operands without conversion through rates.

Hydr rejects positive source contributions that disappear during volume preparation or state assembly.

If a shorter interval is necessary, Hydr returns `ChannelError::RecomputeInterval` without accepting state.

1. Read `suggested_end_s` from that error.
2. Recompute every contributing source for the suggested span.
3. Submit fresh complete forcing with that endpoint.

Never scale or slice original integrated inputs. Successful integrated forcing reaches its requested endpoint with zero internal retries.

## Advance to a requested time

1. Read the accepted time with `reach.time()`.
2. Create a fresh interval whose `start_s` equals that returned value exactly.
3. Supply `end_s` through which all forcing remains valid, respecting exact-volume span requirements.
4. Call `reach.step(requested_end_s, &interval)` within that support.
5. Replace the reach with the returned `.next` and repeat until it reaches your target.

Each successful call accepts one interval. Rate-only forcing supports automatic shortening for wave speeds, donor water, and `max_step`.

When endpoint inputs change, end their support at that change. Supply new states for the next interval.

An error leaves the original reach unchanged. Hydr retries complete calculations without clipping negative water or scaling individual fluxes.

### How sources affect the timestep

Transport uses the accepted-start state. Lateral inflow changes transport on the following interval.

Nonzero lateral inputs also trigger a wave-speed check on the complete candidate state before friction. Candidate fluxes never supply current transport.

This restriction checks start and candidate endpoint states. It does not prove stability along a continuous source path.

Complete zero lateral inputs preserve the original flow arithmetic and timestep sequence. Integrated zeros still require exact spans and recomputation when shortened.

## Read accepted results

| Method | Result |
| --- | --- |
| `state()` | Cell volume, depth, discharge, and longitudinal momentum, with optional stage and velocity |
| `ledger()` | Storage, separate endpoint and lateral totals, the water residual, and cumulative numerical bounds |
| `lateral_ledgers()` | Cumulative accepted water and counter-rounding bounds for each lateral binding |
| `last_step()` | Shared face transfers, pressure and friction changes, wave limits, and receipts for each lateral binding |

Before the first accepted step, `last_step()` returns `None`.

Dry results contain zero depth and an explicit dry flag. Stage and velocity are absent where no represented water exists.

Face transfers follow increasing cell order. The same water volume enters one adjacent cell and leaves the other.

Lateral receipts retain original inputs and accepted volumes. They report applied velocity, signed momentum change, and cumulative counter additions.

A zero withdrawal from a dry cell has no applied velocity.

### Read water and momentum limits

Water balances include lateral inflow and withdrawal. Lateral totals derive from per-binding counters, without competing copies of physical water.

| Numerical bound | What it covers | Existing allowance |
| --- | --- | --- |
| Computed water transfers | Face products rounded to zero, counting interior faces twice | Interval and lifetime water tolerance |
| Lateral water assembly | Source aggregation and additions to cell water | Interval and lifetime water tolerance |
| Lateral counters | Cumulative volume additions for each binding | Lifetime water tolerance |
| Momentum | Pressure/momentum products rounded to zero and lateral assembly | Separate cell momentum tolerance |

Bounds consume the configured tolerance. They never enlarge the physical water scale or remove stored water.

Hydr rejects intervals whose residuals and bound charges exceed the allowance. Water tolerance cannot admit an excessive momentum error.

These bounds cover recorded arithmetic effects only. They do not bound the full numerical solution error.

State and detailed receipt methods copy arrays from the device and return errors if that copy fails. Advancement keeps arrays on the selected runtime.

## Save and restore

Construct the destination with the same definition, config, and original initial state. Restoration returns a new reach.

| Use | Save | Restore |
| --- | --- | --- |
| In-memory checkpoint on the same device | `snapshot()` | `restore(&snapshot)` |
| Application-owned persistence | `snapshot_data()?` | `import_snapshot(&data)?` |

Both restoration paths retain the destination's configured client. Importing saved data recalculates the complete last interval, including lateral inputs and counters.

Persistence must preserve every floating-point bit. Step bounds can contain positive infinity, which ordinary JSON numbers cannot represent.

Exact continuation requires the same build, dependency sources, runtime arithmetic, and sequence of accepted timesteps. Different intermediate targets can change those steps.

Import checks continuation consistency. The application owns content hashes and the authenticity of earlier history.

## Understand the numerical method

The solver uses first-order finite volumes and forward Euler time integration. It represents one depth-averaged longitudinal velocity.

- Pressure: [Roberts–Wilson reconstruction](https://journal.austms.org.au/ojs/index.php/ANZIAMJ/article/view/3948/1542) balances bed and width forces for stationary water.
- Transport: Hydr's existing HLL calculation supplies one shared flux per face, including dry fronts.
- Momentum: The velocity profile is uniform across each section (`β = 1`).
- Step size: Wave limits include changing widths. A separate donor limit counts every outgoing water rate.
- Friction: Split Manning resistance uses water area divided by wetted perimeter, including the rectangular side walls.

Numr supplies runtime storage and tensor operations. Solvr supplies generic root solving within the reused HLL calculation.

## Understand the physical limits

The solver represents hydrostatic, one-dimensional shallow-water flow. It excludes transverse jets and strong channel curvature.

- Geometry: Width and bed can vary between rectangular cells. General surveyed profiles currently support [static queries](cross-sections.md) only.
- Boundaries: Only walls and complete exterior states are supported. Scalar stage/discharge and transmissive channel policies remain unsupported.
- Connections: Junctions, structures, and coordinated exchange with other components remain unsupported.
- Losses: Separate abrupt contraction and expansion losses remain unsupported.
- Drainage: This solver supplies no SWMM drainage, surcharge, or pressurized-pipe model.
- Accuracy: Read [acceptance coverage](acceptance.md) for numerical cases and limits. Numerical agreement establishes no measured-event accuracy.
