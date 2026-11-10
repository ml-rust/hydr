# Rectangular channel flow

Use `ChannelReach<R>` to advance flow along an unbranched rectangular channel. Start with explicit cell geometry and endpoint conditions.

## Construct a reach

Import these types from `hydr::channel`.

1. Order `RectangularChannelCell` values along the channel's positive direction.
2. Supply each cell's length, width, bed elevation, maximum supported depth, and Manning roughness.
3. Create `ChannelReachDefinition` with a common vertical datum, geometry evidence, and `ChannelConfig`.
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

Supply both ends through `ChannelBoundaryInterval`. Its states remain constant over the supplied time interval.

| Policy | Required input | Meaning |
| --- | --- | --- |
| `Wall` | A genuine impermeable end | Reflects flow and transfers no water across that end |
| `Exterior` | Depth and longitudinal velocity | Describes exterior water using the endpoint cell's width and bed |

Exterior depth uses metres above the local bed. Exterior velocity uses m/s in the channel's positive direction.

An exterior state does not prescribe achieved discharge. The numerical interface calculation determines that discharge from both sides.

Missing outlet information requires corrected inputs. A wall cannot replace an unknown outlet.

## Advance to a requested time

1. Read the accepted time with `reach.time()`.
2. Create a fresh interval whose `start_s` equals that returned value exactly.
3. Supply `end_s` through which both endpoint states remain valid.
4. Call `reach.step(requested_end_s, &interval)` within that support.
5. Replace the reach with the returned `.next` and repeat until it reaches your target.

Each call accepts one interval. Wave speeds, donor water, and `max_step` can shorten it.

When endpoint inputs change, end their support at that change. Supply new states for the next interval.

An error leaves the original reach unchanged. Hydr retries complete calculations without clipping negative water or scaling individual fluxes.

## Read accepted results

| Method | Result |
| --- | --- |
| `state()` | Cell volume, depth, discharge, and longitudinal momentum, with optional stage and velocity |
| `ledger()` | Current storage, cumulative endpoint inflow and outflow, and the water residual |
| `last_step()` | The last accepted interval, shared face volumes, momentum transfers, pressure corrections, and friction changes |

Before the first accepted step, `last_step()` returns `None`.

Dry results contain zero depth and an explicit dry flag. Stage and velocity are absent where no represented water exists.

Face transfers follow increasing cell order. The same water volume enters one adjacent cell and leaves the other.

Tiny pressure or momentum products can round to zero. Receipts report their underflow bounds, which consume the existing momentum tolerance.

Tiny computed water transfers can also round to zero. Per-face bounds consume the existing interval and cumulative water allowances.

Interior faces count twice in the cumulative bound because they affect two cells. Neither rounding policy removes stored water.

These bounds exclude other rounding and discretization errors. Hydr rejects an interval when its charge exceeds the remaining allowance.

State and detailed receipt methods copy arrays from the device and return errors if that copy fails. Advancement keeps arrays on the selected runtime.

## Save and restore

Construct the destination with the same definition, config, and original initial state. Restoration returns a new reach.

| Use | Save | Restore |
| --- | --- | --- |
| In-memory checkpoint on the same device | `snapshot()` | `restore(&snapshot)` |
| Application-owned persistence | `snapshot_data()?` | `import_snapshot(&data)?` |

Both restoration paths retain the destination's configured client. Importing saved data checks consistency by recalculating the last accepted interval.

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
- Connections: Lateral inflows, junctions, structures, and exchange with other components remain unsupported.
- Losses: Separate abrupt contraction and expansion losses remain unsupported.
- Drainage: This solver supplies no SWMM drainage, surcharge, or pressurized-pipe model.
- Accuracy: Read [acceptance coverage](acceptance.md) for numerical cases and limits. Numerical agreement establishes no measured-event accuracy.
