# Couple the surface to channel reaches

Use `SurfaceChannelBatch<R>` to advance the 2D surface engine, 1D channel reaches, and level pools over one common interval. Water crosses a bank weir between a surface cell and a channel cell, in either direction. A structure joins any two of a surface cell, a channel cell, and a pool. Water and momentum cross a reach-end port between a line of surface faces and one reach end. A reach end can also meet a pool. Subcatchment runoff can drain into surface cells, reach cells, and pools.

Read [channel flow](channels.md), [numerics](numerics.md), and [structure laws](structures.md#orifices-and-rectangular-weirs) before constructing the inputs. Read [level pools](level-pool.md) and [coupled level pools](coupled-level-pools.md) before declaring pools. Read [controlled gates](controlled-gates.md), [controlled pumps](controlled-pumps.md), and [gate actuators](gate-actuators.md) before declaring operating children.

Read capabilities with `surface_channel_capabilities()`. It returns a `ComponentCapabilities` declaration: runtimes, endpoints, regimes, sources, nested components, unsupported uses, and limits.

## What the coordinator couples

| Part | Supplied as | Notes |
| --- | --- | --- |
| Surface | `SolverSetup` and `InitialState` | Exterior faces, land model, and surface forcing keep their surface rules. |
| Reaches | `SurfaceChannelReach` entries | Rectangular, triangular, or surveyed geometry. |
| Links | `SurfaceChannelLink` entries | One bank weir between one active surface cell and one channel cell. |
| Ports | `SurfaceChannelPort` entries | One line of surface faces against one reach end of rectangular section. See [reach-end ports](#couple-a-reach-end-to-the-surface). |
| Pools | `pools`, `pool_support`, `pool_sources`, `pool_inflows`, `storage` | Level pools with their sources and external inflows. See [level pools](#add-level-pools). |
| Structures | `SurfaceChannelStructure` entries | One weir, orifice, gate, pump, or culvert between two nodes. See [structures](#join-nodes-through-structures). |
| Pool ends | `SurfaceChannelPoolEnd` entries | One reach end against one pool. See [pool ends](#couple-a-reach-end-to-a-pool). |
| Operating children | `SurfaceChannelOperating` | The children that operate gates and pumps. See [controls](#operate-gates-and-pumps). |
| Runoff | `SurfaceChannelRunoff` | Subcatchments whose outlets drain into surface cells, reach cells, and pools. See [native runoff](#attach-native-runoff). |

Links, ports, structures, pool ends, and runoff together must hold at least one entry. A setup with none rejects.

A reach end has one owner: a caller policy, a port, or a pool end. The coordinator owns a ported end and a pool end.

### Couple a surveyed reach

A surveyed reach joins the coordinator through bank links, structure links, ports, and pool ends.

- Endpoints: an end that no port or pool end owns takes any channel endpoint policy, as in a standalone reach. See [channel endpoints](channels.md#choose-both-endpoint-conditions).
- Port: only a surveyed rectangle takes a port. Its width is the distance between the first and last bottom points of the profile. See [reach-end ports](#couple-a-reach-end-to-the-surface).
- Pool end: the bed is the section minimum. A dry end cell at the accepted start rejects with `pool-end-regime`.
- Structure link: a structure endpoint can name a surveyed channel cell.
- Bed: a surveyed cell's bed is the minimum elevation of the reach's surveyed section. The crest check and the reported stage use this bed.
- Stage: the bed plus the depth from the reach's certified area-to-stage recovery.
- Donor water: the lower bound of the two-part stored volume, as the reach's own step uses.

Surveyed reaches keep the [surveyed limits](channels.md#set-surveyed-numerical-controls). The CUDA runtime evidence also covers a surveyed rectangle through a port with a bank link, and a three-band reach through a pool end.

A reach without links advances on the same common interval with its own forcing.

## Declare each link

Import the coordinator types from `hydr::coupling`.

| Field | Meaning |
| --- | --- |
| `id` | Unique nonempty identity. Ledgers and synthesized bindings use it. |
| `surface_cell` | Active row-major surface cell index |
| `reach_id`, `channel_cell` | Linked setup reach and its ordered cell index |
| `crest_elevation_m` | Crest in the common vertical datum, at or above both beds |
| `crest_length_m`, `weir_coefficient` | Finite positive weir length, in m, and coefficient, in m^(1/2)/s |
| `applicability` | `WeirApplicability` evidence, with `symmetric_reversal = true` |
| `reach_tangent` | Unit horizontal tangent of the reach's increasing coordinate, in surface x/y axes |
| `mapping_evidence` | Your evidence that both cells share this bank location |

`SURFACE_CHANNEL_TANGENT_TOLERANCE` bounds `|hypot(reach_tangent) - 1|`.

`SurfaceChannelSetup.storage_exclusion_evidence` records why surface cells and channel cells store no water twice. Hydr records this evidence. It proves no geographic separation.

### Construction checks

`SurfaceChannelDefinition::new` and `SurfaceChannelBatch::new` return `SurfaceChannelError::Input` for each case:

- Crest below the surface bed or the channel bed. An unobstructed connection is unsupported. A surveyed channel bed is the section minimum.
- Repeated link identity, or a repeated `(surface_cell, reach_id, channel_cell)` triple.
- Surface cell outside the grid or inactive.
- Nonfinite reach tangent, or a tangent off unit length.
- Reach vertical datum or gravity bits that differ from the surface.
- Caller lateral binding under the reserved `surface-exchange:` or `structure-exchange:` prefix.
- `applicability.symmetric_reversal = false`. Links carry water both ways.
- Empty evidence or identity text, or text with control characters.
- Initial surface or reach time that differs from the common time bit for bit.

The coordinator appends two bindings per link to each linked reach: `surface-exchange:<link id>:inflow` and `surface-exchange:<link id>:withdrawal`. They follow the caller bindings, in ascending link identity. A structure with a channel endpoint adds `structure-exchange:<structure id>:inflow` and `structure-exchange:<structure id>:withdrawal` the same way. Bank links and structures sort together. Ports and pool ends add no binding. `coordinated_reach(id)` returns the result.

### Set explicit controls

`SurfaceChannelConfig` has no default. Choose allowances in physical units.

| Controls | Purpose |
| --- | --- |
| `head_absolute_m`, `head_relative` | Limit changes in iterated end stages |
| `transfer_absolute_m3`, `transfer_relative` | Limit changes in iterated signed link volumes |
| `mass_absolute_m3`, `mass_relative` | Limit composite water residuals |
| `minimum_step_s`, `maximum_step_s` | Bound the common interval |
| `iteration_limit`, `retry_limit` | Bound iterations and complete recalculations |
| `contraction_target` | Target fixed-point contraction per update, in `(0, 1)`. Sets the shorter retry of a stalled attempt |

Each absolute/relative pair needs a positive combined allowance.

| Rule against the children | Error |
| --- | --- |
| `maximum_step_s` above the surface or a reach `max_step` | Input |
| `minimum_step_s` below the surface or a reach `min_step` | Input |
| `retry_limit` above the surface `max_retries` | Input |

## Exchange law

Each link is a suppressed sharp-crested rectangular weir with the Villemonte submergence correction. [Structure laws](structures.md#numerical-formulas) give the formulas.

The weir uses midpoint heads: the mean of start and end stage on each side. Its discharge times the interval gives the link volume. Positive volumes flow from surface to channel.

| Regime | Heads at the midpoint | Link volume |
| --- | --- | --- |
| `BelowCrest` | No supported side above the crest | Zero |
| `Free` | One side above the crest | Free weir discharge |
| `Submerged` | Both sides above the crest at different heads | Villemonte-reduced discharge |
| `EqualHeads` | Both sides above the crest. The start heads and the zero-exchange end heads each sit within the head allowance | Zero |

The head allowance is `head_absolute_m + head_relative * max(|H_s|, |H_c|)`. The zero-exchange end heads come from one completion of every child without exchange, at the start of each attempt. A link whose heads are level, with nothing else moving them apart, carries no water. When other water moves level heads apart, the Villemonte law carries the exchange.

The capability declaration names these `below-crest`, `free-weir-flow`, and `submerged-weir-flow`. `EqualHeads` counts as submerged flow.

### Stop at level heads

Near level heads, the Villemonte discharge changes without bound. A full weir volume then carries the receiver above the donor within one interval. The levelling cap ends the exchange there instead of the weir law. Capability limit `levelling-cap` records this.

Each attempt first completes the surface and every reach with zero exchange. The end stages `H` of that completion hold every other interval term. Each side then changes stage through a plan area that bounds the change from above:

| Side | Plan area |
| --- | --- |
| Surface cell | Cell area |
| Channel cell | Cell length times water-surface width at the lowest stage the bound covers |

For a link from donor `d` to receiver `r`:

- `O_d` is the donor's summed outgoing weir volume.
- `I_r` is the receiver's summed incoming weir volume.
- `B = max(H_r, z_c)`, with `z_c` the crest.

The link keeps the fraction `θ = min(1, max(θ_below, θ_level, 0))` of its weir volume:

| Term | Formula | Holds |
| --- | --- | --- |
| `θ_below` | `(z_c - H_r) A_r(H_r) / I_r` | The receiver ends at or below the crest |
| `θ_level` | `(H_d - B) / (O_d / A_d(B) + I_r / A_r(B))` | The donor ends at or above the receiver |

A `θ` below one shrinks by 16 units in the last place. Each link then takes the smallest `θ` among the links that share its donor or its receiver. Several links into one cell therefore never cross level together.

A term whose stage lies within one head allowance of its bound keeps nothing. A link whose terms all keep nothing is stopped. Its `θ` is exactly zero, and it leaves the shared minimum and the donor totals of every other link. [Structures](#cap-each-structure-law) add terms for their laws.

The donor limit applies after the cap. `level_factor` reports the applied `θ`. It is exactly `1.0` when the cap does not bind, and the weir volume then keeps every bit.

On the next interval the heads sit within the head allowance. Without other water to move them apart, the link reports `EqualHeads` and carries no water.

### Limit each donor to its water

Each donor gives only water it holds for this interval.

| Donor | Available water |
| --- | --- |
| Surface cell | Post-transport water of the cell |
| Channel cell | Start water, minus native face outflow over the interval, pool-end export, and caller withdrawals, with a rounding margin |
| Pool | Post-source storage: local rain, then capped bottom loss and evaporation |

Donor groups run surface cells, then channel cells, then pools, with pools in ascending identity.

One factor per donor scales every link and structure that donor feeds. When demand exceeds the available water, the factor is `(available - margin) / (demand + bound)`, rounded down once more. Incoming water gives no same-interval donor credit.

### Momentum closure

Bank link water carries momentum along the reach tangent only. Ports carry normal momentum under the [port flux law](#port-flux-law). Structure water enters either side with zero velocity, and jets dissipate. Capability limit `structure-momentum-closure` records this.

| Direction | Velocity carried |
| --- | --- |
| Surface to channel | Surface post-transport velocity projected on `reach_tangent` |
| Channel to surface | Channel accepted-start velocity, along `reach_tangent` in surface x/y |

The normal component dissipates at the bank. Several links into one surface cell carry their summed momentum.

## Couple a reach end to the surface

A port joins one reach end to a line of surface faces. A river mouth that discharges onto a floodplain is one use. Surface flow that enters a reach head is another.

| Field | Meaning |
| --- | --- |
| `id` | Unique nonempty identity, distinct from every link identity |
| `reach_id` | Ported setup reach. It must be rectangular, or surveyed with a rectangular profile. |
| `end` | `ChannelEnd::Left` or `ChannelEnd::Right`. One port binds each end at most. |
| `faces` | Surface faces in any order. Admission sorts them along their face line. |
| `mapping_evidence` | Your evidence that the faces and the reach end share this location |

Each face has one active surface cell. Its other side is an inactive cell or lies off the grid. The faces share one axis, one face line, and one surface side, with no gap. A face is never a caller open face, and it belongs to one port only.

The face widths must sum to the end cell width `b` within `SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE · b`. The tolerance is `16 · f64::EPSILON`. Set the end cell width to the face width sum. A rectangular reach can vary its width by cell. A surveyed rectangle has one width for every cell, its bottom width `b`.

A surveyed profile is a rectangle when its lowest points form one flat run of at least two points. Every point before the run shares the run's first station. Every point after it shares the run's last station. Both end points sit at `maximum_stage_m`. Any other surveyed profile rejects with `port-section-mismatch`. Such a section changes width with depth, so it cannot match a strip of equal-width surface faces.

### Port flux law

Each port face carries one hydrostatic HLL flux, the same flux as any surface face. The reach end cell is the ghost neighbour of every face.

- Reach side: the end cell's accepted-start depth `A / b` and discharge per width `Q / b`.
- Surface side: the cell state after sources, as the surface's own faces use.
- Bed step: hydrostatic reconstruction at the higher of the two beds.
- Timing: the flux depends on start states only. It stays fixed for one attempt and recalculates after a shorter retry.

The reach end takes the summed face flux as a `ChannelBoundary::FluxPort`. The surface CFL limit, the reach CFL limit, and both depletion limits cover the port. A breach shortens the attempt. The coordinator never scales port flux.

| Side | Momentum it receives |
| --- | --- |
| Surface cell | Shared normal flux plus the surface-side pressure, and the shared tangential flux |
| Reach end | Shared normal flux plus the ghost-side pressure, summed over the faces along the outward reach normal |

Water entering the surface carries no tangential velocity. Tangential momentum leaving the surface ends at the junction. `junction_reaction_m4_per_s` reports the difference between both sides: the bed-step reaction and any tangential loss.

### Port regimes

`SurfaceChannelPortTransfer.regime` names the port regime of each interval.

| Regime | Port behaviour |
| --- | --- |
| `Subcritical` | Both waves straddle the port. The flux mixes both states. |
| `SupercriticalToSurface` | Every wave runs into the surface. The flux is the reach state's own flux. |
| `SupercriticalToReach` | Every wave runs into the reach. The flux is the surface state's own flux. |
| `Transcritical` | A rarefaction fan holds its sonic point at the port. |
| `WetDryFront` | One side holds no water above the higher bed. |
| `Dry` | Neither side holds water above the higher bed. The port moves nothing. |

A port takes one regime from its faces:

- Every face `Dry`: the port is `Dry`. Otherwise dry faces drop out.
- Any `WetDryFront` face: the port is `WetDryFront`.
- Else any `Transcritical` face: the port is `Transcritical`.
- Else wet faces that share one regime give that regime. Any other mix gives `Subcritical`.

`reversed` is true when the signed volume changes sign against the previous interval.

### Port forcing and checks

Set `left` or `right` of `SurfaceChannelReachForcing` to `None` at a ported end. Set `Some(policy)` at every other end. Either mistake rejects the trial with `port-endpoint-policy`.

`SurfaceChannelDefinition::new` rejects a port with `SurfaceChannelError::Input`. The message names the capability entry.

| Case | Entry |
| --- | --- |
| Triangular reach end, or a surveyed end whose profile is not a rectangle | `port-section-mismatch` |
| Face width sum off the end cell width | `port-width-mismatch` |
| Bad face set, two ports on one end, or an unknown reach | `port-face-ownership` |
| Empty or repeated port identity, or empty evidence | `duplicate-links`, `missing-link-evidence` |

A ported reach also needs the surface datum and gravity bits, as every reach does. That rejection names the reach and the differing value.

## Add level pools

A pool is a level-pool storage node. One `LevelPoolBatch` holds every pool in ascending identity. `batch.pools()` returns it. Every pool joins at least one structure or pool end.

| Setup field | Meaning |
| --- | --- |
| `pools` | `LevelPoolElement` entries. Their curves use the surface vertical datum. |
| `pool_support` | One `CoupledPoolSupport` per pool, in pool order. The zero-volume stage equals the curve origin. |
| `pool_sources` | One `LevelPoolSourcePolicy` per pool, or `None` without local rain and losses. |
| `pool_inflows` | `CoupledPoolInflowBinding` entries for external inflow into pools. |
| `storage` | `LevelPoolConfig`, the dimensional controls of the pool child. |

`SurfaceChannelInitialState.pools` holds one `LevelPoolState` per pool, in setup pool order. `storage_exclusion_evidence` covers pools as well as surface and channel cells.

[Coupled level pools](coupled-level-pools.md#supply-local-rain-and-losses) define the source policies and the external inflow bindings.

### Supply pool forcing

`SurfaceChannelForcingInterval` carries two pool fields:

- `pool_sources`: one `LevelPoolSources` per pool, in setup pool order, exactly when the setup declares policies. `None` otherwise.
- `pool_inflows`: one `CoupledPoolInflow` per declared binding, zeros included. An integrated volume ties the interval to the requested span, as an integrated lateral volume does.

Capability `tensor-forcing` records that pool forcing is host data only.

### Read the pool water order

Each pool gives water in this order:

1. Local rain.
2. Capped bottom loss and capped evaporation.
3. Withdrawals against the post-source storage.
4. Incoming water.

Incoming link water, pool-end water, and external inflow fund no withdrawal in the same interval.

### Stay within pool capacity

- Source check: post-source water plus external inflow above the top curve knot rejects before iteration.
- Iterates: an iterate above the top knot takes the top stage. The fixed-point map stays continuous.
- Converged candidate: a converged volume above the top knot rejects without retry.

Each rejection is `SurfaceChannelError::Input`. Its message starts with `pool-capacity-exceeded` and names the pool. Declare a spill structure or extend the curve.

Three further pool checks reject and retry on a shorter interval:

- An export through attachments at or above a cut stage exceeds the original water above that stage.
- A transfer crosses a raised bed or a law cutoff.
- A pool end loses wet support over the interval.

## Join nodes through structures

A structure carries one admitted structure law between two nodes. It shares the link table of the bank links. One fixed point, one donor tree, one levelling cap, and one ledger shape serve both.

`SurfaceChannelNode` names a node:

| Node | Identity |
| --- | --- |
| `SurfaceCell(usize)` | `surface-cell:<n>` |
| `ChannelCell { reach_id, cell }` | `reach:<id>:cell:<k>` |
| `Pool(String)` | `pool:<id>` |

A structure joins two distinct nodes. The pairs are surface cell and channel cell, surface cell and pool, channel cell and pool, or two pools. `node.identity()` returns the identity string.

| Field | Meaning |
| --- | --- |
| `element` | `StructureElement` with its identity, law, direction, and datum. Its endpoint strings equal the node identities. |
| `endpoint_a`, `endpoint_b` | `SurfaceChannelAttachment`: the node, `bed_elevation_m`, and `physical_connection_assertion`. Positive volumes move water from A to B. |
| `scenario` | `StructureScenario` with `id` and `evidence`. It names the fixed blockage or condition whose geometry the law parameters hold. |
| `operation` | `SurfaceChannelOperationOwner` |

The laws are `SuppressedRectangularWeir`, `SubmergedOrifice`, `HydrostaticRectangularGate`, `Pump`, and `Culvert`. See [structure laws](structures.md).

An attachment bed sits at or above its node bed, or at or above the curve origin of a pool. A weir crest, gate sill, or orifice sill sits at or above both attachment beds. A pump takes a one-way direction. A bidirectional law needs symmetric reversal evidence.

### Attach a culvert

A culvert joins two nodes as any other structure does. Declare its law with [`CulvertParameters`](structures.md#declare-a-culvert) and the `Passive` operation. The coordinator builds the culvert rows once at construction.

| Rule | Requirement |
| --- | --- |
| Inverts | The invert at each end sits at or above that end's attachment bed. A dry end then acts as a free outfall at its invert. |
| Gravity | The culvert gravity equals the surface gravity bit for bit |
| Direction | `Bidirectional` needs both ends. `AtoBOnly` and `BtoAOnly` need the upstream end. |
| Roadway | A road over the culvert is a separate weir on the same two nodes |
| Client | The client implements `CulvertClient`. Every client that runs this coordinator does. |

A culvert between a channel cell and a pool joins a reach to a pool through a barrel, where a pool end joins it through an open reach end.

A donor at or below its invert gives no water. A one-end culvert whose heads drive flow into its undeclared end reports `CulvertUndeclaredEnd` and moves exact zero.

A blockage is a fixed scenario with evidence. The element's `condition` holds the blockage, and the `StructureScenario` names it. See [structures](structures.md#declare-a-blocked-or-closed-structure). `SurfaceChannelBatch::structure_geometry()` returns the base and effective geometry of each structure in ascending identity order. A `Closed` structure keeps its attachment beds as cutoffs, carries no cap or stop, and moves exact zero volume. A sedimented culvert places its cutoffs and stops above its bed elevations `fl(z + h)`. The coordinator infers no blockage. Capability limit `blockage-as-scenario` records this.

### Choose the operation owner

| Owner | Drives | Operation of each interval |
| --- | --- | --- |
| `Manual(point)` | Any law | The supplied point. A weir or orifice takes `Passive`, a gate takes `GateOpening`, a pump takes `PumpRunning` or `PumpStoppedIsolated`. |
| `Prescribed` | Gate | The midpoint of a linear opening path |
| `Actuated` | Gate | The midpoint of the actuator path |
| `ControlledPump` | Pump | The accepted pump operation at the start, held over the interval |

Each structure has exactly one owner. A dynamic owner binds to a child of [`SurfaceChannelOperating`](#operate-gates-and-pumps).

### Check structure admission

`SurfaceChannelDefinition::new` returns `SurfaceChannelError::Input`. The message names the capability entry.

| Case | Entry |
| --- | --- |
| Element endpoint differs from its node identity, an unknown node, an inactive surface cell, a missing channel cell, an unknown pool, or both endpoints on one node | `structure-endpoint-identity` |
| Element datum differs from the surface datum, or a gate or orifice gravity differs from the surface gravity | `datum-or-gravity-mismatch` |
| Attachment bed below its node bed or pool origin, a crest or sill below either attachment bed, or a culvert invert below its attachment bed | `unobstructed-pool-connection` |
| Bidirectional pump, bidirectional law without symmetric reversal evidence, or bidirectional culvert without both ends | `non-symmetric-structure` |
| Repeated pool, structure, pool-end, or scenario identity | `duplicate-links` |
| Empty evidence, scenario, or identity text | `missing-link-evidence` |
| Pool support, source policies, or inflows that differ from the pools, a repeated support, a zero-volume stage off the curve origin, or a pool that joins nothing | `pool-support` |
| Caller lateral binding under `structure-exchange:` | `reserved-binding-prefix` |
| Operation that does not drive its law, a dynamic owner without its child, a structure with no owner or two, a child off the common time, or a signal on no coordinator node, bound twice, or missing | `operation-ownership` |

Bank links keep `non-symmetric-weir`. Entry `orifice-partial-submergence` rejects an orifice with a side below its crown at the accepted start. Raise both sides above the crown, or declare a weir or gate.

Dryness is state, so admission cannot see it. Two rejections follow at the trial, and neither retries:

| Case | Entry |
| --- | --- |
| A running pump whose donor stands at or below its suction bound at the start | `pump-without-suction` |
| A running pump that delivers to a dry receiver without a declared [free outlet](structures.md#declare-a-free-outlet) | `pump-dry-delivery` |

### Cap each structure law

Equal heads apply to every submerged passive law. These are the Villemonte weir, the submerged orifice, the mixed and submerged gate apertures, and a flowing culvert with both sides above their inverts. The rule holds when the start heads and the zero-exchange end heads each sit within one head allowance. The link then reports `EqualHeads` and carries no water. A pump never takes this rule, because a pump moves water at equal heads.

Each link keeps `θ = min(1, max(terms, 0))` of its law volume. `z_r` is the receiver cutoff: a weir crest, a gate sill, an orifice crown, or the culvert invert at the receiver. `z_s` is a pump's suction bound. `z_d` is the culvert invert at the donor.

| Law | Terms | Holds |
| --- | --- | --- |
| Weir | `θ_below`, `θ_level` at `z_r` | A receiver at or below the crest keeps free flow. The donor never ends below the receiver. |
| Gate | The same terms at the sill | A receiver at or below the sill keeps free flow. The donor never ends below the receiver. |
| Submerged orifice | `θ_level` only | The donor never ends below the receiver. A stage that crosses the crown rejects and retries. A cap never hides it. |
| Pump | `θ_suction = (H_d - z_s) A_d / O_d` | The donor ends at or above its suction stop. A pump needs no levelling term, because it lifts water. |
| Culvert | `min(max(θ_below, θ_level), θ_invert)`, with `θ_invert = (H_d - z_d) A_d / O_d` at the invert stop | A receiver at or below its invert is a free outfall. The donor never ends below the receiver or below its invert stop. |

A term within one head allowance of its bound keeps nothing. A stopped link leaves the shared minimum. A culvert also stops when its donor stands within one head allowance of its invert.

The invert stop sits above the donor invert by half the head allowance, as the suction stop does. A draining culvert stops there, so a pool donor never crosses its raised invert. Capability limit `culvert-invert-cap` records this.

The suction band is `[z_s, z_s + allowance]`. A running pump in the band moves no water. A draining pump stops at its suction stop, inside the band. Capability limit `suction-cap` records this. Without the cap, every shorter attempt crosses the bound and the run ends `Unresolved`.

The cap and response bounds of a pool use its smallest curve segment area over the covered stage range. A dry pool uses its first segment. Capability limit `pool-plan-area-bound` records this. Surface cells keep the cell area. Channel cells keep the width at the lowest covered stage.

The response step holds for every admitted law. A weir, a gate, an orifice, and a culvert discharge more as the head difference rises. A pump discharges less as its lift rises. A pool end takes the plain update, because the reach sets its volume.

A culvert law is costly to evaluate, so a culvert link takes no bisection. It takes one linear step from its certified head sensitivity `s` at the proposal heads: `V = V_g + (P - V_g) / (1 + s |P| (r_A + r_B) / 2)`. `P` is the kept proposal and `r` the stage response per cubic metre of each side. The bound `s` lies at or above the true slope, so the step stops between `V_g` and `P`. A culvert lane at the sensitivity cap of `2^256` per metre takes the plain update and adds no resolution term. Capability limit `culvert-sensitivity-step` records this.

The resolution term of each allowance reads each law's relative head sensitivity. [Structure laws](structures.md#read-head-sensitivity) define it.

## Couple a reach end to a pool

A pool end joins one reach end to one pool. The reach end takes `QuiescentReservoir` at the pool's iterated temporal midpoint stage. Capability limit `pool-end-reservoir-head` records this.

| Field | Meaning |
| --- | --- |
| `id` | Unique nonempty identity, distinct from every link, port, structure, and pool identity |
| `reach_id` | Rectangular, triangular, or surveyed setup reach |
| `end` | `ChannelEnd::Left` or `ChannelEnd::Right` |
| `pool_id` | Coordinator pool |
| `entrance_loss_coefficient` | Finite nonnegative loss against the incoming channel velocity head |
| `physical_connection_assertion` | Evidence that the reach end samples this pool at the end cell bed |
| `quiescent_reservoir_assertion` | Evidence for quiescent reservoir inflow and complete exit mixing |

The reach's own transport sets the end volume. The coordinator never scales it, so a pool end is no link. It joins the iterate through its pool's end stage and its signed volume. Each evaluation reserves the reach's start operands under the reservoir heads of its iterate.

### Own each reach end once

A reach end has one owner: a caller policy, a port, or a pool end. Two owners reject with `reach-end-ownership`. Set the caller policy at a pool end to `None`, as at a port end. A policy there rejects the trial with the same entry.

### Check pool-end support

A pool end needs wet subcritical support at both interval ends.

- A dry pool or a dry end cell at the accepted start rejects without retry, with `pool-end-regime`. A nonfinite or negative entrance loss rejects with the same entry.
- A reach end that turns critical or loses wet support inside the interval rejects. The retry uses a shorter interval.

Other admission cases:

| Case | Entry |
| --- | --- |
| Unknown reach or pool, or a reach without cells | `structure-endpoint-identity` |
| End cell bed below the pool origin | `unobstructed-pool-connection` |
| Port and pool end on one reach end, or two pool ends | `reach-end-ownership` |

A pool holds no momentum, so no port meets a pool. Capability `pool-port` records this.

## Operate gates and pumps

`SurfaceChannelOperating` declares the children that operate dynamic structures. Construction builds each child fresh at the common initial time. Their structures equal the setup's structure links, in ascending identity.

| Field | Child | Owner it serves |
| --- | --- | --- |
| `gates` | `ControlledGateBatch` from controls and actuators | `Actuated` gates |
| `pumps` | `ControlledPumpBatch` from controls and response models | `ControlledPump` pumps |
| `trajectories` | `GateTrajectoryBatch` from one linear opening path per gate | `Prescribed` gates |
| `signals` | `SurfaceChannelSignal` entries | Every controller signal of `gates` and `pumps` |

`SurfaceChannelSignal` holds `controller_id`, `signal_id`, and `node`. The node is the pool, surface cell, or channel cell whose stage the signal samples. Each controller signal has exactly one binding.

`batch.controlled_gates()` and `batch.controlled_pumps()` return the accepted children.

### Follow the interval steps

1. The common end is the earliest of four times. They are the requested end, the start plus `maximum_step_s`, the surface forcing end, and the next operating event. Events are prescribed knots, actuator events, control decisions, and the common terminal bound.
2. An actuator arrival window counts as one event at its upper bound. No interval contains an event. A retry only shortens the interval.
3. Each attempt prepares every child to its end without samples. A gate takes the midpoint of its start and end openings. A gate closed at both ends takes exact closure. A pump holds its accepted operation from the start. Capability limits `interval-operation` and `event-aligned-intervals` record these rules.
4. The fixed point solves the hydraulics under those operations. They depend on no iterate.
5. At a decision end, each signal samples its node's stage in the converged candidate. Observation and availability time equal the decision time. A dry node supplies no sample and rejects.
6. Each child completes with its samples. Its left positions must equal the prepared bits.
7. The new command applies on the decision's right side. The samples never change the interval just solved. No inner iteration joins controls and hydraulics.
8. Acceptance replaces every child at once. A rejected attempt leaves every child unchanged.

A dry signal node at a decision rejects without retry. The error names the controller, the signal, and the node. Bind each signal to a node that stays wet at its decisions.

A control or actuator admission error rejects without retry and names the child. Capability `continuous-control` records that controls decide at sampled times only.

## Schedule structure geometry

`SurfaceChannelSetup::geometry_schedules` attaches caller-owned `StructureGeometrySchedule` entries to declared structure links. Each schedule binds one `structure_id`, one `StructureGeometryTrack`, and non-empty `evidence`.

| Track | Knot type | Evaluation over `[start, end]` |
| --- | --- | --- |
| `Scalar { parameter, knots }` | `StructureGeometryKnot` (piecewise-linear, optional `left != right` step) | Midpoint `(start + end) / 2` |
| `CulvertSediment { knots, bed_manning_n }` | `StructureGeometryKnot` (piecewise-linear depth) | Midpoint `(start + end) / 2` |
| `CulvertClosedBarrels { knots }` | `DiscreteGeometryKnot<u32>` (step) | Constant on `[start, end)` |
| `Closed { knots }` | `DiscreteGeometryKnot<bool>` (step) | Constant on `[start, end)` |

Every knot time joins the coordinator event set alongside operating events, so no accepted interval straddles a geometry knot. Geometry changes never alter stored water volume. Each interval records `SurfaceChannelGeometryReceipt` entries in `last_step()?.geometry` with `storage_before_m3 == storage_after_m3` bit for bit.

`SurfaceChannelSetup::pool_curve_schedules` attaches caller-owned `PoolCurveSchedule` entries to declared pools. Each epoch holds a `time_s`, a checked `LevelPoolCurve` in the pool's vertical datum, and non-empty `evidence`. Epoch times join the coordinator event set. At an epoch, stored volume and cumulative counters stay bit-identical and stage recomputes from the new curve. A pool whose volume exceeds its new top curve knot rejects without retry (`pool-capacity-exceeded`). An epoch whose new curve origin rises above an attached structure bed or pool-end bed rejects at admission (`unobstructed-pool-connection`).

## Attach native runoff

The coordinator can own one `SubcatchmentBatch`. Its outlets drain into surface cells, reach cells, and pools. Read [subcatchments](subcatchment.md) and [contributing support](contributing-support.md) first.

### Declare runoff

Set `SurfaceChannelSetup.runoff` to a `SurfaceChannelRunoff`:

| Field | Holds |
| --- | --- |
| `support` | A `SupportPlanInput` over the surface domain. It names the modelled sources, the surface cells they own, and pool rain support. |
| `definitions` | Every `SubcatchmentDefinition` the plan names, in numerical order |
| `config` | The `SubcatchmentConfig` of the runoff child |
| `outlets` | One `SurfaceChannelRunoffOutlet` per active modelled outlet |

Each outlet holds:

- `outlet_id`: the `outlet_id` of one active `ModelledRunoff` contribution.
- `destination`: one `SurfaceChannelRunoffDestination`.
- `free_outlet`: the outlet invert in the surface datum, in metres, and evidence that receiving water below it cannot change the export.

| Destination | Receives |
| --- | --- |
| `ChannelCells { reach_id, cells }` | Cells of one reach. Each `SurfaceChannelRunoffShare` holds a cell index and a positive weight. |
| `Pool { pool_id }` | One coordinator pool |
| `SurfaceCells { cells }` | Active surface cells by global row-major index. Each `SurfaceChannelRunoffShare` holds a cell index and a positive weight. |

`SurfaceChannelInitialState.runoff` holds one `SubcatchmentState` per active definition, in definition order. The runoff child starts at `initial.time_s`.

Each forcing interval sets `SurfaceChannelForcingInterval.runoff`. It holds one `SubcatchmentContributionInput` per active source over exactly `[start_s, end_s]`. Each record holds rain and evaporation rates. External runon must be zero.

### Delivery lanes

Each potential subarea of an outlet's definition is one lane. Each lane delivers to every target of its outlet. One lane and one target form a delivery lane.

- Reach target: the coordinator adds one `Inflow` lateral binding per delivery lane, `runoff-attachment:<outlet>:<subarea>:<cell>`.
- Pool target: the coordinator adds one pool inflow binding per lane, `runoff-attachment:<outlet>:<subarea>`.
- Surface target: the delivery lane joins the cell's exchange inflow. No binding is added. A cell without a link or structure gets its own exchange lane after the link cells.

Delivery lanes into one surface cell add in delivery-lane order. One delivery lane keeps its volume in every bit. Several pair with the surface exchange within a summation bound.

Caller bindings and caller inputs with the prefix `runoff-attachment:` reject with `reserved-binding-prefix`.

Weights normalize once. One target takes each lane volume in every bit. Several targets split it by weight. The parts sum to the lane within a rounding bound.

### What one attempt does

1. The trial admits the runoff rates once. A receiving stage above its outlet invert rejects with `Input`, `runoff-outlet-submerged`.
2. After the surface step limits, the runoff child trials from its accepted state to the attempt end.
3. Each delivery lane takes its interval volume. Reaches take it as inflow with zero velocity. Pools take it beside the caller inflows.
4. Surface cells take it after surface transport, as exchange inflow with zero velocity. On a cell with link inflow, the link momentum spreads over the link and runoff water together.
5. The volumes stay fixed through the link fixed point.
6. A receiving stage above its invert at the candidate end shortens the interval. A surface cell's stage is its bed plus its depth.
7. Each receiving binding must hold its delivered volume in every bit. The surface exchange must hold the surface runoff within its bound.

A coordinator with runoff and no links still steps. Its surface takes the runoff through the exchange alone.

A shorter attempt trials the runoff child again to the new end. It never scales a volume. Runoff rates need no recompute: they hold over every sub-span. A `SubcatchmentError::Numerical` from the runoff child halves the interval. Other runoff errors return as `SurfaceChannelError::Runoff`.

Incoming runoff funds no withdrawal in the same interval.

### Admission checks

Each rule below rejects with `SurfaceChannelError::Input`. The message starts with the rule name. Construction checks every rule except forcing data, which each trial checks.

| Rule | Rejects |
| --- | --- |
| `unmapped-runoff-outlet` | An active outlet without a destination, or a destination for an unknown or replaced outlet |
| `duplicate-runoff-outlet` | An outlet with two destinations |
| `runoff-destination` | An unknown reach or pool, no target cells, a cell outside the grid or reach, a repeated cell, or a weight that is not finite and positive |
| `runoff-free-outlet` | A nonfinite invert, or empty evidence |
| `runoff-duplicate-writer` | A caller pool inflow whose `physical_support_id` names a selected modelled source or one of its atoms |
| `suppressed-rain` | Nonzero surface rain on a cell a selected source owns. At each trial, nonzero pool rain on a pool a selected source owns. |
| `runoff-plan` | A plan that does not compile against the surface domain, or runoff definitions that differ from the plan |
| `runoff-without-setup` | Runoff states without `setup.runoff`. At each trial, forcing records without it. |

`suppressed-rain` also applies to the window `continue_surface_forcing` and `resume` receive. Rate records that miss a source, repeat one, or carry runon reject at each trial with `runoff-forcing`.

### Read runoff results

| Call or field | Holds |
| --- | --- |
| `runoff()` | The read-only runoff child, or `None` without active runoff |
| `runoff_ledgers()?` | One `SurfaceChannelRunoffLedger` per delivery lane: outlet, subarea, target, lifetime volume, and rounding bound |
| `SurfaceChannelStepDiagnostics.runoff` | One `SurfaceChannelRunoffTransfer` per delivery lane: interval volume, receiving stage at the start and end, and invert |
| `SurfaceChannelWaterBalance.runoff_storage_m3` | Runoff child storage. `storage_m3` includes it. |
| `SurfaceChannelWaterBalance.runoff_delivered_m3` | Runoff the receiving reaches and pools recorded, plus the lifetime surface lane volumes |

The composite balance adds the runoff child's storage, rain, infiltration, evaporation, and soil water adjustment. Delivered runoff stays inside the system, so it leaves the reach lateral totals, the external pool inflow totals, and the surface exchange totals. Over the lifetime, the runoff child's outlet volume, the lane ledgers, and the receivers' totals agree within their bounds.

### Runoff limits

- `runoff-free-outlet-attested`: You attest each outlet's invert. Receiving water never feeds back on the export.
- `runoff-stage-sampled`: The receiving stage is checked at the accepted start and the candidate end only. Request shorter intervals where the receiving water nears the invert.
- `runoff-recomputed-on-shortening`: A shorter attempt recomputes the runoff.
- `runoff-zero-momentum`: Runoff enters a reach cell or a surface cell with zero velocity. On a surface cell with link inflow, it slows the link water.
- `runoff-after-transport`: Runoff enters a surface cell after the interval's surface transport. Surface fluxes and step limits of that interval never see it.
- `runoff-span-rounding`: A split over several cells pairs within a rounding bound. Use one target for exact pairing.
- `pool-rain-atoms-unmeasured`: The plan checks pool rain atom ownership, never pool area.

Run `cargo run --example surface_runoff` for one subcatchment into a reach cell, one into a pool, and one into a surface cell without a link.

## Advance one common interval

1. Create `SurfaceChannelForcingInterval` with `start_s = batch.time()` and one `SurfaceChannelReachForcing` per reach. Leave each ported end and each pool end `None`. Add `pool_sources` and `pool_inflows` when the setup has pools.
2. Call `trial_to(requested_end_s, &forcing)`.
3. Read the trial's `time()` and `diagnostics()`.
4. Replace the batch with `trial.into_next()` when accepting the candidate.
5. Repeat from the returned time.

Surface forcing stays in `setup.surface.forcing`. Supply every caller lateral binding once per interval, including zeros. Omit the `surface-exchange:` and `structure-exchange:` bindings. The coordinator supplies them.

The common end is the smallest of the requested end, the start plus `maximum_step_s`, and the end of the covering surface forcing interval. The next operating event also bounds it. See [the interval steps](#follow-the-interval-steps).

### What one attempt calculates

1. Bind caller forcing and reserve channel donor water from the accepted start. Each reach calculates its accepted-start face fluxes once here. Every evaluation reuses them. A pool-ended reach reserves again per evaluation.
2. Prepare the surface transport and the pool sources. Reject a duration above the surface or channel step limit. Reject a pool above its top knot.
3. Prepare the operating children to the attempt end. They fix each gate opening and pump operation.
4. Complete every child once with zero exchange. Read the levelling stages and plan areas.
5. Iterate end stages, signed link and structure volumes, and pool-end volumes from the accepted state. Each evaluation:
   - Calculates midpoint heads at every node and each law's volume under its interval operation.
   - Applies the equal-heads rule, then the levelling, crest, sill, and suction caps.
   - Applies the donor limit over surface cells, channel cells, and pools.
   - Completes the surface with the summed volume of each linked cell.
   - Advances each linked reach with each link's volume on its synthesized bindings. A pool-ended reach takes `QuiescentReservoir` at its pool's midpoint stage.
   - Completes the pools with the summed incoming and outgoing volumes. Incoming volumes include pool ends and external inflow.
6. Accept the iterate when both budgets pass and the support and regime flags repeat exactly. Flags cover pool wet flags and pool-end support. The evaluation of the accepted iterate is the candidate. An unconverged iterate moves on by the [response step](#response-step).
7. Reject heads that cross beyond the head budget over the interval. The cap keeps this check from binding unless a child's stage departs from its plan-area bound.
8. Check pool capacity, pool cuts, and pool-end support.
9. Complete each operating child at a decision end. See [the interval steps](#follow-the-interval-steps).
10. Check pairing and the composite water balance.

Each update `k` has a residual norm `r_k`: the largest change between an iterate and its evaluation, over its allowance. Convergence needs `r_k <= 1`. The last update within `iteration_limit` counts.

#### Response step

Near level heads the Villemonte discharge changes without bound as the heads move. A plain update `x ← g(x)` then overshoots, and it stops contracting at the step limits. Each unconverged update instead takes a response step:

1. Model each end stage as the evaluated stage plus `±(V - V_g) / A`. `V_g` is the evaluated link volume. `A` is the plan area of the [levelling cap](#stop-at-level-heads), split evenly among the links of a shared cell.
2. Clamp each weir volume to the most the levelling cap and the donor limit keep in its direction.
3. Find by bisection the volume `V` at which the clamped weir volume at the modelled midpoint heads equals `V`. The root lies between `V_g` and the plain proposal.
4. Take `V` and the modelled end stages as the next iterate.

The children then correct the model at the next evaluation. Convergence still compares each iterate with its own evaluation, so the accepted fixed point and every budget stay those of the plain update.

#### Resolution term

Near level heads one rounding of the heads can move a weir volume by more than a tight transfer budget. Each allowance therefore adds the change one rounding of the midpoint heads causes:

| Allowance | Added term |
| --- | --- |
| Transfer, per link | `ε (|H_s| + |H_c|) |V| (1.5 / h_u + 0.385 / gap)` |
| Head, per cell | The transfer term of each link at the cell, over its plan area, summed |

`ε` is `f64::EPSILON`, `h_u` the upstream head above the crest, and `gap` the head difference of submerged flow. Away from level heads the term stays far below any positive budget.

The factor `(1.5 / h_u + 0.385 / gap)` is the weir's relative head sensitivity. Each structure law supplies its own in its place:

| Law | Sensitivity per metre |
| --- | --- |
| Weir | `1.5 / h_u + 0.385 / gap` |
| Submerged orifice | `0.5 / gap` |
| Gate | `1.5 / h_u + 0.5 / gap`, with `h_u` above the sill |
| Pump | `\|dQ/dH\| / Q` on the active curve segment at the running speed, zero where `Q` is zero |
| Culvert | `1 / E` of the governing control. See [culvert sensitivity](structures.md#read-culvert-sensitivity). |

#### Stalled attempts

The contraction `ρ = sqrt(r_k / r_{k-2})` spans two updates, so a 2-cycle shows its true rate. Two checks end a stalled attempt at once:

| Check | Rejection | Recorded `ρ` |
| --- | --- | --- |
| The iterate repeats the iterate two updates earlier, bit for bit | Iteration exhaustion | `1` |
| `ρ >= 1`, or `k + ln(r_k)/(-ln ρ)` exceeds `iteration_limit` | Contraction forecast | Measured |

The forecast needs `k >= 2` and `r_k > 1`. An accepted attempt reports its last `ρ` as `diagnostics().contraction`. Below two updates it reports zero.

### Recalculation

| Rejection | Next attempt |
| --- | --- |
| Duration above the surface or channel step limit | `0.95` of the smallest limit |
| Child suggests a shorter end | That end |
| Stalled fixed point, either check | The largest of `1/2`, `1/4`, `1/8` of the interval whose product with `ρ` is at most `contraction_target`, else `1/8` |
| Other retryable rejection, including exhaustion of every update | Half the interval |
| `retry_limit` reached, or a duration below `minimum_step_s` | `SurfaceChannelError::Unresolved` |

`SurfaceChannelError::is_retryable` names the retryable rejections. Every other rejection returns unchanged. A rejected trial leaves the batch unchanged.

### Recompute integrated inputs

An integrated caller lateral volume belongs to its exact supplied interval.

- Forcing end: `end_s` must equal `requested_end_s`. Otherwise the trial returns `Input`.
- Shorter interval: Hydr returns `SurfaceChannelError::RecomputeInterval`. Recompute every input for `suggested_end_s` and submit fresh forcing from the unchanged start.

Never slice or scale old integrated inputs.

## Read exchange and water accounting

| Call | Result |
| --- | --- |
| `last_step()?` | `SurfaceChannelStepDiagnostics` of the last interval, absent before the first |
| `link_ledgers()?` | Lifetime surface-to-channel and channel-to-surface volume per link, with a rounding bound |
| `port_ledgers()?` | Lifetime surface-to-channel and channel-to-surface reach volume per port, with a rounding bound |
| `structure_ledgers()?` | Lifetime A-to-B and B-to-A volume per structure, with a rounding bound |
| `pool_end_ledgers()?` | Lifetime reach-to-pool and pool-to-reach volume per pool end, with a rounding bound |
| `pool_inflow_ledgers()?` | Lifetime `CoupledPoolInflowLedger` per external pool inflow binding |
| `ledger()?` | `SurfaceChannelWaterBalance` of the surface, every reach, and every pool as one system |
| `surface()`, `reach(id)`, `pools()` | Read-only children |
| `controlled_gates()`, `controlled_pumps()` | Read-only operating children |

Each `SurfaceChannelLinkTransfer` holds the signed volume, weir volume, levelling factor, donor factor, regime, and both midpoint stages.

`SurfaceChannelStepDiagnostics` adds three lists:

- `structures`: one `SurfaceChannelStructureTransfer` per structure, in ascending identity. Each holds these fields:
  - `signed_volume_m3` and `law_volume_m3`
  - `cap_factor` and `donor_factor`
  - `regime` and both midpoint stages
  - `operation` and `receipt`, a `SurfaceChannelOperationReceipt`
- `pool_ends`: one `SurfaceChannelPoolEndTransfer` per pool end, in ascending identity. It holds the signed volume, positive from the reach into the pool, and the reservoir head.
- `pools`: one `SurfaceChannelPoolStep` per pool, in ascending identity. It holds the end volume, the end stage, and the native incoming and outgoing volumes.

The receipt names the source of each operation:

| Receipt | Holds |
| --- | --- |
| `Manual` | The supplied operation |
| `PrescribedGate` | Start, end, and midpoint openings, the larger endpoint error, and whether the gate stays closed |
| `ActuatedGate` | Start and old-target end positions, the midpoint opening and its bounds, and the larger endpoint error |
| `ControlledPump` | The accepted latch at the start and the declared response model |

`SurfaceChannelStepDiagnostics.ports` holds one `SurfaceChannelPortTransfer` per port, in ascending port identity. Each holds:

- `signed_volume_m3`: the reach volume, positive from the surface into the channel
- `reach_momentum_impulse_m4_per_s`: normal momentum leaving the reach, pressure included
- `surface_momentum_impulse_m4_per_s`: momentum entering the surface cells, in grid x and y
- `junction_reaction_m4_per_s`: the difference between both sides, in grid x and y
- `signal_speed_m_per_s`, `regime`, and `reversed`
- `paired_gap_m3` and `paired_bound_m3`: the surface face volume plus the reach outward volume, and its bound

Each accepted interval passes these checks:

- Binding: Each channel binding holds its link or structure volume bit for bit.
- Pairing: Each side's net exchange matches the summed link and structure volumes within `paired_bound_m3`.
- Pool pairing: Each pool's native incoming and outgoing volumes match the structure, pool-end, and external totals within the bound.
- Pool-end pairing: A pool end's volume equals its reach end volume bit for bit. Both come from one tensor.
- Port pairing: Each port's `|paired_gap_m3|` stays within its `paired_bound_m3`.
- Mass: Storage change minus external water stays within `mass_bound_m3`. `port_paired_bound_m3` joins it.

Storage and external water include the pools. External water adds pool rain and external pool inflow. External losses add pool bottom loss and evaporation. External pool water comes from the inflow ledgers, never from native pool counter differences.

Exchange stays inside the composite system. The surface and channel exchange totals report link and structure water only. Port water also stays inside. The surface ledger counts it as exchange, the reach ledger as boundary water, and the composite balance removes it from both. Pool-end water stays inside too.

## Save, restore, and continue

| Use | Save | Restore |
| --- | --- | --- |
| In-memory continuation | `snapshot()` | `restore(&snapshot)` |
| Application-owned persistence | `snapshot_data()?` | `import_snapshot(&data)?` |
| Restart on a new client | `snapshot_data()?` | `SurfaceChannelBatch::resume(client, &data, forcing)` |

Construct a `restore` or `import_snapshot` destination with the same setup and original initial state. Each call returns a new batch.

Pass `data.surface.setup.forcing.clone()` to `resume` for a plain restart.

`SurfaceChannelSnapshotData` carries `schema` (`SURFACE_CHANNEL_SNAPSHOT_SCHEMA`, `hydr-surface-channel-checkpoint-11`) and `contract` (`SURFACE_CHANNEL_CONTRACT_VERSION`, `hydr-surface-channel-numr-cpu-13`). Import checks both first. A package of an earlier format rejects by identity: schema `hydr-surface-channel-checkpoint-10` or contract `hydr-surface-channel-numr-cpu-12` predates pool curve epochs, schema `hydr-surface-channel-checkpoint-9` or contract `hydr-surface-channel-numr-cpu-11` predates structure geometry schedules, schema `hydr-surface-channel-checkpoint-8` predates the structure element condition, contract `hydr-surface-channel-numr-cpu-10` predates runoff into surface cells, contract `hydr-surface-channel-numr-cpu-9` predates the runoff package, runoff counters, runoff transfers, and runoff balance totals, schema `hydr-surface-channel-checkpoint-7` predates the runoff setup, original-state, and forcing fields, contract `hydr-surface-channel-numr-cpu-8` predates warm culvert root searches, schema `hydr-surface-channel-checkpoint-6` predates culvert structures, and schema `-5` or contract `hydr-surface-channel-numr-cpu-7` predates secant probes in surveyed roots. Capture a new package. Import checks every nested package identity, checks `geometry_epochs` and `pool_curve_epochs` at `counters.time_s`, then recalculates the last interval and compares every bit. An error names the first differing field or the package path.

The replay includes decision completion. At a decision end, the recalculation samples the converged stages and completes each operating child, then compares the child packages.

The package holds the ports in `setup.ports` and the lifetime port counters in `counters.ports`. The counters include each port's last signed volume, so a replayed `reversed` flag matches. The last interval's diagnostics hold every port receipt. Port ghost states rebuild from the start-frame reach packages.

The package also holds these parts:

| Field | Holds |
| --- | --- |
| `setup` | Pools, pool support, sources, inflows, `storage`, structures, pool ends, the operating declaration, `geometry_schedules`, and `pool_curve_schedules` |
| `geometry_epochs`, `pool_curve_epochs` | Active epoch indices of each structure geometry schedule and pool curve schedule at `counters.time_s` |
| `last_interval.diagnostics.geometry` | Geometry schedule receipts of the last interval, each with `storage_before_m3 == storage_after_m3` |
| `initial.pools` | The original volume of every pool |
| `pools` | The accepted `LevelPoolSnapshotData` |
| `operating` | The accepted `SurfaceChannelOperatingData` of every operating child |
| `counters` | `structures`, `pool_ends`, and `pool_inflows` ledgers beside the link and port ledgers |
| `last_interval.previous` | The start-frame pool, operating, and runoff packages |
| `setup.runoff`, `initial.runoff` | The runoff declaration and the original runoff states |
| `runoff` | The accepted `SubcatchmentSnapshotData`, or `None` without active runoff |
| `counters.runoff` | `SurfaceChannelRunoffCounterData`: one lane ledger per delivery lane and the summed split bound |
| `last_interval.forcing.runoff`, `last_interval.diagnostics.runoff` | The runoff rate records and transfers of the last interval |

Import rejects a pool whose original volume differs from `initial.pools`. It rejects a runoff package whose original storage differs from the storage of `initial.runoff`. Import and `resume` restore the start-frame runoff package, then replay the last interval and compare the runoff package and counters bit for bit.

The nested surface package carries exchange water, port water included. It restores only through the coordinator. `Solver::restore_data`, `Solver::restore`, and `Solver::resume` reject it.

`continue_surface_forcing(schedule)` splices a surface forecast after the accepted time. It follows the [surface splice rule](integration.md#continue-a-forecast-under-new-forcing). The original batch stays unchanged.

Preserve every package field and floating-point value exactly. Follow the [persistence requirements](integration.md#save-complete-restart-data).

## Stay within the supported scope

| Supported | Excluded |
| --- | --- |
| Bank weirs between active surface cells and rectangular, triangular, or surveyed reach cells | Unobstructed connections |
| Ports between a face line and a rectangular reach end, or a surveyed rectangle end, of the face width | Ports on triangular reach ends and on surveyed ends that are not rectangles |
| Surveyed reaches with any channel endpoint policy at an unowned end | Scalar endpoints on dry, critical, or supercritical water, as on every reach |
| Reversal through one link, structure, pool end, or port between intervals | Normal momentum exchange through bank links, and child subcycling |
| Surface rain, impervious land, Modified Green-Ampt soil, evaporation, and withdrawal | Device-resident forcing |
| Caller lateral inflow and withdrawal on reaches | Rain on channel water surfaces |
| Subcatchment runoff into surface cells, reach cells, and pools through free outlets | Tailwater feedback on runoff, external runon, and routing between subcatchments |
| Level pools with local rain, bottom loss, evaporation, and external inflow | Pool ports. A pool holds no momentum. |
| Weirs, submerged orifices, gates, pumps, and HDS-5 culverts of every admitted barrel shape between surface cells, channel cells, and pools | Partial orifice flow, pump priming, net positive suction head, motor dynamics, and culvert barrel storage |
| Fixed blockage scenarios, caller-scheduled structure geometry (`geometry_schedules`), and pool curve epochs (`pool_curve_schedules`) with evidence | Blockage probability, inferred blockage, and surface bed or channel section geometry epochs |
| Pool ends at a quiescent reservoir head on rectangular, triangular, or surveyed reaches | Dry or critical pool ends |
| Prescribed, actuated, and sampled-pump operation at sampled decisions | Continuous control feedback, minimum dwell, and native SWMM rules |

Time integration remains first order. The weir law establishes no energy guarantee for the complete calculation.

Structure and pool limits:

- `suction-cap`: A draining pump stops where its donor reaches the suction bound. A cap ends its flow, never a rejection.
- `structure-momentum-closure`: Structure water enters either side with zero velocity. Jets dissipate.
- `interval-operation`: A gate takes the midpoint opening of its linear path over each interval. A pump holds its accepted operation.
- `event-aligned-intervals`: Every interval ends at or before the next prescribed knot, actuator event, or control decision.
- `pool-plan-area-bound`: A pool's cap and response bounds use its smallest segment plan area over the covered stage range.
- `pool-end-reservoir-head`: A pool end takes the pool's temporal midpoint stage as its reservoir head.
- `blockage-as-scenario`: A blockage is a fixed caller scenario with evidence. The element's condition holds it.
- `culvert-invert-cap`: A draining culvert donor stops just above its invert. A cap ends its flow, never a rejection.
- `culvert-sensitivity-step`: A culvert link's response step comes from its certified head sensitivity, without bisection.

Each culvert evaluation runs at least one outlet march. A partly full outlet costs far more than a drowned full barrel, which takes a closed form. Every fixed-point update evaluates each culvert twice: once in the map and once for the response step. The first iterate reuses the culvert laws of the accepted start.

Inside one attempt, each culvert evaluation starts its outlet root search in a window around the root of the evaluation before it. The accepted start laws seed the first window. A window only sets where the search starts. Its ends must certify their residual signs, and a lane whose window misses the root searches from zero. So the accepted bits follow the attempt's evaluation order, and a restart repeats that order bit for bit.

Port limits:

- `port-explicit-flux`: The flux uses the accepted-start states for one attempt. Request shorter intervals where the states change fast.
- `port-bed-step`: Hydrostatic reconstruction keeps a lake at rest across a bed step. It gives no energy guarantee across the step.
- `port-tangential-closure`: Reach water enters the surface with no tangential velocity. Tangential surface momentum that leaves through a port ends at the junction. Read `junction_reaction_m4_per_s` for it.

Numerical agreement establishes no measured-event accuracy and no field culvert accuracy. CUDA runs execute without a GPU accuracy qualification.

Read [surface/channel checks](acceptance.md#surfacechannel-checks) for the checked cases and runtime support. Read [reference and refinement](acceptance.md#reference-and-refinement) for the closed-form drain and the split-bank refinement. Read [pools, structures, and controls](acceptance.md#pools-structures-and-controls) for the pool and structure cases.
