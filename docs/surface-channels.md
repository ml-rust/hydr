# Couple the surface to channel reaches

Use `SurfaceChannelBatch<R>` to advance the 2D surface engine and 1D channel reaches over one common interval. Water crosses a bank weir between a surface cell and a channel cell, in either direction. Water and momentum also cross a reach-end port between a line of surface faces and one reach end.

Read [channel flow](channels.md), [numerics](numerics.md), and [structure laws](structures.md#orifices-and-rectangular-weirs) before constructing the inputs.

Read capabilities with `surface_channel_capabilities()`. It returns a `ComponentCapabilities` declaration: runtimes, link endpoints, weir regimes, sources, nested components, unsupported uses, and limits.

## What the coordinator couples

| Part | Supplied as | Notes |
| --- | --- | --- |
| Surface | `SolverSetup` and `InitialState` | Exterior faces, land model, and surface forcing keep their surface rules. |
| Reaches | `SurfaceChannelReach` entries | Rectangular, triangular, or surveyed geometry. |
| Links | `SurfaceChannelLink` entries | One bank weir between one active surface cell and one channel cell. |
| Ports | `SurfaceChannelPort` entries | One line of surface faces against one rectangular reach end. See [reach-end ports](#couple-a-reach-end-to-the-surface). |

`links` can be empty when `ports` is not. A setup without links and without ports rejects.

A reach end without a port takes the nested reach's endpoint policy. The coordinator owns a ported end.

### Couple a surveyed reach

A surveyed reach joins the coordinator through lateral links only.

- Endpoints: supply `Wall` or complete `Exterior` at both ends of every interval. Any other endpoint rejects the trial with `SurfaceChannelError::Channel`.
- Bed: a surveyed cell's bed is the minimum elevation of the reach's surveyed section. The crest check and the reported stage use this bed.
- Stage: the bed plus the depth from the reach's certified area-to-stage recovery.
- Donor water: the lower bound of the two-part stored volume, as the reach's own step uses.

Surveyed reaches keep the [surveyed limits](channels.md#set-surveyed-numerical-controls). Surveyed links run in the CPU checks only. The CUDA runtime evidence covers rectangular reaches.

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
- Caller lateral binding under the reserved `surface-exchange:` prefix.
- `applicability.symmetric_reversal = false`. Links carry water both ways.
- Empty evidence or identity text, or text with control characters.
- Initial surface or reach time that differs from the common time bit for bit.

The coordinator appends two bindings per link to each linked reach: `surface-exchange:<link id>:inflow` and `surface-exchange:<link id>:withdrawal`. They follow the caller bindings, in ascending link identity. `coordinated_reach(id)` returns the result.

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

The donor limit applies after the cap. `level_factor` reports the applied `θ`. It is exactly `1.0` when the cap does not bind, and the weir volume then keeps every bit.

On the next interval the heads sit within the head allowance. Without other water to move them apart, the link reports `EqualHeads` and carries no water.

### Limit each donor to its water

Each donor gives only water it holds for this interval.

| Donor | Available water |
| --- | --- |
| Surface cell | Post-transport water of the cell |
| Channel cell | Start water, minus native face outflow over the interval and caller withdrawals, with a rounding margin |

One factor per donor scales every link that donor feeds. When demand exceeds the available water, the factor is `(available - margin) / (demand + bound)`, rounded down once more. Incoming water gives no same-interval donor credit.

### Momentum closure

Bank link water carries momentum along the reach tangent only. Ports carry normal momentum under the [port flux law](#port-flux-law).

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
| `reach_id` | Ported setup reach. It must be rectangular. |
| `end` | `ChannelEnd::Left` or `ChannelEnd::Right`. One port binds each end at most. |
| `faces` | Surface faces in any order. Admission sorts them along their face line. |
| `mapping_evidence` | Your evidence that the faces and the reach end share this location |

Each face has one active surface cell. Its other side is an inactive cell or lies off the grid. The faces share one axis, one face line, and one surface side, with no gap. A face is never a caller open face, and it belongs to one port only.

The face widths must sum to the end cell width `b` within `SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE · b`. The tolerance is `16 · f64::EPSILON`. Set the end cell width to the face width sum. A rectangular reach can vary its width by cell.

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

A port with mixed face regimes reports `WetDryFront`, then `Transcritical`, before `Subcritical`. `reversed` is true when the signed volume changes sign against the previous interval.

### Port forcing and checks

Set `left` or `right` of `SurfaceChannelReachForcing` to `None` at a ported end. Set `Some(policy)` at every other end. Either mistake rejects the trial with `port-endpoint-policy`.

`SurfaceChannelDefinition::new` rejects a port with `SurfaceChannelError::Input`. The message names the capability entry.

| Case | Entry |
| --- | --- |
| Triangular reach end | `port-section-mismatch` |
| Surveyed reach end | `surveyed-port-end` |
| Face width sum off the end cell width | `port-width-mismatch` |
| Bad face set, two ports on one end, or an unknown reach | `port-face-ownership` |
| Empty or repeated port identity, or empty evidence | `duplicate-links`, `missing-link-evidence` |
| Reach datum or gravity bits that differ from the surface | `datum-or-gravity-mismatch` |

## Advance one common interval

1. Create `SurfaceChannelForcingInterval` with `start_s = batch.time()` and one `SurfaceChannelReachForcing` per reach. Leave each ported end `None`.
2. Call `trial_to(requested_end_s, &forcing)`.
3. Read the trial's `time()` and `diagnostics()`.
4. Replace the batch with `trial.into_next()` when accepting the candidate.
5. Repeat from the returned time.

Surface forcing stays in `setup.surface.forcing`. Supply every caller lateral binding once per interval, including zeros. Omit the `surface-exchange:` bindings. The coordinator supplies them.

The common end is the smallest of the requested end, the start plus `maximum_step_s`, and the end of the covering surface forcing interval.

### What one attempt calculates

1. Bind caller forcing and reserve channel donor water from the accepted start. Each reach calculates its accepted-start face fluxes once here. Every evaluation reuses them.
2. Prepare the surface transport. Reject a duration above the surface or channel step limit.
3. Complete every child once with zero exchange. Read the levelling stages and plan areas.
4. Iterate end stages and signed link volumes from the accepted state. Each evaluation:
   - Calculates midpoint heads and weir volumes.
   - Applies the levelling cap, then the donor limit.
   - Completes the surface with the summed volume of each linked cell.
   - Advances each linked reach with each link's volume on its synthesized bindings.
5. Accept the iterate when both budgets pass and the support and regime flags repeat exactly. The evaluation of the accepted iterate is the candidate. An unconverged iterate moves on by the [response step](#response-step).
6. Reject heads that cross beyond the head budget over the interval. The cap keeps this check from binding unless a child's stage departs from its plan-area bound.
7. Check pairing and the composite water balance.

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
| `ledger()?` | `SurfaceChannelWaterBalance` of the surface and every reach as one system |
| `surface()`, `reach(id)` | Read-only children |

Each `SurfaceChannelLinkTransfer` holds the signed volume, weir volume, levelling factor, donor factor, regime, and both midpoint stages.

`SurfaceChannelStepDiagnostics.ports` holds one `SurfaceChannelPortTransfer` per port, in ascending port identity. Each holds:

- `signed_volume_m3`: the reach volume, positive from the surface into the channel
- `reach_momentum_impulse_m4_per_s`: normal momentum leaving the reach, pressure included
- `surface_momentum_impulse_m4_per_s`: momentum entering the surface cells, in grid x and y
- `junction_reaction_m4_per_s`: the difference between both sides, in grid x and y
- `signal_speed_m_per_s`, `regime`, and `reversed`
- `paired_gap_m3` and `paired_bound_m3`: the surface face volume plus the reach outward volume, and its bound

Each accepted interval passes these checks:

- Binding: Each channel binding holds its link volume bit for bit.
- Pairing: Each side's net exchange matches the summed link volumes within `paired_bound_m3`.
- Port pairing: Each port's `|paired_gap_m3|` stays within its `paired_bound_m3`.
- Mass: Storage change minus external water stays within `mass_bound_m3`. `port_paired_bound_m3` joins it.

Exchange stays inside the composite system. The surface and channel exchange totals report link water only. Port water also stays inside. The surface ledger counts it as exchange, the reach ledger as boundary water, and the composite balance removes it from both.

## Save, restore, and continue

| Use | Save | Restore |
| --- | --- | --- |
| In-memory continuation | `snapshot()` | `restore(&snapshot)` |
| Application-owned persistence | `snapshot_data()?` | `import_snapshot(&data)?` |
| Restart on a new client | `snapshot_data()?` | `SurfaceChannelBatch::resume(client, &data, forcing)` |

Construct a `restore` or `import_snapshot` destination with the same setup and original initial state. Each call returns a new batch.

Pass `data.surface.setup.forcing.clone()` to `resume` for a plain restart.

`SurfaceChannelSnapshotData` carries `schema` (`SURFACE_CHANNEL_SNAPSHOT_SCHEMA`, `hydr-surface-channel-checkpoint-4`) and `contract` (`SURFACE_CHANNEL_CONTRACT_VERSION`, `hydr-surface-channel-numr-cpu-5`). Import checks both first. A package of an earlier format rejects by identity; capture a new package. Import recalculates the last interval and compares every bit. An error names the first differing field.

The package holds the ports in `setup.ports` and the lifetime port counters in `counters.ports`. The counters include each port's last signed volume, so a replayed `reversed` flag matches. The last interval's diagnostics hold every port receipt. Port ghost states rebuild from the start-frame reach packages.

The nested surface package carries exchange water, port water included. It restores only through the coordinator. `Solver::restore_data`, `Solver::restore`, and `Solver::resume` reject it.

`continue_surface_forcing(schedule)` splices a surface forecast after the accepted time. It follows the [surface splice rule](integration.md#continue-a-forecast-under-new-forcing). The original batch stays unchanged.

Preserve every package field and floating-point value exactly. Follow the [persistence requirements](integration.md#save-complete-restart-data).

## Stay within the supported scope

| Supported | Excluded |
| --- | --- |
| Bank weirs between active surface cells and rectangular, triangular, or surveyed reach cells | Unobstructed connections |
| Ports between a face line and a rectangular reach end of the face width | Ports on triangular or surveyed reach ends |
| Surveyed reaches with `Wall` or complete `Exterior` endpoints | Surveyed reaches with any other endpoint or a port |
| Reversal through one link or one port between intervals | Normal momentum exchange through bank links, and child subcycling |
| Surface rain, impervious land, Modified Green-Ampt soil, evaporation, and withdrawal | Level pools, structures, and device-resident forcing |
| Caller lateral inflow and withdrawal on reaches | |

Time integration remains first order. The weir law establishes no energy guarantee for the complete calculation.

Port limits:

- `port-explicit-flux`: The flux uses the accepted-start states for one attempt. Request shorter intervals where the states change fast.
- `port-bed-step`: Hydrostatic reconstruction keeps a lake at rest across a bed step. It gives no energy guarantee across the step.
- `port-tangential-closure`: Reach water enters the surface with no tangential velocity. Tangential surface momentum that leaves through a port ends at the junction. Read `junction_reaction_m4_per_s` for it.

Numerical agreement establishes no measured-event accuracy. CUDA runs execute without a GPU accuracy qualification.

Read [surface/channel checks](acceptance.md#surfacechannel-checks) for the checked cases and runtime support. Read [reference and refinement](acceptance.md#reference-and-refinement) for the closed-form drain and the split-bank refinement.
