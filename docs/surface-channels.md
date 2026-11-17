# Couple the surface to channel reaches

Use `SurfaceChannelBatch<R>` to advance the 2D surface engine and 1D channel reaches over one common interval. Water crosses a bank weir between a surface cell and a channel cell, in either direction.

Read [channel flow](channels.md), [numerics](numerics.md), and [structure laws](structures.md#orifices-and-rectangular-weirs) before constructing the inputs.

Read capabilities with `surface_channel_capabilities()`. It returns a `ComponentCapabilities` declaration: runtimes, link endpoints, weir regimes, sources, nested components, unsupported uses, and limits.

## What the coordinator couples

| Part | Supplied as | Notes |
| --- | --- | --- |
| Surface | `SolverSetup` and `InitialState` | Exterior faces, land model, and surface forcing keep their surface rules. |
| Reaches | `SurfaceChannelReach` entries | Rectangular, triangular, or surveyed geometry. |
| Links | `SurfaceChannelLink` entries | One bank weir between one active surface cell and one channel cell. |

Reach ends take the nested reach's endpoint policies. No reach end exchanges water with the surface.

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
| `EqualHeads` | Both sides above the crest, heads within the head allowance | Zero |

The head allowance is `head_absolute_m + head_relative * max(|H_s|, |H_c|)`, at the midpoint heads.

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

On the next interval the heads sit within the head allowance, so the link reports `EqualHeads` and carries no water.

### Limit each donor to its water

Each donor gives only water it holds for this interval.

| Donor | Available water |
| --- | --- |
| Surface cell | Post-transport water of the cell |
| Channel cell | Start water, minus native face outflow over the interval and caller withdrawals, with a rounding margin |

One factor per donor scales every link that donor feeds. When demand exceeds the available water, the factor is `(available - margin) / (demand + bound)`, rounded down once more. Incoming water gives no same-interval donor credit.

### Momentum closure

Exchange water carries momentum along the reach tangent only:

| Direction | Velocity carried |
| --- | --- |
| Surface to channel | Surface post-transport velocity projected on `reach_tangent` |
| Channel to surface | Channel accepted-start velocity, along `reach_tangent` in surface x/y |

The normal component dissipates at the bank. Several links into one surface cell carry their summed momentum.

## Advance one common interval

1. Create `SurfaceChannelForcingInterval` with `start_s = batch.time()` and one `SurfaceChannelReachForcing` per reach.
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
5. Accept the iterate when both budgets pass and the support and regime flags repeat exactly. The evaluation of the accepted iterate is the candidate.
6. Reject heads that cross beyond the head budget over the interval. The cap keeps this check from binding unless a child's stage departs from its plan-area bound.
7. Check pairing and the composite water balance.

The iteration is a plain fixed-point (Picard) update without history. Stiff links can stall it. Each update `k` has a residual norm `r_k`: the largest change over its allowance. Convergence needs `r_k <= 1`. The contraction `ρ = sqrt(r_k / r_{k-2})` spans two updates, so a 2-cycle shows its true rate. Two checks end a stalled attempt at once:

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
| `ledger()?` | `SurfaceChannelWaterBalance` of the surface and every reach as one system |
| `surface()`, `reach(id)` | Read-only children |

Each `SurfaceChannelLinkTransfer` holds the signed volume, weir volume, levelling factor, donor factor, regime, and both midpoint stages.

Each accepted interval passes three checks:

- Binding: Each channel binding holds its link volume bit for bit.
- Pairing: Each side's net exchange matches the summed link volumes within `paired_bound_m3`.
- Mass: Storage change minus external water stays within `mass_bound_m3`.

Exchange stays inside the composite system. It appears only in the surface and channel exchange totals.

## Save, restore, and continue

| Use | Save | Restore |
| --- | --- | --- |
| In-memory continuation | `snapshot()` | `restore(&snapshot)` |
| Application-owned persistence | `snapshot_data()?` | `import_snapshot(&data)?` |
| Restart on a new client | `snapshot_data()?` | `SurfaceChannelBatch::resume(client, &data, forcing)` |

Construct a `restore` or `import_snapshot` destination with the same setup and original initial state. Each call returns a new batch.

Pass `data.surface.setup.forcing.clone()` to `resume` for a plain restart.

`SurfaceChannelSnapshotData` carries `schema` (`SURFACE_CHANNEL_SNAPSHOT_SCHEMA`) and `contract` (`SURFACE_CHANNEL_CONTRACT_VERSION`). Import checks both first. It recalculates the last interval and compares every bit. An error names the first differing field.

The nested surface package carries exchange water. It restores only through the coordinator. `Solver::restore_data`, `Solver::restore`, and `Solver::resume` reject it.

`continue_surface_forcing(schedule)` splices a surface forecast after the accepted time. It follows the [surface splice rule](integration.md#continue-a-forecast-under-new-forcing). The original batch stays unchanged.

Preserve every package field and floating-point value exactly. Follow the [persistence requirements](integration.md#save-complete-restart-data).

## Stay within the supported scope

| Supported | Excluded |
| --- | --- |
| Bank weirs between active surface cells and rectangular, triangular, or surveyed reach cells | Unobstructed connections and reach-end ports |
| Surveyed reaches with `Wall` or complete `Exterior` endpoints | Surveyed reaches with any other endpoint |
| Reversal through one link between intervals | Normal momentum exchange and child subcycling |
| Surface rain, impervious land, Modified Green-Ampt soil, evaporation, and withdrawal | Level pools, structures, and device-resident forcing |
| Caller lateral inflow and withdrawal on reaches | |

Time integration remains first order. The weir law establishes no energy guarantee for the complete calculation.

Numerical agreement establishes no measured-event accuracy. CUDA runs execute without a GPU accuracy qualification.

Read [surface/channel checks](acceptance.md#surfacechannel-checks) for the checked cases and runtime support. Read [reference and refinement](acceptance.md#reference-and-refinement) for the closed-form drain and the split-bank refinement.
