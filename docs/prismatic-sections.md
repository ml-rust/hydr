# Get bounds for surveyed channel geometry

Use `certify_prismatic` when you need lower and upper bounds on section geometry. Start with the [surveyed-profile inputs](cross-sections.md#supply-the-surveyed-profile).

These queries use depth above the lowest bed. Depth stays separate from the absolute elevation datum, including for very shallow water.

## Construct once and choose a query

1. Construct `CrossSectionBatch::new(client, definitions)` with the configured Numr client.
2. Call `certify_prismatic()` once to obtain `PrismaticSectionBatch<R>`.
3. Choose the query below and supply native tensors on that client's device.
4. Call `properties()` when you need owned host results.

| Known input | Method |
| --- | --- |
| Depth above the bed, in metres | `evaluate_depths_tensor(&owners, &depths_m)` |
| Conserved cross-sectional area, in m² | `recover_depths_tensor(&owners, &areas_m2, &config)` |

Both tensors have shape `[N]`, with one section index and one value per query.

- `owners`: Zero-based 64-bit integer (`I64`) indices into `definitions()`. Repeated and unordered indices are valid.
- Values: Finite, nonnegative 64-bit floating-point (`F64`) depths or areas. Positive and negative zero both describe dry geometry.

Each result owns snapshots of both inputs. Original value bits survive, including signed zero.

The supplied client determines execution. Read [runtime admission](integration.md#runtime-selection-and-admission) before selecting a backend.

## Check whether the profile qualifies

Certification requires a pressure curve whose slope never decreases as area increases. Numerical uncertainty in this check returns `Numerical`.

| Profile feature | Admission |
| --- | --- |
| Wet region | One connected region with banks rising away from the bottom |
| Bottom and walls | A bottom point or plateau, with vertical walls allowed |
| Horizontal bench at `maximum_stage_m` | Admitted. Width uses its limit from below. |
| Valid geometry above `maximum_stage_m` | Irrelevant unless it hides another wet pocket below that level |
| Bench below `maximum_stage_m` or an interior ridge | Rejected |

The original definitions remain unchanged. No query extends their supported depth or area.

## Read the bounds

Native getters return Solvr `Interval<R>` values. Keep their endpoints and all returned tensor storage immutable.

Host properties contain `[lower, upper]` arrays in original query order.

| Quantity | Units |
| --- | --- |
| Geometric depth above the bed | m |
| Wet cross-sectional area | m² |
| Water-surface width | m |
| Hydrostatic integral | m³ |
| Hydraulic depth, area divided by width | m |

See [geometric properties](cross-sections.md#read-the-geometric-properties) for field meanings and the pressure-to-force relation.

The native `dry_mask()` contains F64 `1` for dry rows and `0` for wet rows. Dry width and hydraulic depth contain zero placeholders. Host properties return `None` for these fields.

Recovery retains the exact original area through `requested_areas_m2()` and `requested_area_m2`. Keep that original area as the conserved value. `forward_area_m2` bounds geometry across the recovered depth interval.

Wetted-perimeter and hydraulic-radius bounds are outside this API. The [stage-based methods](cross-sections.md) supply point estimates of those quantities.

## Set the recovery tolerance

`PrismaticRecoveryConfig::default()` uses the following depth-width allowances:

| Field | Default | Meaning |
| --- | --- | --- |
| `depth_atol_m` | `1e-12` | Absolute allowance in metres |
| `depth_rtol` | `1e-12` | Relative allowance times the largest absolute depth endpoint |

Both values must be finite and nonnegative. Setting both to zero admits only a zero-width depth result.

Recovery checks the entire depth-interval width against the combined allowance. Wider tolerances cannot resolve uncertain capacity or an unrepresented positive depth.

## Bound allocation work

`maximum_band_count()` reports the number of height bands allocated for each section and query.

Both limits use `PRISMATIC_BAND_WORK_LIMIT = 2^20` entries:

- Construction: Section count times `maximum_band_count()`.
- Query: Query count times `maximum_band_count()`.

If a limit is exceeded, partition independent profiles or query rows into smaller batches.

## Handle a rejected request

| Error | Next action |
| --- | --- |
| `InvalidInput` | Correct the named operand or supply a profile within the certification requirements. |
| `Numerical` | Inspect the unresolved support, positivity, or width check before retrying. |
| `Runtime` | Read the native diagnostic and correct the selected runtime conditions. |

The capacity check admits positive area at or below its lower bound. Above the upper bound returns `InvalidInput`. Remaining uncertainty returns `Numerical`.

`section()` returns the source section index when known. Rejected queries leave the immutable batch usable. [Acceptance checks](acceptance.md#certified-surveyed-section-bounds) describe the tested cases and limits.

These queries calculate geometry only. They add no surveyed-profile flow advancement or measured-event accuracy claim.
