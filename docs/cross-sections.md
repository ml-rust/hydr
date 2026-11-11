# Evaluate channel cross-sections and recover stages

Use `CrossSectionBatch<R>` to calculate submerged geometry at supplied water levels. Start with an ordered station/elevation profile and an explicit supported stage limit.

Use `recover_stages` when submerged areas are known and water levels are needed.

A stage is a water-surface elevation. This component calculates geometry only. It does not advance flow or establish a hydraulic connection.

## Supply the surveyed profile

The public types live under `hydr::channel`. Each `CrossSectionDefinition` describes one profile.

Station means horizontal position across the section, in metres. Elevation means bed or bank height, in metres, relative to the declared vertical datum.

| Field | What you supply |
| --- | --- |
| `id` | Unique, nonempty section identity |
| `vertical_datum` | Nonempty elevation-reference identity |
| `points` | Ordered `CrossSectionPoint` values with `station_m` and `elevation_m` |
| `maximum_stage_m` | Explicit highest supported water level, in metres |
| `profile_provenance` | Nonempty survey or analytical-profile evidence |

Identity, datum, and provenance text must not contain control characters.

Every query stage must use its profile's declared datum. The batch does not convert elevations between datums.

### Order the points

1. Supply at least three points with finite coordinates.
2. Order stations without decreasing them across the profile.
3. Keep a positive finite horizontal extent between the first and last stations.
4. Remove identical consecutive points and any retracing within a vertical run.

All coordinate differences must also remain finite. Overhangs are unsupported.

Repeated stations allow vertical walls when their elevations differ. Each vertical run must move in one elevation direction.

### Set the supported stage limit

`maximum_stage_m` must lie above the minimum bed elevation. It cannot exceed either surveyed endpoint elevation.

This explicit limit prevents queries from inventing bank extensions. A query above the limit rejects, even if you expect water outside the supplied profile.

## Construct and evaluate a batch

1. Call `CrossSectionBatch::new(client, definitions)` with the configured Numr client.
2. Supply one finite stage per definition, in the same order, to `evaluate(&stages)`.
3. Call the returned query's `properties()` when you need owned results.

The stage count must exactly equal the definition count. `definitions()` exposes the immutable inputs.

Evaluation retains result tensors on the selected runtime and avoids full-array copies to the host. Scalar admission checks still read selected values.

`properties()` explicitly copies ordered results and can return an error. F64 means 64-bit floating-point arithmetic. Unsupported arithmetic returns an error without changing the selected runtime.

## Recover stages from submerged areas

1. Create a `CrossSectionRecoveryConfig`, or use `CrossSectionRecoveryConfig::default()`.
2. Supply one area per ordered definition to `batch.recover_stages(&areas_m2, &config)`.
3. Call the returned recovery's `properties()` to copy owned results to the host.

Each `CrossSectionRecoveredProperties` retains the original target bits in `requested_area_m2`. Its `geometry` field contains the recovered `CrossSectionProperties`, including `query_stage_m`.

Targets must be finite and nonnegative. Each target must lie within its profile's area at `maximum_stage_m`.

| Target | Recovered stage |
| --- | --- |
| Positive or negative zero | Original minimum bed elevation |
| Exact area at the supported cap | Original `maximum_stage_m` |
| Positive interior area | Represented wet stage within the declared support |
| Any area above capacity | Input error, including the next represented area above capacity |

Zero targets retain their sign in `requested_area_m2`. Their geometry reports the bed elevation. Zero area cannot identify an original level below the bed.

### Set stage and area tolerances

Solvr repeatedly halves a height interval above the bed to find the requested area. Recovery never searches above the supported cap.

| Config field | Default | Meaning |
| --- | --- | --- |
| `height_atol_m` | `1e-14` | Absolute local-height tolerance in metres |
| `height_rtol` | `1e-14` | Relative local-height tolerance |
| `area_atol_m2` | `1e-12` | Absolute area residual tolerance in m² |
| `area_rtol` | `1e-12` | Relative area residual tolerance |
| `max_iter` | `256` | Maximum bisection iterations |

Tolerance values must be finite and nonnegative. Each absolute/relative pair needs at least one positive value. `max_iter` must be positive.

A narrow height interval alone cannot establish an acceptable result. Recovery checks the resulting area against the original target.

The area residual must satisfy:

```text
abs(recovered_area_m2 - requested_area_m2)
    <= area_atol_m2 + area_rtol * abs(requested_area_m2)
```

A positive target always requires represented wet geometry, even when the absolute area tolerance exceeds that target.

### Account for benches and arithmetic limits

An exact recovered bench elevation uses the existing left-limit width and perimeter convention. Nearby accepted stages can lie across the bench.

Stage and area tolerances do not bound discontinuous top width or wetted perimeter near benches.

Recovery checks full geometry at every trial stage, including the pressure integral. It rejects unresolved positive integrals even when area alone remains representable.

Extreme coordinate scales can therefore reject an intermediate stage before the area solver converges. Recovery does not replace unresolved geometry with dry zero.

Large datum offsets can prevent a positive local height from producing a distinct represented stage. Recovery rejects when no represented stage meets the area tolerance.

An exhausted iteration budget or inadequate final area residual returns `Numerical`. Earlier query and recovery results remain usable after rejection.

## Read the geometric properties

Each result identifies its source through `section_id` and retains the supplied stage in `query_stage_m`.

Wetted perimeter means the length of bed and banks touching the water. It excludes the free water surface.

| Property | Meaning | Units |
| --- | --- | --- |
| `area_m2` | Submerged cross-sectional area, `A` | m² |
| `top_width_m` | Sum of water-surface widths of wet portions, `T` | m |
| `wetted_perimeter_m` | Wet bed and bank contact length, `P` | m |
| `hydraulic_radius_m` | Hydraulic radius, `A/P` | m |
| `hydraulic_depth_m` | Hydraulic depth, `A/T` | m |
| `hydrostatic_integral_m3` | Depth-weighted submerged area, `I1` | m³ |

Hydraulic radius and hydraulic depth divide area by different lengths. They are not interchangeable. The [USBR Water Measurement Manual](https://www.usbr.gov/tsc/techreferences/mands/wmm/chap02_11.html) defines these geometric ratios.

A flow calculation still needs its own roughness and hydraulic assumptions.

### Convert the pressure integral to force

`I1` weights each part of the submerged area by its depth below the water surface.

For hydrostatic force in newtons, use:

```text
F = rho * g * I1
```

Here `rho` is water density in kg/m³, and `g` is gravitational acceleration in m/s². Hydr returns the geometric integral in m³.

This follows the pressure-area relation in the [HEC-RAS 6.5 momentum derivation](https://www.hec.usace.army.mil/confluence/rasdocs/ras1dtechref/6.5/theoretical-basis-for-one-dimensional-and-two-dimensional-hydrodynamic-calculations/1d-unsteady-flow-hydrodynamics/momentum-equation).

## Interpret dry results and exact bench levels

### Dry geometry

A finite stage at or below the minimum bed returns `dry = true`.

| Result | Dry value |
| --- | --- |
| Area, wetted perimeter, and hydrostatic integral | Zero |
| Top width, hydraulic radius, and hydraulic depth | `None` |

For a dry result, `query_stage_m` still records the original requested level. It does not report a physical dry water surface.

### Horizontal benches

At an exact bench elevation, that horizontal bench contributes neither top width nor wetted perimeter. These quantities use the limits approached from below.

Immediately above the bench, its full horizontal length contributes. The calculation applies no smoothing across that change.

Vertical walls add submerged contact length to wetted perimeter. Their horizontal width is zero.

### Separated wet portions

A profile can contain several separated wetted portions. Results sum their geometry at the requested stage.

That sum establishes neither a hydraulic connection nor a common velocity. A flow model must make those physical decisions separately.

## Handle rejected definitions or queries

| Error | Meaning and next action |
| --- | --- |
| `InvalidInput` | Correct the named definition/query input or select admitted F64 arithmetic conditions. |
| `UnsupportedStage` | Supply a stage within the explicit limit, or extend the surveyed profile and its definition. |
| `Numerical` | Inspect coordinate scale, profile resolution, and stage. Positive geometry cannot be represented reliably. |
| `Runtime` | Read the operation diagnostic and correct the selected runtime conditions. |

`section()` returns the affected ordered section index when known. The immutable batch remains usable after an error.

Unresolved positive wet geometry returns a `Numerical` error. Hydr does not report it as dry zero.

## Current limits

This component supplies static geometry only. It provides no:

- Flow advancement, discharge, roughness, or conveyance calculation.
- Hydraulic connectivity decision for separated wet portions.
- Extrapolated bank geometry or above-limit stage support.
- Dynamic profiles or recovery coupled to flow state.
- Proven error bounds for individual results.

Extreme coordinate scales can reject even when the geometry is mathematically valid. Store the immutable definitions to reconstruct this static component. It has no timestep or water-ledger continuation state.
