# Evaluate channel cross-section geometry

Use `CrossSectionBatch<R>` to calculate submerged geometry at supplied water levels. Start with an ordered station/elevation profile and an explicit supported stage limit.

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
- Proven error bounds for individual results.

Extreme coordinate scales can reject even when the geometry is mathematically valid. Store the immutable definitions to reconstruct this static component. It has no timestep or water-ledger continuation state.
