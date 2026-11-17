# Advance a standalone reservoir

Use `hydr::runoff::ReservoirBatch` to advance storage and calculate the volume exported during an interval. Start by choosing a drainage branch and supplying explicit geometry.

The component supplies no hydraulic attachment or internal routing. Soil and evaporation require separate components.

## Choose the drainage branch

Depression storage is the depth retained before drainage begins.

| Geometry and roughness | Result |
| --- | --- |
| Positive roughness, width, and slope | Nonlinear drainage with finite roughness |
| Positive roughness, zero width or slope | Closed storage with zero export and endpoint discharge |
| Zero roughness, positive width and slope | Instantaneous overflow above depression storage |
| Zero roughness with zero width or slope | Error: ambiguous closure |

The method retains the native-derived runoff coefficient and exponent. Its storage equation is independently first-order implicit.

It reproduces neither SWMM's adaptive ODE controller nor endpoint-discharge volume bookkeeping.

## Construct the reservoir

1. Choose `from_si` or `from_native` for the supplied geometry.
2. Supply actual area and class area.
3. Supply width, slope, Manning coefficient, and depression depth.
4. Give each element a unique nonempty ID without control characters.
5. Supply explicit finite nonnegative antecedent depth.
6. Construct the batch with your configured Numr client.

| Constructor | Geometry units | Preserved values |
| --- | --- | --- |
| `from_si` | Areas in m², lengths in metres | Supplied SI geometry bits |
| `from_native` | Areas in ft², lengths in feet | Checked native geometry with SI conversion |

Slope and the Manning parameter remain dimensionless inputs under either constructor's convention. Both constructors cache separately checked native geometry and closure coefficients.

The stored foot bridge is `c=0.3048`. Area conversion uses `c*c`.

### Distinguish the two areas

- Actual area: Must be positive and no larger than class area. Storage and physical volumes use this area.
- Class area: Determines the runoff coefficient.

IDs establish identity without proving GIS disjointness. The component supplies no implicit rainfall or provider provenance.

## Advance and accept an interval

1. Choose a finite endpoint strictly after the accepted time.
2. Supply an explicit finite nonnegative incoming rate.
3. Keep that rate constant throughout the requested interval.
4. Call `trial_to(end_time, inputs)`.
5. Read candidate outputs or diagnostics.
6. Consume `into_next` to accept the candidate.

The represented duration is `dt=end_time-accepted_time`. The primitive supplies no duration alias or automatic timestep controller.

Trials borrow accepted state immutably. Rejection or discarding a candidate preserves:

- Depths and accepted time.
- Counters and water ledger.

`into_next` transfers ownership without numerical readback or state reconstruction.

### Respond to rejected calculations

Hydr clips no state and applies no numerical water adjustment. Choose a different admitted interval or correct the physical scales.

Errors include:

- Root exhaustion or stagnation.
- Unrepresentable root tolerances or event times.
- Positive supply or export that disappears through rounding.
- Nonfinite or unrepresentable positive physical quantities.

## Read outputs

| API | Returned information |
| --- | --- |
| Batch `states()` | Current represented depths |
| Batch `last_step()` | Last accepted tensor receipt, projected on request |
| Trial `outputs()` | Storage, interval input/export, endpoint discharge, and signed residuals |
| Trial `diagnostics()` | Exact interval, root work, aggregate volumes/scales, and ordered receipts |
| `ledger()` | Original/current storage, lifetime input/export, and cumulative residual |

Array output methods return fallible projections. Numerical continuation never consumes projected receipt arrays.

### Interpret export and endpoint discharge

Export is the water volume removed during the accepted interval. Endpoint discharge describes the flow rate at its end.

Finite-roughness export uses represented storage depletion. It never equals endpoint discharge multiplied by the full interval by definition.

Instantaneous overflow has no defined endpoint rate or discharge. Public outputs return `None`. Its inactive constitutive receipt fields record zero.

### Use the selected runtime

Numerical state remains SI F64 `Tensor<R>` fields on the configured client. The checked CPU reference uses one thread.

Other runtimes require [F64 arithmetic admission](integration.md#runtime-selection-and-admission).

Arithmetic probes require:

- Round-to-nearest ties-to-even.
- Gradual underflow.
- No FTZ/DAZ execution.

## Save and restore

| Task | API |
| --- | --- |
| Retain complete persistent state without projection | `snapshot()` |
| Create application-owned values | Fallible `snapshot_data()` |
| Restore trusted state | `restore` |
| Import owned values | `restore_data` |

Restore checks exact definition bits. It replaces accepted state only after the complete candidate passes admission.

Owned data retains:

- Elements and config.
- Time, depths, ledger, and accepted-step count.
- Complete last diagnostics.

Import reuses numerical receipt checks without replaying roots. Aggregate interval and cumulative gates remain independent.

Receipt consistency cannot authenticate historical input or export. Every package carries `schema` and `contract` identity fields. `snapshot_data()` fills them from `RESERVOIR_SNAPSHOT_SCHEMA` and `RESERVOIR_CONTRACT_VERSION`. `restore_data` compares both before any other check and rejects a foreign value with an input error. The error names the found and expected values. Capture a new package to recover. Packages without the identity fields are rejected.

Exact restart requires the same build/runtime, ordered batch layout, and accepted interval partition.

Different batch shapes carry no bitwise equality promise.

## Understand the calculation

### Drainage law

Finite-roughness parameters retain this operation order:

```text
p = 1.6666667
alphaFt = 1.49 * widthFt / classAreaFt2 * sqrt(slope) / n
alphaSI = alphaFt * c^(1-p)
q(d) = alphaSI * max(d-ds, 0)^p
```

`q` is a depth-law rate in m/s. Endpoint discharge is `actualArea*q`.

Positive drainage geometry requires positive representable native and SI coefficients. Underflow cannot silently select closed storage.

No native `ZERO` cutoff applies. Every represented positive film remains eligible for its declared branch.

### Supply and depression crossing

Supply is calculated once:

```text
incomingDepth = incomingRate * dt
B = oldDepth + incomingDepth
```

When `B<=ds`, storage retains `B` and export is zero. Closed and instantaneous branches use `B` algebraically.

Instantaneous overflow exports `actualArea*max(B-ds,0)` once. It retains `min(B,ds)`.

Finite-roughness drainage uses `tau=dt` when old depth already reaches depression storage. Otherwise, a crossing interval uses:

```text
fillTime = (ds-oldDepth)/incomingRate
tau = dt-fillTime
```

Crossing times must lie strictly inside the original interval. Collapsed or unrepresentable event times reject. Accounting always retains original `B`.

### Solve the excess depth

Existing Solvr batched bisection solves the unique bracketed root:

```text
X = B-ds
G(x) = x + tau*alphaSI*x^p - X
0 <= x <= X
depth = ds+x
```

Bracket expansion stays disabled. Root controls cannot relax or replace physical acceptance gates.

After reconstruction, active drainage requires `ds<depth<B` and positive represented excess. A root rounded back to depression storage rejects. Unresolved positive export also rejects.

### Check storage and law agreement

Export uses `actualArea*(B-depth)`. The independent constitutive check uses endpoint-law volume `actualArea*(tau*q(depth))`.

| Control | Default |
| --- | --- |
| `root_absolute_depth` | One F64 subnormal quantum in metres |
| `root_relative_depth` | `64*f64::EPSILON` |
| `root_iterations` | `512` bisection rounds |
| `volume_absolute` | `1e-9 m³` |
| `volume_relative` | `1e-12` |

Tolerance pairs must be finite and nonnegative. At least one value per pair must be positive. Work limits must be positive.

Constitutive and storage residuals use `absolute+relative*scale` without a `1 m³` floor. Gates apply to:

- Each element.
- Aggregate intervals.
- Cumulative storage accounting.

Positive depths and incoming/export depths require positive finite physical volumes. Positive supply cannot disappear through interval underflow or absorption into unchanged available depth.

Positive endpoint laws require represented positive rates and discharges.

Receipts retain the original numerical evidence:

- Old depth, supply, and active duration.
- Root bracket, accepted depth, and export.
- Law quantities, physical volumes, and independent residual scales.

## Try the existing overflow example

Run the [standalone example](../examples/reservoir.rs):

```sh
cargo run --example reservoir
```

The instantaneous-overflow fixture starts at zero antecedent depth with explicit incoming water.

| Input | Value |
| --- | --- |
| Actual and class areas | `1024 ft²` |
| Width | `32 ft` |
| Slope | `1/64` |
| Roughness | Zero |
| Depression depth | `0.125 ft` |
| Incoming rate | `1/1024 ft/s` |
| Duration | `256 s` |

Declared supply is `256 ft³`. Storage retains `128 ft³`. Integrated export removes `128 ft³` once.

The example prints:

- Accepted time and incoming volume.
- Storage, export, and water residual.
- Endpoint discharge `None`.

## Check timestep accuracy

The archive-free acceptance case uses common native geometry and the zero-input recession below.

| Input | Value |
| --- | --- |
| Actual and class areas | `1024 ft²` each |
| Width | `32 ft` |
| Slope | `1/64` |
| Manning coefficient | `1/32` |
| Depression depth | `0.125 ft` |
| Initial depth | `0.25 ft` |
| Incoming rate | Zero |
| Horizon | `64 s` |

The analytical reference solves recession under the stored SI coefficient and exponent `1.6666667`.

| Error target | Maximum |
| --- | --- |
| Depth | `1.476400894632169e-5 m` |
| Integrated export | `5.844668635035667e-3 m³` |

Refinement requires decreasing errors throughout and both targets at the finest interval.

| Timestep, s | Analytical depth error, m | Analytical integrated-export error, m³ | Targets met |
| --- | --- | --- | --- |
| `1` | `1.56427743892011140e-4` | `1.48813956586595353e-2` | None |
| `0.5` | `7.84531671304961042e-5` | `7.46346262942854111e-3` | None |
| `0.25` | `3.92871201448860563e-5` | `3.73749032376835544e-3` | Export only |
| `0.125` | `1.96587877247453302e-5` | `1.87019380976272132e-3` | Export only |
| `0.0625` | `9.83321277039927999e-6` | `9.35460207961202883e-4` | Depth and export |

The `0.125 s` interval misses the depth target. The `0.0625 s` interval meets both unchanged targets.

The coarser rows remain part of the measured record and establish no passing accuracy claim.

Run the existing refinement check without external archives:

```sh
cargo test --test nonlinear_reservoir accuracy::recession_refinement_reduces_depth_and_integrated_export_errors -- --nocapture
```

The native comparison uses the same geometry and initial depth over eight `8 s` SWMM steps. Depth and endpoint-rate differences also decrease through this sequence.

Hydr's first-order implicit method and integrated export differ from native adaptive stepping and endpoint quadrature.

This selected recession case establishes no general timestep requirement, native bookkeeping equivalence, or field-accuracy guarantee.

## Compare methods and respect scope limits

The [untouched native oracle](runoff-reference.md) remains an independent comparison. Compare depth and endpoint-rate behavior under refinement.

Coarse-step identity is unsupported because the methods and export definitions differ.

This primitive adds no engine capability claim. These mechanisms require separately checked units:

- Soil and evaporation.
- Paired routing.
- Hydraulic attachment.
