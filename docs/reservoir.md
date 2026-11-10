# Conservative reservoir primitive

`hydr::runoff::ReservoirBatch` advances standalone storage and conservative interval export. It supplies no hydraulic attachment, soil, evaporation, or internal routing.

The method preserves the native-derived runoff coefficient and exponent. It uses an independent first-order implicit storage equation. It does not reproduce SWMM's adaptive ODE controller or endpoint-discharge volume bookkeeping.

## Explicit construction

Run the [standalone example](../examples/reservoir.rs):

```sh
cargo run --example reservoir
```

The example uses the established instantaneous-overflow fixture, with zero antecedent depth and explicit incoming water.
Its native areas are `1024 ft²`, width `32 ft`, slope `1/64`, roughness zero, and depression depth `0.125 ft`.
Incoming rate `1/1024 ft/s` acts for `256 s`.
The declared supply is `256 ft³`. Storage retains `128 ft³`, and integrated export removes `128 ft³` once.
The example prints accepted time, incoming volume, storage, export, water residual, and endpoint discharge `None`.

`from_si` accepts actual area, class area, width, slope, Manning coefficient, and depression depth. Areas use square metres and lengths use metres. It preserves supplied SI geometry bits.

`from_native` accepts square-foot areas and foot lengths with the same dimensionless slope and Manning parameter. Both constructors cache separately checked native geometry and closure coefficients. The stored foot bridge is `c=0.3048`, with area conversion `c*c`.

Actual storage area must be positive and no larger than class area. The coefficient uses class area. Storage and physical volume products use actual area. Element IDs must be unique, nonempty, and free of control characters. IDs establish identity without asserting GIS disjointness.

Antecedent depth is explicit, finite, and nonnegative. Incoming rate is explicit, finite, nonnegative, and constant over the requested interval. The primitive supplies no implicit rainfall or provider provenance.

## Closure branches

| Branch | Physical contract |
| --- | --- |
| Positive roughness, positive width/slope | Nonlinear finite-roughness drainage |
| Positive roughness, zero width or slope | Closed storage with zero export and endpoint discharge |
| Zero roughness, positive width/slope | Explicit instantaneous overflow above depression storage |
| Zero roughness with zero width/slope | Rejected ambiguous closure |

Finite-roughness parameters retain this operation order:

```text
p = 1.6666667
alphaFt = 1.49 * widthFt / classAreaFt2 * sqrt(slope) / n
alphaSI = alphaFt * c^(1-p)
q(d) = alphaSI * max(d-ds, 0)^p
```

`q` is a depth-law rate in metres/second. Endpoint discharge is `actualArea*q`. Positive drainage geometry requires positive representable native and SI coefficients. Underflow cannot silently select closed storage.

Instantaneous overflow exports `actualArea*max(B-ds,0)` once and retains `min(B,ds)`. Its endpoint rate and discharge are undefined. Public outputs use `None`. Its inactive constitutive receipt fields record zero.

The production law applies no native `ZERO` cutoff. Every representable positive film remains eligible for its declared branch.

## Exact interval and depression event

`trial_to(end_time, inputs)` uses the represented interval `dt=end_time-accepted_time`. It requires a finite strictly increasing endpoint. It supplies no duration alias or automatic timestep controller.

Supply is calculated once:

```text
incomingDepth = incomingRate * dt
B = oldDepth + incomingDepth
```

When `B<=ds`, the primitive retains `B` and exports zero. Closed and instantaneous branches use `B` algebraically.

Active finite-roughness drainage uses `tau=dt` when the old depth already reaches depression storage. A crossing interval uses:

```text
fillTime = (ds-oldDepth)/incomingRate
tau = dt-fillTime
```

Crossing times must remain strictly inside the original interval. Collapsed or unrepresentable event times reject the trial. Accounting always retains original `B`.

## Root and independent admission

Existing solvr batched bisection solves the unique bracketed excess-depth root:

```text
X = B-ds
G(x) = x + tau*alphaSI*x^p - X
0 <= x <= X
depth = ds+x
```

Bracket expansion stays disabled. Root controls cannot silently relax or replace physical acceptance gates.

Accepted state is checked independently after reconstruction. Active drainage requires `ds<depth<B` and positive represented excess. A root that rounds back to depression storage rejects. Unresolved positive export also rejects.

Export is defined from represented storage depletion, `actualArea*(B-depth)`. It never uses endpoint discharge multiplied by the full interval. Endpoint-law volume uses `actualArea*(tau*q(depth))` for its independent constitutive check.

| Control | Default |
| --- | --- |
| `root_absolute_depth` | One F64 subnormal quantum in metres |
| `root_relative_depth` | `64*f64::EPSILON` |
| `root_iterations` | `512` bisection rounds |
| `volume_absolute` | `1e-9 m³` |
| `volume_relative` | `1e-12` |

Tolerance pairs are finite and nonnegative, with at least one positive value per pair. Work limits are positive. Root exhaustion, stagnation, and unrepresentable tolerances return errors.

Both constitutive and storage residuals use `absolute+relative*scale`, without a `1 m³` floor. Checks apply per element, to aggregate intervals, and to cumulative storage accounting.

Positive depths and incoming/export depths require positive finite represented volumes. Positive supply cannot disappear through interval underflow or absorption into unchanged available depth. Positive endpoint laws require representable positive rates and discharges.

No clipping or numerical water adjustment repairs rejected state. The caller can choose a different admitted interval or correct physical scales.

## Persistent state and outputs

The primitive keeps SI F64 `Tensor<R>` fields on a caller-configured Numr client. The checked CPU reference uses one thread. Other runtimes require [F64 arithmetic admission](integration.md#runtime-selection-and-admission). Shared environment probes require round-to-nearest ties-to-even and gradual underflow. FTZ/DAZ execution is unsupported.

Trials borrow accepted state immutably. Rejection or discarding a trial preserves every depth, time, counter, and ledger. `into_next` accepts ownership without numerical readback or state reconstruction.

| API | Output boundary |
| --- | --- |
| Batch `states()` | Current represented depths |
| Batch `last_step()` | Complete last accepted tensor receipt projected on request |
| Trial `outputs()` | Per-element storage, interval input/export, endpoint discharge, and signed residuals |
| Trial `diagnostics()` | Exact interval, root work, aggregate volumes/scales, and ordered receipts |
| `ledger()` | Original/current storage, lifetime input/export, and cumulative residual |

Array output methods return fallible projections. Receipts retain old depth, supply, active duration, root bracket, represented accepted depth, and export. They also retain law quantities, physical volumes, and independent residual scales. Numerical continuation never consumes projected receipt arrays.

## Complete restart and limits

`snapshot()` clones complete persistent state without projection. `snapshot_data()` produces fallible caller-owned values. `restore` and `restore_data` check exact definition bits and replace state only after complete candidate admission.

Owned data retains elements, config, time/depths, ledger, accepted-step count, and complete last diagnostics. Import reuses numerical receipt checks without replaying roots. Aggregate interval and cumulative gates remain independent.

Receipt consistency cannot authenticate historical input or export. Exact restart requires the same build/runtime, ordered batch layout, and accepted interval partition. Different batch shapes carry no bitwise equality promise.

The [untouched native oracle](runoff-reference.md) remains an independent comparison. Compare depth and endpoint-rate behavior under refinement. Coarse-step identity is unsupported because the methods and export definitions differ.

## Measured recession refinement

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
The frozen depth target is `1.476400894632169e-5 m`. The integrated-export target is `5.844668635035667e-3 m³`.
The test requires decreasing errors throughout refinement and both targets at the finest interval.

| Timestep, s | Analytical depth error, m | Analytical integrated-export error, m³ | Depth target | Export target |
| --- | --- | --- | --- | --- |
| `1` | `1.56427743892011140e-4` | `1.48813956586595353e-2` | Above | Above |
| `0.5` | `7.84531671304961042e-5` | `7.46346262942854111e-3` | Above | Above |
| `0.25` | `3.92871201448860563e-5` | `3.73749032376835544e-3` | Above | Within |
| `0.125` | `1.96587877247453302e-5` | `1.87019380976272132e-3` | Above | Within |
| `0.0625` | `9.83321277039927999e-6` | `9.35460207961202883e-4` | Within | Within |

The `0.125 s` interval misses the depth target. The `0.0625 s` interval meets both unchanged targets.
The coarser rows remain part of the measured record and establish no passing accuracy claim.

Run the existing refinement check without external archives:

```sh
cargo test --test nonlinear_reservoir accuracy::recession_refinement_reduces_depth_and_integrated_export_errors -- --nocapture
```

The native comparison uses the same geometry and initial depth over eight `8 s` SWMM steps.
Depth and endpoint-rate differences also decrease through this sequence.
Hydr's first-order implicit method and integrated export differ from native adaptive stepping and endpoint quadrature.
This selected recession case establishes no general timestep requirement, native bookkeeping equivalence, or field-accuracy guarantee.

This primitive adds no engine capability claim. Soil, evaporation, paired routing, and hydraulic attachment require separately checked units.
