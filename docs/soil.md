# Calculate soil infiltration

Use `ModifiedGreenAmptBatch<R>` to calculate infiltration while retaining complete soil memory. Start by choosing standalone or coupled calculation.

| Your task | Component |
| --- | --- |
| Calculate raw infiltration rates and soil state | This standalone soil component |
| Apply infiltration to surface water with evaporation, accounting, and retries | [Land sources](land-sources.md) |

Infiltration means water entering the soil. This component follows the Modified Green–Ampt method in EPA SWMM v5.2.4.

Read capabilities with `soil_capabilities()`. It returns a `ComponentCapabilities` declaration: runtimes, sources, parent components, unsupported uses, and limits.

## Construct the soil batch

1. Supply soil parameters in SI or native EPA units.
2. Choose explicit EPA initialization or complete antecedent state.
3. Supply the configured Numr `R::Client`.

`hydr::soil::ModifiedGreenAmptBatch<R>` keeps its numerical state in Numr `Tensor<R>` fields. The checked CPU reference uses one thread. Other runtimes require [F64 arithmetic admission](integration.md#runtime-selection-and-admission).

### Choose parameter units

Use `ModifiedGreenAmptParameters`. Both constructors take a dimensionless maximum deficit. Native refers to EPA's internal units and state fields.

| Method | Suction | Conductivity |
| --- | --- | --- |
| `from_si` | Metres | Metres/second |
| `from_native` | Feet | Feet/second |

Parameters require finite nonnegative suction, positive conductivity, and maximum deficit in `[0,1]`. Derived `Lu` and recovery time must remain finite and positive. Positive SI/native parameter conversions must retain their values.

### Choose initial soil memory

Use `ModifiedGreenAmptBatch<R>` with the configured `client` and soil `parameters`.

| Method | Initial state |
| --- | --- |
| `initialized` | Explicit EPA initialization from `client` and `parameters` |
| `from_native_states` | All five checked native `states`, alongside `client` and `parameters` |

## Advance and accept an interval

1. Call `trial` or `trial_native` with one input per cell.
2. Read `outputs()` when you need interval results.

Then choose one outcome:

| Decision | Action |
| --- | --- |
| Accept the candidate | Consume `into_next()` to obtain the complete next batch. |
| Keep the original state | Discard the trial without calling `into_next()`. |

```rust
use hydr::soil::{ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, SoilInput};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

let parameters = ModifiedGreenAmptParameters::from_si(0.12, 3e-6, 0.3)?;
let client = CpuClient::new(CpuDevice::new())
    .with_parallelism(ParallelismConfig::new(Some(1), None));
let accepted = ModifiedGreenAmptBatch::<CpuRuntime>::initialized(client, &[parameters])?;
let trial = accepted.trial(60.0, &[SoilInput {
    incoming_rate_m_per_s: 1e-5,
    ponded_depth_m: 0.01,
}])?;
let output = trial.outputs()?;
let next = trial.into_next();
let restart_state = next.native_states()?;
# Ok::<(), hydr::soil::SoilError>(())
```

Both trial methods require a positive finite duration.

### Supply water once

Inputs describe nonnegative incoming rain/runon and nonnegative ponding. They exclude evaporation. Requested forcing never overrides native branch rules.

| Input path | Arithmetic contract |
| --- | --- |
| `trial` | SI input tensors divide by `0.3048` |
| `trial_native` | Inputs retain exact feet/second and feet branch probes |
| SI output | Native rate multiplies by `0.3048` before interval-depth multiplication |

Selected interval rates, conversions, depths, and final state must remain representable. A source-returned zero rate stays zero independently of rounded state differences.

### Read results or continue

| API | Result |
| --- | --- |
| `outputs()` | Interval rates and depths in native and SI units |
| `into_next()` | Complete accepted candidate |
| `native_states()` | All five state fields for output or reconstruction |

Reading result arrays copies them from the selected runtime and can return an error.

Repeated trials borrow the same immutable state. Discarding a trial preserves every accepted field. Numerical continuation uses persistent tensors without copying state to the host between accepted intervals.

## Save complete soil memory

Reconstruction requires the same checked parameters and every field below. Do not reconstruct state from cumulative infiltration alone.

Soil memory persists only inside surface-solver and subcatchment packages, under their schema and contract identities.

| Field | Meaning and admitted range |
| --- | --- |
| `imd` | Dimensionless initial moisture deficit, from zero through its exact recovery ceiling |
| `f_ft` | Recoverable event infiltration memory in feet, including legal negative values |
| `fu_ft` | Upper-zone memory in feet, from zero through native capacity |
| `saturated` | Independent saturation flag |
| `t_s` | Recovery timer in seconds, including negative expired values |

`F` tracks recoverable event memory. Use a separate ledger for lifetime infiltration. A reachable wet/dry sequence produces negative `F` while `Fu` remains positive.

Saturated dry intervals preserve moisture memory and saturation while decrementing `T`. Finite signed `F` and `T` remain admissible independently of saturation. Import adds no guessed history restrictions.

### Recovery ceiling and small capacities

The `IMD` ceiling follows the implemented arithmetic:

```text
C=fl(IMDmax*Lu)
M_upper=max(IMDmax,fl(C/Lu))
Recovery: fl(fl(C-Fu)/Lu), with 0<=Fu<=C
```

Here `fl` means the represented floating-point result. Monotonic subtraction and division bound recovery by `fl(C/Lu)`. The rounding band is `M_upper-IMDmax`.

Admission preserves endpoint state without clamping. Positive subnormal capacity (below the normal floating-point range) remains admissible. A positive maximum deficit with capacity rounded to zero rejects.

## Understand the preserved numerical rules

The implementation is pinned to EPA commit `7952ca837988b1c32f791812eccc9fd64547e093`. The [reference harness](soil-reference.md) records original C results.

Solvr supplies generic Newton iteration. Hydr supplies soil equations, native branches, and the prescribed fallback.

| Mechanism | Preserved behavior |
| --- | --- |
| Upper-zone depth | Sequential `Ks*12*3600`, then `4*sqrt(ksat)/12` |
| Saturation threshold | `Fs=((ks*(S+depth))*IMD)/(ia-ks)` |
| Hydraulic coefficient | `c1=(S+depth)*IMD` |
| Effective availability | `ia<1e-10 ft/s` selects effective zero |
| Saturation equality | Native strict comparisons preserve equality branches |
| Saturated event update | `F += F2-F`, retaining the original rounding order |
| Unsaturated crossing | Direct assignment `F=F2` |
| Root iteration | Solvr batched Newton, at most 20 corrections, strict pre-update `abs(correction)<1e-5 ft` |
| Root fallback | Every nonconverged or nonfinite Newton lane selects native `F2min` |

`InfilFactor` and `Evap.recoveryFactor` equal one. Inactive tensor expressions use safe operands. Active invalid logarithms or singular derivatives reach native fallback. Legal signed event memory receives no premature logarithm-domain rejection.

Tiny positive ponding can enter effective-zero recovery. Saturated zero-availability intervals preserve saturation.

## Check agreement with EPA results

Generate a fresh 72-case/158-row native corpus with the [reference harness](soil-reference.md). Then run:

```sh
cargo run --example soil_reference -- resource/evidence/soil-component-oracle-extended/soil.csv
cargo test --test soil_component
```

The example reads the full native schema. It compares initialization, every supported modified STEP, interval rate/depth, and all five state fields.

Direct F2 probes run through saturated public states. Original-model cases and the zero-Ks helper receive explicit unsupported reports.

### Comparison criteria

The checked corpus covers 60 modified state sequences, 69 STEP calls, and nine F2 probes.

| Quantity | Frozen comparison tolerance |
| --- | --- |
| Native depths and event/upper-zone memory | `1e-12 ft + 1e-11 relative` |
| Native rate | `1e-14 ft/s + 1e-11 relative` |
| Moisture deficit | `1e-12 + 1e-11 relative` |
| Recovery time | `1e-9 s + 1e-11 relative` |
| Saturation and zero/reset branch witnesses | Exact |

Rotated batches check 17,550 lanes at lengths 31, 32, 33, 64, and 65. Branch probes occupy SIMD positions and scalar tails.

Lifecycle checks cover mixed/independent batches, SI/native conversion, immutable trials, complete restart, negative memory, and rejected-trial rollback.

### What these checks establish

The infiltration logarithm is correctly rounded, so its bits do not depend on batch length or lane position. The checked layouts meet the frozen tolerances. Bitwise restart requires a fixed batch layout, build, and runtime. Cross-shape bitwise equality remains unestablished: no test compares the same cell across batch shapes.

## Current limits

Standalone outputs describe raw empirical soil infiltration. This API excludes:

- Surface-water availability handling and conservative surface coupling.
- Evaporation, coupled retries, and lifetime water ledgers.
- Checkpoint packaging and the original Green–Ampt model.

Use [land sources](land-sources.md) for atomic surface coupling, numerical admission, evaporation, ledgers, and complete-state checkpoints.
