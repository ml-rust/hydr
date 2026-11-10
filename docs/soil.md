# Standalone Modified Green–Ampt soil

`hydr::soil::ModifiedGreenAmptBatch<R>` advances complete soil memory on persistent Numr `Tensor<R>` fields. Construction receives the configured `R::Client`. The checked CPU reference uses one thread. Other runtimes require [F64 arithmetic admission](integration.md#runtime-selection-and-admission).

The component follows EPA SWMM v5.2.4 Modified Green–Ampt, pinned at `7952ca837988b1c32f791812eccc9fd64547e093`. The [reference harness](soil-reference.md) records original C results. Generic Newton iteration belongs to solvr. Hydr supplies the soil correction, native branches, and prescribed fallback.

## Explicit construction and trials

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

| API | Contract |
| --- | --- |
| `ModifiedGreenAmptParameters::from_si` | Suction in metres, conductivity in metres/second, dimensionless maximum deficit |
| `ModifiedGreenAmptParameters::from_native` | Suction in feet, conductivity in feet/second, dimensionless maximum deficit |
| `ModifiedGreenAmptBatch::<R>::initialized(client, parameters)` | Explicit EPA antecedent initialization on the selected client |
| `ModifiedGreenAmptBatch::<R>::from_native_states(client, parameters, states)` | Checked complete antecedent state without inferred history |
| `trial` / `trial_native` | Immutable interval computation with positive finite duration and one input per cell |
| `outputs` | Native/SI interval rates and depths, projected at the application output boundary |
| `into_next` | Accept the complete candidate batch |
| `native_states` | Project complete state for output or reconstruction |

Inputs represent nonnegative incoming rain/runon and nonnegative ponding. They exclude evaporation. Native trials retain exact feet/second and feet branch probes. SI trials divide input tensors by `0.3048`. SI output rate multiplies native rate by `0.3048` before interval-depth multiplication.

Discarding a trial preserves every accepted field. Repeated trials borrow the same immutable state. Reconstruction requires the same checked parameters and all five native fields. Numerical continuation uses persistent tensors, without host-state projection between accepted intervals.

## Complete native memory

| Field | Meaning |
| --- | --- |
| `imd` | Dimensionless initial moisture deficit |
| `f_ft` | Event infiltration memory in feet, including legal negative values |
| `fu_ft` | Upper-zone memory in feet, bounded by native capacity |
| `saturated` | Independent saturation flag |
| `t_s` | Recovery timer in seconds, including negative expired values |

`F` is recoverable event memory. It is not a lifetime infiltration ledger. A reachable wet/dry sequence produces negative `F` while `Fu` remains positive. Saturated dry intervals preserve moisture memory and saturation while decrementing `T`.

Finite signed `F` and `T` remain admissible independently of saturation. Imported state receives no guessed history restrictions. `Fu` must lie between zero and native upper-zone capacity. `IMD` must lie between zero and its exact recovery ceiling.

The ceiling follows the implemented recovery arithmetic. Let `C=fl(IMDmax*Lu)` and `M_upper=max(IMDmax,fl(C/Lu))`. Recovery uses `fl(fl(C-Fu)/Lu)` with `0<=Fu<=C`. Monotonic subtraction and division bound that result by `fl(C/Lu)`. The rounding band is `M_upper-IMDmax`. Admission preserves endpoint state without clamping. Positive subnormal capacity remains admissible. Positive maximum deficit with capacity rounded to zero is rejected.

Parameters require finite nonnegative suction, positive conductivity, and maximum deficit in `[0,1]`. Derived `Lu` and recovery time must remain finite and positive. Positive SI/native parameter bridges must retain their values. Selected interval rates, unit conversions, depths, and final state must remain representable. Source-returned zero rate remains zero, independently of rounded state differences.

## Numerical semantics

| Mechanism | Preserved behavior |
| --- | --- |
| Upper-zone depth | Sequential `Ks*12*3600`, then `4*sqrt(ksat)/12` |
| Saturation threshold | `Fs=((ks*(S+depth))*IMD)/(ia-ks)` |
| Hydraulic coefficient | `c1=(S+depth)*IMD` |
| Effective availability | `ia<1e-10 ft/s` selects effective zero |
| Saturation equality | Native strict comparisons preserve equality branches |
| Saturated event update | `F += F2-F`, retaining the original rounding order |
| Unsaturated crossing | Direct assignment `F=F2` |
| Root iteration | solvr batched Newton, at most 20 corrections, strict pre-update `abs(correction)<1e-5 ft` |
| Root fallback | Every nonconverged or nonfinite Newton lane selects native `F2min` |

`InfilFactor` and `Evap.recoveryFactor` equal one. Inactive tensor expressions use safe operands. Active invalid logarithms or singular derivatives reach native fallback. Legal signed event memory receives no premature logarithm-domain rejection.

Tiny positive ponding can enter effective-zero recovery. Saturated zero-availability intervals preserve saturation. Requested forcing never overrides these native branch rules.

## Differential checks

Generate a fresh 72-case/158-row native corpus with the reference harness. Run the public comparison example:

```sh
cargo run --example soil_reference -- resource/evidence/soil-component-oracle-extended/soil.csv
cargo test --test soil_component
```

The example reads the full native schema. It compares initialization, every supported modified STEP, interval rate/depth, and all five fields. Direct F2 probes run through saturated public states. Original-model cases and the zero-Ks helper receive explicit unsupported reports.

The checked corpus covers 60 modified state sequences, 69 STEP calls, and nine F2 probes.

| Quantity | Frozen comparison tolerance |
| --- | --- |
| Native depths and event/upper-zone memory | `1e-12 ft + 1e-11 relative` |
| Native rate | `1e-14 ft/s + 1e-11 relative` |
| Moisture deficit | `1e-12 + 1e-11 relative` |
| Recovery time | `1e-9 s + 1e-11 relative` |
| Saturation and zero/reset branch witnesses | Exact |

Rotated oracle batches check 17,550 lanes at lengths 31, 32, 33, 64, and 65. These place branch probes in SIMD and scalar tail positions. Public lifecycle checks cover mixed/independent batches, SI/native conversion, immutable trials, complete-state restart, negative memory, and rejected-trial rollback.

Checked scalar and SIMD logarithm paths conform within the frozen comparison tolerances. Bitwise restart requires a fixed batch layout, build, and runtime. No cross-shape bitwise equality is established.

## Scope

Outputs are raw empirical soil infiltration. This component supplies no conservative surface-water coupling claim. Surface availability handling, evaporation, coupled retries, lifetime ledgers, checkpoint packaging, and the original Green–Ampt model remain outside this standalone API.

The separate [land-source engine integration](land-sources.md) supplies atomic surface coupling, numerical admission, evaporation, ledgers, and complete-state checkpoints.
