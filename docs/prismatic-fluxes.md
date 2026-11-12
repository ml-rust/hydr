# Calculate interface fluxes for surveyed channels

Use `riemann_flux_tensor` to bound water and momentum fluxes between supplied channel states. First [certify your surveyed sections](prismatic-sections.md).

A Riemann query calculates the waves where two constant states meet. The query samples their solution at the fixed interface. For reach advancement, `ChannelReach` currently admits rectangular and triangular reaches through the [channel-flow API](channels.md).

## Supply the states

Construct `CrossSectionBatch` with your configured Numr client, then call `certify_prismatic()` once. The returned `PrismaticSectionBatch<R>` owns the geometry.

All five input tensors must have matching rank-one shape `[N]` on that client's runtime and device.

| Input | Type | Meaning |
| --- | --- | --- |
| `owners` | `I64` | Zero-based section-row indices into `definitions()`. Repeated and unordered indices are valid. |
| `left_area_m2`, `right_area_m2` | `F64` | Finite, nonnegative wet areas, in m² |
| `left_discharge_m3_per_s`, `right_discharge_m3_per_s` | `F64` | Finite signed discharges, in m³/s |

Each owner selects one geometry for both sides. Owners identify section rows, never channel cells or interface numbers.

An exactly dry state requires `A = 0` and `Q = 0`. Positive area remains wet without a dry threshold.

The configured client must support F64 interval arithmetic and Solvr's certified root and integration operations. See [runtime admission](integration.md#runtime-selection-and-admission).

## Call the query

With a certified `batch` and its configured `client`, these tensors describe one interface using section row zero:

```rust
use hydr::channel::PrismaticRiemannConfig;
use numr::{runtime::RuntimeClient, tensor::Tensor};

let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
let left_area = Tensor::from_slice(&[1.0_f64], &[1], client.device())?;
let left_discharge = Tensor::from_slice(&[0.0_f64], &[1], client.device())?;
let right_area = Tensor::from_slice(&[0.5_f64], &[1], client.device())?;
let right_discharge = Tensor::from_slice(&[0.0_f64], &[1], client.device())?;
let config = PrismaticRiemannConfig::default();
let result = batch.riemann_flux_tensor(
    &owners, &left_area, &left_discharge,
    &right_area, &right_discharge, &config,
)?;
```

Replace the example areas with states inside your profile's supported capacity. The query also checks support throughout the wave solution.

## Read fluxes and interface traces

Results retain original query order and independent input snapshots. Native bounds use Solvr `Interval<R>` with `lower()` and `upper()` tensors.

| Getter | Meaning and units |
| --- | --- |
| `water_flux_m3_per_s()` | Water flux across the interface, m³/s |
| `momentum_flux_m4_per_s2()` | Momentum flux across the interface, m⁴/s² |
| `trace_area_m2()`, `trace_discharge_m3_per_s()`, `trace_depth_m()` | Bounds on admissible interface states, in m², m³/s, and m |
| `trace_dry_possible()`, `trace_wet_possible()` | F64 masks. `1` means that trace type remains possible, `0` excludes it. |
| `wave_speed_m_per_s()` | Signed envelope of wave speeds, m/s, including zero. Negative speeds travel left and positive speeds travel right. |

A trace is the state immediately beside the interface. A stationary shock has two traces with the same flux. Trace bounds can therefore remain wide even when flux bounds are narrow. Judge numerical accuracy using flux widths. Trace widths have no separate acceptance allowance.

- Result storage: Keep every tensor and endpoint alias read-only, including cloned aliases.
- Host values: Use `try_to_vec::<f64>()` on endpoints and handle device-copy errors.

## Choose bounds and work limits

`PrismaticRiemannConfig::default()` supplies these defaults:

| Field | Default |
| --- | --- |
| `gravity_m_per_s2` | `9.81` m/s² |
| `recovery.depth_atol_m`, `recovery.depth_rtol` | `1e-12` m and `1e-12` |
| `water_flux_atol_m3_per_s` | `1e-9` m³/s |
| `momentum_flux_atol_m4_per_s2` | `1e-8` m⁴/s² |
| `flux_rtol` | `1e-10` |
| `initial_simpson_panels` | `4` integration panels per active height band |
| `max_simpson_panels` | `4096` integration panels per active height band |
| `max_refinements` | `10` panel doublings |
| `max_native_query_points` | `1 << 24` root and integration evaluation points |

Each complete flux-interval width must fit `atol + flux_rtol * max(abs(lower), abs(upper))`, checked conservatively with interval arithmetic. Depth recovery uses the analogous absolute-plus-relative width allowance. Root convergence alone never accepts a flux.

Numerical bounds describe the supplied profile and states. They establish neither survey accuracy nor measured-event accuracy.

Star roots start from the supplied water depths. Solvr expands the search within the surveyed cap before bisection.

| Control | Meaning |
| --- | --- |
| `root.atol`, `root.rtol` | Root-width allowances in normalized depth |
| `root.max_iterations` | Maximum bisection midpoint rounds |
| `max_bracket_expansions` | Maximum search expansions. Default: `4096`. Zero checks the initial endpoints only. |
| `root.max_points` | Shared equation-evaluation ceiling for search and bisection, including repeated endpoint evaluations |

`result.work()` reports executed work across every refinement. Bracket invocations and bisection invocations have separate counters. Expansion rounds have separate total and maximum counters. The global point ceiling includes equation evaluations, integration values, and derivative bounds.

## Act on rejection

`PrismaticRiemannError::Rejected` discards every partial result. Read `reason`, `context`, and cumulative `work` before retrying. When known, `query` identifies the input row and `section_owner` identifies its section row.

| Reason | Next action |
| --- | --- |
| `CapExceeded` | Check input units or extend surveyed support. |
| `CapUnresolved`, `GeometryUnresolved` | Check the named profile quantities and depth allowances. Preserve the supplied conserved areas. |
| `QuadratureUnresolved` | Check the reported integration uncertainty before increasing panel limits. |
| `RootBracketUnresolved` | Read the diagnostic for the unresolved water state. Check profile support and supplied states. |
| `SamplingUnresolved` | Read the diagnostic for the unresolved wave or interface state. Review the supplied states. |
| `FluxBudgetExceeded` | Inspect flux bounds and allowances. Increase allowances only when your application accepts wider bounds. |
| `WorkLimit` | Inspect counters before increasing ceilings or partitioning independent query rows. |

`PrismaticRiemannError::Channel` carries input, numerical, runtime, or arithmetic diagnostics with no channel-cell identity. Correct the named condition before retrying.
