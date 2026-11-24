<div align="center">

# hydr

<h3>Hydrology and hydraulic simulation in Rust. Conservative. Reproducible. Multi-backend.</h3>

<p>
2D shallow-water surface flow, Modified Green-Ampt infiltration, subcatchment runoff, 1D river channels,
storage pools, hydraulic structures and HDS-5 culverts, piped drainage (pinned EPA SWMM v5.2.4), and
conservative coupled advancement on <a href="https://github.com/ml-rust/numr">numr</a> and <a href="https://github.com/ml-rust/solvr">solvr</a>.
</p>

<p>
  <a href="https://docs.rs/hydr"><strong>Docs</strong></a>
  ·
  <a href="https://crates.io/crates/hydr"><strong>Crate</strong></a>
  ·
  <a href="#component-catalog"><strong>Components</strong></a>
  ·
  <a href="#backends"><strong>Backends</strong></a>
  ·
  <a href="#installation"><strong>Install</strong></a>
  ·
  <a href="#quick-start"><strong>Quick Start</strong></a>
  ·
  <a href="#examples"><strong>Examples</strong></a>
  ·
  <a href="CONTRIBUTING.md"><strong>Contributing</strong></a>
</p>

<p>
  <a href="https://discord.gg/jBhFk9kHPg">
    <img src="https://img.shields.io/discord/1453357769720594543?label=Discord&logo=discord&logoColor=white&color=5865F2" alt="Join the Discord">
  </a>
</p>

<p>
  <a href="https://github.com/ml-rust/hydr/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/ml-rust/hydr/ci.yml?branch=main&label=ci" alt="CI status">
  </a>
  <a href="https://crates.io/crates/hydr">
    <img src="https://img.shields.io/crates/v/hydr" alt="crates.io">
  </a>
  <a href="https://docs.rs/hydr">
    <img src="https://img.shields.io/docsrs/hydr" alt="docs.rs">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/crates/l/hydr" alt="License">
  </a>
  <a href="https://github.com/ml-rust/hydr/stargazers">
    <img src="https://img.shields.io/github/stars/ml-rust/hydr?style=social" alt="GitHub stars">
  </a>
</p>

</div>

`hydr` is a pure-Rust (`#![forbid(unsafe_code)]`) numerical engine for building flood, river, urban drainage, and water-resources applications.
Every component operates in canonical SI units (`m`, `m³`, `m³/s`, `s`) with `f64` precision, atomic accepted steps, explicit capability declarations, and bitwise-reproducible checkpoints.

- **No GUI or proprietary solver lock-in**: Native Rust workers call `hydr` directly as a library.
- **Built on the `ml-rust` stack**: [numr](https://github.com/ml-rust/numr) provides tensor storage and CPU/CUDA kernels; [solvr](https://github.com/ml-rust/solvr) provides certified root-finding, quadrature, and interpolation; `hydr` owns hydraulic equations, admission rules, and conservation.
- **Conservative by construction**: Negative water rejects without silent clipping. Coupled coordinators reserve native outflow before interface withdrawals and commit shared signed transfer volumes atomically.
- **Explicit scope**: Every component exports `<component>_capabilities() -> &'static ComponentCapabilities` declaring precision, arithmetic contracts, runtimes, regimes, unsupported cases, and limits.

## Installation

Requires Rust 1.95 or newer. Add `hydr` and `numr`:

```bash
cargo add hydr numr                  # CPU F64 reference runtime
cargo add hydr --features cuda       # NVIDIA GPUs (CUDA 12.8+ or 13.x)
cargo add hydr --features wgpu       # WebGPU feature gate (rejects f64 without fallback)
```

For piped urban drainage (`hydr::drainage` and `SurfaceDrainageBatch`), build the isolated pinned EPA SWMM v5.2.4 worker binary once on the host:

```bash
python3 tools/reference/swmm_drainage_build.py --help
```

See [docs/drainage-reference.md](docs/drainage-reference.md) for the pinned commit, build steps, and rollback verification.

## Backends

All hydraulic calculations require IEEE-754 `f64` precision. Pass a configured `numr` client when constructing any component; nested components reuse that runtime and client.

| Runtime | Feature | Support | Notes |
| --- | --- | --- | --- |
| `CpuRuntime` | default | Checked `f64` reference | Single-threaded deterministic reference profile; required for `SwmmDrainage` and `SurfaceDrainageBatch` |
| `CudaRuntime` | `cuda` | Executed `f64` parity | Verified on NVIDIA L4 (`sm_89`, CUDA 13.0) across surface, channels, pools, runoff, structures, culverts, and surface-channel coupling |
| `WgpuRuntime` | `wgpu` | Rejected cleanly | WebGPU computes in `f32` only; `hydr` rejects `f64` profiles without silent `f32` or CPU fallback |

## Quick Start

Advance a 2D closed basin under constant rainfall and inspect the accepted water depth and conservation residual:

```rust
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandCell, LandModel, SoilState};
use hydr::solver::{
    AdvanceStatus, BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, Solver,
    SolverConfig, SolverSetup,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(1), None));

    let grid = CartesianGrid::new(5, 5, 10.0, 10.0)?;
    let n = grid.cell_count();
    let reference = SpatialReference::new(
        "projected metres".into(),
        "NAVD88".into(),
        [0.0, 0.0],
    )?;
    let domain = Domain::new(grid, vec![0.0; n], vec![0.03; n], vec![true; n], reference)?;
    let land = LandModel::new(&domain, vec![LandCell::Impervious; n])?;
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(&domain, vec![CellState { h: 0.0, qx: 0.0, qy: 0.0 }; n])?;

    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 60.0,
            rain: vec![1e-4; n],
            potential_evaporation: vec![0.0; n],
            withdrawal: vec![0.0; n],
            boundaries: Vec::new(),
        }],
    )?;

    let mut solver = Solver::new(
        client,
        SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config: SolverConfig::default(),
        },
        InitialState { time: 0.0, flow, soil },
    )?;

    let report = solver.advance_to(60.0, 10_000)?;
    assert_eq!(report.status, AdvanceStatus::ReachedTarget);

    let depth = solver.state()?.cells()[12].h;
    let residual = solver.ledger().residual;
    println!("t={} s, center depth={depth:.6} m, mass residual={residual:e} m³", solver.time());
    Ok(())
}
```

## Component Catalog

Applications depend only on the `hydr` facade crate. Every module re-exports its types and capability declarations:

| Module | What it computes | Guide |
| --- | --- | --- |
| `hydr::solver` | 2D shallow-water finite-volume flow (hydrostatic bed reconstruction, HLL fluxes, Manning friction, 5 physical open boundary kinds, forecast splicing, checkpoints) | [Integration](docs/integration.md), [Numerics](docs/numerics.md), [Boundaries](docs/boundaries.md) |
| `hydr::soil` | Pinned Modified Green-Ampt infiltration, capped evaporation, and prescribed withdrawals with complete soil memory | [Soil](docs/soil.md), [Land sources](docs/land-sources.md) |
| `hydr::runoff` | Nonlinear reservoir and three-subarea subcatchment runoff with explicit rainfall ownership and contributing-support checks | [Subcatchment](docs/subcatchment.md), [Reservoir](docs/reservoir.md), [Support](docs/contributing-support.md) |
| `hydr::channel` | 1D rectangular, triangular, and certified surveyed river reaches with 6 endpoint policies, lateral momentum, and section epoch schedules | [Channels](docs/channels.md), [Cross-sections](docs/cross-sections.md), [Prismatic bounds](docs/prismatic-sections.md) |
| `hydr::storage` | Closed level-pool storage with piecewise-linear stage-volume curves, local rain/losses, and curve epoch schedules | [Level pools](docs/level-pool.md) |
| `hydr::structures` | Weir, orifice, hydrostatic gate, pump, and full HDS-5 inlet/outlet culvert laws (6 barrel shapes, blockage, sediment, closure), sampled controls, gate actuators, and geometry schedules | [Structures](docs/structures.md), [Controls](docs/operating-controls.md), [Actuators](docs/gate-actuators.md) |
| `hydr::drainage` | Safe Rust wrapper over the isolated pinned EPA SWMM v5.2.4 piped drainage worker (`EXTRAN`/`SLOT` surcharge, live external link actuation, complete hidden-state rollback and restart) | [Piped drainage](docs/drainage.md), [Reference worker](docs/drainage-reference.md) |
| `hydr::coupling` | Conservative coordinators (`CoupledLevelPoolBatch`, `ChannelPoolBatch`, `RunoffPoolBatch`, `SurfaceChannelBatch`, `SurfaceDrainageBatch`) with exclusive parameter/control authority and certified crossing localization | [Surface/channels](docs/surface-channels.md), [Surface/drainage](docs/surface-drainage.md), [Channel/pools](docs/channel-pools.md) |
| `hydr::capability` | Stable machine-readable `ComponentCapabilities` declarations backed by admission and runtime evidence tests | [Read capabilities](docs/integration.md#read-component-capabilities) |

## Numerical Guarantees and Scope Limits

- **Atomic steps and rollback**: Every engine advances via trial steps that shorten on CFL, donor-depletion, or convergence limits. Unaccepted or failed trials leave accepted state unchanged.
- **Complete portable checkpoints**: Every component exports typed `*SnapshotData` packages with explicit `schema` and `contract` headers for exact bitwise restart and forecast splicing.
- **Exclusive parameter and control authority**: Coordinators reject duplicate writers across structure operations, static blockage/closure conditions, geometry schedules, native SWMM `[CONTROLS]`, and external `hydr` link commands.
- **Declared limits**: First-order finite volumes solve the 2D shallow-water equations; scalar stage/discharge boundaries require wet-subcritical flow; synthetic and analytical benchmarks establish numerical contracts rather than site-specific field calibration. See [docs/acceptance.md](docs/acceptance.md).

## Examples

Run any example from a checkout:

```bash
cargo run --example grid
cargo run --example rain_basin
cargo run --example surface_runoff
cargo run --example native_worker    # capability check, cancellation, checkpoint/restore, forecast splice
```

Run the full test suite:

```bash
cargo test --workspace --all-targets
```

## Documentation

| Task | Guide |
| --- | --- |
| Integrate `hydr` into a native Rust worker | [docs/integration.md](docs/integration.md) |
| Review equations, step limits, and conservation | [docs/numerics.md](docs/numerics.md) |
| Inspect analytical and composite acceptance coverage | [docs/acceptance.md](docs/acceptance.md) |
| Contribute code, tests, or docs | [CONTRIBUTING.md](CONTRIBUTING.md) |

## References

- [Hydroflux](https://github.com/franciscoparrao/hydroflux): Rust shallow-water implementation reference.
- [GeoClaw](https://www.clawpack.org/geoclaw.html): Independent depth-averaged flow reference.
- [EPA SWMM 5.2](https://www.epa.gov/water-research/storm-water-management-model-swmm): Dynamic-wave piped drainage and Modified Green-Ampt reference.
- [FHWA HDS-5 / HEC-RAS](https://www.hec.usace.army.mil/software/hec-ras/documentation.aspx): Culvert hydraulics and channel flow methods.

## License

Apache-2.0. See [LICENSE](LICENSE).
