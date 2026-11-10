# hydr

hydr is a Rust library for building hydraulic applications. Choose a component below, then follow its guide.

For your first integration, follow [Library integration](docs/integration.md).

## What you can use

The library is unpublished. Its components support the following calculations:

| Component | What it calculates | Guide |
| --- | --- | --- |
| Surface flow | Water depth and horizontal flow across a two-dimensional grid | [Numerical method](docs/numerics.md) and [boundaries](docs/boundaries.md) |
| Soil and land sources | Rain, Modified Green–Ampt infiltration, capped evaporation, and withdrawals with complete soil memory | [Soil](docs/soil.md) and [land sources](docs/land-sources.md) |
| Rainfall runoff | Stored water and runoff from three catchment subareas | [Reservoir](docs/reservoir.md) and [subcatchment](docs/subcatchment.md) |
| Channel cross-sections | Submerged area, surface width, wetted perimeter, and pressure geometry at supplied stages | [Cross-section geometry](docs/cross-sections.md) |
| Rectangular channel flow | Water volume and longitudinal flow along an unbranched channel with varying width and bed | [Channel flow](docs/channels.md) |
| Storage pools | Water volume and level from supplied storage curves | [Level pools](docs/level-pool.md) |
| Coupled storage pools | Accepted structure transfers, local rain and losses, and prescribed external rates or integrated volumes | [Coupled level pools](docs/coupled-level-pools.md) |
| Runoff into pools | Native runoff generation and pool storage accepted over one common interval | [Runoff-to-pool connections](docs/runoff-pools.md) |
| Structure laws | Flow proposals for openings, weirs, pumps, and gates | [Structures](docs/structures.md) |
| Operating controls | Sampled commands and explicitly assumed ideal pump or gate responses | [Controls](docs/operating-controls.md), [actuators](docs/gate-actuators.md), [controlled gates](docs/controlled-gates.md), and [controlled pumps](docs/controlled-pumps.md) |

Connected level pools support accepted water transfers, local sources, prescribed external inflows, and static manual operations. Runoff-to-pool coordination requires free outlets without downstream feedback. Surface-to-network coupling remains unsupported.

The [operating-source registry](docs/operating-sources.md) assigns one source of operating inputs to each structure. [Contributing-support checks](docs/contributing-support.md) check which areas receive rainfall and where measured inflows replace generated runoff completely. Applications supply the ownership evidence. These checks do not establish hydraulic connections or prove that geographic areas do not overlap.

Operating-source bindings retain uncertainty in actual gate opening beside point flow proposals. These proposals establish no integrated water transfer.

## Try the examples

```rust
use hydr::grid::CartesianGrid;

fn main() -> Result<(), hydr::error::GridError> {
    let grid = CartesianGrid::new(100, 50, 2.0, 2.0)?;
    assert_eq!(grid.cell_count(), 5_000);
    Ok(())
}
```

```sh
cargo run --example grid
cargo run --example rain_basin
cargo test --all-targets
```

Grid spacing uses metres. Construction checks geometry without allocating cell storage.

## Choose a runtime

All hydraulic state uses 64-bit floating-point (`f64`) precision.

A runtime selects where calculations run. Supply its configured Numr client when constructing a component. Nested components keep that runtime and client.

| Runtime | Current support |
| --- | --- |
| CPU | Checked `f64` reference. Tests and reference examples select `CpuRuntime` with one worker thread. |
| CUDA | Enable `cuda` and supply a `CudaRuntime` client. Each algorithm still checks its required arithmetic support. |
| WGPU | WGPU cannot compute `f64`, so Hydr rejects the current hydraulic profile. |

Reading full surface state, soil memory, detailed diagnostics, or checkpoint data copies results from the device. These calls return errors when the copy fails. Snapshots retain complete continuation state for restart. The surface solver commits flow, soil memory, and land-source changes together.

## Understand the numerical limits

- Method: First-order finite volumes solve the full two-dimensional shallow-water equations. Hydrostatic side corrections balance bed forces. Bracketed-bound HLL supplies face fluxes, and split Manning resistance represents bed friction.
- Step checks: CFL limits bound the step using wave speeds. Outflow limits prevent a step from consuming unavailable water. Hydr shortens and recalculates complete trials when needed. Negative water rejects without clipping.
- Boundaries: Scalar stage or discharge inputs require wet-subcritical flow. Other regimes require complete exterior states or an admitted transmissive policy. Read the [boundary guide](docs/boundaries.md) before choosing an outlet.
- Missing inputs: Impermeable walls cannot replace unknown outlets or missing terrain.
- Accuracy: Measured-event accuracy and agreement with HEC-RAS remain unestablished. Higher-order methods, wider GPU checks, and calibration require separate numerical evidence.

## Integrate or contribute

Applications call the Rust library directly. Hydr requires neither a GUI nor a HEC-RAS installation. Numr supplies tensor operations, and Solvr supplies generic solving algorithms. Hydr owns the hydraulic equations and physical checks.

| Reader task | Guide |
| --- | --- |
| Construct inputs, advance the solver, and save results | [Library integration](docs/integration.md) |
| Understand the equations and numerical assumptions | [Numerical contract](docs/numerics.md) |
| Read the checked cases and accuracy limits | [Acceptance coverage](docs/acceptance.md) |
| Modify code or documentation | [Contributing](CONTRIBUTING.md) |

## References

- [Hydroflux](https://github.com/franciscoparrao/hydroflux): Rust shallow-water implementation reference.
- [GeoClaw](https://www.clawpack.org/geoclaw.html): Independent depth-averaged flow reference.
- [HEC-RAS manuals](https://www.hec.usace.army.mil/software/hec-ras/documentation.aspx): Hydraulic equations and numerical methods.

## License

Apache-2.0. See [LICENSE](LICENSE).
