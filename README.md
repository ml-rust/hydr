# hydr

hydr advances two-dimensional shallow-water flow on numr's CPU runtime for hydraulic application developers.

## Status

The first-order CPU `f64` reference implements checked domains and conservative time advancement.
Inputs include terrain, roughness, initial depth/momentum, explicit land/soil memory, rain, potential evaporation, and prescribed withdrawal.
Boundaries include walls, complete flux ports, wet-subcritical stage/discharge traces, complete wet/dry exterior reservoir states, and admitted transmissive outflow.
Snapshots preserve complete numerical continuation state. Diagnostics expose accepted steps and component water budgets.
Measured-event accuracy and agreement with HEC-RAS remain unestablished.
The crate is unpublished. The existing Git remote remains `ml-rust/rivr`.

## Example

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
cargo test --all-features
```

Grid spacing uses metres. Construction checks geometry without allocating cell storage.

## Direction

- Physics: Full two-dimensional shallow-water equations for river and floodplain flow.
- Numerics: First-order finite volumes, hydrostatic side corrections, bracketed-bound HLL, and split Manning resistance.
- Foundation: numr 0.15.2 CPU tensors and solvr 0.3.1 batched root algorithms.
- Acceptance: CFL and all-outgoing depletion bounds trigger full trial recalculation. Negative water rejects without clipping.
- Extensions: Higher-order methods, GPU execution, and calibration require separate numerical evidence.
- Interface: A Rust library without a GUI or HEC-RAS runtime dependency.

Scalar stage/discharge traces require wet-subcritical states. Other regimes require complete exterior data or an admitted transmissive policy.
Atomic land sources integrate pinned Modified Green–Ampt soil, rain, capped evaporation, and prescribed withdrawal.
Structures, networks, and coupled external components remain unsupported.
Impermeable walls cannot replace unknown outlets or missing terrain.
Generic numerical operations belong in numr. Generic solving algorithms belong in solvr. Hydraulic equations and acceptance belong in hydr.
See [Contributing](CONTRIBUTING.md) for development checks.
Read the [numerical contract](docs/numerics.md), [integration contract](docs/integration.md), [physical boundaries](docs/boundaries.md), and [acceptance coverage](docs/acceptance.md).

The standalone [soil component](docs/soil.md) retains native memory. The [land-source contract](docs/land-sources.md) defines atomic hydraulic integration and correction budgets.

A standalone [conservative reservoir](docs/reservoir.md) retains explicit storage and interval export.
The standalone [subcatchment component](docs/subcatchment.md) adds native soil memory, capped evaporation, and same-interval paired internal routing.
Hydraulic attachment and inter-catchment networks remain unsupported.

## References

- [Hydroflux](https://github.com/franciscoparrao/hydroflux): Rust shallow-water implementation reference.
- [GeoClaw](https://www.clawpack.org/geoclaw.html): Independent depth-averaged flow reference.
- [HEC-RAS manuals](https://www.hec.usace.army.mil/software/hec-ras/documentation.aspx): Hydraulic equations and numerical methods.

## License

Apache-2.0. See [LICENSE](LICENSE).
