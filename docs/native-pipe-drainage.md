# Native Rust Pressurized-Pipe Drainage Solver (`NativePipeNetwork`)

`hydr::drainage::NativePipeNetwork` is a pure-Rust (`#![forbid(unsafe_code)]`, zero external C binary dependency) 1D dynamic-wave closed-conduit drainage network solver.

## Method

- **Conduit hydraulics (`PipeCrossSection`)**: circular and rectangular box conduits with continuous Preissmann surcharge slot ($T_s = g A_{\text{full}} / c_{\text{slot}}^2$) so free-surface gravity flow transitions smoothly into pressurized surcharge above the pipe crown.
- **Link-node dynamics**: semi-implicit Manning friction momentum update per conduit coupled with nodal volume continuity across junctions, storage units, and free/fixed-stage outfalls.
- **Control & checkpoints**: supports per-conduit control settings in `[0.0, 1.0]`, transactional `trial_to` / `accept` with zero-cost rollback on drop, and portable `NativePipeSnapshotData` checkpoints.
