# Changelog

All notable changes to hydr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
hydr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.3.0] - 2026-11-25

### Breaking

- `numr`, `solvr`: Requires `numr 0.17.0` and `solvr 0.5.0`. Update callers to the same tensor and solver crate versions.

### Added

- `hydr::solver`: Adds native-async `WgpuRuntime` methods (`advance_to_async`, `advance_controlled_async`, `depth_m_async`, `state_async`, `cumulative_infiltration_m_async`, `snapshot_data_async`) on `SurfaceSolverF32<WgpuRuntime>` for browser `wasm32` WebGPU execution without synchronous readback deadlocks. Use them on `wasm32` after initializing `WgpuClient::new_async`.
- `hydr::solver`: Adds per-face exterior open boundary masks (`SurfaceF32OpenMasks`, `SurfaceSolverF32::with_open_masks`, `set_open_masks`) across CPU, CUDA, and WebGPU. Attach masks to keep closed exterior faces reflecting while opening selected outflow faces.

## [0.2.0] - 2026-11-24

### Added

- `hydr::coupling`: Adds simultaneous 3-way coordinator `SurfaceChannelDrainageBatch` coupling 2D surface flow, 1D river channels (plus level pools, structures, controls, and runoff via `SurfaceChannelBatch`), and pinned EPA SWMM v5.2.4 piped drainage over a single common step with midpoint-head iteration, bit-identical SWMM rollback, and atomic 3-way checkpoints. Construct via `SurfaceChannelDrainageBatch::new`.
- `hydr::drainage`: Adds pure-Rust (`#![forbid(unsafe_code)]`, zero external binary dependency) 1D dynamic-wave Preissmann-slot pressurized-pipe solver `NativePipeNetwork` supporting circular and box conduits, free-surface to pressurized surcharge transitions, link control actuation, transactional rollback, and portable checkpoints. Construct via `NativePipeNetwork::new`.
- `hydr::solver`: Upgrades `SurfaceSolverF32` (CPU, CUDA, and WebGPU) with per-side open exterior boundaries (`SurfaceF32Boundaries`: `Wall`, `Transmissive`, `Stage`, `Discharge`), vectorized `DType::F32` Modified Green-Ampt infiltration (`SurfaceF32Soil`), opt-in 2nd-order well-balanced MUSCL reconstruction (`SurfaceF32SpatialOrder::SecondOrderMuscl`), and coastal wind/Coriolis/pressure forcing (`SurfaceF32CoastalForcing`).
- `hydr::solver`: Adds `SubgridTerrain` and `SubgridCellCurve` for exact fine-DEM subgrid storage-elevation and wet-fraction tables on coarse 2D grids.
- `hydr::solver`: Adds opt-in `CoastalForcing` (wind stress, atmospheric pressure gradient, and energy-preserving Crank-Nicolson Coriolis rotation) to certified `F64` `Solver<R>` via `Solver::with_coastal_forcing` and `Solver::set_coastal_forcing`.

## [0.1.0] - 2026-11-24

Initial public release of `hydr` on `numr 0.16` and `solvr 0.4`.

### Added

- `hydr::solver`: Adds 2D shallow-water finite-volume solver (`Solver`, certified `F64`) with hydrostatic bed reconstruction, HLL fluxes, Manning friction, five physical open boundary kinds, forecast splicing, and complete checkpoints, plus opt-in `SurfaceSolverF32` running natively in `F32` on CPU, CUDA, and WebGPU (`WgpuRuntime`). Call `Solver::new` for certified `F64` or `SurfaceSolverF32::new` for fast `F32`/WebGPU.
- `hydr::soil`, `hydr::runoff`: Adds Modified Green-Ampt infiltration, nonlinear reservoir routing, three-subarea subcatchments, and explicit rainfall ownership plans. Attach soil through `SolverSetup` and runoff through `RunoffPoolBatch` or `SurfaceChannelBatch`.
- `hydr::channel`: Adds 1D rectangular, triangular, and certified surveyed reaches with six endpoint policies, lateral momentum sources, and section epoch schedules. Construct reaches with `ChannelReach::new`.
- `hydr::storage`, `hydr::structures`: Adds closed level pools, weir/orifice/gate/pump/HDS-5 culvert laws with blockage, sediment, and closure conditions, sampled operating controls, gate actuators, and prescribed geometry schedules.
- `hydr::drainage`: Adds safe Rust wrapper (`#![forbid(unsafe_code)]`) over the pinned EPA SWMM v5.2.4 piped drainage worker with canonical SI conversion, EXTRAN/SLOT surcharge checks, live external link actuation, transactional rollback, and signed checkpoints. Build the worker with `tools/reference/swmm_drainage_build.py`.
- `hydr::coupling`: Adds conservative coordinators (`CoupledLevelPoolBatch`, `ChannelPoolBatch`, `RunoffPoolBatch`, `SurfaceChannelBatch`, `SurfaceDrainageBatch`) with exclusive parameter/control authority tables, certified crossing localization, and atomic dual-component checkpoints.
- `hydr::capability`: Adds stable component capability declarations (`<component>_capabilities()`) backed by admission and runtime evidence tests.
