# Changelog

All notable changes to hydr will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
hydr uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
