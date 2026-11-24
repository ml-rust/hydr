# Opt-in `F32` 2D Surface Solver (`SurfaceSolverF32`)

`SurfaceSolverF32<R>` provides an explicit, opt-in binary32 (`DType::F32`) 2D shallow-water surface solver for high-throughput execution on **CPU**, **CUDA**, and **WebGPU (`WgpuRuntime` — Vulkan, Metal, DX12)**.

## When to Use `F32` vs `F64`

- **`Solver<R>` (`F64`, default)**: Certified reference profile on CPU and CUDA with `CorrectlyRoundedOps`, `TwoSum`, Modified Green-Ampt soil, 5 open boundary kinds, and coupled network coordinators. `WgpuRuntime` rejects `F64` without silent fallback.
- **`SurfaceSolverF32<R>` (`F32`, opt-in)**: Native `DType::F32` 2D finite-volume shallow-water solver (hydrostatic bed reconstruction, HLL Riemann fluxes, split semi-implicit Manning friction, rain/evaporation/withdrawal, CFL and positivity step control, water-balance ledger, and `SurfaceSolverF32SnapshotData`). Runs natively on `CpuRuntime`, `CudaRuntime`, and `WgpuRuntime`.

## Capabilities and Limits

- `surface_solver_f32_capabilities()` declares `ComponentId::SurfaceSolverF32`, `Precision::F32`, `ArithmeticRequirement::F32Ieee754Basic`, and `Cpu` (`Reference`), `Cuda` (`Executed`), and `Wgpu` (`Executed`).
- `f32-rounding-envelope`: Single-precision rounding (`~1e-7` relative) governs lake-at-rest and mass conservation.
- `closed-wall-boundaries`: Admits closed impermeable wall boundaries and distributed rain, evaporation, and withdrawal sources.
- `invalid-f32-config` and `tampered-f32-snapshot`: Rejected at admission without mutating state.
