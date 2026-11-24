# Opt-In `F32` 2D Surface Solver (`SurfaceSolverF32`)

`hydr::solver::SurfaceSolverF32<R>` runs the 2D shallow-water finite-volume solver natively in `DType::F32` on **`CpuRuntime`**, **`CudaRuntime`**, and **`WgpuRuntime`** (Vulkan, Metal, DX12).

## Capabilities

- **Open boundaries (`SurfaceF32Boundaries`)**: per-side `Wall`, `Transmissive`, `Stage(f32)`, and `Discharge(f32)` with exact boundary inflow/outflow tracking in `SurfaceSolverF32Ledger`.
- **Modified Green-Ampt infiltration (`SurfaceF32Soil`)**: vectorized `DType::F32` 6-iteration safeguarded Newton solve persisted in `SurfaceSolverF32SnapshotData`.
- **Coastal & wind forcing (`SurfaceF32CoastalForcing`)**: quadratic wind stress, atmospheric pressure gradient (inverse barometer), and energy-preserving Crank-Nicolson Coriolis rotation.
- **Precision contract**: single-precision (`F32`, ~1e-7 relative rounding). Use certified `F64` `hydr::solver::Solver<R>` when reference-grade `F64` guarantees or `TwoSum` / correctly-rounded elementary contracts are required.
