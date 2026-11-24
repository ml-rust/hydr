# 3-Way Surface-Channel-Drainage Coordinator (`SurfaceChannelDrainageBatch`)

`hydr::coupling::SurfaceChannelDrainageBatch<R>` couples **2D shallow-water surface flow**, **1D river channels** (plus level pools, structures, controls, and runoff via `SurfaceChannelBatch`), and **EPA SWMM v5.2.4 piped urban drainage** (`SwmmDrainage`) in a single conservative step.

## Method

- Each trial advances all three subsystems over the exact same common interval `[start_s, end_s]`.
- An outer midpoint-head fixed-point loop computes bidirectional surface↔drainage inlet capture and pressurized surcharge with proportional surface donor limiting.
- Inlet transfers splice conservatively into the surface forcing window (`withdrawal` for capture, `rain` for surcharge) and run the full inner `SurfaceChannelBatch` fixed-point solver so 1D channels and 2D floodplain immediately feel sewer exchange and vice versa.
- If `SurfaceChannelBatch` shortens the step (CFL, channel limit, or operating event), SWMM rolls back bit-for-bit and the outer loop restarts on the exact shortened interval.
- Dropping a `SurfaceChannelDrainageTrial` without `.accept()` leaves `SurfaceChannelBatch` unchanged and rolls back the SWMM worker bit-for-bit.
- `snapshot_data()` / `import_snapshot()` export and atomically restore all three subsystems together.
