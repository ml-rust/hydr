# Surface–Drainage Coupling (`SurfaceDrainageBatch`)

`SurfaceDrainageBatch` couples the 2D shallow-water `Solver` (`hydr-solver`) with the pinned EPA SWMM v5.2.4 piped drainage engine (`SwmmDrainage`, `hydr-drainage`) through conservative manhole/catch-basin inlets.

## Inlet Exchange Law

- Each `SurfaceDrainageInlet` links one active 2D surface cell to one SWMM drainage node at an explicit `rim_elevation_m` above the shared `vertical_datum`.
- **Surface capture** (`H_s > rim` and `H_s > H_d`): `Q = min(C_w * L * (H_s - rim) * sqrt(dh), C_o * A * sqrt(2 g dh))` with `dh = H_s - max(H_d, rim)`.
- **Pressurized surcharge** (`allow_surcharge_to_surface == true`, `H_d > rim`, `H_d > H_s`): `Q = -C_o * A * sqrt(2 g (H_d - max(H_s, rim)))`.
- Both components accept the exact same signed transfer volume (`transfer_volume_m3`).

## Conservative Advancement and Rollback

- `SurfaceDrainageBatch::trial_to` prepares the 2D surface stage (`Solver::coupling_prepare`), reserving native 2D face outflow before interface withdrawals.
- Midpoint-head Picard iteration evaluates inlet exchange, applies proportional surface donor limiting per linked cell, advances SWMM via `SwmmDrainage::trial_to`, and completes the surface step via `Solver::coupling_complete`.
- Every unconverged iteration drops `SwmmDrainageTrial`, invoking SWMM `RESTORE` bit-for-bit before the next iteration.
- Unconverged steps halve `dt` down to `min_step_s` up to `max_retries`.
- Dropping `SurfaceDrainageTrial` without calling `accept()` leaves the 2D surface unchanged and rolls back SWMM bit-for-bit.

## Control and Parameter Authority

- `DrainageControlAuthorityTable` parses SWMM `.inp` `[CONTROLS]` action clauses (`PUMP`, `ORIFICE`, `WEIR`, `OUTLET`, `CONDUIT`) and registers every natively controlled link.
- Any external claim targeting a natively controlled link, any duplicate claim, or any duplicate inlet `id` / `(surface_cell, drainage_node_id)` pair rejects with `duplicate-parameter-authority`.
- External `hydr` claims (`HydrCommandsNativeActuation`, `HydrActuatorImposedPosition`, `PrescribedHydrTrajectory`) on uncontrolled SWMM links execute via `SurfaceDrainageBatch::trial_to_with_commands`. Attempting to command an unclaimed or native-SWMM-owned link rejects.

## Portable Checkpoints

- `SurfaceDrainageBatch::snapshot_data` exports `hydr-surface-drainage-checkpoint-1` containing complete surface and SWMM hidden-state snapshots plus cumulative per-inlet transfer ledgers.
- `SurfaceDrainageBatch::import_snapshot` restores both components atomically and rolls back on any error.

## Limits

- `datum-mismatch`: setup and SWMM vertical datums must match, and every inlet rim must lie at or above both the SWMM node invert and the surface cell bed.
- `midpoint-inlet-closure`: midpoint-head weir/orifice exchange with proportional surface donor limiting.
- `pinned-swmm-v5-2-4` and `gpu-execution`: nested SWMM worker runs on host CPU only.
