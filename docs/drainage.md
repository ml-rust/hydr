# Piped Drainage (`hydr-drainage`)

`hydr-drainage` wraps the pinned EPA SWMM v5.2.4 dynamic-wave worker (`swmm-drainage-worker`) behind a safe Rust API (`#![forbid(unsafe_code)]`).

## Architecture

- Each `SwmmDrainage` instance spawns an isolated `swmm-drainage-worker` subprocess over stdin/stdout.
- Process isolation prevents aliasing of SWMM's process-global C state across concurrent runs.
- Build the worker binary with `python3 tools/reference/swmm_drainage_build.py`. See [drainage-reference.md](drainage-reference.md) for the pinned commit, snapshot layout, and verification suite.

## Units and Vertical Datum

- SWMM stores all internal routing quantities in US customary feet (`ft`), cubic feet (`ft³`), and cubic feet per second (`cfs`), regardless of the `.inp` file's `FLOW_UNITS`.
- `hydr-drainage` exposes canonical SI units (`m`, `m³`, `m³/s`, `s`) using SWMM's exact `MperFT = 0.3048` conversion factor (`FT3_TO_M3 = 0.3048³`).
- Every `SwmmDrainageDefinition` declares an explicit `vertical_datum`, `provenance`, and `surcharge_evidence`.

## Surcharge Formulation

- Callers must declare `SwmmSurchargeMethod::Extran` or `SwmmSurchargeMethod::Slot`.
- `SwmmDrainage::open` checks the `.inp` file's active `SURCHARGE_METHOD` reported by SWMM and rejects any mismatch (`surcharge-method-mismatch`).

## Control Authority

- Standalone `SwmmDrainage` admits `SwmmControlAuthority::NativeSwmm` only.
- Hybrid `hydr`/SWMM control-authority modes reject at admission (`native-control-authority`) until attached through surface-drainage coupling.

## Transactional Steps and Complete Snapshots

- `SwmmDrainage::trial_to` executes a step against the worker and returns `SwmmDrainageTrial`.
- Calling `SwmmDrainageTrial::accept` commits the step.
- Dropping `SwmmDrainageTrial` without accepting sends `RESTORE` to the worker and rolls back all hidden SWMM state bit-for-bit (hydraulic arrays, `Xnode`, `VariableStep`, control timers, pump and orifice actuation state, and continuity totals).
- `SwmmDrainage::snapshot_data` exports a signed checkpoint (`hydr-swmm-drainage-checkpoint-1`) carrying schema, contract, `.inp` SHA-256, vertical datum, surcharge method, accepted step count, and payload SHA-256 digest.
- `SwmmDrainage::import_snapshot` and `SwmmDrainage::resume` reject tampered or mismatched checkpoints (`tampered-snapshot`).

## Limits

- `pinned-swmm-v5-2-4`: requires the pinned EPA SWMM v5.2.4 worker binary (`missing-worker-binary` returns `DrainageError::BackendUnavailable`).
- `si-boundary-conversion`: all quantities convert at the worker boundary using exact `0.3048 m/ft`.
- `gpu-execution`: the pinned C worker runs on host CPU only.
