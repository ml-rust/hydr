# Pinned EPA SWMM drainage worker

Use [the drainage builder](../tools/reference/swmm_drainage_build.py) and [the verification script](../tools/reference/swmm_drainage_verify.py) to compile and check the pinned EPA SWMM v5.2.4 dynamic-wave drainage worker with complete hidden-state restoration.

## Pinned source

| Property | Value |
| --- | --- |
| Engine | EPA SWMM v5.2.4 (public domain) |
| Commit | `7952ca837988b1c32f791812eccc9fd64547e093` |
| Archive SHA256 | `0c882c315b64d5236b3569f891e9ad71421bc5b7d92a5710ceb1ae3b7f34c6cf` |

No original SWMM C source is copied into tracked files. The builder checks every solver translation unit against the pinned archive before and after compilation.

## Hidden-state snapshot scope

The worker includes untouched `swmm5.c`, `routing.c`, `dynwave.c`, `controls.c`, and `massbal.c` and links all remaining solver translation units. `SNAPSHOT` and `RESTORE` serialize every mutable field in exact IEEE-754 bits:

- Clocks and counters: `OldRoutingTime`, `NewRoutingTime`, `RoutingDuration`, `ElapsedTime`, `ReportTime`, `TotalStepCount`, `ReportStepCount`, `NonConvergeCount`, `FlowError`.
- Routing statics: `NextEvent`, `BetweenEvents`, `NewRuleTime`.
- Dynamic-wave statics: `VariableStep`, `Omega`, `Steps`, and `Xnode[0..N]` (`converged`, `newSurfArea`, `oldSurfArea`, `sumdqdh`, `dYdT`).
- Controls: `CurrentDate`, `CurrentTime`, `ControlValue`, `SetPoint`, every rule action's `value`, PID errors `e1` and `e2`, and every link's `setting`, `targetSetting`, and `timeLastSet`.
- Mass balance: `FlowTotals`, `StepFlowTotals`, `OldStepFlowTotals`, `NodeInflow[]`, and `NodeOutflow[]`.
- Hydraulics: `Node[]`, `Storage[]`, `Outfall[]`, `Link[]` (including mutated `xsect`), `Conduit[]`, `Orifice[]`, `Weir[]`, and `Curve[]`/`Tseries[]` cursors.

## Surcharge treatment

Each `.inp` model declares its surcharge treatment explicitly via `SURCHARGE_METHOD` (`EXTRAN` or `SLOT`). Both treatments are checked under pressurizing inflow above the conduit crown.

## Reproduce

```sh
python3 tools/reference/swmm_drainage_verify.py
```
