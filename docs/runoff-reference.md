# Pinned EPA runoff reference

`tools/reference/swmm_runoff.py` records original EPA runoff behavior independently of hydr production numerics. It adds no public runoff API or conservative hydraulic attachment claim.

The harness uses SWMM v5.2.4, commit `7952ca837988b1c32f791812eccc9fd64547e093`. Its [original archive](https://codeload.github.com/USEPA/Stormwater-Management-Model/tar.gz/7952ca837988b1c32f791812eccc9fd64547e093) requires SHA256 `0c882c315b64d5236b3569f891e9ad71421bc5b7d92a5710ceb1ae3b7f34c6cf`.

## Source and execution

The C adapter includes unchanged `subcatch.c` and `infil.c`. It links every remaining original solver translation unit, including `odesolve.c`. The build checks every original file against the checksum-pinned archive before and after compilation. Native execution repeats that complete source check.

Only the included infiltration source maps `UCF` to a native feet/seconds parameter bridge. Original engine unit conversion remains unchanged elsewhere. Requested rain passes through the original rain-gage units. Evidence records both requested and actual native precipitation.

`InfilFactor`, `Adjust.hydconFactor`, and `Evap.recoveryFactor` equal one. Seasonal patterns and excluded physical components remain disabled explicitly.

The [pinned README](https://github.com/USEPA/Stormwater-Management-Model/blob/7952ca837988b1c32f791812eccc9fd64547e093/README.md) declares SWMM source public domain. The [original ODE header](https://github.com/USEPA/Stormwater-Management-Model/blob/7952ca837988b1c32f791812eccc9fd64547e093/src/solver/odesolve.c) separately records Numerical Recipes in C, Cambridge University Press, 1992, as its implementation provenance. No original numerical source body is copied into tracked tooling. This observation harness establishes no production algorithm reuse decision.

## Reproduce

Requirements: Python3.10 or newer with its standard library, and a C11 compiler supporting GNU section garbage collection and `libm`. Use the original extracted archive described in the [soil reference guide](soil-reference.md).

Run from the repository root with fresh destinations:

```sh
python3 tools/reference/swmm_runoff.py build \
  --source-root resource/reference-engines/swmm/Stormwater-Management-Model-7952ca837988b1c32f791812eccc9fd64547e093 \
  --archive resource/reference-engines/swmm/swmm-v5.2.4.tar.gz \
  --build-dir resource/reference-engines/swmm/runoff-build \
  --cc gcc
python3 tools/reference/swmm_runoff.py run \
  --source-root resource/reference-engines/swmm/Stormwater-Management-Model-7952ca837988b1c32f791812eccc9fd64547e093 \
  --archive resource/reference-engines/swmm/swmm-v5.2.4.tar.gz \
  --build-dir resource/reference-engines/swmm/runoff-build \
  --output resource/reference-engines/swmm/runoff-reference.csv
```

The canonical compiler flags include `-std=c11 -D_GNU_SOURCE -O2 -ffp-contract=off -ffunction-sections -fdata-sections -Wl,--gc-sections`. The feature-test macro exposes original POSIX interfaces under strict C11. Original CMake leaves GNU C extensions enabled. Compiler include/search overrides are removed. Build and output paths never overwrite existing evidence.

The deterministic catalog covers depression filling and endpoint thresholds, equilibrium, recession/refinement, zero branches, antecedent state, intermittent forcing, competing losses, routing, and original adaptive ODE behavior. Direct endpoint and ODE probes operate on copied native state. They remain separate from advancing timesteps.

## Native protocol

The adapter accepts one command per line. Inputs use feet, square feet, seconds, feet/second, and cubic feet/second.

```text
INIT A fracImperv fracZeroImperv W slope nImp nPerv dStoreImp dStorePerv route routedFraction S Ks IMDmax evapDryOnly
STATE time d0 in0 q0 d1 in1 q1 d2 in2 q2 oldQ newQ runon rainfall evapLoss infilLoss IMD F Fu Sat T
STEP dt requestedRain E externalRunonCfs
RATE subarea tRunoff
ODE subarea dt tolerance initialStep
```

| Field | Meaning |
| --- | --- |
| Subareas0/1/2 | Impervious without storage, impervious with storage, pervious |
| `route` | `0`: outlet, `1`: pervious to impervious, `2`: impervious to pervious |
| `fracZeroImperv` | Fraction within the impervious class receiving zero depression storage |
| `routedFraction` | Fraction of selected subarea runoff sent internally |
| `d`, `in`, `q` | Depth, inflow scratch, and previous endpoint runoff rate for each subarea |
| `IMD F Fu Sat T` | Complete original native soil state |
| `runon` | Zero at the restricted component's accepted endpoint |

`INIT` records actual original stored parameters and derives original class-area runoff coefficients. `STATE` restores the complete emitted restricted witness. `STEP` follows original old-state, internal-routing, external-runon, and runoff calls.

The harness repeats every case in a fresh process. Stdout and stderr must match byte-for-byte. Multi-step cases restart halfway from the complete restricted witness. Following physical observations must retain identical floating-point bits.

Each multi-step case also advances one trial, restores the saved witness in that process, and repeats the remaining commands. Every following physical/diagnostic field must match, excluding only the protocol index. Evidence retains trial commands, restored witness, and raw hashes.

This restart check covers the isolated component phase. It excludes full-driver weather cursors, graphs, hydraulic routing, and automatic wet/dry timestep selection.

## Evidence and interpretation

| Artifact | Contents |
| --- | --- |
| Build manifest | Archive/file identity, compiler/version/flags, unit bridge, linked sources, adapter/header/binary hashes |
| CSV | Full three-subarea config/state, native soil, routing history, original losses/endpoint volumes, and independent storage diagnostics |
| JSON | Runtime identity, exact catalog commands, repeated hashes, restart witnesses, observed discrepancies, and unsupported scope |
| `.native` directory | Exact hexadecimal commands, original17-digit stdout, stderr, and restricted restart runs |

The original code reports endpoint discharge multiplied by the full interval. This quantity can differ materially from storage-implied transfer. Original evaporation and infiltration can independently claim the same available water. These discrepancies remain visible in evidence.

Gross subarea transfer includes previous-endpoint internal delivery. Whole-catchment implied outlet export includes external runon only. The output separately records internal routing lag between prior delivered rates and current endpoint rates. A coarse native timestep does not guarantee cancellation.

Python independently sums represented subarea storage and endpoint transfers. It compares bookkeeping identities and frozen analytical recession/equilibrium gates. It never forces the physical mass residual to zero.

The consumer binds emitted forcing, clocks, parameters, and state to submitted commands. Ten archive-free corruption regressions check these gates:

```sh
python3 -m unittest discover -s tools/reference -p test_swmm_runoff.py
```

Material endpoint-quadrature, routing-lag, or competing-loss discrepancies are not arithmetic water adjustments. The harness never clips losses, repairs native state, or substitutes another ODE/controller.

The checked catalog produces 214 rows across 29 cases. Fourteen multi-step cases pass fresh restart and same-process trial restoration.

| Case | Observed native accounting |
| --- | --- |
| `impervious_zero_roughness` | Rain supplies 256 ft³. Final storage holds 128 ft³. Endpoint bookkeeping exports 256 ft³, exceeding storage-implied export by 128 ft³. |
| `pervious_competing_losses` | Initial storage holds 128 ft³. Evaporation and infiltration each claim 128 ft³. Final storage and runoff equal zero. |

These measured discrepancies prevent unchanged native bookkeeping from defining conservative hydraulic transfers.

Excluded scope includes snow, groundwater, LID, pollutants, seasonal patterns, inter-subcatchment graphs, hydraulic feedback, and full SWMM-driver rewind. Production transfer semantics and competing-loss admission require a separately checked implementation.
