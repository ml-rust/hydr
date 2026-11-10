# Pinned EPA runoff reference

Use this guide to reproduce original EPA runoff behavior with [the reference harness](../tools/reference/swmm_runoff.py).

The harness records observations independently of Hydr production numerics. It establishes no public runoff API or conservative hydraulic attachment claim.

## Pinned source

| Property | Value |
| --- | --- |
| Engine | SWMM v5.2.4 |
| Commit | `7952ca837988b1c32f791812eccc9fd64547e093` |
| [Original archive](https://codeload.github.com/USEPA/Stormwater-Management-Model/tar.gz/7952ca837988b1c32f791812eccc9fd64547e093) SHA256 | `0c882c315b64d5236b3569f891e9ad71421bc5b7d92a5710ceb1ae3b7f34c6cf` |

## Source and execution

| Source handling | Rule |
| --- | --- |
| Included files | Unchanged `subcatch.c` and `infil.c` |
| Linked files | Every remaining original solver translation unit, including `odesolve.c` |
| Build integrity | Every original file checked against the checksum-pinned archive before and after compilation |
| Execution integrity | Repeats the complete source check |

### Units and factors

Only the included infiltration source maps `UCF` to a native feet/seconds parameter bridge. Original engine conversion remains unchanged elsewhere.

Requested rain passes through original rain-gage units. Evidence records both requested and actual native precipitation.

| Setting | Value |
| --- | --- |
| `InfilFactor`, `Adjust.hydconFactor`, `Evap.recoveryFactor` | One |
| Seasonal patterns and excluded physical components | Explicitly disabled |

### Reuse limits

The [pinned README](https://github.com/USEPA/Stormwater-Management-Model/blob/7952ca837988b1c32f791812eccc9fd64547e093/README.md) declares SWMM source public domain.

The [original ODE header](https://github.com/USEPA/Stormwater-Management-Model/blob/7952ca837988b1c32f791812eccc9fd64547e093/src/solver/odesolve.c) separately records implementation provenance: Numerical Recipes in C, Cambridge University Press, 1992.

No original numerical source body is copied into tracked tooling. This observation harness establishes no production algorithm reuse decision.

## Reproduce

| Requirement | Value |
| --- | --- |
| Python | Python3.10 or newer, with its standard library |
| Compiler | C11, GNU section garbage collection, and `libm` |
| Source | Original extracted archive from [the soil reference guide](soil-reference.md) |
| Working directory | Repository root |
| Destinations | Fresh build and output paths. Existing evidence is never overwritten. |

1. Build the original component.
2. Run the catalog with fresh output paths.

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

### Compiler environment

Canonical flags are `-std=c11 -D_GNU_SOURCE -O2 -ffp-contract=off -ffunction-sections -fdata-sections -Wl,--gc-sections`.

- POSIX interfaces: The feature-test macro exposes them under strict C11.
- Original CMake: Leaves GNU C extensions enabled.
- Compiler overrides: Include/search overrides are removed.

### Catalog coverage

| Group | Cases |
| --- | --- |
| Storage | Depression filling and endpoint thresholds |
| Flow | Equilibrium, recession/refinement, zero branches, and routing |
| Forcing and memory | Antecedent state, intermittent forcing, and competing losses |
| Solver | Original adaptive ODE behavior |

Direct endpoint and ODE probes use copied native state. They remain separate from advancing timesteps.

## Native protocol

The adapter accepts one command per line.

| Input quantity | Native unit |
| --- | --- |
| Length | Feet |
| Area | Square feet |
| Time | Seconds |
| Depth rate | Feet/second |
| Discharge | Cubic feet/second |

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

### Command behavior

| Command | Action |
| --- | --- |
| `INIT` | Records original stored parameters and derives original class-area runoff coefficients |
| `STATE` | Restores the complete emitted restricted witness |
| `STEP` | Follows original old-state, internal-routing, external-runon, and runoff calls |

### Repeatability and restoration

1. Repeat every case in a fresh process. Stdout and stderr must match byte-for-byte.
2. Restart multi-step cases halfway from their complete restricted witness. Following physical observations must retain identical floating-point bits.
3. Advance one trial, restore its saved witness in-process, then repeat the remaining commands.

The final check requires identical physical/diagnostic fields, excluding only the protocol index. Evidence retains trial commands, restored witnesses, and raw hashes.

This checks the isolated component phase only. Full-driver weather cursors, graphs, hydraulic routing, and automatic wet/dry timestep selection remain excluded.

## Evidence and interpretation

| Artifact | Contents |
| --- | --- |
| Build manifest | Archive/file identity, compiler/version/flags, unit bridge, linked sources, adapter/header/binary hashes |
| CSV | Full three-subarea config/state, native soil, routing history, original losses/endpoint volumes, and independent storage diagnostics |
| JSON | Runtime identity, exact catalog commands, repeated hashes, restart witnesses, observed discrepancies, and unsupported scope |
| `.native` directory | Exact hexadecimal commands, original17-digit stdout, stderr, and restricted restart runs |

### Accounting discrepancies remain visible

| Original behavior | Interpretation risk |
| --- | --- |
| Endpoint discharge multiplied by the full interval | Can differ materially from transfer implied by storage changes |
| Independent evaporation and infiltration claims | Both processes can claim the same available water |

The harness retains these discrepancies in its evidence.

| Reported quantity | Meaning |
| --- | --- |
| Gross subarea transfer | Includes previous-endpoint internal delivery |
| Whole-catchment implied outlet export | Includes external runon only |
| Internal routing lag | Difference between prior delivered rates and current endpoint rates |

A coarse native timestep does not guarantee cancellation.

Python independently sums represented subarea storage and endpoint transfers. It checks bookkeeping identities and frozen analytical recession/equilibrium criteria.

The harness never forces the physical mass residual to zero.

The consumer binds emitted forcing, clocks, parameters, and state to submitted commands. Ten archive-free corruption regressions check these gates:

```sh
python3 -m unittest discover -s tools/reference -p test_swmm_runoff.py
```

Material endpoint-quadrature, routing-lag, and competing-loss discrepancies are not arithmetic water adjustments.

The harness preserves original losses and native state. It substitutes no ODE or controller.

The checked catalog produces 214 rows across 29 cases. Fourteen multi-step cases pass fresh restart and same-process trial restoration.

| Case | Observed native accounting |
| --- | --- |
| `impervious_zero_roughness` | Rain supplies 256 ft³. Final storage holds 128 ft³. Endpoint bookkeeping exports 256 ft³, exceeding storage-implied export by 128 ft³. |
| `pervious_competing_losses` | Initial storage holds 128 ft³. Evaporation and infiltration each claim 128 ft³. Final storage and runoff equal zero. |

These measured discrepancies prevent unchanged native bookkeeping from defining conservative hydraulic transfers.

## Scope limits

| Excluded scope | Limit |
| --- | --- |
| Snow, groundwater, LID, and pollutants | No observation of those components |
| Seasonal patterns | Explicitly disabled |
| Inter-subcatchment graphs and hydraulic feedback | No connected-system claim |
| Full SWMM-driver rewind | Only the restricted component phase is restored |

Production transfer semantics and competing-loss admission require a separately checked implementation.
