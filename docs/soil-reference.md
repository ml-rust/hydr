# EPA soil reference harness

`tools/reference/swmm_soil.py` builds and runs the original EPA Green–Ampt functions independently of hydr. The harness records native results for future implementation comparisons. It establishes no hydr soil conformance claim.

## Source and reuse

| Property | Value |
| --- | --- |
| Engine | EPA Storm Water Management Model, v5.2.4 |
| Commit | `7952ca837988b1c32f791812eccc9fd64547e093` |
| Archive SHA256 | `0c882c315b64d5236b3569f891e9ad71421bc5b7d92a5710ceb1ae3b7f34c6cf` |
| Source | [Pinned EPA archive](https://codeload.github.com/USEPA/Stormwater-Management-Model/tar.gz/7952ca837988b1c32f791812eccc9fd64547e093) |
| Reuse statement | The [pinned EPA README](https://github.com/USEPA/Stormwater-Management-Model/blob/7952ca837988b1c32f791812eccc9fd64547e093/README.md) releases SWMM source in the Public Domain. |

The adapter includes the unchanged `src/solver/infil.c` through an absolute compile-time path. It calls `grnampt_getInfil` and the original static `grnampt_getF2` within that translation unit. The harness checks the hardcoded archive checksum and every original regular file against archive bytes. A mutable local hash manifest cannot replace this check.

## Reproduce

Requirements: Python 3.10 or newer, its standard library, and a C11 compiler supporting GNU section garbage collection and `libm`. The canonical build uses `-O2 -ffp-contract=off -ffunction-sections -fdata-sections -Wl,--gc-sections`. Compiler include and search override variables are removed.

Download and extract the pinned archive into an ignored directory:

```sh
mkdir -p resource/reference-engines/swmm || exit 1
curl --fail --location https://codeload.github.com/USEPA/Stormwater-Management-Model/tar.gz/7952ca837988b1c32f791812eccc9fd64547e093 --output resource/reference-engines/swmm/swmm-v5.2.4.tar.gz || exit 1
tar -xzf resource/reference-engines/swmm/swmm-v5.2.4.tar.gz -C resource/reference-engines/swmm || exit 1
```

Run each command from the repository root. Choose fresh build and output paths:

```sh
python3 tools/reference/swmm_soil.py build \
  --source-root resource/reference-engines/swmm/Stormwater-Management-Model-7952ca837988b1c32f791812eccc9fd64547e093 \
  --archive resource/reference-engines/swmm/swmm-v5.2.4.tar.gz \
  --build-dir resource/reference-engines/swmm/soil-build \
  --cc gcc || exit 1
python3 tools/reference/swmm_soil.py run \
  --source-root resource/reference-engines/swmm/Stormwater-Management-Model-7952ca837988b1c32f791812eccc9fd64547e093 \
  --archive resource/reference-engines/swmm/swmm-v5.2.4.tar.gz \
  --build-dir resource/reference-engines/swmm/soil-build \
  --output resource/reference-engines/swmm/soil-reference.csv || exit 1
```

The build refuses an existing build directory. The run refuses existing CSV, JSON, or raw evidence paths. Both commands preserve unsuccessful attempts for inspection. Choose new paths after correcting an error.

## Native protocol and units

The adapter accepts one command per line. It rejects malformed shapes, nonfinite inputs, unsupported model codes, and nonfinite native outputs with nonzero status.

```text
INIT S Ks IMDmax
STATE S Ks IMDmax IMD F Fu Sat T
STEP model dt irate ponded_depth
F2 f1 c1 ks ts
```

| Field | Native meaning |
| --- | --- |
| `S`, `F`, `Fu`, `Lu`, `ponded_depth`, `f1`, `c1`, `F2` | Feet |
| `Ks`, `irate`, returned interval rate | Feet per second |
| `dt`, `T`, `ts` | Seconds |
| `IMD`, `IMDmax` | Dimensionless moisture deficit |
| `Sat` | Exact zero or one |
| `model` | `2`: original Green–Ampt, `3`: modified Green–Ampt |

`INIT` invokes original parameter parsing and initialization. `STATE` invokes original parameter parsing and restores all five state fields. `STEP` and `F2` require initialization. `F2` leaves the soil state unchanged.

`UCF`, `InfilFactor`, and `Evap.recoveryFactor` equal one. Inputs already use native units. The original source computes `Lu` through its own conductivity conversion and square-root formula. The adapter never substitutes a rewritten infiltration solver.

`F` is event memory. Recovery can produce negative `F` while `Fu` remains positive. Saturated dry intervals preserve saturation and event memory while decrementing the timer. The CSV separately accumulates interval infiltration from returned rates. That accumulation does not replace native state.

## Cases and evidence

The fixed catalog covers initialized wetting and drying, original/modified divergence, seeded and initialized negative event memory, saturation transitions, and recovery. Exact and adjacent floating-point cases cover `ZERO=1e-10 ft/s`, `Ks`, saturation equality, `ts=10 s`, and `f1=0.01*c1`. Other cases cover zero suction, zero maximum deficit, Newton convergence, and singular fallback.

The original helper uses strict short-step and front comparisons. Its Newton increment threshold is `1e-5 ft`, with at most 20 iterations. Its fallback returns `F2min`. The harness records those source semantics without replacing their numerical tolerances.

Every case runs twice in serialized native processes. Serialization preserves the shared `Fumax` and `InfilFactor` assumptions. Repeated stdout must match byte-for-byte. Branch checks assert source consequences without implementing another infiltration algorithm.

| Artifact | Contents |
| --- | --- |
| Build manifest | Source hashes, compiler identity, exact flags, adapter hashes, binary hash, factors |
| CSV | Exact initial state, preceding state, model, each call input, interval rate, direct root, all five outputs |
| JSON | Source/build provenance, runtime identity, helper hashes, case inputs, repeat hashes, CSV checksum |
| `.native` directory | Hexadecimal input commands, two native stdout CSV files, and stderr per case |

CSV floating-point fields use 17 significant digits. Raw inputs use hexadecimal encodings for exact round trips. Source, compiler, adapter, binary, and build-manifest hashes are checked around the relevant build or execution envelope.

This harness isolates native soil functions. It excludes SWMM catchment routing, rainfall interpolation, evaporation schedules, groundwater coupling, and finite surface-water availability. It supplies reference evidence for future hydr soil integration and restart checks.
