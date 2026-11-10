# EPA soil reference harness

Use this guide to run the original EPA Green–Ampt functions and retain independent reference results.

[The soil harness](../tools/reference/swmm_soil.py) supplies observations for implementation comparisons. The harness alone establishes no Hydr soil conformance claim.

## Source and reuse

| Property | Value |
| --- | --- |
| Engine | EPA Storm Water Management Model, v5.2.4 |
| Commit | `7952ca837988b1c32f791812eccc9fd64547e093` |
| Archive SHA256 | `0c882c315b64d5236b3569f891e9ad71421bc5b7d92a5710ceb1ae3b7f34c6cf` |
| Source | [Pinned EPA archive](https://codeload.github.com/USEPA/Stormwater-Management-Model/tar.gz/7952ca837988b1c32f791812eccc9fd64547e093) |
| Reuse statement | The [pinned EPA README](https://github.com/USEPA/Stormwater-Management-Model/blob/7952ca837988b1c32f791812eccc9fd64547e093/README.md) releases SWMM source in the Public Domain. |

### Original functions and source checks

| Item | Behavior |
| --- | --- |
| Included source | Unchanged `src/solver/infil.c`, through an absolute compile-time path |
| Native calls | `grnampt_getInfil` and original static `grnampt_getF2` within that translation unit |
| Integrity check | Hardcoded archive checksum and every original regular file checked against archive bytes |

A mutable local hash manifest cannot replace the archive check.

## Reproduce

### 1. Prepare the source

| Requirement | Value |
| --- | --- |
| Python | 3.10 or newer, with its standard library |
| Compiler | C11, GNU section garbage collection, and `libm` |
| Canonical flags | `-O2 -ffp-contract=off -ffunction-sections -fdata-sections -Wl,--gc-sections` |
| Compiler environment | Include and search override variables are removed |

Fetch and extract the pinned archive into an ignored directory:

```sh
mkdir -p resource/reference-engines/swmm || exit 1
curl --fail --location https://codeload.github.com/USEPA/Stormwater-Management-Model/tar.gz/7952ca837988b1c32f791812eccc9fd64547e093 --output resource/reference-engines/swmm/swmm-v5.2.4.tar.gz || exit 1
tar -xzf resource/reference-engines/swmm/swmm-v5.2.4.tar.gz -C resource/reference-engines/swmm || exit 1
```

### 2. Build and run

Run these commands from the repository root. Choose fresh build and output paths:

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

| Existing destination | Result |
| --- | --- |
| Build directory | Build refuses it |
| CSV, JSON, or raw evidence path | Run refuses it |

Both commands preserve unsuccessful attempts. After correcting an error, choose new paths before repeating the commands.

## Native protocol and units

The adapter accepts one command per line. Invalid input or output returns nonzero status.

Rejected cases include malformed shapes and nonfinite values. Unsupported model codes and nonfinite native outputs also reject.

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

### Command behavior

| Command | Action |
| --- | --- |
| `INIT` | Runs original parameter parsing and initialization |
| `STATE` | Runs original parameter parsing, then restores all five state fields |
| `STEP` | Requires initialization |
| `F2` | Requires initialization and leaves soil state unchanged |

### Factors and state interpretation

- Factors: `UCF`, `InfilFactor`, and `Evap.recoveryFactor` equal one.
- Units: Inputs already use native units.
- Derived `Lu`: The original source uses its conductivity conversion and square-root formula.

The adapter never substitutes a rewritten infiltration solver.

`F` records event memory. Recovery can make `F` negative while `Fu` remains positive.

Saturated dry intervals preserve saturation and event memory while decrementing the timer. The CSV separately accumulates infiltration from interval rates. This accumulation never replaces native state.

## Cases and evidence

### Case coverage

| Group | Cases |
| --- | --- |
| State transitions | Initialized wetting/drying, saturation transitions, and recovery |
| Model differences | Original versus modified Green–Ampt |
| Event memory | Seeded and initialized negative memory |
| Exact and adjacent floating-point inputs | `ZERO=1e-10 ft/s`, `Ks`, saturation equality, `ts=10 s`, and `f1=0.01*c1` |
| Other numerical branches | Zero suction, zero maximum deficit, Newton convergence, and singular fallback |

| Original helper rule | Value |
| --- | --- |
| Short-step and front comparisons | Strict |
| Newton increment threshold | `1e-5 ft` |
| Newton limit | At most 20 iterations |
| Fallback | Returns `F2min` |

The harness records these rules without replacing their numerical tolerances.

### Repeatability and artifacts

Every case runs twice in serialized native processes. Serialization preserves shared `Fumax` and `InfilFactor` assumptions.

Repeated stdout must match byte-for-byte. Branch checks assert consequences of the original source without implementing another infiltration algorithm.

| Artifact | Contents |
| --- | --- |
| Build manifest | Source hashes, compiler identity, exact flags, adapter hashes, binary hash, factors |
| CSV | Exact initial state, preceding state, model, each call input, interval rate, direct root, all five outputs |
| JSON | Source/build provenance, runtime identity, helper hashes, case inputs, repeat hashes, CSV checksum |
| `.native` directory | Hexadecimal input commands, two native stdout CSV files, and stderr per case |

| Encoding or identity | Rule |
| --- | --- |
| CSV floating-point fields | 17 significant digits |
| Raw inputs | Hexadecimal encodings for exact round trips |
| Build/execution checks | Source, compiler, adapter, binary, and build-manifest hashes checked around the relevant operation |

## Scope limits

This harness isolates native soil functions. Its observations support Hydr soil-integration and restart comparisons.

| Excluded process | Consequence |
| --- | --- |
| SWMM catchment routing | No whole-catchment claim |
| Rainfall interpolation and evaporation schedules | Inputs remain explicit native calls |
| Groundwater coupling | No groundwater feedback |
| Finite surface-water availability | No conservative surface-water coupling claim |
