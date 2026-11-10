# Independent exactly dry comparison

Use this guide to reproduce the three dry-dam comparisons. Check the front diagnostic before interpreting agreement between engines.

hydr and Clawpack agree within the declared field tolerances on three exactly dry Ritter cases. Front accuracy and convergence remain unestablished.

The analytical front error, evaluated on matching cell-mean support, does not decrease monotonically.

## Front accuracy diagnostic

The reported front is the last cell centre whose mean depth exceeds `1e-3 m`. Both engines select the same cell at every resolution.

The exact Ritter cell means select different cells:

| Cells | Numerical front, m | Exact cell-mean threshold front, m | Signed error, m | Absolute error, m |
| --- | --- | --- | --- | --- |
| 100 | 25 | 23 | +2 | 2 |
| 200 | 22.5 | 23.5 | -1 | 1 |
| 400 | 21.25 | 23.75 | -2.5 | 2.5 |

### How the analytical front is calculated

| Quantity | Definition or result |
| --- | --- |
| Exact cell mean inside the rarefaction fan | `((2*c*t-x_i)²+dx²/12)/(9*g*t²)`, with `c=sqrt(g*hL)` |
| Cells crossing a fan boundary | Use the clipped integral |
| Continuous `h=1e-3 m` crossing | `23.86819029 m` at `4 s` |
| Numerical `400`-cell front | Lags the continuous crossing by about `2.618 m` |

The table compares matching cell-mean support. Point sampling therefore does not explain the nonmonotonic errors.

**Acceptance limit:** The frozen front criterion checks agreement between engines only. No additional analytical front criterion applies.

Passing this criterion establishes no analytical front accuracy or convergence.

## Case and external policy

| Input | Value |
| --- | --- |
| Initial depth | `hL=1 m`, `hR=0 m` exactly, with zero discharge |
| Geometry | Flat `[-100,100] m` channel, width `1 m`, one transverse cell |
| Resolution | 100, 200, and 400 uniform cells in the flow direction |
| Physics | Full shallow water, `g=9.81 m/s²`, no rainfall, withdrawals, or friction |
| Endpoint | `4 s`, maximum timestep `0.01 s` |
| Boundaries | Wall state boundaries and extrapolated zero-bed auxiliary boundaries |
| External method | First-order `ClawSolver2D`, dimensional splitting, augmented f-waves, three equations and three waves |
| Precision | CPU `float64` |

Clawpack uses its original `classic2`, `rpn2_sw_aug`, and `rpt2_sw_aug` routines. The adapter checks compiled gravity equals `9.81`.

Read [the Clawpack solver guide](https://www.clawpack.org/riemann/Shallow_water_Riemann_solvers.html) for the available shallow-water Riemann solvers.

### Dry-state handling

The external dry policy differs from hydr's policy:

- Both-dry faces skip the external Riemann calculation.
- Other wet/dry interfaces reset local depth and discharge when `h<=1e-14 m`.
- Negative external inputs print `Negative input` and become zero depth and discharge.
- hydr rejects negative trials and preserves positive films under its declared numerical contract.

| Recorded check | Result |
| --- | --- |
| Accepted states | Nonnegative depth and zero discharge at zero depth |
| Transverse discharge | Zero in both engines |
| External logs | Zero `Negative input` messages |
| Initial dry bed | Exactly dry, without substitution of a positive film |

These checks establish no absence of internal dry-threshold adjustments.

## Frozen criteria and measured fields

| Metric | Required value |
| --- | --- |
| Mean absolute depth difference | `<=0.03 m` |
| Mean absolute discharge difference | `<=0.1 m²/s` |
| Spatial refinement | Both differences strictly decrease through 100, 200, and 400 cells |
| Front agreement | `<=4*dx`, using the last centre where `h>1e-3 m` |
| Reference water residual | `<=1e-8 m² + 1e-12*initial_storage`, per unit width |
| hydr water residual | Existing default tolerance, with zero numerical water adjustment |
| Accepted states | Finite nonnegative depth, finite discharge, and zero discharge at zero depth |
| Transverse discharge | Absolute `qy<=1e-12 m²/s` throughout accepted states and at output |

| Cells | Mean depth difference, m | Mean discharge difference, m²/s | Front difference, m |
| --- | --- | --- | --- |
| 100 | `7.84106e-4` | `1.87709e-3` | 0 |
| 200 | `4.05037e-4` | `8.84786e-4` | 0 |
| 400 | `2.20843e-4` | `4.48079e-4` | 0 |

### Time and water accounting

| Measurement | Result |
| --- | --- |
| Endpoint | Both engines reach `4 s` in 401 accepted steps |
| Minimum accepted interval | `4.13003e-14 s` |
| Maximum represented interval | `0.010000000000000009 s` |
| Minimum accepted depth | Zero, with no negative accepted depth or zero-depth discharge |
| Rust CSV maximum absolute water residual | `5.68e-14` for either engine at unit width |
| NumPy sidecar maximum absolute reference residual | `1.42e-14 m²` |

Different reduction orders explain the last-bit difference.

## Reproduce the compiled reference

### Prepare the source and toolchain

1. Follow [the wet-bed guide](reference-comparison.md) to fetch, check, and extract the pinned archives.
2. Select matching C and Fortran compiler installations through `CC` and `FC`.
3. Run the commands below from the repository root, using fresh build and output directories.

Both components use unchanged Clawpack v5.12.0 sources under BSD-3-Clause licences. All 396 original source-file hashes match the archives in the recorded run.

| Recorded dependency | Version |
| --- | --- |
| Python | `3.12.7` |
| NumPy | `2.4.0` |
| Meson | `1.12.1` |
| Matching GCC C/Fortran compilers | `16.2.1` |

Generated binaries, logs, and manifests remain under ignored `resource/` directories. Existing CSV and JSON destinations reject writes.

```sh
reference_root="$PWD/resource/reference-engines"
pyclaw_root="$reference_root/clawpack/pyclaw-60de625fb28b2e19db828d06db0cc3247109bd24"
riemann_root="$reference_root/clawpack/riemann-c29dcab4af30913fc16e4406fe56fe5f0a322b1f"
build_root="$reference_root/clawpack/build-aug-reproduction"
toolchain_root="$reference_root/toolchain-reproduction"
mkdir "$build_root" "$toolchain_root" || exit 1
python3 -m pip install --target "$toolchain_root/python-tools" meson==1.12.1 || exit 1
CC="${CC:-gcc}" FC="${FC:-gfortran}" python3 - "$pyclaw_root" "$riemann_root" "$build_root" "$toolchain_root" <<'PY' || exit 1
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

pyclaw, riemann, build, tools = map(Path, sys.argv[1:])
environment = dict(os.environ)
environment["PATH"] = str(tools / "python-tools/bin") + os.pathsep + environment["PATH"]
environment["PYTHONPATH"] = str(tools / "python-tools") + os.pathsep + environment.get("PYTHONPATH", "")
environment["CCACHE_DIR"] = str(tools / "ccache")
classic = pyclaw / "src/pyclaw/classic"
sources = {
    "classic2": [classic / name for name in
                 ("limiter.f90", "philim.f90", "flux2.f90", "step2ds.f90", "step2.f90")],
    "sw_aug_2D": [riemann / "src" / name for name in
                  ("rpn2_sw_aug.f90", "rpt2_sw_aug.f90")],
}
compiler = environment["FC"]
manifest = {
    "compiler": compiler,
    "compiler_version": subprocess.check_output(shlex.split(compiler) + ["--version"], text=True),
    "c_compiler": environment["CC"],
    "meson": "1.12.1",
    "commands": [],
}
for module, paths in sources.items():
    command = [sys.executable, "-m", "numpy.f2py", "-c", "--backend", "meson", "-m", module]
    command.extend(map(str, paths))
    with (build / f"{module}-build.log").open("x") as log:
        subprocess.run(command, cwd=build, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    manifest["commands"].append({
        "module": module, "argv": command, "exit_code": 0,
        "sources_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in paths},
    })
with (build / "build-manifest.json").open("x") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
PY
mkdir -p resource/evidence || exit 1
comparison_root=resource/evidence/clawpack-dry-reproduction
mkdir "$comparison_root" || exit 1
mkdir "$comparison_root/reference" "$comparison_root/hydr" || exit 1
for cells in 100 200 400; do
    python3 tools/reference/clawpack_dam.py \
        --pyclaw-root "$pyclaw_root" --riemann-root "$riemann_root" \
        --compiled-root "$build_root" --case dry-ritter --cells "$cells" \
        --output "$comparison_root/reference/dry-ritter_${cells}.csv" \
        > "$comparison_root/dry-ritter_${cells}.log" 2>&1 || exit 1
done
cargo run --example reference_comparison -- dry-suite "$comparison_root/reference" "$comparison_root/hydr" || exit 1
```

### Recorded Arch compiler route

The recorded environment supplies Fortran through an isolated [GCC package](https://geo.mirror.pkgbuild.com/core/os/x86_64/gcc-fortran-16.2.1+r23+gd564253eb6c8-1-x86_64.pkg.tar.zst).

| Property | Requirement |
| --- | --- |
| Package SHA256 | `008443629461abc92ff3ea6d363a5d951e2b314998a4165b4d6f0f7dc5a97007` |
| Compiler search | Extracted GCC `16` and matching system GCC `16` directories through `-B` |
| Host toolchain | Compatible Arch C toolchain |
| Installation | No system install |

The build uses original [classic routines](https://github.com/clawpack/pyclaw/tree/60de625fb28b2e19db828d06db0cc3247109bd24/src/pyclaw/classic) and [normal](https://github.com/clawpack/riemann/blob/c29dcab4af30913fc16e4406fe56fe5f0a322b1f/src/rpn2_sw_aug.f90) and [transverse](https://github.com/clawpack/riemann/blob/c29dcab4af30913fc16e4406fe56fe5f0a322b1f/src/rpt2_sw_aug.f90) augmented routines.

### Retain the evidence

Each execution sidecar records binary/source hashes, build identity, accepted-state checks, and the external dry policy.

Retain captured Fortran output to check `Negative input` messages. Hashes identify executed files. They establish no historical authenticity.

## Interpretation limits

- Whole-domain means include undisturbed cells and dilute local differences.
- This single-row case establishes no genuine two-dimensional transverse-flow agreement.
- Independent terrain, broader hydraulic regimes, and measured-event accuracy remain open.
