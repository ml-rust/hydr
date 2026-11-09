# Independent wet-bed comparison

Hydraulic developers can compare hydr against an independently executed Clawpack solver with the reference harness.

The six checked runs use untouched PyClaw and Riemann sources from Clawpack v5.12.0.
These norms measure differences between engines. Analytical accuracy remains a separate gate.

## Cases and methods

| Input | Value |
| --- | --- |
| Cases | `wet-quarter`: `hL=1 m`, `hR=0.25 m`. `wet-tenth`: `hL=1 m`, `hR=0.1 m` |
| Geometry | Flat channel over `[-100,100] m`, width `1 m`, initial interface at `x=0` |
| Resolution | 100, 200, and 400 uniform cells, with matched finite-volume cell means |
| Initial momentum | Zero |
| Gravity | `9.81 m/s²` |
| Endpoint | `4 s`, with maximum timestep `0.01 s` |
| Sources | No rainfall, withdrawals, or friction |
| Boundaries | Impermeable remote walls outside the disturbed region |

hydr executes its CPU `f64` method on one transverse row.
Clawpack executes classic first-order wave propagation with its Python HLL routine and Roe/Einfeldt speed estimates.
hydr uses conservative HLL fluxes with physical bracketed wave bounds.
Neither run uses second-order corrections or subgrid terrain.

The Python adapter loads original PyClaw, static metadata, and shallow-water HLL modules through namespace packages.
This route bypasses unrelated compiled package imports and requires no Fortran compiler.
The adapter changes no upstream numerical source.
Upstream logging creates a local file and socket. The adapter places its log beside the generated reference fields.

## Frozen criteria

| Metric | Required value |
| --- | --- |
| Mean absolute depth difference | `<=0.03 m` |
| Mean absolute discharge difference | `<=0.1 m²/s` |
| Spatial refinement | Both differences strictly decrease through 100, 200, and 400 cells for each case |
| Front position | Difference `<=4*dx`, using the last centre where `h>hR+0.05*(1-hR)` |
| Reference water residual | `<=1e-8 m² + 1e-12*initial_storage`, per unit width |
| hydr water residual | Existing default absolute-plus-relative tolerance, with zero water adjustment |
| Wet state | Strictly positive depth at every accepted reference state and exported endpoint |
| Input agreement | Matching case metadata, endpoint, cell centres, and row count |

The suite rejects missing fronts and malformed or mismatched reference files.
Exactly dry cases remain excluded because this Python HLL routine divides by depth.
The harness never replaces a dry bed with a positive film.

## Reproduce the comparison

The recorded environment uses Python 3.12.7 and NumPy 2.4.0.
Other environments require repeating the checks. Follow [Contributing](../CONTRIBUTING.md) for hydr's Rust dependencies and local patches.

Run these commands from the repository root.
Sources and generated fields remain under ignored `resource/` directories.
Choose a fresh `comparison_root` for each run. Existing CSV and JSON destinations reject writes.

```sh
mkdir -p resource/reference-engines/clawpack || exit 1
curl --fail --location --output resource/reference-engines/clawpack/pyclaw.tar.gz https://codeload.github.com/clawpack/pyclaw/tar.gz/60de625fb28b2e19db828d06db0cc3247109bd24 || exit 1
curl --fail --location --output resource/reference-engines/clawpack/riemann.tar.gz https://codeload.github.com/clawpack/riemann/tar.gz/c29dcab4af30913fc16e4406fe56fe5f0a322b1f || exit 1
sha256sum --check <<'EOF' || exit 1
5b05db7fc44dae916095f1b778dd123d0aa7f3201e9135cb0e18dc5d773d8452  resource/reference-engines/clawpack/pyclaw.tar.gz
3f4675c34cd16f58ceebde46d0b9ff52e437382184b918d1d908e642a8862ef0  resource/reference-engines/clawpack/riemann.tar.gz
EOF
tar -xzf resource/reference-engines/clawpack/pyclaw.tar.gz -C resource/reference-engines/clawpack || exit 1
tar -xzf resource/reference-engines/clawpack/riemann.tar.gz -C resource/reference-engines/clawpack || exit 1
mkdir -p resource/evidence || exit 1
comparison_root=resource/evidence/pyclaw-wet-reproduction
mkdir "$comparison_root" || exit 1
mkdir "$comparison_root/reference" "$comparison_root/hydr" || exit 1
for case_name in wet-quarter wet-tenth; do
    for cells in 100 200 400; do
        python3 tools/reference/pyclaw_wet_dam.py \
            --pyclaw-root resource/reference-engines/clawpack/pyclaw-60de625fb28b2e19db828d06db0cc3247109bd24 \
            --riemann-root resource/reference-engines/clawpack/riemann-c29dcab4af30913fc16e4406fe56fe5f0a322b1f \
            --case "$case_name" --cells "$cells" \
            --output "$comparison_root/reference/${case_name}_${cells}.csv" \
            > "$comparison_root/${case_name}_${cells}.log" 2>&1 || exit 1
    done
done
cargo run --example reference_comparison -- suite "$comparison_root/reference" "$comparison_root/hydr" || exit 1
cargo test --example reference_comparison || exit 1
```

Each reference CSV declares its case and method before the `x,h,qx` header.
Its JSON sidecar records source hashes, command, Python/NumPy versions, water accounting, and accepted timestep extrema.
The sidecar also records the adapter hash and output hash.
Retain these files with the source archives and their licences.
Hashes identify the executed files. They establish no historical authenticity.

Both pinned components carry BSD-3-Clause licences: [PyClaw](https://github.com/clawpack/pyclaw/blob/60de625fb28b2e19db828d06db0cc3247109bd24/LICENSE) and [Riemann](https://github.com/clawpack/riemann/blob/c29dcab4af30913fc16e4406fe56fe5f0a322b1f/LICENSE).
The executed [PyClaw solver](https://github.com/clawpack/pyclaw/blob/60de625fb28b2e19db828d06db0cc3247109bd24/src/pyclaw/classic/solver.py) calls the original [Python HLL routine](https://github.com/clawpack/riemann/blob/c29dcab4af30913fc16e4406fe56fe5f0a322b1f/riemann/shallow_1D_py.py).

## Measured results

| Case | Cells | Mean depth difference, m | Mean discharge difference, m²/s | Front difference, m |
| --- | --- | --- | --- | --- |
| `wet-quarter` | 100 | `3.54130e-5` | `1.05277e-4` | 0 |
| `wet-quarter` | 200 | `1.41325e-5` | `3.20972e-5` | 0 |
| `wet-quarter` | 400 | `7.36057e-6` | `1.40974e-5` | 0 |
| `wet-tenth` | 100 | `6.23146e-5` | `1.61959e-4` | 0 |
| `wet-tenth` | 200 | `3.58245e-5` | `7.28412e-5` | 0 |
| `wet-tenth` | 400 | `2.27618e-5` | `3.96578e-5` | 0 |

Both engines reach `4 s` in 401 accepted steps in every case.
The smallest interval is `4.13003e-14 s`, completing the endpoint after accumulated floating-point time additions.
The largest represented interval is `0.010000000000000009 s`.
The harness records actual intervals instead of inferring equal partitions from equal timestep ceilings.

Rust's CSV comparison reports maximum absolute reference and hydr water residuals of `4.69e-13 m²` and `5.68e-14 m³`.
The unit width makes their numerical storage values comparable.
Python sidecars use NumPy reductions, so their last bits differ from Rust's CSV sums.

Whole-domain means include undisturbed cells and dilute local differences.
Equal threshold-defined fronts establish no exact shock-position accuracy.
These runs establish selected wet-bed agreement only.
Independent dry-bed, terrain, two-dimensional transverse-flow, and measured-event comparisons remain open.
