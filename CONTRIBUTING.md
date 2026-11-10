# Contributing

hydr develops a Rust library for hydraulic simulation.

## Development checks

Rust 2024 and Rust 1.95 or later are required.

CPU baseline checks:

```sh
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo run --example grid
cargo run --example rain_basin
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

Optional all-feature compilation and strict linting require the CUDA toolkit and backend build dependencies:

```sh
cargo check --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Actual backend tests require a working CUDA device and driver, plus a Vulkan adapter and driver for WGPU.
These tests use actual backend clients without CPU fallback or skipped execution.

```sh
cargo test --all-features --test runtime_backends
```

## Engineering conventions

- Module roots contain wiring and module documentation.
- Responsibility-specific modules contain implementation.
- Public quantities use SI units unless their API states otherwise.
- Runtime errors return typed results.
- Numerical changes include analytical or independent reference checks.
- Performance comparisons report hardware, precision, numerical error, and memory use.
- Internal plans belong in `resource/`.

## Dependencies

The engine uses numr 0.15.2 and solvr 0.3.1 through caller-selected runtime clients.
State, geometry, forcing, face reconstruction, transport, source updates, and reductions use numr tensors.
Owned arrays serve input checks and explicit accepted-output projections.
Public tensor-bearing components use `R: Runtime<DType = DType>` without a default backend.
Constructors receive `R::Client` before hydraulic inputs. Device placement comes from that client.
Factories inherit their parent's runtime and client. Algorithms never create a CPU fallback.
Tests and reference executables share `tests/support/runtime.rs` for explicit sequential CPU selection.
Backend execution checks use actual backend clients and retain all numerical admission requirements.
The `cuda` and `wgpu` features forward backend support into numr and solvr.
WGPU rejects the current `f64` profile. CUDA admission retains its algorithm-specific arithmetic restrictions.
Generic numerical primitives belong in numr or solvr.
Hydraulic equations and acceptance policy belong in hydr.

The current local overrides use the sibling numr and solvr checkouts:

```toml
# .cargo/config.toml (ignored)
[patch.crates-io]
numr = { path = "../numr" }
solvr = { path = "../solvr" }
```

CI checks both dependency repositories through the same local patch mechanism.
numr owns stable elementwise magnitude through `HypotOps::hypot`.
solvr owns batched bracket expansion/bisection through `BatchedBisectionAlgorithms::bisect_batched`.
hydr supplies the physical Riemann residual and consumes solvr's retained upper endpoint.
Committed manifests must use published versions or explicit Git revisions.
Local overrides belong in ignored `.cargo/config.toml`.
