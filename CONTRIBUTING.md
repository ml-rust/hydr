# Contributing

hydr develops a Rust library for hydraulic simulation.

## Development checks

Rust 2024 and Rust 1.95 or later are required.

```sh
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo run --example grid
cargo run --example rain_basin
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
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

The engine uses numr 0.15.2 and solvr 0.3.1 through a configured sequential CPU client.
State, geometry, forcing, face reconstruction, transport, source updates, and reductions use numr tensors.
Owned arrays serve input checks and accepted-output projections.
Generic numerical primitives belong in numr or solvr.
Hydraulic equations and acceptance policy belong in hydr.

The current local override uses the sibling numr checkout:

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
