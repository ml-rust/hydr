# Contributing

hydr develops a Rust library for hydraulic simulation.

## Development checks

Rust 2024 and Rust 1.95 or later are required.

### CPU baseline

Run these checks before submitting a code change:

```sh
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo run --example grid
cargo run --example rain_basin
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

### Optional backends

All-feature compilation and strict linting require the CUDA toolkit and backend build dependencies:

```sh
cargo check --all-targets --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Actual backend tests require a working CUDA device and driver, plus a Vulkan adapter and driver for WGPU. These tests use actual backend clients without CPU fallback or skipped execution.

```sh
cargo test --all-features --test runtime_backends
```

## Documentation

Write guides that help library users complete a task or understand a constraint.

- Purpose: Start with what the reader can learn or do.
- Language: Use short, direct sentences. Explain necessary technical terms when first used.
- Structure: Use descriptive headings and short paragraphs. Use lists for steps and tables for useful comparisons.
- Content: Give every paragraph and table a reader need. Remove repetition and details without a purpose.
- Accuracy: Preserve assumptions, limits, units, error explanations, and numerical evidence when shortening text.
- Navigation: Link to detailed contracts and reference results instead of repeating them across guides.
- Review: Check that a new user can find prerequisites, follow instructions, and recognize unsupported uses.

Keep public guidance in `docs/`. Keep plans, research notes, and generated evidence outside public documentation.

### Review the source

Review documentation from its Markdown source. Use visual tools only for a specific layout defect or an explicit request.

- Paragraphs: Keep one reader decision per paragraph. Source line breaks do not create separate paragraphs.
- Procedures: Use numbered steps for actions. Give each step a clear input or outcome.
- Choices: Use a table when readers must compare inputs, supported cases, or results. Keep long signatures outside comparison tables.
- Detail: Explain the user action before exact bit rules, internal checks, or numerical derivations.
- Limits: Separate requirements, results, and unsupported uses so readers can find each one.
- Evidence: Link test inventories and detailed numerical results from task guides to the relevant acceptance or reference page.

Read each section as a new user. Check that its purpose, next action, units, and limits are clear. A prose lint or shorter word count does not replace this review.

## Engineering conventions

- Module roots contain wiring and module documentation.
- Responsibility-specific modules contain implementation.
- Public quantities use SI units unless their API states otherwise.
- Runtime errors return typed results.
- Numerical changes include analytical or independent reference checks.
- Performance comparisons report hardware, precision, numerical error, and memory use.

## Dependencies

The engine uses numr 0.15.2 and solvr 0.3.1 through caller-selected runtime clients. State, geometry, forcing, face reconstruction, transport, source updates, and reductions use numr tensors. Owned arrays serve input checks and explicit accepted-output projections.

### Runtime clients

Public tensor-bearing components use `R: Runtime<DType = DType>` without a default backend. Constructors receive `R::Client` before hydraulic inputs. Device placement comes from that client. Factories inherit their parent's runtime and client. Algorithms never create a CPU fallback.

Tests and reference executables share `tests/support/runtime.rs` for explicit sequential CPU selection. Backend execution checks use actual backend clients and retain all numerical admission requirements.

The `cuda` and `wgpu` features forward backend support into numr and solvr. WGPU rejects the current `f64` profile. CUDA admission retains its algorithm-specific arithmetic restrictions.

### Checked dependency sources

The current numerical checks use these local source commits. Match them when reproducing the recorded results.

| Dependency | Version | Source commit |
| --- | --- | --- |
| numr | `0.15.2` | `167652acd7798730987186efd04961aa7cd2b70b` |
| solvr | `0.3.1` | `9bae04662a9e3da2d4913bf9900392fa7a05b913` |

Numr supplies checked square roots and fused multiply-add arithmetic for CPU and CUDA. Solvr supplies checked interval arithmetic, certified integration, and certified root solving. Package versions remain unchanged during this unpublished work.

### Local dependency overrides

The current local overrides use the sibling numr and solvr checkouts:

```toml
# .cargo/config.toml (ignored)
[patch.crates-io]
numr = { path = "../numr" }
solvr = { path = "../solvr" }
```

CI checks both dependency repositories at the recorded commits through the same local patch mechanism. Publish those dependency commits before running remote CI. Committed manifests must use published versions or explicit Git revisions. Local overrides belong in ignored `.cargo/config.toml`.

### Numerical ownership

numr owns stable elementwise magnitude through `HypotOps::hypot`. solvr owns batched bracket expansion/bisection through `BatchedBisectionAlgorithms::bisect_batched`. hydr supplies the physical Riemann residual and consumes solvr's retained upper endpoint.
