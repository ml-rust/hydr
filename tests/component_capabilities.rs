//! Capability declarations checked against identity, documentation, and admission.
//!
//! Each component appends its declaration to `declarations` and adds one module of
//! admission checks.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "support/exterior_fixture.rs"]
mod fixture;

#[path = "component_capabilities/solver.rs"]
mod solver;

use hydr::capability::{ComponentCapabilities, RegimeScope};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// One component's declaration with the identity constants it must carry.
struct Declared {
    capabilities: &'static ComponentCapabilities,
    contract: Option<&'static str>,
    snapshot_schema: Option<&'static str>,
}

fn declarations() -> Vec<Declared> {
    vec![Declared {
        capabilities: hydr::solver::capabilities(),
        contract: Some(hydr::solver::CONTRACT_VERSION),
        snapshot_schema: Some(hydr::solver::SNAPSHOT_SCHEMA),
    }]
}

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn is_kebab(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('-')
        && !id.ends_with('-')
        && !id.contains("--")
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn rust_sources(directory: &Path, found: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_dir() {
            rust_sources(&path, found)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
    Ok(())
}

/// Returns `true` when a test or library source under the repository defines `fn name(`.
fn defines_function(sources: &[String], name: &str) -> bool {
    let needle = format!("fn {name}(");
    sources.iter().any(|source| source.contains(&needle))
}

#[test]
fn declarations_are_unique_and_identified() -> Result<(), Box<dyn std::error::Error>> {
    let declared = declarations();
    let mut components = BTreeSet::new();
    for entry in &declared {
        let id = entry.capabilities.component.id();
        assert!(components.insert(id), "`{id}` is declared twice");
    }
    let mut files = Vec::new();
    rust_sources(&root().join("tests"), &mut files)?;
    rust_sources(&root().join("crates"), &mut files)?;
    let sources = files
        .iter()
        .map(std::fs::read_to_string)
        .collect::<std::io::Result<Vec<_>>>()?;
    for entry in &declared {
        let capabilities = entry.capabilities;
        let id = capabilities.component.id();
        assert_eq!(capabilities.contract, entry.contract, "`{id}` contract");
        assert_eq!(
            capabilities.snapshot_schema, entry.snapshot_schema,
            "`{id}` snapshot schema"
        );
        let mut ids = BTreeSet::new();
        for record in capabilities
            .unsupported
            .iter()
            .map(|u| (u.id, u.doc))
            .chain(capabilities.limits.iter().map(|l| (l.id, l.doc)))
        {
            assert!(
                is_kebab(record.0),
                "`{id}`: `{}` is not kebab-case",
                record.0
            );
            assert!(ids.insert(record.0), "`{id}`: `{}` appears twice", record.0);
            assert!(
                root().join(record.1).is_file(),
                "`{id}`: `{}` names missing doc `{}`",
                record.0,
                record.1
            );
        }
        for linked in capabilities.nested.iter().chain(capabilities.attachable_to) {
            assert!(
                components.contains(linked.id()),
                "`{id}` links undeclared component `{}`",
                linked.id()
            );
        }
        for runtime in capabilities.runtimes {
            assert!(
                defines_function(&sources, runtime.evidence),
                "`{id}` {} evidence `{}` names no function",
                runtime.runtime.id(),
                runtime.evidence
            );
        }
        for regime in capabilities.regimes {
            if let RegimeScope::Boundary(kind) = regime.scope {
                assert!(
                    capabilities.endpoints.iter().any(|e| e.admits(kind)),
                    "`{id}` claims a regime at unadmitted boundary `{}`",
                    kind.id()
                );
            }
        }
    }
    Ok(())
}
