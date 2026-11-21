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

#[path = "component_capabilities/channel_fixture.rs"]
mod channel_fixture;

#[path = "component_capabilities/channel_variants.rs"]
mod channel_variants;

#[path = "component_capabilities/channel.rs"]
mod channel;

#[allow(dead_code)]
#[path = "level_pool/fixture.rs"]
mod level_pool_fixture;

#[allow(dead_code)]
#[path = "coupled_level_pools/fixture.rs"]
mod coupled_pool_fixture;

#[allow(dead_code)]
#[path = "coupled_pool_inflows/inflow_fixture.rs"]
mod inflow_fixture;

#[allow(dead_code)]
#[path = "channel_pools/reference.rs"]
mod reference;

#[allow(dead_code)]
#[path = "channel_pools/fixture.rs"]
mod channel_pool_fixture;

#[path = "component_capabilities/pools_fixture.rs"]
mod pools_fixture;

#[path = "component_capabilities/pools_variants.rs"]
mod pools_variants;

#[path = "component_capabilities/pools.rs"]
mod pools;

#[allow(dead_code)]
#[path = "land_sources/fixture.rs"]
mod land_fixture;

#[allow(dead_code)]
#[path = "nonlinear_reservoir/fixture.rs"]
mod reservoir_fixture;

#[allow(dead_code)]
#[path = "subcatchment/fixture.rs"]
mod subcatchment_fixture;

#[allow(dead_code)]
#[path = "component_capabilities/runoff_pool_support.rs"]
mod runoff_pools;

#[path = "component_capabilities/runoff_fixture.rs"]
mod runoff_fixture;

#[path = "component_capabilities/runoff_variants.rs"]
mod runoff_variants;

#[path = "component_capabilities/runoff.rs"]
mod runoff;

#[allow(dead_code)]
#[path = "controlled_gate_support/fixture.rs"]
mod gate_fixture;

#[allow(dead_code)]
#[path = "controlled_pump_support/fixture.rs"]
mod pump_fixture;

#[path = "component_capabilities/structures_fixture.rs"]
mod structures_fixture;

#[path = "component_capabilities/structures_variants.rs"]
mod structures_variants;

#[path = "component_capabilities/structures.rs"]
mod structures;

#[allow(dead_code)]
#[path = "component_capabilities/surface_channel_support.rs"]
mod surface_channel_support;

#[path = "component_capabilities/surface_channels_fixture.rs"]
mod surface_channels_fixture;

#[path = "component_capabilities/surface_channels_variants.rs"]
mod surface_channels_variants;

#[path = "component_capabilities/surface_channels.rs"]
mod surface_channels;

#[path = "component_capabilities/surface_channel_ports.rs"]
mod surface_channel_ports;

#[path = "component_capabilities/surface_channel_pools.rs"]
mod surface_channel_pools;

#[path = "component_capabilities/culvert_barrels.rs"]
mod culvert_barrels;

#[path = "component_capabilities/culvert_inlets.rs"]
mod culvert_inlets;

#[path = "component_capabilities/culvert_outlets.rs"]
mod culvert_outlets;

#[path = "component_capabilities/culverts.rs"]
mod culverts;

#[path = "component_capabilities/declarations.rs"]
mod declarations;

use declarations::declarations;
use hydr::capability::{ComponentId, RegimeScope, RuntimeKind};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

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

/// Reads every Rust source under `tests/` and `crates/`, with its path.
fn repository_sources() -> std::io::Result<Vec<(PathBuf, String)>> {
    let mut files = Vec::new();
    rust_sources(&root().join("tests"), &mut files)?;
    rust_sources(&root().join("crates"), &mut files)?;
    files
        .into_iter()
        .map(|path| std::fs::read_to_string(&path).map(|source| (path, source)))
        .collect()
}

/// One definition of an evidence function, with the attributes that gate it.
struct Definition<'a> {
    path: &'a Path,
    /// Attribute and comment lines directly above `fn`, plus the file's inner attributes.
    gates: Vec<&'a str>,
}

/// Finds each `fn name(` definition and the attribute lines that gate it.
///
/// A gate counts only on the function's own attribute block or as a file inner attribute.
fn definitions_of<'a>(sources: &'a [(PathBuf, String)], name: &str) -> Vec<Definition<'a>> {
    let needle = format!("fn {name}(");
    let mut found = Vec::new();
    for (path, source) in sources {
        let lines = source.lines().collect::<Vec<_>>();
        let inner = lines
            .iter()
            .copied()
            .filter(|line| line.trim_start().starts_with("#!["));
        for (index, line) in lines.iter().enumerate() {
            let signature = line.trim_start();
            let signature = signature.strip_prefix("pub ").unwrap_or(signature);
            if !signature.starts_with(&needle) {
                continue;
            }
            let mut gates = inner.clone().collect::<Vec<_>>();
            for above in lines[..index].iter().rev() {
                let above = above.trim();
                if above.is_empty() || above.ends_with(['}', ';', '{']) {
                    break;
                }
                gates.push(above);
            }
            found.push(Definition { path, gates });
        }
    }
    found
}

/// Returns the feature gate the runtime's evidence needs: `Some` for a GPU, `None` for CPU.
fn required_gate(runtime: RuntimeKind) -> Result<Option<&'static str>, String> {
    match runtime {
        RuntimeKind::Cpu => Ok(None),
        RuntimeKind::Cuda => Ok(Some("feature = \"cuda\"")),
        RuntimeKind::Wgpu => Ok(Some("feature = \"wgpu\"")),
        other => Err(format!(
            "runtime `{}` has no evidence gate rule. Add one to `required_gate`",
            other.id()
        )),
    }
}

#[test]
fn evidence_gates_come_from_the_attribute_block_only() {
    let source = "fn helper() {}\n\n#[cfg(feature = \"cuda\")]\n#[test]\nfn gated() -> Result {\n}\n\n#[test]\npub fn plain() {\n    let _ = \"#[cfg(feature = \\\"wgpu\\\")]\";\n}\n";
    let sources = vec![(PathBuf::from("synthetic.rs"), source.to_string())];
    let gated = definitions_of(&sources, "gated");
    assert_eq!(gated.len(), 1);
    assert!(gated[0].gates.contains(&"#[cfg(feature = \"cuda\")]"));
    assert!(gated[0].gates.contains(&"#[test]"));
    let plain = definitions_of(&sources, "plain");
    assert_eq!(plain.len(), 1);
    assert_eq!(plain[0].gates, ["#[test]"]);
    assert!(definitions_of(&sources, "missing").is_empty());
    let inner = vec![(
        PathBuf::from("inner.rs"),
        "#![cfg(feature = \"wgpu\")]\nuse x;\n\n#[test]\nfn whole() {}\n".to_string(),
    )];
    let whole = definitions_of(&inner, "whole");
    assert_eq!(whole.len(), 1);
    assert!(whole[0].gates.contains(&"#![cfg(feature = \"wgpu\")]"));
}

/// Each runtime claim names exactly one test, gated by the claimed runtime's feature.
///
/// CUDA evidence carries `cfg(feature = "cuda")` and WGPU evidence `cfg(feature = "wgpu")`.
/// CPU evidence carries neither, so a default build runs it.
#[test]
fn runtime_claims_name_existing_tests() -> Result<(), Box<dyn std::error::Error>> {
    let sources = repository_sources()?;
    let gpu_gates = [RuntimeKind::Cuda, RuntimeKind::Wgpu]
        .into_iter()
        .map(required_gate)
        .collect::<Result<Vec<_>, _>>()?;
    let mut errors = Vec::new();
    for entry in declarations() {
        let id = entry.capabilities.component.id();
        for claim in entry.capabilities.runtimes {
            let (runtime, evidence) = (claim.runtime.id(), claim.evidence);
            let definitions = definitions_of(&sources, evidence);
            let [definition] = definitions.as_slice() else {
                errors.push(format!(
                    "`{id}` {runtime} evidence `{evidence}` has {} definitions. Define it once under tests/ or crates/",
                    definitions.len()
                ));
                continue;
            };
            let place = definition
                .path
                .strip_prefix(root())
                .unwrap_or(definition.path);
            let place = place.display();
            let has = |gate: &str| definition.gates.iter().any(|line| line.contains(gate));
            if !definition.gates.contains(&"#[test]") {
                errors.push(format!(
                    "`{id}` {runtime} evidence `{evidence}` in {place} is not a #[test]. Name a test function"
                ));
            }
            match required_gate(claim.runtime)? {
                Some(gate) if !has(gate) => errors.push(format!(
                    "`{id}` {runtime} evidence `{evidence}` in {place} lacks #[cfg({gate})]. Gate the test on its runtime feature"
                )),
                None if gpu_gates.iter().flatten().any(|gate| has(gate)) => {
                    errors.push(format!(
                        "`{id}` {runtime} evidence `{evidence}` in {place} carries a GPU feature gate. Name an ungated CPU test"
                    ))
                }
                _ => {}
            }
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
    Ok(())
}

#[test]
fn declarations_are_unique_and_identified() -> Result<(), Box<dyn std::error::Error>> {
    let declared = declarations();
    let mut components = BTreeSet::new();
    for entry in &declared {
        let id = entry.capabilities.component.id();
        assert!(components.insert(id), "`{id}` is declared twice");
    }
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
        for option in capabilities.options {
            assert!(is_kebab(option.id), "`{id}`: option `{}`", option.id);
            assert!(
                ids.insert(option.id),
                "`{id}`: `{}` appears twice",
                option.id
            );
            let mut values = BTreeSet::new();
            for value in option.values {
                assert!(is_kebab(value), "`{id}` `{}`: value `{value}`", option.id);
                assert!(
                    values.insert(*value),
                    "`{id}` `{}`: `{value}` twice",
                    option.id
                );
            }
            if let Some(default) = option.default {
                assert!(
                    values.contains(default),
                    "`{id}` `{}`: default `{default}` is not a value",
                    option.id
                );
            }
        }
        for linked in capabilities.nested.iter().chain(capabilities.attachable_to) {
            assert!(
                components.contains(linked.id()),
                "`{id}` links undeclared component `{}`",
                linked.id()
            );
        }
        let find = |component: &hydr::capability::ComponentId| {
            declared
                .iter()
                .map(|d| d.capabilities)
                .find(|c| c.component == *component)
                .ok_or(format!("`{id}` links undeclared `{}`", component.id()))
        };
        // A parent runs its nested components, so it needs their arithmetic too.
        for nested in capabilities.nested {
            let child = find(nested)?;
            for requirement in child.arithmetic {
                assert!(
                    capabilities.arithmetic.contains(requirement),
                    "`{id}` omits `{}` arithmetic `{}`",
                    nested.id(),
                    requirement.id()
                );
            }
            assert!(
                child.attachable_to.contains(&capabilities.component),
                "`{}` omits attachment to `{id}`",
                nested.id()
            );
        }
        for parent in capabilities.attachable_to {
            assert!(
                find(parent)?.nested.contains(&capabilities.component),
                "`{}` does not nest `{id}`",
                parent.id()
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

/// Components that a runtime without correctly rounded elementary functions cannot run.
///
/// Each calls numr `CorrectlyRoundedOps` or the host `correctly_rounded` functions, directly or
/// through a nested component, or needs a `StructureBatch` that does.
const CORRECTLY_ROUNDED_CALLERS: &[ComponentId] = &[
    // Surface Manning `cr_pow_scalar(depth, 7/3)`, and nested soil.
    ComponentId::SurfaceSolver,
    // Reach Manning `cr_pow_scalar(radius, 4/3)`.
    ComponentId::ChannelReachRectangular,
    // Star-depth `cr_log` / `cr_exp` and reach Manning.
    ComponentId::ChannelReachTriangular,
    // Surveyed Manning `cr_pow_scalar(radius, 4/3)`.
    ComponentId::ChannelReachSurveyed,
    // Green-Ampt Newton step `cr_log`.
    ComponentId::Soil,
    // Drainage `cr_pow_scalar` and the host SI coefficient `pow`.
    ComponentId::Reservoir,
    // Weir and orifice log-scale caches and `cr_exp` discharge.
    ComponentId::CoupledLevelPools,
    // Needs a `StructureBatch`, whose construction runs `cr_log` and the correctly rounded probe.
    ComponentId::ControlledGate,
    ComponentId::OperatingControl,
    ComponentId::GateActuator,
    ComponentId::ControlledPump,
    // Interval `acos` for circle angles and interval `log` in the critical and normal residuals.
    ComponentId::CulvertBarrel,
    // Interval `log` and `exp` for the inlet powers, and the barrel's critical depth.
    ComponentId::CulvertInlet,
    // Interval `acos`, `log` and `exp` in the profile integrand and the barrel depth solves.
    ComponentId::CulvertOutlet,
    // Inlet and outlet curves of each direction, and the form-1 critical-depth solve.
    ComponentId::Culvert,
    // Bank weir, host stall forecast `log`, and nested components.
    ComponentId::SurfaceChannels,
    // Nested callers only.
    ComponentId::Subcatchment,
    ComponentId::ChannelPools,
    ComponentId::RunoffPools,
];

#[test]
fn correctly_rounded_elementary_is_declared_by_exactly_its_callers() {
    use hydr::capability::ArithmeticRequirement as A;
    for entry in declarations() {
        let capabilities = entry.capabilities;
        let id = capabilities.component.id();
        let declares = capabilities
            .arithmetic
            .contains(&A::CorrectlyRoundedElementary);
        let calls = CORRECTLY_ROUNDED_CALLERS.contains(&capabilities.component);
        assert_eq!(
            declares, calls,
            "`{id}`: declares correctly-rounded-elementary={declares}, calls it={calls}"
        );
        if declares {
            // The numr family needs IEEE basic arithmetic and fused multiply-add.
            for needed in [
                A::F64Ieee754Basic,
                A::NearestEvenGradualUnderflow,
                A::FusedMulAdd,
            ] {
                assert!(
                    capabilities.arithmetic.contains(&needed),
                    "`{id}` declares correctly-rounded-elementary without `{}`",
                    needed.id()
                );
            }
        }
    }
}

#[test]
fn reference_runtime_meets_correctly_rounded_elementary() {
    assert!(hydr_core::tensor::probe_correctly_rounded::<runtime::Runtime>().is_none());
}
