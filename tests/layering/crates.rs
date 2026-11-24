//! Crate-edge checks. Each member's manifest may depend only on the `hydr-*` crates in
//! `CRATES`, and the root facade holds only `pub use` items.

use super::strip;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Member crate and the `hydr-*` crates it may list under `[dependencies]`.
const CRATES: &[(&str, &[&str])] = &[
    ("hydr-core", &[]),
    ("hydr-soil", &["hydr-core"]),
    ("hydr-storage", &["hydr-core"]),
    ("hydr-structures", &["hydr-core"]),
    ("hydr-channel", &["hydr-core"]),
    ("hydr-drainage", &["hydr-core"]),
    ("hydr-runoff", &["hydr-core", "hydr-soil"]),
    ("hydr-solver", &["hydr-core", "hydr-soil"]),
    (
        "hydr-coupling",
        &[
            "hydr-core",
            "hydr-storage",
            "hydr-structures",
            "hydr-soil",
            "hydr-runoff",
            "hydr-channel",
            "hydr-drainage",
            "hydr-solver",
        ],
    ),
];

#[derive(PartialEq)]
enum Kind {
    Normal,
    Dev,
    Build,
}

/// Dependency kind of a manifest section header, or `None` for other sections.
/// Returns the dependency name when the header is a `[dependencies.<name>]` table.
fn classify(section: &str) -> Option<(Kind, Option<String>)> {
    for (key, kind) in [
        ("dev-dependencies", Kind::Dev),
        ("build-dependencies", Kind::Build),
        ("dependencies", Kind::Normal),
    ] {
        if section == key || section.ends_with(&format!(".{key}")) {
            return Some((kind, None));
        }
        let prefix = format!("{key}.");
        if let Some(name) = section.strip_prefix(&prefix) {
            return Some((kind, Some(name.trim().trim_matches('"').to_string())));
        }
        if let Some(at) = section.find(&format!(".{key}.")) {
            let name = &section[at + key.len() + 2..];
            return Some((kind, Some(name.trim().trim_matches('"').to_string())));
        }
    }
    None
}

/// Returns the `hydr-*` names under `[dependencies]` and one message per `hydr-*` name
/// under `[dev-dependencies]` or `[build-dependencies]`.
fn manifest_edges(manifest: &str) -> (BTreeSet<String>, Vec<String>) {
    let mut deps = BTreeSet::new();
    let mut hidden = Vec::new();
    let mut current: Option<(Kind, Option<String>)> = None;
    let mut record = |kind: &Kind, name: &str, deps: &mut BTreeSet<String>| {
        if !name.starts_with("hydr-") {
            return;
        }
        match kind {
            Kind::Normal => {
                deps.insert(name.to_string());
            }
            Kind::Dev => hidden.push(format!("`{name}` under [dev-dependencies]")),
            Kind::Build => hidden.push(format!("`{name}` under [build-dependencies]")),
        }
    };
    for raw in manifest.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if let Some(header) = line.strip_prefix('[') {
            let section = header.trim_end_matches(']').trim_start_matches('[').trim();
            current = classify(section);
            if let Some((kind, Some(name))) = &current {
                record(kind, name, &mut deps);
            }
            continue;
        }
        if let Some((kind, None)) = &current {
            let key = line
                .split(['=', '.'])
                .next()
                .unwrap_or("")
                .trim()
                .trim_matches('"');
            record(kind, key, &mut deps);
        }
    }
    (deps, hidden)
}

/// Returns one message per manifest edge that the table does not allow.
fn edge_errors(label: &str, allowed: &[&str], manifest: &str) -> Vec<String> {
    let (deps, hidden) = manifest_edges(manifest);
    let allowed: BTreeSet<String> = allowed.iter().map(|s| s.to_string()).collect();
    let mut out = Vec::new();
    for name in deps.difference(&allowed) {
        out.push(format!(
            "{label}: depends on `{name}`, which the crate table does not allow. \
             Remove the edge or change the table in tests/layering/crates.rs."
        ));
    }
    for name in allowed.difference(&deps) {
        out.push(format!(
            "{label}: the crate table lists `{name}`, but the manifest does not depend on it."
        ));
    }
    for what in hidden {
        out.push(format!(
            "{label}: {what}. Members must not name `hydr-*` crates outside [dependencies]."
        ));
    }
    out
}

/// Returns an error message when `src` holds anything but `pub use` items.
fn facade_errors(src: &str) -> Vec<String> {
    let text = strip(src);
    let chars: Vec<char> = text.chars().collect();
    let mut items: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '#' && cur.trim().is_empty() {
            // Skip an attribute `#[..]` or `#![..]`.
            let mut j = i + 1;
            if chars.get(j) == Some(&'!') {
                j += 1;
            }
            if chars.get(j) == Some(&'[') {
                let mut depth = 0usize;
                while j < chars.len() {
                    match chars[j] {
                        '[' => depth += 1,
                        ']' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                i = j + 1;
                continue;
            }
        }
        if ch == ';' {
            items.push(std::mem::take(&mut cur));
        } else {
            cur.push(ch);
        }
        i += 1;
    }
    items.push(cur);
    let mut out = Vec::new();
    for item in items {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        if !item.starts_with("pub use ") {
            let head: String = item.chars().take(40).collect();
            out.push(format!(
                "src/lib.rs: the facade holds only `pub use` items, found `{head}`. \
                 Move the code into a member crate."
            ));
        }
    }
    out
}

fn crate_dirs(root: &Path) -> Result<Vec<String>, String> {
    let dir = root.join("crates");
    let mut names = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| format!("read directory {}: {e}", dir.display()))? {
        let path = entry
            .map_err(|e| format!("read entry in {}: {e}", dir.display()))?
            .path();
        if path.is_dir()
            && let Some(name) = path.file_name().and_then(|n| n.to_str())
        {
            names.push(name.to_string());
        }
    }
    names.sort();
    Ok(names)
}

#[test]
fn every_crate_directory_has_a_row() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut errors = Vec::new();
    for name in crate_dirs(root)? {
        if !CRATES.iter().any(|(row, _)| *row == name) {
            errors.push(format!(
                "crates/{name}: no row in the crate table. \
                 Add a row for `{name}` to CRATES in tests/layering/crates.rs."
            ));
        }
    }
    for (row, _) in CRATES {
        if !root.join("crates").join(row).is_dir() {
            errors.push(format!(
                "crate table row `{row}` has no directory under crates/"
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

#[test]
fn crate_edges_match_the_table() -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut errors = Vec::new();
    for (name, allowed) in CRATES {
        let path = root.join("crates").join(name).join("Cargo.toml");
        let text =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        errors.extend(edge_errors(
            &format!("crates/{name}/Cargo.toml"),
            allowed,
            &text,
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} crate edge error(s):\n{}",
            errors.len(),
            errors.join("\n")
        ))
    }
}

#[test]
fn crate_table_is_acyclic_and_closed() {
    for (name, deps) in CRATES {
        for dep in *deps {
            let row = CRATES.iter().position(|(n, _)| n == dep);
            let me = CRATES.iter().position(|(n, _)| n == name);
            assert!(row.is_some(), "`{name}` allows `{dep}`, which has no row");
            assert!(
                row < me,
                "`{name}` allows `{dep}`, which is not listed before it"
            );
        }
    }
}

#[test]
fn facade_holds_only_pub_use() -> Result<(), String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let text = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let errors = facade_errors(&text);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

#[test]
fn checker_accepts_allowed_manifest() {
    let m = "[package]\nname = \"hydr-soil\"\n\n[dependencies]\nhydr-core = { workspace = true }\nnumr = \"1\"\n\n[features]\ncuda = [\"hydr-core/cuda\"]\n";
    assert_eq!(edge_errors("m", &["hydr-core"], m), Vec::<String>::new());
}

#[test]
fn checker_rejects_forbidden_edge() {
    let m = "[dependencies]\nhydr-core.workspace = true\nhydr-solver = { workspace = true }\n";
    let found = edge_errors("m", &["hydr-core"], m);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains("hydr-solver"), "{}", found[0]);
}

#[test]
fn checker_rejects_missing_edge() {
    let found = edge_errors("m", &["hydr-core"], "[dependencies]\nnumr = \"1\"\n");
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].contains("hydr-core"), "{}", found[0]);
}

#[test]
fn checker_rejects_dev_and_build_dependencies() {
    let m = "[dependencies]\nhydr-core = { workspace = true }\n\n[dev-dependencies]\nhydr-soil = { workspace = true }\n\n[build-dependencies.hydr-runoff]\nworkspace = true\n";
    let found = edge_errors("m", &["hydr-core"], m);
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(found[0].contains("hydr-soil") && found[0].contains("dev-dependencies"));
    assert!(found[1].contains("hydr-runoff") && found[1].contains("build-dependencies"));
}

#[test]
fn checker_reads_dependency_tables() {
    let (deps, _) = manifest_edges("[dependencies.hydr-soil]\nworkspace = true\n");
    assert!(deps.contains("hydr-soil"), "{deps:?}");
}

#[test]
fn facade_checker_accepts_attributed_pub_use_and_comments() {
    let src = "//! doc\n// mod x { }\n#![allow(unused)]\n#[doc(inline)]\npub use hydr_core::{error, grid};\n/* fn f() {} */\n#[doc(inline)]\npub use hydr_soil::soil;\n";
    assert_eq!(facade_errors(src), Vec::<String>::new());
}

#[test]
fn facade_checker_rejects_other_items() {
    for src in [
        "pub use a::b;\nmod x;\n",
        "pub use a::b;\nfn f() {}\n",
        "pub use a::b;\npub struct S;\n",
        "use a::b;\n",
    ] {
        assert!(!facade_errors(src).is_empty(), "accepted: {src}");
    }
}
