//! Fails when a module under `src/` references a module its layer does not allow.
//! The allowed-dependency table keeps the module graph acyclic and in layer order.
//!
//! The scan strips comments, string literals and char literals before it reads
//! `crate::<module>` and `$crate::<module>` paths. Raw strings and nested block
//! comments are handled. Newlines survive stripping, so reported lines are exact.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const CORE: &[&str] = &["error", "grid", "model", "forcing", "contract"];

/// Module and the modules it may reference.
const TABLE: &[(&str, &[&str])] = &[
    ("error", &[]),
    ("grid", &["error"]),
    ("model", &["error", "grid"]),
    ("forcing", &[]),
    ("contract", &["error", "grid", "model", "forcing"]),
    ("tensor", CORE),
    (
        "numerics",
        &["error", "grid", "model", "forcing", "contract", "tensor"],
    ),
    (
        "soil",
        &[
            "error", "grid", "model", "forcing", "contract", "tensor", "numerics",
        ],
    ),
    (
        "storage",
        &[
            "error", "grid", "model", "forcing", "contract", "tensor", "numerics",
        ],
    ),
    (
        "structures",
        &[
            "error", "grid", "model", "forcing", "contract", "tensor", "numerics",
        ],
    ),
    (
        "runoff",
        &[
            "error", "grid", "model", "forcing", "contract", "tensor", "numerics", "soil",
        ],
    ),
    (
        "channel",
        &[
            "error", "grid", "model", "forcing", "contract", "tensor", "numerics",
        ],
    ),
    (
        "solver",
        &[
            "error", "grid", "model", "forcing", "contract", "tensor", "numerics", "soil",
        ],
    ),
    (
        "coupling",
        &[
            "error",
            "grid",
            "model",
            "forcing",
            "contract",
            "tensor",
            "numerics",
            "soil",
            "storage",
            "structures",
            "runoff",
            "channel",
        ],
    ),
];

fn allowed(
    table: &[(&'static str, &'static [&'static str])],
    module: &str,
) -> Option<&'static [&'static str]> {
    table
        .iter()
        .find(|(name, _)| *name == module)
        .map(|(_, deps)| *deps)
}

/// Replaces comments, string literals and char literals with spaces. Newlines stay.
fn strip(src: &str) -> String {
    let c: Vec<char> = src.chars().collect();
    let mut out = String::with_capacity(src.len());
    let blank = |ch: char| if ch == '\n' { '\n' } else { ' ' };
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        let next = c.get(i + 1).copied();
        if ch == '/' && next == Some('/') {
            while i < c.len() && c[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else if ch == '/' && next == Some('*') {
            let mut depth = 0usize;
            while i < c.len() {
                if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(c[i]));
                    i += 1;
                }
            }
        } else if let Some(end) = raw_string_end(&c, i) {
            while i < end {
                out.push(blank(c[i]));
                i += 1;
            }
        } else if ch == '"' {
            out.push(' ');
            i += 1;
            while i < c.len() {
                let s = c[i];
                out.push(blank(s));
                i += 1;
                if s == '\\' && i < c.len() {
                    out.push(blank(c[i]));
                    i += 1;
                } else if s == '"' {
                    break;
                }
            }
        } else if ch == '\'' {
            // Char literal `'x'` or `'\..'`. Anything else is a lifetime and stays.
            let end = if next == Some('\\') {
                (i + 2..c.len().min(i + 12))
                    .find(|&j| c[j] == '\'')
                    .map(|j| j + 1)
            } else if c.get(i + 2) == Some(&'\'') {
                Some(i + 3)
            } else {
                None
            };
            match end {
                Some(end) => {
                    while i < end {
                        out.push(blank(c[i]));
                        i += 1;
                    }
                }
                None => {
                    out.push(ch);
                    i += 1;
                }
            }
        } else {
            out.push(ch);
            i += 1;
        }
    }
    out
}

/// Returns the index after a raw string (`r"…"`, `br#"…"#`) that starts at `i`.
fn raw_string_end(c: &[char], i: usize) -> Option<usize> {
    if i > 0 && (c[i - 1].is_alphanumeric() || c[i - 1] == '_') {
        return None;
    }
    let mut j = i;
    if c.get(j) == Some(&'b') {
        j += 1;
    }
    if c.get(j) != Some(&'r') {
        return None;
    }
    j += 1;
    let hashes = c[j..].iter().take_while(|&&h| h == '#').count();
    j += hashes;
    if c.get(j) != Some(&'"') {
        return None;
    }
    j += 1;
    while j < c.len() {
        if c[j] == '"'
            && c[j + 1..]
                .iter()
                .take(hashes)
                .filter(|&&h| h == '#')
                .count()
                == hashes
        {
            return Some(j + 1 + hashes);
        }
        j += 1;
    }
    Some(c.len())
}

fn is_ident(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}

fn skip_ws(c: &[char], mut i: usize, line: &mut usize) -> usize {
    while i < c.len() && c[i].is_whitespace() {
        if c[i] == '\n' {
            *line += 1;
        }
        i += 1;
    }
    i
}

fn read_ident(c: &[char], i: usize) -> (String, usize) {
    let end = (i..c.len()).find(|&j| !is_ident(c[j])).unwrap_or(c.len());
    (c[i..end].iter().collect(), end)
}

/// Returns `(line, module)` for every `crate::<module>` and `$crate::<module>` in stripped
/// source. A `crate::{a, b::c}` group yields the leading name of each top-level item.
fn references(stripped: &str) -> Vec<(usize, String)> {
    let c: Vec<char> = stripped.chars().collect();
    let mut found = Vec::new();
    let mut line = 1;
    let mut i = 0;
    while i < c.len() {
        if c[i] == '\n' {
            line += 1;
        }
        let starts = c[i..].starts_with(&['c', 'r', 'a', 't', 'e'])
            && (i == 0 || c[i - 1] == '$' || !is_ident(c[i - 1]))
            && !c.get(i + 5).is_some_and(|&n| is_ident(n));
        if !starts {
            i += 1;
            continue;
        }
        let mut at = line;
        let mut j = skip_ws(&c, i + 5, &mut at);
        if !c[j..].starts_with(&[':', ':']) {
            i += 5;
            continue;
        }
        j = skip_ws(&c, j + 2, &mut at);
        if c.get(j) == Some(&'{') {
            let mut depth = 0usize;
            let mut item_start = true;
            while j < c.len() {
                match c[j] {
                    '{' => {
                        depth += 1;
                        item_start = depth == 1;
                    }
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    ',' if depth == 1 => item_start = true,
                    '\n' => at += 1,
                    ch if depth == 1 && item_start && is_ident(ch) => {
                        let (name, end) = read_ident(&c, j);
                        found.push((at, name));
                        item_start = false;
                        j = end;
                        continue;
                    }
                    _ => {}
                }
                j += 1;
            }
        } else if c.get(j).is_some_and(|&ch| is_ident(ch)) {
            let (name, _) = read_ident(&c, j);
            found.push((at, name));
        }
        i += 5;
    }
    found
}

/// Returns one message per forbidden or unknown reference in one source file.
fn violations(
    table: &[(&'static str, &'static [&'static str])],
    rel: &str,
    module: &str,
    src: &str,
) -> Vec<String> {
    let Some(deps) = allowed(table, module) else {
        return vec![format!(
            "src/{rel}: module `{module}` is not in the layering table in tests/layering.rs. \
             Add it with its allowed layer."
        )];
    };
    let mut out = Vec::new();
    for (line, target) in references(&strip(src)) {
        if target == module || deps.contains(&target.as_str()) {
            continue;
        }
        let why = if allowed(table, &target).is_some() {
            "forbidden"
        } else {
            "not a module in the layering table"
        };
        out.push(format!(
            "src/{rel}:{line}: `{module}` references `crate::{target}` ({why}). \
             Allowed modules for `{module}`: {deps:?}"
        ));
    }
    out
}

fn module_of(rel: &Path) -> Option<String> {
    let mut parts = rel.components();
    let first = parts.next()?.as_os_str().to_str()?;
    if parts.next().is_some() {
        Some(first.to_string())
    } else {
        Path::new(first).file_stem()?.to_str().map(str::to_string)
    }
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        fs::read_dir(dir).map_err(|e| format!("read directory {}: {e}", dir.display()))?;
    for entry in entries {
        let path = entry
            .map_err(|e| format!("read entry in {}: {e}", dir.display()))?
            .path();
        if path.is_dir() {
            rust_files(&path, out)?;
        } else if path.extension().is_some_and(|x| x == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Returns the first dependency cycle in the table, or an error for an unknown dependency.
fn find_cycle(
    table: &[(&'static str, &'static [&'static str])],
) -> Result<Option<Vec<String>>, String> {
    fn visit<'a>(
        table: &'a [(&'static str, &'static [&'static str])],
        node: &'a str,
        stack: &mut Vec<&'a str>,
        done: &mut BTreeMap<&'a str, bool>,
    ) -> Result<Option<Vec<String>>, String> {
        if let Some(pos) = stack.iter().position(|&n| n == node) {
            let mut cycle: Vec<String> = stack[pos..].iter().map(|s| s.to_string()).collect();
            cycle.push(node.to_string());
            return Ok(Some(cycle));
        }
        if done.get(node) == Some(&true) {
            return Ok(None);
        }
        let deps = allowed(table, node)
            .ok_or_else(|| format!("`{node}` is a dependency but has no table row"))?;
        stack.push(node);
        for dep in deps {
            if let Some(cycle) = visit(table, dep, stack, done)? {
                return Ok(Some(cycle));
            }
        }
        stack.pop();
        done.insert(node, true);
        Ok(None)
    }
    let mut done = BTreeMap::new();
    for (name, _) in table {
        if let Some(cycle) = visit(table, name, &mut Vec::new(), &mut done)? {
            return Ok(Some(cycle));
        }
    }
    Ok(None)
}

#[test]
fn source_modules_respect_layers() -> Result<(), String> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files)?;
    files.sort();
    let mut errors = Vec::new();
    for file in files {
        let rel = file
            .strip_prefix(&src)
            .map_err(|e| format!("{}: {e}", file.display()))?;
        if rel == Path::new("lib.rs") {
            continue;
        }
        let module =
            module_of(rel).ok_or_else(|| format!("no module name for {}", file.display()))?;
        let text =
            fs::read_to_string(&file).map_err(|e| format!("read {}: {e}", file.display()))?;
        errors.extend(violations(TABLE, &rel.to_string_lossy(), &module, &text));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} layering error(s):\n{}",
            errors.len(),
            errors.join("\n")
        ))
    }
}

#[test]
fn table_is_acyclic() -> Result<(), String> {
    match find_cycle(TABLE)? {
        None => Ok(()),
        Some(cycle) => Err(format!(
            "layering table has a cycle: {}",
            cycle.join(" -> ")
        )),
    }
}

#[test]
fn cyclic_table_is_rejected() -> Result<(), String> {
    const CYCLE: &[(&str, &[&str])] = &[("a", &["b"]), ("b", &["c"]), ("c", &["a"])];
    let cycle = find_cycle(CYCLE)?.ok_or("cycle not found")?;
    assert_eq!(cycle, ["a", "b", "c", "a"]);
    Ok(())
}

#[test]
fn table_rows_name_only_table_modules() {
    for (name, deps) in TABLE {
        for dep in *deps {
            assert!(
                allowed(TABLE, dep).is_some(),
                "`{name}` allows `{dep}`, which has no row"
            );
            assert_ne!(name, dep, "`{name}` lists itself");
        }
    }
}

#[test]
fn checker_rejects_upward_reference() {
    let src = "use crate::error::E;\nfn f() {\n    let _ = crate::solver::run();\n}\n";
    let found = violations(TABLE, "grid/mod.rs", "grid", src);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].starts_with("src/grid/mod.rs:3:"), "{}", found[0]);
    assert!(
        found[0].contains("crate::solver") && found[0].contains("[\"error\"]"),
        "{}",
        found[0]
    );
}

#[test]
fn checker_reads_macro_and_group_paths() {
    let src = "macro_rules! m { () => { $crate::coupling::x() }; }\nuse crate::{error::E,\n    channel::C};\n";
    let found = violations(TABLE, "tensor/a.rs", "tensor", src);
    assert_eq!(found.len(), 2, "{found:?}");
    assert!(found[0].contains(":1:") && found[0].contains("crate::coupling"));
    assert!(found[1].contains(":3:") && found[1].contains("crate::channel"));
}

#[test]
fn checker_ignores_comments_and_literals() {
    let src = concat!(
        "// use crate::solver;\n/// crate::solver\n/* crate::solver /* nested crate::solver */ crate::solver */\n",
        "let s = \"crate::solver \\\" crate::solver\";\nlet r = r#\"crate::solver \" crate::solver\"#;\n",
        "let c = '\"'; let d: &'static str = \"x\"; let _ = crate::error::E;\n",
    );
    assert_eq!(
        violations(TABLE, "grid/a.rs", "grid", src),
        Vec::<String>::new()
    );
}

#[test]
fn checker_names_unlisted_module() {
    let found = violations(TABLE, "newmod/a.rs", "newmod", "");
    assert_eq!(found.len(), 1);
    assert!(
        found[0].contains("Add it with its allowed layer"),
        "{}",
        found[0]
    );
}

#[test]
fn module_name_comes_from_first_component_or_stem() {
    assert_eq!(module_of(Path::new("error.rs")).as_deref(), Some("error"));
    assert_eq!(module_of(Path::new("soil/a/b.rs")).as_deref(), Some("soil"));
}
