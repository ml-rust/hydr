//! Test helper that checks a list of capability ids.

use std::collections::BTreeSet;

/// Returns `true` when `id` is non-empty lowercase kebab-case.
pub(super) fn is_kebab(id: &str) -> bool {
    !id.is_empty()
        && !id.starts_with('-')
        && !id.ends_with('-')
        && !id.contains("--")
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// Panics when an id is not kebab-case or appears twice.
pub(super) fn assert_ids(ids: &[&str]) {
    let mut seen = BTreeSet::new();
    for id in ids {
        assert!(is_kebab(id), "`{id}` is not kebab-case");
        assert!(seen.insert(*id), "`{id}` appears twice");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kebab_rule() {
        assert!(is_kebab("f64-ieee754-basic"));
        assert!(is_kebab("cpu"));
        for bad in ["", "-a", "a-", "a--b", "Wall", "a_b", "a b"] {
            assert!(!is_kebab(bad), "`{bad}` passed");
        }
    }

    #[test]
    #[should_panic(expected = "appears twice")]
    fn duplicate_id_panics() {
        assert_ids(&["a", "b", "a"]);
    }
}
