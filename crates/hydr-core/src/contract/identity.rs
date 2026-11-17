/// Checks a package's schema and contract identities before any other import check.
#[doc(hidden)]
pub fn package_identity(
    component: &str,
    found_schema: &str,
    expected_schema: &str,
    found_contract: &str,
    expected_contract: &str,
) -> Result<(), String> {
    if found_schema != expected_schema {
        return Err(format!(
            "{component} checkpoint schema: found \"{found_schema}\", expected \"{expected_schema}\": capture a new package from a running {component} of this hydr build"
        ));
    }
    if found_contract != expected_contract {
        return Err(format!(
            "{component} checkpoint contract: found \"{found_contract}\", expected \"{expected_contract}\": the package needs a different numerical contract; restore it with the hydr build that produced it, or capture a new package"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_identities_pass() {
        assert_eq!(package_identity("c", "s1", "s1", "k1", "k1"), Ok(()));
    }

    #[test]
    fn schema_mismatch_message() {
        assert_eq!(
            package_identity("surface solver", "old", "new", "k", "k"),
            Err("surface solver checkpoint schema: found \"old\", expected \"new\": capture a new package from a running surface solver of this hydr build".to_string())
        );
    }

    #[test]
    fn empty_found_schema_shows_quotes() {
        let message = package_identity("c", "", "s", "k", "k")
            .err()
            .unwrap_or_default();
        assert!(message.contains("found \"\", expected \"s\""), "{message}");
    }

    #[test]
    fn contract_mismatch_message() {
        assert_eq!(
            package_identity("c", "s", "s", "old", "new"),
            Err("c checkpoint contract: found \"old\", expected \"new\": the package needs a different numerical contract; restore it with the hydr build that produced it, or capture a new package".to_string())
        );
    }

    #[test]
    fn schema_is_reported_before_contract() {
        let message = package_identity("c", "a", "b", "x", "y")
            .err()
            .unwrap_or_default();
        assert!(message.contains("checkpoint schema"), "{message}");
        assert!(!message.contains("contract"), "{message}");
    }
}
