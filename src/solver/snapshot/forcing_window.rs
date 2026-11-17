//! Package forcing-prefix consistency and definition equivalence.

use super::SnapshotData;
use super::check::require;
use crate::contract::EngineError;
use crate::solver::compatibility::Candidate;
use crate::solver::definition::Definition;

/// Checks the package prefix shape, then folds both histories to the later base and compares them.
///
/// The prefix must end at the first retained start bit for bit.
/// An empty prefix has equal start and end bits and the origin digest.
///
/// # Errors
/// Rejects an inconsistent prefix, an engine window that starts after the package window, or any definition difference.
pub(super) fn check(definition: &Definition, data: &SnapshotData) -> Result<(), EngineError> {
    let prefix = &data.forcing_prefix;
    let first = data
        .setup
        .forcing
        .intervals()
        .first()
        .map(|interval| interval.start);
    require(
        first.is_some_and(|first| prefix.consistent(first)),
        "forcing prefix",
        format!("prefix={prefix:?}, first retained start={first:?}"),
    )?;
    let candidate = Candidate {
        domain: &data.setup.domain,
        land: &data.setup.land,
        boundaries: &data.setup.boundaries,
        config: &data.setup.config,
        prefix,
        forcing: data.setup.forcing.intervals(),
        chain: None,
        retained_from: prefix.intervals,
    };
    require(
        definition.equivalent(&candidate)?,
        "definition",
        "domain, boundaries, forcing prefix, forcing window, or config differs",
    )
}
