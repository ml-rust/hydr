//! Command-line choice of the certified root probe.

use solvr::optimize::CertifiedRootProbe;

/// Parses the optional probe argument: `midpoint` or `slack-secant` (default).
pub fn parse(arg: Option<&String>) -> Result<CertifiedRootProbe, String> {
    match arg.map(String::as_str) {
        Some("midpoint") => Ok(CertifiedRootProbe::Midpoint),
        None | Some("slack-secant") => Ok(CertifiedRootProbe::SlackSecant),
        Some(other) => Err(format!("probe={other:?}: use `midpoint` or `slack-secant`")),
    }
}
