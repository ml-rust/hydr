//! Exclusive control and parameter authority table for coupled surface-drainage networks.
//!
//! Parses SWMM `.inp` `[CONTROLS]` to register every link owned by native SWMM control rules,
//! and checks external `hydr` device authority claims and inlet bindings so no link setting or
//! `(surface_cell, drainage_node)` inlet pair ever has two writers.

use super::error::SurfaceDrainageError;
use super::inlet::SurfaceDrainageInlet;
use crate::drainage::SwmmControlAuthority;
use std::collections::{BTreeMap, BTreeSet};

/// External authority claim on a SWMM drainage link setting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrainageControlClaim {
    /// SWMM link identifier (case-insensitive in SWMM `.inp` files).
    pub link_id: String,
    /// Declared control-authority owner for this link's setting.
    pub authority: SwmmControlAuthority,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LinkWriter {
    NativeSwmmControls,
    ExternalClaim(SwmmControlAuthority),
}

impl LinkWriter {
    fn label(&self) -> &'static str {
        match self {
            Self::NativeSwmmControls => "native SWMM [CONTROLS]",
            Self::ExternalClaim(auth) => auth.id(),
        }
    }
}

/// Checked exclusive-writer table for SWMM drainage link controls and surface-drainage inlets.
#[derive(Clone, Debug)]
pub struct DrainageControlAuthorityTable {
    link_writers: BTreeMap<String, LinkWriter>,
    native_controlled_links: BTreeSet<String>,
}

impl DrainageControlAuthorityTable {
    /// Builds and checks the authority table from `.inp` text, valid link IDs, external claims,
    /// and declared surface-drainage inlets.
    ///
    /// Rejects with `duplicate-parameter-authority` if:
    /// - an external claim targets a link already written by native SWMM `[CONTROLS]`
    /// - two external claims target the same SWMM link
    /// - an external claim names a link that does not exist in the SWMM network
    /// - an external claim requests an unattached hybrid actuation mode
    /// - two inlets declare the same `(surface_cell, drainage_node_id)` pair or duplicate `id`
    pub fn build(
        inp_content: &str,
        known_link_ids: &[String],
        claims: &[DrainageControlClaim],
        inlets: &[SurfaceDrainageInlet],
    ) -> Result<Self, SurfaceDrainageError> {
        let known_upper: BTreeSet<String> = known_link_ids
            .iter()
            .map(|id| id.to_ascii_uppercase())
            .collect();
        let native_controlled_links = parse_native_controlled_links(inp_content, &known_upper);

        let mut link_writers = BTreeMap::new();
        for link in &native_controlled_links {
            link_writers.insert(link.clone(), LinkWriter::NativeSwmmControls);
        }

        for claim in claims {
            let trimmed = claim.link_id.trim();
            if trimmed.is_empty() {
                return Err(SurfaceDrainageError::input(
                    "drainage control claim link_id must not be empty",
                ));
            }
            let key = trimmed.to_ascii_uppercase();
            if !known_upper.contains(&key) {
                return Err(SurfaceDrainageError::input(format!(
                    "drainage control claim references unknown SWMM link `{trimmed}`"
                )));
            }
            if let Some(existing) = link_writers.get(&key) {
                if matches!(existing, LinkWriter::NativeSwmmControls)
                    && claim.authority == SwmmControlAuthority::NativeSwmm
                {
                    // Explicit confirmation of native SWMM ownership is allowed once;
                    // replace with ExternalClaim(NativeSwmm) so a second duplicate claim rejects.
                    link_writers.insert(
                        key,
                        LinkWriter::ExternalClaim(SwmmControlAuthority::NativeSwmm),
                    );
                    continue;
                }
                return Err(SurfaceDrainageError::input(format!(
                    "duplicate-parameter-authority: drainage link `{trimmed}` setting already owned by {}; cannot also be written by {}",
                    existing.label(),
                    claim.authority.id()
                )));
            }
            if claim.authority != SwmmControlAuthority::NativeSwmm {
                return Err(SurfaceDrainageError::input(format!(
                    "native-control-authority: external mode `{}` on SWMM link `{trimmed}` is not yet attached to live worker actuation; use `NativeSwmm`",
                    claim.authority.id()
                )));
            }
            link_writers.insert(key, LinkWriter::ExternalClaim(claim.authority));
        }

        let mut inlet_ids = BTreeSet::new();
        let mut cell_node_pairs = BTreeSet::new();
        for inlet in inlets {
            if !inlet_ids.insert(inlet.id.clone()) {
                return Err(SurfaceDrainageError::input(format!(
                    "duplicate-parameter-authority: duplicate inlet id `{}`",
                    inlet.id
                )));
            }
            let pair = (
                inlet.surface_cell,
                inlet.drainage_node_id.to_ascii_uppercase(),
            );
            if !cell_node_pairs.insert(pair) {
                return Err(SurfaceDrainageError::input(format!(
                    "duplicate-parameter-authority: surface cell {} and drainage node `{}` already linked",
                    inlet.surface_cell, inlet.drainage_node_id
                )));
            }
        }

        Ok(Self {
            link_writers,
            native_controlled_links,
        })
    }

    /// Uppercase identifiers of SWMM links controlled by native `[CONTROLS]` actions.
    pub fn native_controlled_links(&self) -> &BTreeSet<String> {
        &self.native_controlled_links
    }

    /// Number of links recorded in the exclusive authority table.
    pub fn link_count(&self) -> usize {
        self.link_writers.len()
    }
}

/// Scans `[CONTROLS]` action clauses (`THEN`/`ELSE`) for SWMM link targets (`PUMP`, `ORIFICE`,
/// `WEIR`, `OUTLET`, `CONDUIT`).
fn parse_native_controlled_links(
    inp_content: &str,
    known_upper: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut controlled = BTreeSet::new();
    let mut in_controls = false;
    let mut in_action = false;
    for raw in inp_content.lines() {
        let line = raw.split(';').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_controls = line.eq_ignore_ascii_case("[CONTROLS]");
            in_action = false;
            continue;
        }
        if !in_controls {
            continue;
        }
        let tokens: Vec<String> = line
            .split_whitespace()
            .map(|t| t.to_ascii_uppercase())
            .collect();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0].as_str() {
            "RULE" | "IF" | "PRIORITY" => {
                in_action = false;
                continue;
            }
            "THEN" | "ELSE" => {
                in_action = true;
            }
            "AND" | "OR" => {}
            _ => {}
        }
        if !in_action {
            continue;
        }
        for window in tokens.windows(2) {
            if matches!(
                window[0].as_str(),
                "PUMP" | "ORIFICE" | "WEIR" | "OUTLET" | "CONDUIT" | "LINK"
            ) && known_upper.contains(&window[1])
            {
                controlled.insert(window[1].clone());
            }
        }
    }
    controlled
}
