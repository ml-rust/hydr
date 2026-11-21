//! Explicit contributing-support declarations and immutable compiled ownership.

use super::SupportError;
use crate::contract::{BoundaryConditions, BoundaryFace};
use crate::model::Domain;
use crate::runoff::SubcatchmentDefinition;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Complete caller-declared physical support and source identities.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportPlanInput {
    /// Explicit geometric partition assertion supplied by the application.
    pub partition: PartitionAssertion,
    /// Identity expected from prepared surface observations.
    pub surface_input_identity: String,
    /// Disjoint application-defined support atoms.
    pub atoms: Vec<SupportAtom>,
    /// Primary sources and additive inclusion declarations.
    pub contributions: Vec<Contribution>,
    /// Static whole-source measured replacements.
    pub replacements: Vec<WholeReplacement>,
    /// Complete active-cell footprints and rainfall assignments.
    pub surface_cells: Vec<SurfaceCellSupport>,
    /// Exactly one identity binding per declared open face.
    pub boundaries: Vec<BoundaryBinding>,
    /// Rain-receiving support of level pools, at most one entry per pool. Empty without pools.
    pub pool_rain: Vec<PoolRainSupport>,
}

/// Rain-receiving support of one level pool and its whole-pool rain ownership.
///
/// The plan checks atom ownership only. It never compares atom area with the pool plan area,
/// which varies with stage.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolRainSupport {
    /// Exact pool identity of the consumer that owns the pool.
    pub pool_id: String,
    /// Nonempty disjoint support atoms that receive the pool's local rain.
    pub atoms: Vec<String>,
    /// Direct local rain, or suppression by the selected source that owns every atom.
    pub rain: CellRainAssignment,
}

/// Application assertion. Hydr checks declared ownership without proving GIS geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionAssertion {
    /// Stable partition identity.
    pub identity: String,
    /// Named complete contributing-support extent.
    pub covered_support: String,
    /// Application evidence for this assertion.
    pub provenance: String,
    /// Explicit assertion that atoms form a complete disjoint partition.
    pub complete_disjoint: bool,
}

/// One indivisible physical support atom with a represented SI area.
#[derive(Clone, Debug, PartialEq)]
pub struct SupportAtom {
    /// Unique support identity.
    pub id: String,
    /// Finite positive represented area in m².
    pub area_m2: f64,
}

/// Named physical source or additive inclusion declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct Contribution {
    /// Unique contribution identity.
    pub id: String,
    /// Exact expected input identity.
    pub input_identity: String,
    /// Source support and acquisition evidence.
    pub provenance: String,
    /// Complete primary support atom identities. Inclusion leaves require empty support.
    pub atoms: Vec<String>,
    /// Explicit additive leaf identities. Modelled sources require an empty list.
    pub includes: Vec<String>,
    /// Physical role and concrete destination identity.
    pub kind: ContributionKind,
}

/// Supported primary sources and declaration-only additive leaves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContributionKind {
    /// Rainfall runoff from exactly one supplied numerical definition.
    ModelledRunoff {
        /// Exact numerical definition ID.
        definition_id: String,
        /// Unique future attachment endpoint, without current hydraulic injection.
        outlet_id: String,
    },
    /// Already composed imported rainfall runoff at a bound face.
    ImportedRunoff {
        /// Exact boundary binding ID.
        boundary_id: String,
    },
    /// Already composed measured flow, eligible for whole-source replacement.
    MeasuredFlow {
        /// Exact boundary binding ID.
        boundary_id: String,
    },
    /// Named additive baseflow declaration, included exactly once by a selected root.
    Baseflow,
    /// Named additive release declaration, included exactly once by a selected root.
    Release,
}

/// Static replacement of complete primary source footprints.
#[derive(Clone, Debug, PartialEq)]
pub struct WholeReplacement {
    /// Selected measured root identity.
    pub selected: String,
    /// Complete modelled or imported source identities replaced by that root.
    pub replaced: Vec<String>,
}

/// Complete active-cell support and indivisible rainfall assignment.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceCellSupport {
    /// Row-major active cell index.
    pub cell: usize,
    /// Complete disjoint support footprint of this cell.
    pub atoms: Vec<String>,
    /// Explicit direct or selected upstream ownership.
    pub rain: CellRainAssignment,
}

/// Static whole-cell rainfall ownership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CellRainAssignment {
    /// Inject observed local rain without another primary owner.
    Direct,
    /// Retain observed rain but inject zero because the selected source owns its support.
    Suppressed {
        /// Selected primary source identity.
        selected_contribution_id: String,
    },
}

/// Exact identity/provenance binding for one physical open face.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryBinding {
    /// Unique boundary identity.
    pub id: String,
    /// Actual declared exterior face.
    pub face: BoundaryFace,
    /// Exact expected physical input identity.
    pub input_identity: String,
    /// Source and boundary provenance.
    pub provenance: String,
    /// Selected hydrological root, or an explicitly nonhydrological boundary.
    pub contribution: Option<String>,
}

#[derive(Debug)]
pub(super) struct CompiledPlan {
    pub input: SupportPlanInput,
    pub domain: Domain,
    pub boundaries: BoundaryConditions,
    pub definitions: Vec<SubcatchmentDefinition>,
    pub active_indices: Vec<usize>,
    pub active_contributions: Vec<usize>,
    pub boundary_indices: BTreeMap<String, usize>,
}

/// Immutable checked ownership and concrete forcing targets.
#[derive(Clone, Debug)]
pub struct SupportPlan {
    pub(super) inner: Arc<CompiledPlan>,
}

impl SupportPlan {
    /// Read the immutable complete application declaration.
    pub fn input(&self) -> &SupportPlanInput {
        &self.inner.input
    }
    /// Read active numerical definition indices in original definition order.
    pub fn active_definition_indices(&self) -> &[usize] {
        &self.inner.active_indices
    }
    /// Read selected definitions in their original numerical order.
    pub fn active_definitions(&self) -> &[SubcatchmentDefinition] {
        &self.inner.definitions
    }
    /// Read the selected active modelled contributions in definition order.
    pub fn active_contributions(&self) -> impl ExactSizeIterator<Item = &Contribution> {
        self.inner
            .active_contributions
            .iter()
            .map(|index| &self.inner.input.contributions[*index])
    }
    /// Read the checked domain the plan's surface targets belong to.
    pub fn domain(&self) -> &Domain {
        &self.inner.domain
    }
    /// Read the checked open-face set of the plan's surface targets.
    pub fn boundaries(&self) -> &BoundaryConditions {
        &self.inner.boundaries
    }
    /// Read whether the pool's local rain belongs to a selected source.
    ///
    /// Returns `false` for direct pool rain and for a pool without declared rain support.
    pub fn pool_rain_suppressed(&self, pool_id: &str) -> bool {
        self.inner.input.pool_rain.iter().any(|support| {
            support.pool_id == pool_id
                && matches!(support.rain, CellRainAssignment::Suppressed { .. })
        })
    }
}

pub(super) fn text(value: &str, field: &str) -> Result<(), SupportError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(SupportError::identity(format!(
            "{field}={value:?}: supply nonempty identity/provenance without control characters"
        )));
    }
    Ok(())
}
