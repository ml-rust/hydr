//! Small two-cell plan inputs shared by the support unit tests.

use super::{
    CellRainAssignment, Contribution, ContributionKind, PartitionAssertion, PoolRainSupport,
    SupportAtom, SupportPlan, SupportPlanInput, SurfaceCellSupport,
};
use crate::contract::BoundaryConditions;
use crate::model::{Domain, SpatialReference};
use crate::runoff::{
    SubcatchmentBatch, SubcatchmentConfig, SubcatchmentContributionInput, SubcatchmentDefinition,
    SubcatchmentGeometry, SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting,
    SubcatchmentState,
};
use hydr_core::grid::CartesianGrid;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub(super) type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The ownership category of a rejected `result`, or `None` when it passes.
pub(super) fn kind<T>(
    result: std::result::Result<T, super::SupportError>,
) -> Option<super::SupportErrorKind> {
    result.err().and_then(|error| error.kind())
}

/// Plan parts: a two-cell domain, one modelled source over cell 0, and one pool.
pub(super) struct Fixture {
    pub domain: Domain,
    pub definitions: Vec<SubcatchmentDefinition>,
    pub input: SupportPlanInput,
}

impl Fixture {
    pub fn plan(&self) -> std::result::Result<SupportPlan, super::SupportError> {
        SupportPlan::new(
            self.input.clone(),
            &self.domain,
            &BoundaryConditions::impermeable_walls(),
            &self.definitions,
        )
    }
}

pub(super) fn definition(id: &str, area: f64) -> Result<SubcatchmentDefinition> {
    Ok(SubcatchmentDefinition {
        id: id.into(),
        parameters: SubcatchmentParameters::new(
            SubcatchmentGeometry {
                total_area_m2: area,
                width_m: 1.0,
                slope: 1.0 / 64.0,
                impervious_fraction: 1.0,
                zero_storage_impervious_fraction: 1.0,
                impervious_manning: 1.0 / 32.0,
                pervious_manning: 1.0 / 32.0,
                impervious_depression_m: 0.0,
                pervious_depression_m: 0.0,
            },
            None,
            SubcatchmentRouting::Direct,
        )?,
    })
}

fn atom(id: &str, area_m2: f64) -> SupportAtom {
    SupportAtom {
        id: id.into(),
        area_m2,
    }
}

/// Cell 0 belongs to `upstream-source`. Cell 1 and pool `basin` take direct rain.
pub(super) fn fixture() -> Result<Fixture> {
    let domain = Domain::new(
        CartesianGrid::new(2, 1, 1.0, 1.0)?,
        vec![0.0; 2],
        vec![0.0; 2],
        vec![true; 2],
        SpatialReference::new(
            "synthetic metric plane".into(),
            "synthetic support datum".into(),
            [0.0; 2],
        )?,
    )?;
    Ok(Fixture {
        domain,
        definitions: vec![definition("upstream-model", 1.0)?],
        input: SupportPlanInput {
            partition: PartitionAssertion {
                identity: "partition".into(),
                covered_support: "two cells and one pool".into(),
                provenance: "Explicit complete disjoint synthetic partition".into(),
                complete_disjoint: true,
            },
            surface_input_identity: "surface-input".into(),
            atoms: vec![atom("upstream", 1.0), atom("local", 1.0), atom("pool", 0.5)],
            contributions: vec![Contribution {
                id: "upstream-source".into(),
                input_identity: "upstream-input".into(),
                provenance: "Explicit synthetic rainfall support".into(),
                atoms: vec!["upstream".into()],
                includes: vec![],
                kind: ContributionKind::ModelledRunoff {
                    definition_id: "upstream-model".into(),
                    outlet_id: "upstream-outlet".into(),
                },
            }],
            replacements: vec![],
            surface_cells: vec![
                SurfaceCellSupport {
                    cell: 0,
                    atoms: vec!["upstream".into()],
                    rain: CellRainAssignment::Suppressed {
                        selected_contribution_id: "upstream-source".into(),
                    },
                },
                SurfaceCellSupport {
                    cell: 1,
                    atoms: vec!["local".into()],
                    rain: CellRainAssignment::Direct,
                },
            ],
            boundaries: vec![],
            pool_rain: vec![pool("basin", &["pool"], CellRainAssignment::Direct)],
        },
    })
}

pub(super) fn pool(id: &str, atoms: &[&str], rain: CellRainAssignment) -> PoolRainSupport {
    PoolRainSupport {
        pool_id: id.into(),
        atoms: atoms.iter().map(|atom| (*atom).into()).collect(),
        rain,
    }
}

pub(super) fn suppressed_by(owner: &str) -> CellRainAssignment {
    CellRainAssignment::Suppressed {
        selected_contribution_id: owner.into(),
    }
}

/// One rate record of `upstream-source` over `[start, end]`.
pub(super) fn record(start: f64, end: f64, rain: f64) -> SubcatchmentContributionInput {
    SubcatchmentContributionInput {
        contribution_id: "upstream-source".into(),
        input_identity: "upstream-input".into(),
        start,
        end,
        input: SubcatchmentInput {
            rain_rate_m_per_s: rain,
            potential_evaporation_m_per_s: 0.0,
            external_runon_volumes_m3: [0.0; 3],
        },
    }
}

/// The active batch of `plan` at `time` with a wet impervious start depth.
pub(super) fn batch(plan: &SupportPlan, time: f64) -> Result<SubcatchmentBatch<CpuRuntime>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let definitions = plan.active_definitions().to_vec();
    let states = definitions
        .iter()
        .map(|_| SubcatchmentState {
            depths_m: [0.01, 0.0, 0.0],
            soil: None,
        })
        .collect();
    Ok(SubcatchmentBatch::new(
        client,
        definitions,
        states,
        time,
        SubcatchmentConfig::default(),
    )?)
}
