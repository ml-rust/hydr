use hydr::grid::CartesianGrid;
use hydr::model::{Domain, SpatialReference};
use hydr::runoff::{
    BoundBoundaryInput, BoundaryBinding, CellRainAssignment, Contribution, ContributionKind,
    PartitionAssertion, SubcatchmentBatch, SubcatchmentConfig, SubcatchmentContributionInput,
    SubcatchmentDefinition, SubcatchmentGeometry, SubcatchmentInput, SubcatchmentParameters,
    SubcatchmentRouting, SubcatchmentState, SupportAtom, SupportIntervalInput, SupportPlan,
    SupportPlanInput, SurfaceCellSupport, SurfaceObservationInput, WholeReplacement,
};
use hydr::solver::{Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, DischargeBoundary};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct Fixture {
    pub domain: Domain,
    pub boundaries: BoundaryConditions,
    pub definitions: Vec<SubcatchmentDefinition>,
    pub input: SupportPlanInput,
    pub interval: SupportIntervalInput,
}

impl Fixture {
    pub fn plan(&self) -> Result<SupportPlan> {
        Ok(SupportPlan::new(
            self.input.clone(),
            &self.domain,
            &self.boundaries,
            &self.definitions,
        )?)
    }
}

pub fn domain(count: usize, spacing: f64, active: Vec<bool>) -> Result<Domain> {
    Ok(Domain::new(
        CartesianGrid::new(count, 1, spacing, spacing)?,
        vec![0.0; count],
        vec![0.0; count],
        active,
        SpatialReference::new(
            "synthetic metric plane".into(),
            "synthetic support datum".into(),
            [0.0; 2],
        )?,
    )?)
}

pub fn definition(id: &str, area: f64) -> Result<SubcatchmentDefinition> {
    Ok(SubcatchmentDefinition {
        id: id.into(),
        parameters: SubcatchmentParameters::new(
            SubcatchmentGeometry {
                total_area_m2: area,
                width_m: 0.0,
                slope: 1.0 / 64.0,
                impervious_fraction: 1.0,
                zero_storage_impervious_fraction: 0.0,
                impervious_manning: 1.0 / 32.0,
                pervious_manning: 1.0 / 32.0,
                impervious_depression_m: 0.125 * 0.3048,
                pervious_depression_m: 0.125 * 0.3048,
            },
            None,
            SubcatchmentRouting::Direct,
        )?,
    })
}

pub fn model(id: &str, definition_id: &str, atom: &str) -> Contribution {
    Contribution {
        id: id.into(),
        input_identity: format!("{id}-input"),
        provenance: "Explicit synthetic rainfall support".into(),
        atoms: vec![atom.into()],
        includes: vec![],
        kind: ContributionKind::ModelledRunoff {
            definition_id: definition_id.into(),
            outlet_id: format!("{id}-outlet"),
        },
    }
}

pub fn model_input(id: &str, rain: f64) -> SubcatchmentContributionInput {
    SubcatchmentContributionInput {
        contribution_id: id.into(),
        input_identity: format!("{id}-input"),
        start: 0.0,
        end: 1.0,
        input: SubcatchmentInput {
            rain_rate_m_per_s: rain,
            potential_evaporation_m_per_s: 0.0,
            external_runon_volumes_m3: [0.0; 3],
        },
    }
}

pub fn suppressed(cell: usize, atom: &str, owner: &str) -> SurfaceCellSupport {
    SurfaceCellSupport {
        cell,
        atoms: vec![atom.into()],
        rain: CellRainAssignment::Suppressed {
            selected_contribution_id: owner.into(),
        },
    }
}

pub fn primary() -> Result<Fixture> {
    Ok(Fixture {
        domain: domain(2, 1.0, vec![true; 2])?,
        boundaries: BoundaryConditions::impermeable_walls(),
        definitions: vec![definition("upstream-model", 1.0)?],
        input: SupportPlanInput {
            partition: PartitionAssertion {
                identity: "partition".into(),
                covered_support: "two synthetic square metres".into(),
                provenance: "Explicit complete disjoint synthetic partition".into(),
                complete_disjoint: true,
            },
            surface_input_identity: "surface-input".into(),
            atoms: vec![
                SupportAtom {
                    id: "upstream".into(),
                    area_m2: 1.0,
                },
                SupportAtom {
                    id: "local".into(),
                    area_m2: 1.0,
                },
            ],
            contributions: vec![model("upstream-source", "upstream-model", "upstream")],
            replacements: vec![],
            surface_cells: vec![
                suppressed(0, "upstream", "upstream-source"),
                SurfaceCellSupport {
                    cell: 1,
                    atoms: vec!["local".into()],
                    rain: CellRainAssignment::Direct,
                },
            ],
            boundaries: vec![],
        },
        interval: SupportIntervalInput {
            start: 0.0,
            end: 1.0,
            surface: SurfaceObservationInput {
                input_identity: "surface-input".into(),
                start: 0.0,
                end: 1.0,
                rain: vec![0.001; 2],
                potential_evaporation: vec![0.0; 2],
                withdrawal: vec![0.0; 2],
            },
            subcatchments: vec![model_input("upstream-source", 0.002)],
            boundaries: vec![],
        },
    })
}

pub fn ordered() -> Result<Fixture> {
    let mut fixture = primary()?;
    fixture.definitions = vec![definition("model-b", 1.0)?, definition("model-a", 1.0)?];
    fixture.input.atoms[0].id = "a".into();
    fixture.input.atoms[1].id = "b".into();
    fixture.input.contributions = vec![
        model("source-a", "model-a", "a"),
        model("source-b", "model-b", "b"),
    ];
    fixture.input.surface_cells = vec![
        suppressed(0, "a", "source-a"),
        suppressed(1, "b", "source-b"),
    ];
    fixture.interval.subcatchments = vec![
        model_input("source-a", 0.002),
        model_input("source-b", 0.003),
    ];
    Ok(fixture)
}

pub fn face() -> BoundaryFace {
    BoundaryFace {
        axis: Axis::X,
        column: 0,
        row: 0,
    }
}

pub fn measured() -> Result<Fixture> {
    let mut fixture = ordered()?;
    fixture.boundaries = BoundaryConditions::with_open_faces(&fixture.domain, vec![face()])?;
    fixture.input.contributions.push(Contribution {
        id: "measurement".into(),
        input_identity: "gauge-input".into(),
        provenance: "Already composed synthetic measured flow".into(),
        atoms: vec!["a".into(), "b".into()],
        includes: vec![],
        kind: ContributionKind::MeasuredFlow {
            boundary_id: "gauge".into(),
        },
    });
    fixture.input.replacements = vec![WholeReplacement {
        selected: "measurement".into(),
        replaced: vec!["source-a".into(), "source-b".into()],
    }];
    fixture.input.surface_cells = vec![
        suppressed(0, "a", "measurement"),
        suppressed(1, "b", "measurement"),
    ];
    fixture.input.boundaries = vec![BoundaryBinding {
        id: "gauge".into(),
        face: face(),
        input_identity: "gauge-input".into(),
        provenance: "Explicit synthetic hydrograph boundary".into(),
        contribution: Some("measurement".into()),
    }];
    fixture.interval.subcatchments.clear();
    fixture.interval.boundaries = vec![BoundBoundaryInput {
        boundary_id: "gauge".into(),
        input_identity: "gauge-input".into(),
        start: 0.0,
        end: 1.0,
        value: BoundaryForcing::Discharge(DischargeBoundary {
            face: face(),
            outward_rate: -0.003,
            inflow_tangential_velocity: Some(0.0),
        }),
    }];
    Ok(fixture)
}

pub fn standalone_measured() -> Result<Fixture> {
    let mut fixture = measured()?;
    fixture.definitions.clear();
    fixture
        .input
        .contributions
        .retain(|contribution| contribution.id == "measurement");
    fixture.input.replacements.clear();
    Ok(fixture)
}

pub fn nonhydrological() -> Result<Fixture> {
    let mut fixture = standalone_measured()?;
    fixture.input.contributions.clear();
    for cell in &mut fixture.input.surface_cells {
        cell.rain = CellRainAssignment::Direct;
    }
    fixture.input.boundaries[0].contribution = None;
    fixture.input.boundaries[0].input_identity = "boundary-input".into();
    fixture.interval.boundaries[0].input_identity = "boundary-input".into();
    Ok(fixture)
}

pub fn batch(plan: &SupportPlan, time: f64) -> Result<SubcatchmentBatch> {
    let definitions = plan.active_definitions().to_vec();
    let states = definitions
        .iter()
        .map(|definition| SubcatchmentState::initialized(&definition.parameters))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(SubcatchmentBatch::new(
        definitions,
        states,
        time,
        SubcatchmentConfig::default(),
    )?)
}
