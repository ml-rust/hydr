//! Explicit support ownership feeds real surface and standalone runoff consumers.

use hydr::grid::CartesianGrid;
use hydr::model::{Domain, SpatialReference};
use hydr::runoff::{
    CellRainAssignment, Contribution, ContributionKind, PartitionAssertion, SubcatchmentBatch,
    SubcatchmentConfig, SubcatchmentContributionInput, SubcatchmentDefinition,
    SubcatchmentGeometry, SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting,
    SubcatchmentState, SupportAtom, SupportIntervalInput, SupportPlan, SupportPlanInput,
    SurfaceCellSupport, SurfaceObservationInput,
};
use hydr::solver::BoundaryConditions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Domain::new(
        CartesianGrid::new(2, 1, 1.0, 1.0)?,
        vec![0.0; 2],
        vec![0.03; 2],
        vec![true; 2],
        SpatialReference::new("synthetic plane".into(), "synthetic datum".into(), [0.0; 2])?,
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let parameters = SubcatchmentParameters::new(
        SubcatchmentGeometry {
            total_area_m2: 1.0,
            width_m: 0.0,
            slope: 0.0,
            impervious_fraction: 1.0,
            zero_storage_impervious_fraction: 0.0,
            impervious_manning: 0.03,
            pervious_manning: 0.03,
            impervious_depression_m: 0.0,
            pervious_depression_m: 0.0,
        },
        None,
        SubcatchmentRouting::Direct,
    )?;
    let definitions = vec![SubcatchmentDefinition {
        id: "upstream-model".into(),
        parameters,
    }];
    let plan = SupportPlan::new(
        SupportPlanInput {
            partition: PartitionAssertion {
                identity: "synthetic-partition".into(),
                covered_support: "two disjoint square metres".into(),
                provenance: "explicit synthetic fixture".into(),
                complete_disjoint: true,
            },
            surface_input_identity: "surface-observations".into(),
            atoms: vec![
                SupportAtom {
                    id: "local".into(),
                    area_m2: 1.0,
                },
                SupportAtom {
                    id: "upstream".into(),
                    area_m2: 1.0,
                },
            ],
            contributions: vec![Contribution {
                id: "upstream-rainfall".into(),
                input_identity: "upstream-observations".into(),
                provenance: "explicit synthetic upstream support".into(),
                atoms: vec!["upstream".into()],
                includes: vec![],
                kind: ContributionKind::ModelledRunoff {
                    definition_id: "upstream-model".into(),
                    outlet_id: "future-upstream-outlet".into(),
                },
            }],
            replacements: vec![],
            surface_cells: vec![
                SurfaceCellSupport {
                    cell: 0,
                    atoms: vec!["local".into()],
                    rain: CellRainAssignment::Direct,
                },
                SurfaceCellSupport {
                    cell: 1,
                    atoms: vec!["upstream".into()],
                    rain: CellRainAssignment::Suppressed {
                        selected_contribution_id: "upstream-rainfall".into(),
                    },
                },
            ],
            boundaries: vec![],
        },
        &domain,
        &boundaries,
        &definitions,
    )?;
    let packet = plan.admit_interval(SupportIntervalInput {
        start: 0.0,
        end: 60.0,
        surface: SurfaceObservationInput {
            input_identity: "surface-observations".into(),
            start: 0.0,
            end: 60.0,
            rain: vec![1e-5; 2],
            potential_evaporation: vec![0.0; 2],
            withdrawal: vec![0.0; 2],
        },
        subcatchments: vec![SubcatchmentContributionInput {
            contribution_id: "upstream-rainfall".into(),
            input_identity: "upstream-observations".into(),
            start: 0.0,
            end: 60.0,
            input: SubcatchmentInput {
                rain_rate_m_per_s: 1e-5,
                potential_evaporation_m_per_s: 0.0,
                external_runon_volumes_m3: [0.0; 3],
            },
        }],
        boundaries: vec![],
    })?;
    let states = plan
        .active_definitions()
        .iter()
        .map(|d| SubcatchmentState::initialized(&d.parameters))
        .collect::<Result<Vec<_>, _>>()?;
    let batch = SubcatchmentBatch::new(
        plan.active_definitions().to_vec(),
        states,
        0.0,
        SubcatchmentConfig::default(),
    )?;
    let schedule = plan.surface_schedule(std::slice::from_ref(&packet))?;
    println!("injected_surface_rain={:?}", schedule.intervals()[0].rain);
    println!("suppression_receipts={:?}", packet.suppression_receipts());
    if let Some(trial) = plan.trial_subcatchments(&packet, Some(&batch))? {
        println!(
            "model_applied_rain_m3={}",
            trial.diagnostics()?.applied_rain_volume_m3
        );
    }
    Ok(())
}
