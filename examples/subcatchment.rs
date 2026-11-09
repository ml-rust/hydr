//! Standalone explicit three-subarea rainfall, soil, and same-interval routing.

use hydr::runoff::{
    SubcatchmentBatch, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentGeometry,
    SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
};
use hydr::soil::ModifiedGreenAmptParameters;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parameters = SubcatchmentParameters::new(
        SubcatchmentGeometry {
            total_area_m2: 100.0,
            width_m: 10.0,
            slope: 0.01,
            impervious_fraction: 0.5,
            zero_storage_impervious_fraction: 0.5,
            impervious_manning: 0.03,
            pervious_manning: 0.04,
            impervious_depression_m: 0.002,
            pervious_depression_m: 0.004,
        },
        Some(ModifiedGreenAmptParameters::from_si(0.04, 1e-5, 0.3)?),
        SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
    )?;
    let state = SubcatchmentState::initialized(&parameters)?;
    let batch = SubcatchmentBatch::new(
        vec![SubcatchmentDefinition {
            id: "roof-and-soil".into(),
            parameters,
        }],
        vec![state],
        0.0,
        SubcatchmentConfig::default(),
    )?;
    let trial = batch.trial_to(
        60.0,
        &[SubcatchmentInput {
            rain_rate_m_per_s: 1e-4,
            potential_evaporation_m_per_s: 1e-7,
            external_runon_volumes_m3: [0.0; 3],
        }],
    )?;
    let diagnostics = trial.diagnostics()?;
    println!(
        "rain_m3={}, infiltration_m3={}, evaporation_m3={}, outlet_m3={}, residual_m3={}",
        diagnostics.applied_rain_volume_m3,
        diagnostics.infiltration_volume_m3,
        diagnostics.evaporation_volume_m3,
        diagnostics.outlet_volume_m3,
        diagnostics.water_residual_m3,
    );
    let accepted = trial.into_next();
    println!("accepted_time_s={}", accepted.time());
    Ok(())
}
