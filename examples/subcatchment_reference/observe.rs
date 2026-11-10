use super::{cases::Case, records};
use crate::Result;
use hydr::runoff::{SubcatchmentBatch, SubcatchmentConfig, SubcatchmentDefinition};
use std::io::Write;

pub fn run(case: &Case, dt: f64, csv: &mut impl Write, soil: &mut impl Write) -> Result {
    let mut batch = SubcatchmentBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![SubcatchmentDefinition {
            id: case.name.into(),
            parameters: case.parameters.clone(),
        }],
        vec![case.initial],
        0.0,
        SubcatchmentConfig::default(),
    )?;
    let mut gross = [0.0; 3];
    let mut outlet = [0.0; 3];
    let mut routed = [0.0; 3];
    let mut rates = [None; 3];
    records::endpoint(
        csv,
        records::Endpoint {
            case,
            dt,
            step: 0,
            batch: &batch,
            gross,
            outlet,
            routed,
            rates,
        },
    )?;
    for step in 1..=(case.horizon / dt) as usize {
        let before = batch.states()?;
        let old_soil = before.first().ok_or("missing interval-start state")?.soil;
        let trial = batch.trial_to(step as f64 * dt, &[case.input])?;
        let diagnostics = trial.diagnostics()?;
        let catchment = diagnostics
            .receipts
            .first()
            .ok_or("missing accepted catchment receipt")?;
        for source in catchment.subareas.iter().flatten() {
            let index = source.subarea.index();
            gross[index] += source.drainage.export_volume_m3;
            outlet[index] += source.partition.outlet_volume_m3;
            routed[index] += source.partition.routed_volume_m3;
            rates[index] = source.drainage.endpoint_rate_m_per_s;
        }
        batch = trial.into_next();
        let accepted = batch.states()?;
        let accepted_soil = accepted.first().ok_or("missing accepted state")?.soil;
        records::soil_call(
            soil,
            records::SoilCall {
                case,
                dt,
                step,
                diagnostics: &diagnostics,
                old: old_soil,
                accepted: accepted_soil,
            },
        )?;
        records::endpoint(
            csv,
            records::Endpoint {
                case,
                dt,
                step,
                batch: &batch,
                gross,
                outlet,
                routed,
                rates,
            },
        )?;
    }
    Ok(())
}
