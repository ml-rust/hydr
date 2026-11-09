use super::fixture::{Result, geometry, single};
use hydr::runoff::{
    Subarea, SubcatchmentConfig, SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting,
    SubcatchmentState,
};
use hydr::soil::ModifiedGreenAmptParameters;

fn volume(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-9 + 1e-12 * expected.abs(),
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}

#[test]
fn instantaneous_routed_equilibria_match_independent_physical_volumes_at_every_resolution() -> Result
{
    for impervious_donors in [true, false] {
        for dt in [64.0, 32.0, 16.0, 8.0] {
            let mut g = geometry(0.5, 0.5);
            g.impervious_manning = 0.0;
            g.pervious_manning = 0.0;
            let rain = 0.3048 / 1024.0;
            let conductivity = if impervious_donors {
                1.0 / 1024.0
            } else {
                1.0 / 2048.0
            };
            let soil = ModifiedGreenAmptParameters::from_native(0.125, conductivity, 0.0)?;
            let routing = if impervious_donors {
                SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 }
            } else {
                SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 }
            };
            let p = SubcatchmentParameters::new(g, Some(soil), routing)?;
            let areas = Subarea::ALL.map(|subarea| p.area_m2(subarea));
            let mut state = SubcatchmentState::initialized(&p)?;
            state.depths_m = [0.0, 0.125 * 0.3048, 0.125 * 0.3048];
            let depths = state.depths_m;
            let mut batch = single(p, state, SubcatchmentConfig::default())?;
            for step in 1..=(256.0 / dt) as usize {
                let end = step as f64 * dt;
                let trial = batch.trial_to(
                    end,
                    &[SubcatchmentInput {
                        rain_rate_m_per_s: rain,
                        potential_evaporation_m_per_s: 0.0,
                        external_runon_volumes_m3: [0.0; 3],
                    }],
                )?;
                let diagnostics = trial.diagnostics()?;
                assert!(
                    diagnostics.water_residual_m3.abs()
                        <= 1e-9 + 1e-12 * diagnostics.water_scale_m3
                );
                batch = trial.into_next();
                let applied = g.total_area_m2 * rain * end;
                let (infiltration, outlet, internal) = if impervious_donors {
                    (0.5 * applied, 0.5 * applied, 0.25 * applied)
                } else {
                    (0.25 * applied, 0.75 * applied, 0.125 * applied)
                };
                volume(batch.ledger().infiltration_volume_m3, infiltration);
                volume(batch.ledger().outlet_volume_m3, outlet);
                volume(batch.ledger().internal_transfer_volume_m3, internal);
                volume(batch.ledger().evaporation_volume_m3, 0.0);
                let states = batch.states()?;
                for (index, area) in areas.into_iter().enumerate() {
                    volume(area * states[0].depths_m[index], area * depths[index]);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn closed_constant_sources_match_depth_and_loss_volumes_at_every_resolution() -> Result {
    for dt in [64.0, 32.0, 16.0, 8.0] {
        let mut g = geometry(0.0, 0.0);
        g.width_m = 0.0;
        let soil = ModifiedGreenAmptParameters::from_native(0.125, 1.0 / 1024.0, 0.0)?;
        let p = SubcatchmentParameters::new(g, Some(soil), SubcatchmentRouting::Direct)?;
        let mut state = SubcatchmentState::initialized(&p)?;
        state.depths_m[2] = 0.125 * 0.3048;
        let initial = state.depths_m[2];
        let rain = 0.3048 / 512.0;
        let conductivity = 0.3048 / 1024.0;
        let evaporation = 0.3048 / 8192.0;
        let mut batch = single(p, state, SubcatchmentConfig::default())?;
        for step in 1..=(256.0 / dt) as usize {
            let end = step as f64 * dt;
            batch = batch
                .trial_to(
                    end,
                    &[SubcatchmentInput {
                        rain_rate_m_per_s: rain,
                        potential_evaporation_m_per_s: evaporation,
                        external_runon_volumes_m3: [0.0; 3],
                    }],
                )?
                .into_next();
            let exact_depth = initial + (rain - conductivity - evaporation) * end;
            volume(
                g.total_area_m2 * batch.states()?[0].depths_m[2],
                g.total_area_m2 * exact_depth,
            );
            volume(
                batch.ledger().infiltration_volume_m3,
                g.total_area_m2 * conductivity * end,
            );
            volume(
                batch.ledger().evaporation_volume_m3,
                g.total_area_m2 * evaporation * end,
            );
            volume(batch.ledger().outlet_volume_m3, 0.0);
            volume(batch.ledger().internal_transfer_volume_m3, 0.0);
        }
    }
    Ok(())
}
