use super::exports::assert_paired;
use super::fixture::{Batch, Result, forcing, impervious, setup, shortened_case};
use super::laws::{pump, running};
use super::pool_fixture::{assemble, near};
use hydr::coupling::{
    CoupledLevelPoolBatch, CoupledLevelPoolForcingInterval, CoupledLevelPoolInitialState,
    CoupledPoolInflow, CoupledPoolInflowValue, RunoffPoolForcingInterval, RunoffPoolInitialState,
    RunoffPoolSetup,
};
use hydr::runoff::{SubcatchmentBatch, SubcatchmentInput};
use hydr::storage::{LevelPoolSourcePolicy, LevelPoolSources};

#[test]
fn common_recomputation_matches_isolated_children_including_soil_and_routing() -> Result {
    let (setup, initial, forcing) = shortened_case()?;
    let batch = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?;
    let before = batch.snapshot_data()?;
    let next = batch.trial_with_forcing_to(1.0, &forcing)?.into_next();
    assert_eq!(batch.snapshot_data()?, before);
    assert!(next.time() > 0.0 && next.time() < 1.0);
    let step = next
        .last_step()?
        .ok_or("Missing recomputed common receipt")?;
    assert!(step.retries > 0);
    assert_eq!(step.requested_end_s, 1.0);
    compare_isolated(&next, &setup, &initial, &forcing)?;
    assert_paired(&next)
}

pub fn compare_isolated(
    batch: &Batch,
    setup: &RunoffPoolSetup,
    initial: &RunoffPoolInitialState,
    forcing: &RunoffPoolForcingInterval,
) -> Result {
    let native = SubcatchmentBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup.subcatchments.clone(),
        initial.subcatchments.clone(),
        initial.time_s,
        setup.runoff_config.clone(),
    )?;
    let inputs = forcing
        .subcatchments
        .iter()
        .map(|input| SubcatchmentInput {
            rain_rate_m_per_s: input.rain_rate_m_per_s,
            potential_evaporation_m_per_s: input.potential_evaporation_m_per_s,
            external_runon_volumes_m3: [0.0; 3],
        })
        .collect::<Vec<_>>();
    let native = native.trial_to(batch.time(), &inputs)?.into_next();
    assert_eq!(batch.runoff().snapshot_data()?, native.snapshot_data()?);
    let receipts = native
        .last_step()?
        .ok_or("Missing isolated native receipts")?;
    let mut inflows = Vec::new();
    for boundary in &setup.pools.external_inflows {
        let route = setup
            .outlets
            .iter()
            .find(|route| route.boundary_id == boundary.boundary_id)
            .ok_or("Missing isolated outlet mapping")?;
        let catchment = setup
            .subcatchments
            .iter()
            .position(|entry| entry.id == route.subcatchment_id)
            .ok_or("Missing isolated catchment")?;
        let volume_m3 = receipts.receipts[catchment].subareas[route.subarea.index()]
            .as_ref()
            .ok_or("Missing isolated outlet lane")?
            .partition
            .outlet_volume_m3;
        inflows.push(CoupledPoolInflow {
            boundary_id: boundary.boundary_id.clone(),
            input_id: boundary.input_id.clone(),
            value: CoupledPoolInflowValue::IntervalVolume { volume_m3 },
        });
    }
    let pools = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup.pools.clone(),
        CoupledLevelPoolInitialState {
            time_s: initial.time_s,
            states: initial.pools.clone(),
        },
    )?;
    let pools = pools
        .trial_with_forcing_to(
            batch.time(),
            &CoupledLevelPoolForcingInterval {
                start_s: initial.time_s,
                end_s: batch.time(),
                sources: forcing.pool_sources.clone(),
                inflows,
            },
        )?
        .into_next();
    assert_eq!(batch.pools().snapshot_data()?, pools.snapshot_data()?);
    Ok(())
}

#[test]
fn pool_sources_and_native_exports_preserve_original_pump_donor_order() -> Result {
    let (mut pools, initial) = assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[4.0, 2.0],
        vec![pump("pump", "a", "b", &[(-16.0, 1.0), (16.0, 1.0)])?],
        vec![running("pump")],
    )?;
    pools.source_policies = Some(vec![
        LevelPoolSourcePolicy {
            dry_basal_footprint_assertion:
                "Synthetic unit-area pool receives local rain over its declared basal area".into(),
            bottom_loss_provenance: "Supplied synthetic bottom demand without inferred soil state"
                .into(),
        };
        2
    ]);
    let (mut setup, initial) = setup(
        vec![impervious(2.0, 1.0)?, impervious(1.0, 1.0)?],
        pools,
        initial.states,
    )?;
    setup.pools.external_inflows[1].pool_id = "b".into();
    let mut forcing = forcing(2, 0.25, 0.0);
    forcing.pool_sources = Some(vec![
        LevelPoolSources {
            rain_rate_m_per_s: 1.0,
            bottom_loss_rate_m3_per_s: 0.25,
            potential_evaporation_rate_m_per_s: 0.5,
        },
        LevelPoolSources {
            rain_rate_m_per_s: 0.0,
            bottom_loss_rate_m3_per_s: 0.0,
            potential_evaporation_rate_m_per_s: 0.0,
        },
    ]);
    let next = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(1.0, &forcing)?
        .into_next();
    near(next.pools().pools().states()?[0].volume_m3, 3.75);
    near(next.pools().pools().states()?[1].volume_m3, 3.25);
    let balance = next
        .last_step()?
        .ok_or("Missing source-pump balance")?
        .balance;
    near(balance.rain_volume_m3, 1.75);
    near(balance.bottom_loss_volume_m3, 0.25);
    near(balance.evaporation_volume_m3, 0.5);
    near(balance.internal_outlet_volume_m3, 0.75);
    near(balance.storage_after_m3 - balance.storage_before_m3, 1.0);
    near(balance.water_residual_m3, 0.0);
    assert_paired(&next)
}

#[test]
fn global_balance_uses_original_losses_and_soil_adjustment_without_internal_export() -> Result {
    let (setup, initial, mut forcing) = shortened_case()?;
    forcing.subcatchments[0].potential_evaporation_m_per_s = 1e-6;
    let next = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?
        .trial_with_forcing_to(1.0, &forcing)?
        .into_next();
    compare_isolated(&next, &setup, &initial, &forcing)?;
    let balance = next
        .last_step()?
        .ok_or("Missing full physical balance")?
        .balance;
    let runoff = next.runoff().ledger();
    assert!(runoff.infiltration_volume_m3 > 0.0);
    assert!(runoff.evaporation_volume_m3 > 0.0);
    near(
        balance.infiltration_volume_m3,
        runoff.infiltration_volume_m3,
    );
    near(balance.evaporation_volume_m3, runoff.evaporation_volume_m3);
    near(
        balance.soil_water_adjustment_volume_m3,
        runoff.soil_water_adjustment_volume_m3,
    );
    near(balance.internal_outlet_volume_m3, runoff.outlet_volume_m3);
    let residual = (balance.storage_after_m3 - balance.storage_before_m3) - balance.rain_volume_m3
        + balance.infiltration_volume_m3
        + balance.evaporation_volume_m3
        + balance.bottom_loss_volume_m3
        - balance.soil_water_adjustment_volume_m3;
    near(balance.water_residual_m3, residual);
    assert!(balance.roundoff_bound_m3.is_finite() && balance.roundoff_bound_m3 >= 0.0);
    assert!(balance.water_residual_m3.abs() <= 1e-9 + 1e-12 * balance.physical_scale_m3);
    Ok(())
}

#[test]
fn native_incoming_water_cannot_finance_same_interval_pool_withdrawals() -> Result {
    let (pools, initial) = assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[1.0, 1.0],
        vec![pump("pump", "a", "b", &[(-16.0, 2.0), (16.0, 2.0)])?],
        vec![running("pump")],
    )?;
    let (setup, initial) = setup(vec![impervious(4.0, 0.25)?], pools, initial.states)?;
    let next = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(1.0, &forcing(1, 0.25, 0.0))?
        .into_next();
    let step = next
        .pools()
        .last_step()?
        .ok_or("Missing native-funded pool interval")?;
    near(step.transfers[0].proposed_volume_m3, 2.0);
    assert!(step.transfers[0].signed_volume_m3 <= 1.0);
    near(step.transfers[0].signed_volume_m3, 1.0);
    near(next.pools().pools().states()?[0].volume_m3, 1.0);
    near(next.pools().pools().states()?[1].volume_m3, 2.0);
    assert_paired(&next)
}

#[test]
fn native_incoming_water_arrives_after_capped_pool_losses() -> Result {
    let (mut pools, initial) = super::fixture::inactive_pools([0.0, 1.0])?;
    pools.source_policies = Some(vec![
        LevelPoolSourcePolicy {
            dry_basal_footprint_assertion: "Declared unit-area synthetic dry rainfall footprint"
                .into(),
            bottom_loss_provenance:
                "Explicit supplied potential demand without inferred soil state".into(),
        };
        2
    ]);
    let (setup, initial) = setup(vec![impervious(4.0, 0.25)?], pools, initial)?;
    let mut forcing = forcing(1, 0.25, 0.0);
    forcing.pool_sources = Some(vec![
        LevelPoolSources {
            rain_rate_m_per_s: 0.0,
            bottom_loss_rate_m3_per_s: 1.0,
            potential_evaporation_rate_m_per_s: 1.0,
        },
        LevelPoolSources {
            rain_rate_m_per_s: 0.0,
            bottom_loss_rate_m3_per_s: 0.0,
            potential_evaporation_rate_m_per_s: 0.0,
        },
    ]);
    let next = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(1.0, &forcing)?
        .into_next();
    near(next.pools().pools().states()?[0].volume_m3, 1.0);
    let balance = next
        .last_step()?
        .ok_or("Missing capped pool losses")?
        .balance;
    assert_eq!(balance.bottom_loss_volume_m3, 0.0);
    assert_eq!(balance.evaporation_volume_m3, 0.0);
    assert_paired(&next)
}
