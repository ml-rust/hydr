use super::fixture::{
    Batch, Result, exact_case, forcing, impervious, inactive_pools, pervious, setup,
};
use super::pool_fixture::near;
use hydr::coupling::CoupledPoolInflowValue;
use hydr::runoff::{Subarea, SubcatchmentRouting};

pub fn assert_paired(batch: &Batch) -> Result {
    let common = batch.last_step()?.ok_or("Missing common interval")?;
    let runoff = batch
        .runoff()
        .last_step()?
        .ok_or("Missing runoff interval")?;
    let pools = batch.pools().last_step()?.ok_or("Missing pool interval")?;
    for route in &common.outlets {
        let index = batch
            .runoff()
            .definitions()
            .iter()
            .position(|definition| definition.id == route.subcatchment_id)
            .ok_or("Unknown paired catchment")?;
        let native = runoff.receipts[index].subareas[route.subarea.index()]
            .as_ref()
            .ok_or("Outlet bound to absent subarea")?
            .partition
            .outlet_volume_m3;
        let input = pools
            .external_inflows
            .iter()
            .find(|input| input.boundary_id == route.boundary_id)
            .ok_or("Missing paired pool input")?;
        assert_eq!(native.to_bits(), input.volume_m3.to_bits());
        match input.input {
            CoupledPoolInflowValue::IntervalVolume { volume_m3 } => {
                assert_eq!(native.to_bits(), volume_m3.to_bits())
            }
            other => return Err(format!("Native outlet changed input kind: {other:?}").into()),
        }
    }
    assert_eq!(batch.time(), batch.runoff().time());
    assert_eq!(batch.time(), batch.pools().time());
    assert_eq!(
        batch.runoff().accepted_steps(),
        batch.pools().pools().accepted_steps()
    );
    Ok(())
}

#[test]
fn impervious_rain_exports_one_original_cubic_metre_into_a_dry_pool() -> Result {
    let (setup, initial) = exact_case()?;
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    let trial = batch.trial_with_forcing_to(2.0, &forcing(1, 0.25, 0.0))?;
    assert_eq!(batch.snapshot_data()?, before);
    let next = trial.into_next();
    assert_eq!(next.time(), 2.0);
    assert_eq!(
        next.pools().pools().states()?[0].volume_m3.to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(next.pools().pools().outputs()?[0].stage_m, Some(1.0));
    let step = next.last_step()?.ok_or("Missing exact common receipt")?;
    assert_eq!(
        step.balance.internal_outlet_volume_m3.to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(step.balance.rain_volume_m3.to_bits(), 1.0_f64.to_bits());
    assert_eq!(
        step.balance.storage_after_m3 - step.balance.storage_before_m3,
        1.0
    );
    assert_eq!(step.balance.water_residual_m3, 0.0);
    assert_paired(&next)
}

#[test]
fn unequal_native_lanes_keep_their_destinations_when_boundary_order_reverses() -> Result {
    let (mut pools, states) = inactive_pools([0.0, 0.0])?;
    pools.config.maximum_step_s = 2.0;
    let (mut setup, initial) = setup(vec![impervious(4.0, 0.25)?], pools, states)?;
    setup.pools.external_inflows[1].pool_id = "b".into();
    let mut reversed = setup.clone();
    reversed.pools.external_inflows.reverse();
    let a = Batch::new(crate::runtime::client(), setup, initial.clone())?
        .trial_with_forcing_to(2.0, &forcing(1, 0.25, 0.0))?
        .into_next();
    let b = Batch::new(crate::runtime::client(), reversed, initial)?
        .trial_with_forcing_to(2.0, &forcing(1, 0.25, 0.0))?
        .into_next();
    for batch in [&a, &b] {
        near(batch.pools().pools().states()?[0].volume_m3, 0.5);
        near(batch.pools().pools().states()?[1].volume_m3, 1.5);
        assert_paired(batch)?;
    }
    assert_eq!(
        a.pools().pools().snapshot_data()?,
        b.pools().pools().snapshot_data()?
    );
    Ok(())
}

#[test]
fn potential_outlets_follow_area_and_routing_fraction_even_when_currently_dry() -> Result {
    for routing in [
        SubcatchmentRouting::Direct,
        SubcatchmentRouting::ImperviousToPervious { fraction: 0.0 },
        SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
        SubcatchmentRouting::ImperviousToPervious { fraction: 1.0 },
        SubcatchmentRouting::PerviousToImpervious { fraction: 0.0 },
        SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 },
        SubcatchmentRouting::PerviousToImpervious { fraction: 1.0 },
    ] {
        let expected = match routing {
            SubcatchmentRouting::ImperviousToPervious { fraction: 1.0 } => {
                vec![Subarea::Pervious]
            }
            SubcatchmentRouting::PerviousToImpervious { fraction: 1.0 } => {
                vec![Subarea::ImperviousZeroStorage, Subarea::ImperviousStorage]
            }
            _ => Subarea::ALL.to_vec(),
        };
        let (pools, states) = inactive_pools([1.0, 1.0])?;
        let (setup, initial) = setup(vec![pervious(routing)?], pools, states)?;
        assert_eq!(
            setup
                .outlets
                .iter()
                .map(|outlet| outlet.subarea)
                .collect::<Vec<_>>(),
            expected
        );
        let next = Batch::new(crate::runtime::client(), setup, initial)?
            .trial_with_forcing_to(0.125, &forcing(1, 0.0, 0.0))?
            .into_next();
        assert_paired(&next)?;
        let receipt = next.runoff().last_step()?.ok_or("Missing routed receipt")?;
        for subarea in Subarea::ALL {
            if !expected.contains(&subarea) {
                assert_eq!(
                    receipt.receipts[0].subareas[subarea.index()]
                        .as_ref()
                        .ok_or("Missing represented routed donor")?
                        .partition
                        .outlet_volume_m3,
                    0.0
                );
            }
        }
    }
    Ok(())
}
