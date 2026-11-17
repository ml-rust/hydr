use super::exports::assert_paired;
use super::fixture::{Batch, Result, exact_case, forcing, shortened_case};
use hydr::coupling::{
    CoupledPoolInflowValue, RUNOFF_POOL_CONTRACT_VERSION, RUNOFF_POOL_SNAPSHOT_SCHEMA,
    RunoffPoolError, RunoffPoolSnapshotData,
};
use hydr::runoff::{SUBCATCHMENT_SNAPSHOT_SCHEMA, SubcatchmentBatch, SubcatchmentInput};
use hydr::storage::LEVEL_POOL_CONTRACT_VERSION;

fn advanced() -> Result<Batch> {
    let (setup, initial) = exact_case()?;
    Ok(Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(0.5, &forcing(1, 0.25, 0.0))?
        .into_next())
}

/// Import `changed`, then check the input error names every `needle` and the destination is unchanged.
fn assert_rejected(changed: &RunoffPoolSnapshotData, needles: &[&str]) -> Result {
    let mut destination = advanced()?;
    let before = destination.snapshot_data()?;
    let Err(error) = destination.restore_data(changed) else {
        return Err("foreign identity was accepted".into());
    };
    assert!(
        matches!(error, RunoffPoolError::Input(_)),
        "unexpected kind: {error}"
    );
    let message = error.to_string();
    for needle in needles {
        assert!(message.contains(needle), "missing {needle:?}: {message}");
    }
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

#[test]
fn runoff_pool_package_carries_identity() -> Result {
    assert_eq!(RUNOFF_POOL_SNAPSHOT_SCHEMA, "hydr-runoff-pool-checkpoint-1");
    assert_eq!(RUNOFF_POOL_CONTRACT_VERSION, "hydr-runoff-pool-numr-cpu-1");
    let package = advanced()?.snapshot_data()?;
    assert_eq!(package.schema, RUNOFF_POOL_SNAPSHOT_SCHEMA);
    assert_eq!(package.contract, RUNOFF_POOL_CONTRACT_VERSION);
    Ok(())
}

#[test]
fn runoff_pool_import_rejects_foreign_schema() -> Result {
    let package = advanced()?.snapshot_data()?;
    for found in ["hydr-runoff-pool-checkpoint-0", ""] {
        let mut changed = package.clone();
        changed.schema = found.into();
        assert_rejected(
            &changed,
            &[
                "runoff pool checkpoint schema",
                &format!("found \"{found}\""),
                RUNOFF_POOL_SNAPSHOT_SCHEMA,
                "capture a new package",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn runoff_pool_import_rejects_foreign_contract() -> Result {
    let mut changed = advanced()?.snapshot_data()?;
    changed.contract = "hydr-runoff-pool-numr-gpu-1".into();
    assert_rejected(
        &changed,
        &[
            "runoff pool checkpoint contract",
            "found \"hydr-runoff-pool-numr-gpu-1\"",
            RUNOFF_POOL_CONTRACT_VERSION,
        ],
    )
}

#[test]
fn runoff_pool_import_rejects_nested_runoff_schema() -> Result {
    let mut changed = advanced()?.snapshot_data()?;
    changed.runoff.schema = "hydr-subcatchment-checkpoint-0".into();
    assert_rejected(
        &changed,
        &[
            "runoff pool checkpoint runoff",
            "subcatchment checkpoint schema",
            "found \"hydr-subcatchment-checkpoint-0\"",
            SUBCATCHMENT_SNAPSHOT_SCHEMA,
        ],
    )
}

#[test]
fn runoff_pool_import_rejects_nested_level_pool_contract() -> Result {
    let mut changed = advanced()?.snapshot_data()?;
    changed.pools.pools.contract = "hydr-level-pool-numr-gpu-1".into();
    assert_rejected(
        &changed,
        &[
            "runoff pool checkpoint pools: coupled level pool checkpoint pools",
            "level pool checkpoint contract",
            "found \"hydr-level-pool-numr-gpu-1\"",
            LEVEL_POOL_CONTRACT_VERSION,
        ],
    )
}

#[test]
fn pervious_soil_routing_and_both_children_repeat_after_trusted_and_cold_restore() -> Result {
    let (setup, initial, forcing) = shortened_case()?;
    let mut batch = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?
        .trial_with_forcing_to(1.0, &forcing)?
        .into_next();
    let trusted = batch.snapshot();
    let owned = batch.snapshot_data()?;
    assert_eq!(trusted.data()?, owned);
    let end = batch.time() + 0.03125;
    let expected = batch
        .trial_with_forcing_to(end, &forcing)?
        .into_next()
        .snapshot_data()?;
    batch = batch.trial_with_forcing_to(end, &forcing)?.into_next();
    batch.restore(&trusted)?;
    assert_eq!(batch.snapshot_data()?, owned);
    assert_eq!(
        batch
            .trial_with_forcing_to(end, &forcing)?
            .into_next()
            .snapshot_data()?,
        expected
    );
    let mut cold = Batch::new(crate::runtime::client(), setup, initial)?;
    cold.restore_data(&owned)?;
    assert_eq!(cold.snapshot_data()?, owned);
    let next = cold.trial_with_forcing_to(end, &forcing)?.into_next();
    assert_eq!(next.snapshot_data()?, expected);
    assert_paired(&next)
}

#[test]
fn parent_forcing_pairing_epochs_and_child_continuity_reject_atomically() -> Result {
    let (setup, initial) = exact_case()?;
    let forcing = forcing(1, 0.25, 0.0);
    let mut batch = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(0.5, &forcing)?
        .into_next();
    let original = batch.snapshot_data()?;
    for mode in 0..9 {
        let mut altered = original.clone();
        match mode {
            0 => {
                altered
                    .last_step
                    .as_mut()
                    .ok_or("Missing common receipt")?
                    .forcing
                    .subcatchments[0]
                    .rain_rate_m_per_s = 0.5
            }
            1 => {
                altered
                    .last_step
                    .as_mut()
                    .ok_or("Missing common receipt")?
                    .outlets[0]
                    .boundary_id = "other".into()
            }
            2 => altered.initial_time_s = 0.125,
            3 => altered.runoff.accepted_steps += 1,
            4 => altered.runoff.time = 0.25,
            5 => altered.last_step = None,
            6 => {
                altered
                    .last_step
                    .as_mut()
                    .ok_or("Missing common receipt")?
                    .end_s = 0.25
            }
            7 => {
                altered
                    .pools
                    .last_step
                    .as_mut()
                    .ok_or("Missing pool receipt")?
                    .external_inflows[0]
                    .volume_m3 += 0.25
            }
            8 => {
                altered
                    .last_step
                    .as_mut()
                    .ok_or("Missing common receipt")?
                    .balance
                    .internal_outlet_volume_m3 += 0.25
            }
            _ => return Err("Unknown common checkpoint mutation".into()),
        }
        assert!(
            batch.restore_data(&altered).is_err(),
            "Accepted common checkpoint mutation={mode}"
        );
        assert_eq!(batch.snapshot_data()?, original);
    }
    Ok(())
}

#[test]
fn valid_native_history_cannot_hide_unpaired_exports_or_external_runon() -> Result {
    for mode in 0..3 {
        let (mut setup, initial) = exact_case()?;
        setup.config.mass_absolute_m3 = 1e6;
        let forcing = forcing(1, 0.25, 0.0);
        let mut batch = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?;
        for end in [0.25, 0.5] {
            batch = batch.trial_with_forcing_to(end, &forcing)?.into_next();
        }
        let original = batch.snapshot_data()?;
        let mut native = SubcatchmentBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            setup.subcatchments,
            initial.subcatchments,
            initial.time_s,
            setup.runoff_config,
        )?;
        for (index, end) in [0.25, 0.5].into_iter().enumerate() {
            let mut input = SubcatchmentInput {
                rain_rate_m_per_s: 0.25,
                potential_evaporation_m_per_s: 0.0,
                external_runon_volumes_m3: [0.0; 3],
            };
            if mode == 0 && index == 0 {
                input.rain_rate_m_per_s = 0.5;
            }
            if (mode == 1 && index == 0) || (mode == 2 && index == 1) {
                input.external_runon_volumes_m3[0] = 0.125;
            }
            native = native.trial_to(end, &[input])?.into_next();
        }
        let mut altered = original.clone();
        let native_data = native.snapshot_data()?;
        native.restore_data(&native_data)?;
        altered.runoff = native_data;
        assert!(
            batch.restore_data(&altered).is_err(),
            "Accepted incompatible native history mode={mode}"
        );
        assert_eq!(batch.snapshot_data()?, original);
    }
    Ok(())
}

#[test]
fn trusted_and_owned_restore_require_the_destination_initial_epoch() -> Result {
    let (setup, initial) = exact_case()?;
    let source = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?;
    let trusted = source.snapshot();
    let owned = source.snapshot_data()?;
    for epoch in [0.125, -0.0] {
        let mut different = initial.clone();
        different.time_s = epoch;
        let mut destination = Batch::new(crate::runtime::client(), setup.clone(), different)?;
        let before = destination.snapshot_data()?;
        assert!(destination.restore(&trusted).is_err());
        assert!(destination.restore_data(&owned).is_err());
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn independently_admitted_rate_receipts_cannot_replace_paired_native_volume_inputs() -> Result {
    let (setup, initial) = exact_case()?;
    let mut batch = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(0.5, &forcing(1, 0.25, 0.0))?
        .into_next();
    let original = batch.snapshot_data()?;
    let mut altered = original.clone();
    let receipt = altered
        .pools
        .last_step
        .as_mut()
        .ok_or("Missing paired pool receipt")?;
    let volume = receipt.external_inflows[0].volume_m3;
    let dt = receipt.external_inflows[0].dt_s;
    let rate_m3_per_s = volume / dt;
    assert_eq!((rate_m3_per_s * dt).to_bits(), volume.to_bits());
    let equivalent = CoupledPoolInflowValue::Rate { rate_m3_per_s };
    receipt.external_inflows[0].input = equivalent;
    receipt
        .forcing_interval
        .as_mut()
        .ok_or("Missing paired forcing")?
        .inflows[0]
        .value = equivalent;

    let mut child = batch.pools().clone();
    child.restore_data(&altered.pools)?;
    assert_eq!(child.snapshot_data()?, altered.pools);
    assert!(batch.restore_data(&altered).is_err());
    assert_eq!(batch.snapshot_data()?, original);
    Ok(())
}
