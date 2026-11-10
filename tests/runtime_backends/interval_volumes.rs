use super::{Result, coupled_pool_fixture, inflow_fixture, runtime};

macro_rules! interval_volume_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::coupling::{CoupledLevelPoolBatch, CoupledPoolInflowValue};

        let client = $client;
        let (mut setup, initial) = coupled_pool_fixture::orifice_case([4.0, 2.0])?;
        setup.external_inflows = vec![
            inflow_fixture::binding("integrated", "a"),
            inflow_fixture::binding("rate", "b"),
        ];
        let forcing = inflow_fixture::span(
            0.0,
            0.25,
            vec![
                inflow_fixture::volume("integrated", 0.0625),
                inflow_fixture::rate("rate", 0.25),
            ],
        );
        let fresh = inflow_fixture::span(
            0.25,
            0.5,
            vec![
                inflow_fixture::volume("integrated", 0.0625),
                inflow_fixture::rate("rate", 0.25),
            ],
        );
        let batch =
            CoupledLevelPoolBatch::<$runtime>::new(client.clone(), setup.clone(), initial.clone())?;
        let original = batch.snapshot_data()?;
        let mut next = batch.trial_with_forcing_to(0.25, &forcing)?.into_next();
        assert_eq!(batch.snapshot_data()?, original);
        assert_eq!(next.time(), 0.25);
        let dt: f64 = 0.25;
        let transfer = dt * 0.5 * ((dt * dt + 8.0).sqrt() - dt);
        let states = next.pools().states()?;
        coupled_pool_fixture::near(states[0].volume_m3, 4.0625 - transfer);
        coupled_pool_fixture::near(states[1].volume_m3, 2.0625 + transfer);
        let step = next
            .last_step()?
            .ok_or("Missing runtime integrated receipt")?;
        assert_eq!(step.retries, 0);
        assert_eq!(
            step.external_inflows[0].input,
            CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.0625 }
        );
        let checkpoint = next.snapshot();
        let owned = checkpoint.data()?;
        next = next.trial_with_forcing_to(0.5, &fresh)?.into_next();
        let expected = next.snapshot_data()?;
        coupled_pool_fixture::near(next.pools().totals()?.volume_m3, 6.25);
        next.restore(&checkpoint)?;
        assert_eq!(next.snapshot_data()?, owned);
        assert_eq!(
            next.trial_with_forcing_to(0.5, &fresh)?
                .into_next()
                .snapshot_data()?,
            expected
        );
        let mut cold = CoupledLevelPoolBatch::<$runtime>::new(client, setup, initial)?;
        cold.restore_data(&owned)?;
        assert_eq!(cold.snapshot_data()?, owned);
        assert_eq!(
            cold.trial_with_forcing_to(0.5, &fresh)?
                .into_next()
                .snapshot_data()?,
            expected
        );
    }};
}

#[test]
fn selected_cpu_client_executes_mixed_interval_volumes_and_complete_restart() -> Result {
    interval_volume_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_mixed_interval_volumes_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    interval_volume_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}
