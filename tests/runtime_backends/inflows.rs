use super::{Result, coupled_pool_fixture, inflow_fixture, runtime};

macro_rules! external_inflow_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::coupling::CoupledLevelPoolBatch;

        let client = $client;
        let (mut setup, initial) = coupled_pool_fixture::orifice_case([4.0, 2.0])?;
        setup.external_inflows = vec![
            inflow_fixture::binding("left", "a"),
            inflow_fixture::binding("right", "b"),
        ];
        let forcing = inflow_fixture::forcing(vec![
            inflow_fixture::rate("left", 0.25),
            inflow_fixture::rate("right", 0.25),
        ]);
        let batch =
            CoupledLevelPoolBatch::<$runtime>::new(client.clone(), setup.clone(), initial.clone())?;
        let original = batch.snapshot_data()?;
        let mut next = batch.trial_with_forcing_to(0.25, &forcing)?.into_next();
        assert_eq!(batch.snapshot_data()?, original);
        assert_eq!(next.time(), 0.25);
        let dt: f64 = 0.25;
        let transfer = dt * 0.5 * ((dt * dt + 8.0).sqrt() - dt);
        let states = next.pools().states()?;
        coupled_pool_fixture::near(states[0].volume_m3, 4.0 + 0.25 * dt - transfer);
        coupled_pool_fixture::near(states[1].volume_m3, 2.0 + 0.25 * dt + transfer);
        let checkpoint = next.snapshot();
        let checkpoint_data = checkpoint.data()?;
        next = next.trial_with_forcing_to(0.5, &forcing)?.into_next();
        let continued = next.snapshot_data()?;
        coupled_pool_fixture::near(next.pools().totals()?.volume_m3, 6.25);
        let ledgers = next.external_inflows()?;
        coupled_pool_fixture::near(
            ledgers
                .iter()
                .map(|ledger| ledger.cumulative_volume_m3)
                .sum(),
            0.25,
        );
        next.restore(&checkpoint)?;
        assert_eq!(next.snapshot_data()?, checkpoint_data);
        assert_eq!(
            next.trial_with_forcing_to(0.5, &forcing)?
                .into_next()
                .snapshot_data()?,
            continued
        );
        let mut cold = CoupledLevelPoolBatch::<$runtime>::new(client, setup, initial)?;
        cold.restore_data(&checkpoint_data)?;
        assert_eq!(cold.snapshot_data()?, checkpoint_data);
        assert_eq!(
            cold.trial_with_forcing_to(0.5, &forcing)?
                .into_next()
                .snapshot_data()?,
            continued
        );
    }};
}

#[test]
fn selected_cpu_client_executes_external_inflows_and_complete_restart() -> Result {
    external_inflow_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_executes_external_inflows_and_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    external_inflow_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}
