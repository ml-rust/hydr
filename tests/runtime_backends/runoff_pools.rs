use super::{Result, coupled_pool_fixture, runtime};

fn exact_case() -> Result<(
    hydr::coupling::RunoffPoolSetup,
    hydr::coupling::RunoffPoolInitialState,
)> {
    use hydr::coupling::*;
    use hydr::runoff::*;
    use hydr::structures::*;

    let element = StructureElement {
        id: "inactive".into(),
        endpoint_a: "a".into(),
        endpoint_b: "b".into(),
        datum: coupled_pool_fixture::DATUM.into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            1.0,
            16.0,
            1.0,
            WeirApplicability {
                coefficient_provenance: "Declared synthetic inactive high crest".into(),
                free_nappe_and_approach_assertion:
                    "Quiescent synthetic approaches remain below the crest".into(),
                submergence_correction_assertion:
                    "Existing weir correction applies within its declared support".into(),
                symmetric_reversal: true,
            },
        )?),
    };
    let (mut pools, initial) = coupled_pool_fixture::assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[0.0, 1.0],
        vec![element],
        vec![StructureOperation {
            structure_id: "inactive".into(),
            operating_point: StructureOperatingPoint::Passive,
        }],
    )?;
    pools.config.maximum_step_s = 2.0;
    pools.external_inflows = vec![CoupledPoolInflowBinding {
        boundary_id: "native-outlet".into(), pool_id: "a".into(), input_id: "native-zero-storage-export".into(),
        physical_support_id: "separate-impervious-area".into(),
        boundary_assertion: "Declared native free outlet enters the dry synthetic pool".into(),
        ownership_assertion: "Native runoff excludes represented pool storage and pool rainfall without duplicate inputs".into(),
    }];
    let parameters = SubcatchmentParameters::new(
        SubcatchmentGeometry {
            total_area_m2: 2.0,
            width_m: 1.0,
            slope: 0.01,
            impervious_fraction: 1.0,
            zero_storage_impervious_fraction: 1.0,
            impervious_manning: 0.0,
            pervious_manning: 0.0,
            impervious_depression_m: 0.0,
            pervious_depression_m: 0.0,
        },
        None,
        SubcatchmentRouting::Direct,
    )?;
    let state = SubcatchmentState::initialized(&parameters)?;
    Ok((RunoffPoolSetup {
        subcatchments: vec![SubcatchmentDefinition { id: "catchment".into(), parameters }],
        runoff_config: SubcatchmentConfig::default(), pools,
        outlets: vec![RunoffPoolOutletBinding { subcatchment_id: "catchment".into(),
            subarea: Subarea::ImperviousZeroStorage, boundary_id: "native-outlet".into(),
            free_outlet_assertion: "Free export excludes downstream backwater and hydraulic feedback".into(),
            nonoverlap_assertion: "Separate runoff area excludes pool storage and rainfall, external runon, and disposal routes".into() }],
        config: RunoffPoolConfig { minimum_step_s: 1e-8, maximum_step_s: 2.0,
            retry_limit: 16, mass_absolute_m3: 1e-9, mass_relative: 1e-12 },
    }, RunoffPoolInitialState { time_s: 0.0, subcatchments: vec![state], pools: initial.states }))
}

macro_rules! native_runoff_execution {
    ($runtime:ty, $client:expr) => {{
        use hydr::coupling::{
            RunoffPoolBatch, RunoffPoolError, RunoffPoolForcingInterval, RunoffPoolMeteorology,
        };

        let client = $client;
        let (setup, initial) = exact_case()?;
        let forcing = RunoffPoolForcingInterval {
            start_s: 0.0,
            end_s: 4.0,
            subcatchments: vec![RunoffPoolMeteorology {
                rain_rate_m_per_s: 0.25,
                potential_evaporation_m_per_s: 0.0,
            }],
            pool_sources: None,
        };
        let batch =
            RunoffPoolBatch::<$runtime>::new(client.clone(), setup.clone(), initial.clone())?;
        let original = batch.snapshot_data()?;
        let mut next = batch.trial_with_forcing_to(2.0, &forcing)?.into_next();
        assert_eq!(batch.snapshot_data()?, original);
        assert_eq!(next.time(), 2.0);
        assert_eq!(
            next.pools().pools().states()?[0].volume_m3.to_bits(),
            1.0_f64.to_bits()
        );
        let native = next
            .runoff()
            .last_step()?
            .ok_or("Missing runtime native receipt")?;
        let exported = native.receipts[0].subareas[0]
            .as_ref()
            .ok_or("Missing runtime native outlet")?
            .partition
            .outlet_volume_m3;
        let imported = next
            .pools()
            .last_step()?
            .ok_or("Missing runtime pool receipt")?
            .external_inflows[0]
            .volume_m3;
        assert_eq!(exported.to_bits(), 1.0_f64.to_bits());
        assert_eq!(exported.to_bits(), imported.to_bits());
        let trusted = next.snapshot();
        let owned = trusted.data()?;
        next = next.trial_with_forcing_to(3.0, &forcing)?.into_next();
        let expected = next.snapshot_data()?;
        next.restore(&trusted)?;
        assert_eq!(next.snapshot_data()?, owned);
        assert_eq!(
            next.trial_with_forcing_to(3.0, &forcing)?
                .into_next()
                .snapshot_data()?,
            expected
        );
        let (mut rejected_setup, rejected_initial) = exact_case()?;
        rejected_setup.config.retry_limit = 1;
        let rejected =
            RunoffPoolBatch::<$runtime>::new(client.clone(), rejected_setup, rejected_initial)?;
        let before_rejection = rejected.snapshot_data()?;
        let mut unrepresentable = forcing.clone();
        unrepresentable.subcatchments[0].rain_rate_m_per_s = 1e308;
        let error = rejected
            .trial_with_forcing_to(2.0, &unrepresentable)
            .err()
            .ok_or("Unrepresentable native forcing was accepted")?;
        match error {
            RunoffPoolError::Unresolved {
                retries,
                dt_s,
                reason,
            } => {
                assert_eq!(retries, 1);
                assert_eq!(dt_s, 1.0);
                assert!(reason.contains("subcatchment arithmetic"), "{reason}");
                assert!(reason.contains("catchment=Some(0)"), "{reason}");
                assert!(
                    reason.contains("subarea=Some(ImperviousZeroStorage)"),
                    "{reason}"
                );
            }
            other => return Err(format!("Unexpected native forcing rejection: {other}").into()),
        }
        assert_eq!(rejected.snapshot_data()?, before_rejection);

        let mut cold = RunoffPoolBatch::<$runtime>::new(client, setup, initial)?;
        cold.restore_data(&owned)?;
        assert_eq!(cold.snapshot_data()?, owned);
        assert_eq!(
            cold.trial_with_forcing_to(3.0, &forcing)?
                .into_next()
                .snapshot_data()?,
            expected
        );
    }};
}

#[test]
fn selected_cpu_client_pairs_native_runoff_and_preserves_complete_restart() -> Result {
    native_runoff_execution!(runtime::Runtime, runtime::client());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn selected_cuda_client_pairs_native_runoff_and_preserves_complete_restart() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    native_runoff_execution!(CudaRuntime, CudaClient::new(CudaDevice::new(0))?);
    Ok(())
}
