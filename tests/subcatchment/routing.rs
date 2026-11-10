use super::fixture::{Result, geometry, single};
use hydr::runoff::{
    Subarea, SubcatchmentConfig, SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting,
    SubcatchmentState,
};
use hydr::soil::{ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, SoilInput};

#[test]
fn both_routing_directions_credit_identical_interface_volumes_with_unequal_areas() -> Result {
    for impervious_donors in [true, false] {
        for fraction in [0.0, 0.5, 1.0] {
            let mut g = geometry(0.5, 0.5);
            g.impervious_manning = 0.0;
            g.pervious_manning = 0.0;
            let conductivity = if impervious_donors {
                1.0 / 1024.0
            } else {
                1.0 / 2048.0
            };
            let soil = ModifiedGreenAmptParameters::from_native(0.125, conductivity, 0.0)?;
            let routing = if impervious_donors {
                SubcatchmentRouting::ImperviousToPervious { fraction }
            } else {
                SubcatchmentRouting::PerviousToImpervious { fraction }
            };
            let p = SubcatchmentParameters::new(g, Some(soil), routing)?;
            let mut state = SubcatchmentState::initialized(&p)?;
            state.depths_m = [0.0, 0.125 * 0.3048, 0.125 * 0.3048];
            let batch = single(p, state, SubcatchmentConfig::default())?;
            let trial = batch.trial_to(
                64.0,
                &[SubcatchmentInput {
                    rain_rate_m_per_s: 0.3048 / 1024.0,
                    potential_evaporation_m_per_s: 0.0,
                    external_runon_volumes_m3: [0.0; 3],
                }],
            )?;
            let step = trial.diagnostics()?;
            assert_eq!(
                step.transfers.len(),
                if fraction == 0.0 {
                    0
                } else if impervious_donors {
                    2
                } else {
                    1
                }
            );
            for transfer in &step.transfers {
                let catchment = step
                    .receipts
                    .get(transfer.catchment)
                    .ok_or("missing transfer catchment")?;
                let donor = catchment.subareas[transfer.donor.index()]
                    .as_ref()
                    .ok_or("missing donor receipt")?;
                let receiver = catchment.subareas[transfer.receiver.index()]
                    .as_ref()
                    .ok_or("missing receiver receipt")?;
                assert_eq!(
                    transfer.volume_m3.to_bits(),
                    donor.partition.routed_volume_m3.to_bits()
                );
                assert_eq!(
                    transfer.volume_m3.to_bits(),
                    receiver.received_by_donor_m3[transfer.donor.index()].to_bits()
                );
                assert_eq!(
                    transfer.donor_volume_m3.to_bits(),
                    donor.drainage.export_volume_m3.to_bits()
                );
                assert_eq!(
                    transfer.outlet_volume_m3.to_bits(),
                    donor.partition.outlet_volume_m3.to_bits()
                );
                assert_eq!(
                    (transfer.volume_m3 + transfer.outlet_volume_m3).to_bits(),
                    transfer.donor_volume_m3.to_bits()
                );
                if fraction > 0.0 && fraction < 1.0 {
                    assert!(transfer.volume_m3 > 0.0 && transfer.outlet_volume_m3 > 0.0);
                }
            }
            assert!(step.water_residual_m3.abs() <= 1e-9 + 1e-12 * step.water_scale_m3);
        }
    }
    Ok(())
}

#[test]
fn routed_receiver_soil_replays_the_exact_constructed_input_sequence() -> Result {
    let p = super::fixture::parameters(
        0.5,
        0.5,
        SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
    )?;
    let soil = p.soil_parameters().ok_or("missing pervious parameters")?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m = [0.125 * 0.3048, 0.25 * 0.3048, 0.0];
    let native = state.soil.ok_or("missing native antecedent")?;
    let mut independent = ModifiedGreenAmptBatch::<crate::runtime::Runtime>::from_native_states(
        crate::runtime::client(),
        &[soil],
        &[native],
    )?;
    let mut batch = single(p, state, SubcatchmentConfig::default())?;
    for end in [64.0, 128.0, 192.0, 256.0] {
        let trial = batch.trial_to(
            end,
            &[SubcatchmentInput {
                rain_rate_m_per_s: 0.3048 / 1024.0,
                potential_evaporation_m_per_s: 0.0,
                external_runon_volumes_m3: [0.0; 3],
            }],
        )?;
        let step = trial.diagnostics()?;
        let receiver = step.receipts[0].subareas[Subarea::Pervious.index()]
            .as_ref()
            .ok_or("missing pervious receiver")?;
        let raw = independent.trial(
            step.dt,
            &[SoilInput {
                incoming_rate_m_per_s: receiver.bridge.total_rate_m_per_s,
                ponded_depth_m: receiver.old_depth_m,
            }],
        )?;
        let expected_output = raw.outputs()?[0];
        let receipt = receiver
            .soil
            .as_ref()
            .ok_or("missing native soil receipt")?;
        assert_eq!(
            receipt.native_rate_ft_per_s.to_bits(),
            expected_output.native_rate_ft_per_s.to_bits()
        );
        assert_eq!(
            receiver.infiltration_depth_m.to_bits(),
            expected_output.depth_m.to_bits()
        );
        independent = raw.into_next();
        batch = trial.into_next();
        assert_eq!(
            batch.states()?[0].soil,
            Some(independent.native_states()?[0])
        );
    }
    Ok(())
}
