use super::super::{
    ChannelBoundary, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
    ChannelReach,
    surveyed_update::tests::{TestResult, cpu, moving_reach},
};
use super::{ChannelCouplingTrial, SURFACE_EXCHANGE_BINDING_PREFIX};
use crate::forcing::WaterAmount;
use numr::runtime::cpu::CpuRuntime;

/// Coordinated surveyed reach with one reserved withdrawal binding at the middle cell.
fn coordinated() -> TestResult<ChannelReach<CpuRuntime>> {
    let standalone = moving_reach::<CpuRuntime>(cpu(), &[10.0, 6.0, 0.0], &[0.0; 3], &[0.0; 3])?;
    let mut definition = standalone.definition().clone();
    definition.lateral_bindings = vec![ChannelLateralBinding {
        boundary_id: format!("{SURFACE_EXCHANGE_BINDING_PREFIX}bank:withdrawal"),
        cell_id: "cell-1".into(),
        direction: ChannelLateralDirection::Withdrawal,
        ownership_assertion: "coordinator surface exchange".into(),
    }];
    Ok(ChannelReach::new_coordinated(
        cpu(),
        definition,
        standalone.context.original_initial.clone(),
    )?)
}

fn withdrawal(rate_m3_per_s: f64) -> ChannelLateralInput {
    ChannelLateralInput::Withdrawal {
        boundary_id: format!("{SURFACE_EXCHANGE_BINDING_PREFIX}bank:withdrawal"),
        amount: WaterAmount::Rate { rate_m3_per_s },
    }
}

#[test]
fn surveyed_lateral_only_coordination_advances() -> TestResult {
    let reach = coordinated()?;
    let wall = ChannelBoundary::Wall;
    let exterior = ChannelBoundary::Exterior {
        depth_m: 0.0,
        velocity_m_per_s: 0.0,
    };
    let inputs = [withdrawal(0.5)];
    let forcing = reach.bind_coupling_host(0.0, 0.05, Some(&wall), Some(&exterior), &inputs)?;
    let reservation = reach.coupling_reservation(&forcing)?;
    let donor = reservation.donor_volume_m3.try_to_vec::<f64>()?;
    let stored = reach
        .coupling_accounting()
        .cell_volume_m3
        .try_to_vec::<f64>()?;
    assert!(donor.iter().zip(&stored).all(|(d, s)| d <= s && *d >= 0.0));
    assert!(reservation.start_cfl_s > 0.0);

    // Stage is the section minimum plus the certified recovered depth.
    let (stage, wet) = reach.coupling_cell_stages(&[2, 0, 1])?;
    let stage = stage.try_to_vec::<f64>()?;
    let wet = wet.try_to_vec::<f64>()?;
    let state = reach.state()?;
    for (slot, cell) in [2, 0, 1].into_iter().enumerate() {
        let result = &state[cell];
        assert_eq!(wet[slot], if result.dry { 0.0 } else { 1.0 });
        let expected = result.stage_m.unwrap_or(0.0);
        assert_eq!(stage[slot].to_bits(), expected.to_bits(), "cell={cell}");
    }

    let next = match reach.trial_coupled(0.05, &forcing, None, None)? {
        ChannelCouplingTrial::Accepted(next) => next,
        ChannelCouplingTrial::Shorter { reason, .. } => {
            return Err(format!("a short surveyed interval must accept: {reason}").into());
        }
    };
    let ledger = next.lateral_ledgers()?;
    assert_eq!(ledger.len(), 1);
    assert!(next.ledger().lateral_out_m3 > 0.0);
    Ok(())
}

#[test]
fn surveyed_connected_or_scalar_endpoint_rejects() -> TestResult {
    let reach = coordinated()?;
    let wall = ChannelBoundary::Wall;
    let stage = ChannelBoundary::Stage { elevation_m: 1.0 };
    let inputs = [withdrawal(0.0)];
    for (left, right, names) in [
        (None, Some(&wall), "left=connected, right=Wall"),
        (Some(&wall), None, "left=Wall, right=connected"),
        (Some(&stage), Some(&wall), "left=Stage, right=Wall"),
    ] {
        let error = reach
            .bind_coupling_host(0.0, 0.05, left, right, &inputs)
            .err()
            .ok_or("a surveyed reach without external Wall or Exterior endpoints must reject")?;
        let message = error.to_string();
        assert!(
            message.contains("surveyed coordinator endpoints"),
            "{message}"
        );
        assert!(message.contains(names), "{message}");
        assert!(message.contains("lateral bindings only"), "{message}");
    }
    Ok(())
}
