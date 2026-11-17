use super::super::{
    ChannelError, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
    ChannelReach, ChannelTensorBoundary,
    surveyed_update::tests::{TestResult, cpu, moving_reach, walls},
};
use crate::forcing::WaterAmount;
use numr::runtime::cpu::CpuRuntime;

/// A fast inflow into one cell drives it into the left wall.
#[test]
fn candidate_speed_entry_rejection_shortens_the_interval() -> TestResult {
    // The interval stays inside the accepted-start CFL limit.
    let (rate, velocity, end) = (20.0, -50.0, 0.3);
    // The full-interval candidate holds V = 10 + rate·end and M = rate·end·velocity.
    let full = moving_reach::<CpuRuntime>(cpu(), &[16.0], &[-300.0], &[0.0])?;
    let wall = ChannelTensorBoundary::Wall;
    let fields = full
        .context
        .fields(&full.accepted.volume, &full.accepted.momentum)?;
    let error = full
        .context
        .candidate_speed(&fields, &wall, &wall)
        .err()
        .ok_or("the full-interval candidate must reject")?;
    assert!(
        matches!(&error, ChannelError::Numerical { context, .. }
            if context.contains("surveyed reach Riemann")),
        "{error}"
    );
    let rest = moving_reach::<CpuRuntime>(cpu(), &[10.0], &[0.0], &[0.0])?;
    let mut definition = rest.definition().clone();
    definition.lateral_bindings = vec![ChannelLateralBinding {
        boundary_id: "in".into(),
        cell_id: "cell-0".into(),
        direction: ChannelLateralDirection::Inflow,
        ownership_assertion: "Synthetic fast inflow".into(),
    }];
    let origin =
        ChannelReach::<CpuRuntime>::new(cpu(), definition, rest.context.original_initial.clone())?;
    let mut forcing = walls(0.0, end);
    forcing.lateral_inputs = vec![ChannelLateralInput::Inflow {
        boundary_id: "in".into(),
        amount: WaterAmount::Rate {
            rate_m3_per_s: rate,
        },
        longitudinal_velocity_m_per_s: velocity,
    }];
    let next = origin.step(end, &forcing)?.next;
    let receipt = next.last_step()?.ok_or("missing accepted interval")?;
    assert!(receipt.retries > 0);
    assert!(next.time() > 0.0 && next.time() < end);
    assert!(
        receipt.accepted_start_cfl_limit_s > end,
        "start CFL={}, end={}, retries={}",
        receipt.accepted_start_cfl_limit_s,
        next.time(),
        receipt.retries
    );
    Ok(())
}
