//! Pool stage-volume curve epochs with exact volume conservation.

mod apply;
mod schedule;

pub use schedule::{
    PoolCurveEpoch, PoolCurveSchedule, active_curve_at, active_pool_curve_epochs,
    check_pool_curve_schedules, curve_digest_bits, next_pool_curve_event_after,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{
        LevelPoolBatch, LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolError,
        LevelPoolKnot, LevelPoolState,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn curve(max_vol: f64, max_stage: f64) -> Result<LevelPoolCurve, LevelPoolError> {
        LevelPoolCurve::new(
            vec![
                LevelPoolKnot {
                    volume_m3: 0.0,
                    stage_m: 0.0,
                },
                LevelPoolKnot {
                    volume_m3: max_vol,
                    stage_m: max_stage,
                },
            ],
            "datum".into(),
            "synthetic".into(),
        )
    }

    #[test]
    fn curve_epoch_conserves_volume_and_rejects_above_new_top() -> Result<(), LevelPoolError> {
        let client = CpuClient::new(CpuDevice::new());
        let element = LevelPoolElement {
            id: "pond".into(),
            curve: curve(10.0, 2.0)?,
            connected_pool_assertion: "connected".into(),
            storage_support_identity: "pond-support".into(),
        };
        let smaller = curve(6.0, 2.0)?;
        let schedule = PoolCurveSchedule {
            schedule_id: "silt".into(),
            pool_id: "pond".into(),
            epochs: vec![PoolCurveEpoch {
                time_s: 1.0,
                curve: smaller.clone(),
                evidence: "Siltation survey".into(),
            }],
        };
        check_pool_curve_schedules(
            std::slice::from_ref(&element),
            std::slice::from_ref(&schedule),
        )?;
        assert_eq!(
            next_pool_curve_event_after(std::slice::from_ref(&schedule), 0.0),
            Some(1.0)
        );
        assert_eq!(
            next_pool_curve_event_after(std::slice::from_ref(&schedule), 1.0),
            None
        );
        assert_eq!(
            active_pool_curve_epochs(std::slice::from_ref(&schedule), 0.5),
            vec![0]
        );
        assert_eq!(
            active_pool_curve_epochs(std::slice::from_ref(&schedule), 1.0),
            vec![1]
        );

        // Volume 3.0 fits the new curve [0, 6]: volume stays bit-identical, stage rises 0.6 -> 1.0.
        let batch = LevelPoolBatch::<CpuRuntime>::new(
            client.clone(),
            vec![element.clone()],
            vec![LevelPoolState { volume_m3: 3.0 }],
            0.0,
            LevelPoolConfig::default(),
        )?;
        let updated = batch.coupling_with_curves(std::slice::from_ref(&smaller))?;
        assert_eq!(updated.states()?[0].volume_m3.to_bits(), 3.0_f64.to_bits());
        let stage = updated.stages_tensor()?.try_to_vec::<f64>()?;
        assert_eq!(stage[0], 1.0);

        // Volume 8.0 exceeds the new curve [0, 6]: must reject.
        let full = LevelPoolBatch::<CpuRuntime>::new(
            client,
            vec![element],
            vec![LevelPoolState { volume_m3: 8.0 }],
            0.0,
            LevelPoolConfig::default(),
        )?;
        assert!(full.coupling_with_curves(&[smaller]).is_err());
        Ok(())
    }
}
