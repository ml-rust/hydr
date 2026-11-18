use super::{
    CoupledPoolInflow, CoupledPoolInflowBinding, CoupledPoolInflowLedger, CoupledPoolInflowValue,
};
use crate::storage::{LevelPoolCurve, LevelPoolElement, LevelPoolKnot};
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};

pub(super) fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), Some(64)))
}
pub(super) fn pools(
    capacity: f64,
) -> Result<Vec<LevelPoolElement>, crate::storage::LevelPoolError> {
    (0..2)
        .map(|index| {
            Ok(LevelPoolElement {
                id: format!("pool-{index}"),
                curve: LevelPoolCurve::new(
                    vec![
                        LevelPoolKnot {
                            volume_m3: 0.0,
                            stage_m: 0.0,
                        },
                        LevelPoolKnot {
                            volume_m3: capacity,
                            stage_m: capacity,
                        },
                    ],
                    "analytic datum".into(),
                    "unit area".into(),
                )?,
                connected_pool_assertion: "one connected level pool".into(),
                storage_support_identity: format!("storage-{index}"),
            })
        })
        .collect()
}
pub(super) fn binding(boundary: &str) -> CoupledPoolInflowBinding {
    CoupledPoolInflowBinding {
        boundary_id: boundary.into(),
        pool_id: "pool-0".into(),
        input_id: format!("input-{boundary}"),
        physical_support_id: "shared receiving aperture".into(),
        boundary_assertion: "prescribed external flow enters pool-0".into(),
        ownership_assertion:
            "independent selected input excludes represented water and local rainfall".into(),
    }
}
pub(super) fn rate(boundary: &str, value: f64) -> CoupledPoolInflow {
    CoupledPoolInflow {
        boundary_id: boundary.into(),
        input_id: format!("input-{boundary}"),
        value: CoupledPoolInflowValue::Rate {
            rate_m3_per_s: value,
        },
    }
}
pub(super) fn ledger(value: f64, bound: f64) -> CoupledPoolInflowLedger {
    CoupledPoolInflowLedger {
        boundary_id: "boundary".into(),
        input_id: "input-boundary".into(),
        cumulative_volume_m3: value,
        roundoff_bound_m3: bound,
    }
}
pub(super) fn volume(boundary: &str, value: f64) -> CoupledPoolInflow {
    CoupledPoolInflow {
        boundary_id: boundary.into(),
        input_id: format!("input-{boundary}"),
        value: CoupledPoolInflowValue::IntervalVolume { volume_m3: value },
    }
}
